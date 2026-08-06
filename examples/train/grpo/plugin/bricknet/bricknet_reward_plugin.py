"""BrickNet-MM GRPO rewards aligned with BrickNet hard-mining semantics.

The plugin intentionally imports the verifier helpers from the BrickNet checkout so
hard mining and online GRPO use the same parser, inventory definition and pose matcher.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, List

import numpy as np

from swift.rewards import ORM, orms


DEFAULT_BRICKNET_ROOT = Path("/home/jiahao/task/BrickNet")


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean, got {value!r}")


def _bricknet_root() -> Path:
    return Path(os.getenv("BRICKNET_ROOT", str(DEFAULT_BRICKNET_ROOT))).expanduser().resolve()


@lru_cache(maxsize=1)
def _load_rl_utils():
    root = _bricknet_root()
    helper_path = root / "data_preprocess" / "rl_utils.py"
    if not helper_path.is_file():
        raise FileNotFoundError(
            f"BrickNet verifier helper not found: {helper_path}. "
            "Set BRICKNET_ROOT to the BrickNet checkout."
        )
    module_name = "_ms_swift_bricknet_rl_utils"
    spec = importlib.util.spec_from_file_location(module_name, helper_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import BrickNet verifier helper: {helper_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    module.setup_bricknet_import()
    return module


def _normalise_inventory(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"target_inventory must be a list, got {type(value).__name__}")
    return [
        {
            "part_id": int(row["part_id"]),
            "color_code": int(row["color_code"]),
            "count": int(row["count"]),
        }
        for row in value
    ]


def _inventory_key(value: Any) -> str:
    rows = _normalise_inventory(value)
    rows.sort(key=lambda row: (row["part_id"], row["color_code"], row["count"]))
    return json.dumps(rows, separators=(",", ":"), sort_keys=True)


@lru_cache(maxsize=256)
def _parse_target(solution: str):
    utils = _load_rl_utils()
    from bricknet.tree import parse_sample

    parsed = parse_sample(utils.normalize_path_text(solution))
    if parsed.error is not None:
        raise ValueError(f"Invalid BrickNet reference solution: {parsed.error}")
    return parsed.tree


@dataclass(frozen=True)
class BrickNetScore:
    parse_prefix: float
    inventory_f1: float
    length: float
    collision_prefix: float
    pose_match: float
    strict_success: float

    @property
    def dense_reward(self) -> float:
        return (
            0.20 * self.parse_prefix
            + 0.20 * self.inventory_f1
            + 0.10 * self.length
            + 0.20 * self.collision_prefix
            + 0.30 * self.pose_match
        )


@lru_cache(maxsize=512)
def _score_cached(
    completion: str,
    solution: str,
    target_inventory_json: str,
    target_part_count: int,
    collision_check: bool,
    translation_tolerance: float,
    rotation_tolerance: float,
    pose_success_threshold: float,
) -> BrickNetScore:
    utils = _load_rl_utils()
    from bricknet.tree import parse_sample

    text = utils.normalize_path_text(completion)
    parsed = parse_sample(text)
    tree = parsed.tree
    generated_parts = len(tree.parts)

    inventory_rows = json.loads(target_inventory_json)
    target_inventory = utils.inventory_counter_from_rows(inventory_rows)
    predicted_inventory = utils.inventory_counter_from_tree(tree)
    inventory_f1 = utils.multiset_f1(predicted_inventory, target_inventory)
    exact_inventory = predicted_inventory == target_inventory

    if generated_parts or target_part_count:
        length_score = min(generated_parts, target_part_count) / max(generated_parts, target_part_count)
    else:
        length_score = 1.0
    parse_prefix = (
        1.0
        if parsed.error is None
        else min(generated_parts / max(target_part_count, 1), 1.0)
    )

    if collision_check:
        try:
            from bricknet.score import check_tree

            collisions = check_tree(tree)
        except (FileNotFoundError, ModuleNotFoundError) as error:
            raise RuntimeError(
                "BrickNet collision reward is unavailable. Install the local BrickNet package "
                "(including meshlib) in the ms-swift environment and set BRICKNET_DATA to a "
                "directory containing inset/*.ply."
            ) from error
        collision_free = not collisions
        if collisions:
            first_collision = min(collisions)
            collision_prefix = min(first_collision / max(target_part_count, 1), 1.0)
        else:
            collision_prefix = 1.0
    else:
        collision_free = True
        collision_prefix = 1.0

    try:
        pose_match = utils.pose_match_score(
            tree,
            _parse_target(solution),
            translation_tolerance=translation_tolerance,
            rotation_tolerance_degrees=rotation_tolerance,
        )
    except (ValueError, np.linalg.LinAlgError):
        pose_match = 0.0

    strict_success = float(
        parsed.error is None
        and generated_parts == target_part_count
        and exact_inventory
        and collision_free
        and pose_match >= pose_success_threshold
    )
    return BrickNetScore(
        parse_prefix=float(parse_prefix),
        inventory_f1=float(inventory_f1),
        length=float(length_score),
        collision_prefix=float(collision_prefix),
        pose_match=float(pose_match),
        strict_success=strict_success,
    )


def score_completion(
    completion: str,
    solution: str,
    target_inventory: Any,
    target_part_count: int,
) -> BrickNetScore:
    """Score one completion with the hard-mining verifier configuration."""
    return _score_cached(
        str(completion or ""),
        str(solution or ""),
        _inventory_key(target_inventory),
        int(target_part_count),
        _env_bool("BRICKNET_COLLISION_CHECK", True),
        float(os.getenv("BRICKNET_POSE_TRANSLATION_TOLERANCE", "0.5")),
        float(os.getenv("BRICKNET_POSE_ROTATION_TOLERANCE", "5.0")),
        float(os.getenv("BRICKNET_POSE_SUCCESS_THRESHOLD", "1.0")),
    )


def _score_batch(
    completions: Iterable[str],
    solutions: Iterable[str],
    inventories: Iterable[Any],
    part_counts: Iterable[int],
) -> list[BrickNetScore]:
    values = [list(items) for items in (completions, solutions, inventories, part_counts)]
    lengths = {len(items) for items in values}
    if len(lengths) != 1:
        raise ValueError(
            "completions, solution, target_inventory and target_part_count must have equal lengths, "
            f"got {[len(items) for items in values]}"
        )
    return [
        score_completion(completion, solution, inventory, part_count)
        for completion, solution, inventory, part_count in zip(*values)
    ]


class _BrickNetComponentORM(ORM):
    component: str

    def __call__(
        self,
        completions,
        solution,
        target_inventory,
        target_part_count,
        **kwargs,
    ) -> List[float]:
        scores = _score_batch(completions, solution, target_inventory, target_part_count)
        return [float(getattr(score, self.component)) for score in scores]


class BrickNetParsePrefixORM(_BrickNetComponentORM):
    component = "parse_prefix"


class BrickNetInventoryF1ORM(_BrickNetComponentORM):
    component = "inventory_f1"


class BrickNetLengthORM(_BrickNetComponentORM):
    component = "length"


class BrickNetCollisionPrefixORM(_BrickNetComponentORM):
    component = "collision_prefix"


class BrickNetPoseMatchORM(_BrickNetComponentORM):
    component = "pose_match"


class BrickNetStrictSuccessORM(_BrickNetComponentORM):
    component = "strict_success"


class BrickNetDenseRewardORM(ORM):
    """Single-function equivalent of the five configured component rewards."""

    def __call__(
        self,
        completions,
        solution,
        target_inventory,
        target_part_count,
        **kwargs,
    ) -> List[float]:
        scores = _score_batch(completions, solution, target_inventory, target_part_count)
        return [score.dense_reward for score in scores]


orms["bricknet_parse_prefix"] = BrickNetParsePrefixORM
orms["bricknet_inventory_f1"] = BrickNetInventoryF1ORM
orms["bricknet_length"] = BrickNetLengthORM
orms["bricknet_collision_prefix"] = BrickNetCollisionPrefixORM
orms["bricknet_pose_match"] = BrickNetPoseMatchORM
orms["bricknet_strict_success"] = BrickNetStrictSuccessORM
orms["bricknet_dense_reward"] = BrickNetDenseRewardORM
