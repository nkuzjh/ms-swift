#!/usr/bin/env python3
"""Smoke-test BrickNet reward registration and perfect-reference scoring."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/BrickNet-MM-RL_n2000_seed42.jsonl"),
    )
    parser.add_argument(
        "--plugin",
        type=Path,
        default=Path(__file__).with_name("bricknet_reward_plugin.py"),
    )
    return parser.parse_args()


def load_plugin(path: Path):
    spec = importlib.util.spec_from_file_location("bricknet_reward_plugin", path.resolve())
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    args = parse_args()
    with args.dataset.open("r", encoding="utf-8") as handle:
        row = json.loads(next(handle))
    plugin = load_plugin(args.plugin)
    score = plugin.score_completion(
        row["solution"],
        row["solution"],
        row["target_inventory"],
        row["target_part_count"],
    )
    values = {
        "parse_prefix": score.parse_prefix,
        "inventory_f1": score.inventory_f1,
        "length": score.length,
        "collision_prefix": score.collision_prefix,
        "pose_match": score.pose_match,
        "strict_success": score.strict_success,
        "dense_reward": score.dense_reward,
    }
    print(json.dumps(values, indent=2))
    if any(abs(value - 1.0) > 1e-8 for value in values.values()):
        raise SystemExit("Perfect reference must score 1.0 for every component.")


if __name__ == "__main__":
    main()
