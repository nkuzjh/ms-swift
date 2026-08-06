#!/usr/bin/env python3
"""Run reproducible, end-to-end BrickNet-MM evaluation for an ms-swift checkpoint.

The public ``run`` command performs these stages:

1. Batch inference over a labeled BrickNet-MM validation JSONL.
2. LlamaFactory-compatible BLEU-4 and ROUGE-1/2/L text metrics.
3. The upstream BrickNet parse/collision/LDR/render/PE/SigLIP2/VQAScore pipeline.
4. BrickNet-MM alignment metrics used by GRPO hard mining and online rewards.

All inputs and sampling parameters are recorded in ``run_manifest.json``. Completed
stages are reused only when their row counts and input identity match, which makes the
script suitable for later SFT/RL experiments as well as exp0.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean
from typing import Any, Iterable


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[5]
DEFAULT_BRICKNET_ROOT = Path("/home/jiahao/task/BrickNet")
DEFAULT_VAL_DATASET = (
    DEFAULT_BRICKNET_ROOT
    / "outputs_preprocess/BrickNet-MM/sharegpt/BrickNet-MM_VAL.jsonl"
)
DEFAULT_CAPTIONS = DEFAULT_BRICKNET_ROOT / "data/bricknet_datasets/captions_val.jsonl"
DEFAULT_SWIFT_BIN = Path("/home/jiahao/miniconda3/envs/swift/bin/swift")
DEFAULT_LLAMFACTORY_PYTHON = Path("/home/jiahao/miniconda3/envs/llamafactory/bin/python")
DEFAULT_BRICKNET_PYTHON = Path("/home/jiahao/miniconda3/envs/bricknet/bin/python")
ALIGNMENT_SECTION = "## BrickNet-MM Alignment"
CONDITION_SECTION = "## Condition Generation"
CAPTION_RE = re.compile(r"Caption:\n(?P<caption>.*?)\n\nInventory of parts:", re.DOTALL)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def resolve(path: Path) -> Path:
    return path.expanduser().resolve()


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path}: invalid JSON: {exc}") from exc


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, 1):
            if not raw.strip():
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no}: expected a JSON object")
            rows.append(row)
    if not rows:
        raise ValueError(f"{path}: no JSONL rows")
    return rows


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")


def require_executable(path: Path, label: str) -> None:
    if not path.is_file() or not os.access(path, os.X_OK):
        raise FileNotFoundError(f"{label} is not executable: {path}")


def run_command(
    label: str,
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str] | None = None,
    dry_run: bool = False,
) -> float:
    import shlex

    print(f"\n[{label}]\n$ {shlex.join(command)}", flush=True)
    if dry_run:
        return 0.0
    started = time.monotonic()
    result = subprocess.run(command, cwd=cwd, env=env, check=False)
    elapsed = time.monotonic() - started
    if result.returncode:
        raise RuntimeError(f"{label} failed with exit code {result.returncode}")
    print(f"[{label}] completed in {elapsed:.1f}s", flush=True)
    return elapsed


def normalize_path_text(value: Any) -> str:
    text = str(value or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    return text + "\n"


def final_assistant_content(row: dict[str, Any], *, row_index: int) -> str:
    messages = row.get("messages")
    if not isinstance(messages, list):
        raise ValueError(f"dataset row {row_index}: missing messages list")
    assistant = [
        message.get("content")
        for message in messages
        if isinstance(message, dict) and message.get("role") == "assistant"
    ]
    if len(assistant) != 1 or not isinstance(assistant[0], str):
        raise ValueError(
            f"dataset row {row_index}: expected exactly one assistant reference"
        )
    return assistant[0]


def dataset_caption(row: dict[str, Any], *, row_index: int) -> str:
    messages = row.get("messages")
    if not isinstance(messages, list):
        raise ValueError(f"dataset row {row_index}: missing messages")
    users = [
        message.get("content")
        for message in messages
        if isinstance(message, dict) and message.get("role") == "user"
    ]
    if len(users) != 1 or not isinstance(users[0], str):
        raise ValueError(f"dataset row {row_index}: expected one user message")
    match = CAPTION_RE.search(users[0])
    if not match:
        raise ValueError(f"dataset row {row_index}: cannot extract caption")
    return match.group("caption").strip()


def result_label(row: dict[str, Any], *, row_index: int) -> str:
    label = row.get("labels")
    if isinstance(label, str):
        return label
    raise ValueError(f"inference row {row_index}: labels must be a string")


def validate_inputs(
    dataset_rows: list[dict[str, Any]],
    caption_rows: list[dict[str, Any]],
    result_rows: list[dict[str, Any]] | None = None,
) -> None:
    if len(dataset_rows) != len(caption_rows):
        raise ValueError(
            f"dataset/caption count mismatch: {len(dataset_rows)} != {len(caption_rows)}"
        )
    if result_rows is not None and len(result_rows) != len(dataset_rows):
        raise ValueError(
            f"inference result count mismatch: expected {len(dataset_rows)}, "
            f"found {len(result_rows)}"
        )

    ids: set[str] = set()
    for index, (dataset_row, caption_row) in enumerate(
        zip(dataset_rows, caption_rows)
    ):
        sample_id = str(dataset_row.get("id", index))
        if sample_id in ids:
            raise ValueError(f"dataset row {index}: duplicate id {sample_id!r}")
        ids.add(sample_id)
        expected_caption = str(caption_row.get("caption", "")).strip()
        if not expected_caption:
            raise ValueError(f"caption row {index}: missing caption")
        actual_caption = dataset_caption(dataset_row, row_index=index)
        if actual_caption != expected_caption:
            raise ValueError(
                f"caption alignment mismatch at row {index}: "
                f"{actual_caption!r} != {expected_caption!r}"
            )
        if result_rows is not None:
            label = normalize_path_text(result_label(result_rows[index], row_index=index))
            reference = normalize_path_text(
                final_assistant_content(dataset_row, row_index=index)
            )
            if label != reference:
                raise ValueError(
                    f"inference label/reference mismatch at row {index}"
                )
            response = result_rows[index].get("response")
            if not isinstance(response, str):
                raise ValueError(
                    f"inference row {index}: response must be a string"
                )


def infer_image_root(dataset: Path, dataset_rows: list[dict[str, Any]]) -> Path:
    images = dataset_rows[0].get("images")
    if not isinstance(images, list) or not images or not isinstance(images[0], str):
        raise ValueError("first validation row does not contain an image path")
    image = Path(images[0])
    if image.is_absolute():
        require_file(image, "validation image")
        return image.parent
    for parent in (dataset.parent, *dataset.parents):
        candidate = parent / image
        if candidate.is_file():
            return parent
    raise FileNotFoundError(
        f"cannot resolve validation image {image} from ancestors of {dataset}"
    )


def resolve_checkpoint_model(checkpoint: Path, base_model: Path | None) -> tuple[str, Path]:
    if (checkpoint / "adapter_model.safetensors").is_file():
        if base_model is None:
            args_path = checkpoint / "args.json"
            require_file(args_path, "checkpoint args.json")
            model_value = load_json(args_path).get("model")
            if not isinstance(model_value, str) or not model_value:
                raise ValueError(
                    f"{args_path}: cannot infer the adapter base model; pass --base-model"
                )
            candidate = Path(model_value).expanduser()
            base_model = candidate if candidate.is_absolute() else REPO_ROOT / candidate
        base_model = resolve(base_model)
        require_file(base_model / "config.json", "base model config")
        return "adapter", base_model
    require_file(checkpoint / "config.json", "full checkpoint config")
    return "full", checkpoint


def convert_results(
    result_rows: list[dict[str, Any]],
    caption_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    converted = []
    for index, (result, caption) in enumerate(zip(result_rows, caption_rows)):
        converted.append(
            {
                "id": index,
                "sample": 0,
                "source": caption.get("source", caption.get("id", index)),
                "caption": str(caption["caption"]).strip(),
                "text": normalize_path_text(result["response"]),
            }
        )
    return converted


def manifest_identity(args: argparse.Namespace, model_kind: str, model: Path) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "experiment": args.experiment_name,
        "checkpoint": str(resolve(args.checkpoint)) if args.checkpoint else None,
        "checkpoint_adapter_sha256": (
            sha256(resolve(args.checkpoint) / "adapter_model.safetensors")
            if args.checkpoint
            and (resolve(args.checkpoint) / "adapter_model.safetensors").is_file()
            else None
        ),
        "model_kind": model_kind,
        "base_or_full_model": str(model),
        "dataset": str(resolve(args.dataset)),
        "dataset_sha256": sha256(resolve(args.dataset)),
        "captions": str(resolve(args.captions)),
        "captions_sha256": sha256(resolve(args.captions)),
        "supplied_inference_results": (
            str(resolve(args.inference_results)) if args.inference_results else None
        ),
        "supplied_inference_results_sha256": (
            sha256(resolve(args.inference_results)) if args.inference_results else None
        ),
        "sampling": {
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "seed": args.seed,
        },
        "pose": {
            "translation_tolerance": args.pose_translation_tolerance,
            "rotation_tolerance_degrees": args.pose_rotation_tolerance,
            "success_threshold": args.pose_success_threshold,
        },
    }


def validate_run_manifest(
    path: Path, identity: dict[str, Any], *, force: bool
) -> None:
    if path.is_file() and not force:
        existing = load_json(path)
        comparable = {key: existing.get(key) for key in identity}
        if comparable != identity:
            raise RuntimeError(
                f"{path.parent} belongs to a different evaluation configuration; "
                "use a different --output-dir or pass --force"
            )
    write_json(path, {**identity, "status": "running", "updated_at": utc_now()})


def remove_wrapper_outputs(output_dir: Path) -> None:
    for name in (
        "swift_inference.jsonl",
        "predictions.jsonl",
        "text_metrics.json",
        "alignment.jsonl",
        "run_manifest.json",
    ):
        (output_dir / name).unlink(missing_ok=True)


def run_pipeline(args: argparse.Namespace) -> int:
    dataset = resolve(args.dataset)
    captions = resolve(args.captions)
    output_dir = resolve(args.output_dir)
    checkpoint = resolve(args.checkpoint) if args.checkpoint else None
    supplied_results = resolve(args.inference_results) if args.inference_results else None
    base_model = resolve(args.base_model) if args.base_model else None
    bricknet_root = resolve(args.bricknet_root)
    swift_bin = resolve(args.swift_bin)
    llamafactory_python = resolve(args.llamafactory_python)
    bricknet_python = resolve(args.bricknet_python)

    require_file(dataset, "validation dataset")
    require_file(captions, "caption metadata")
    require_file(bricknet_root / "scripts/evaluate_experiment.py", "BrickNet evaluator")
    require_executable(llamafactory_python, "LlamaFactory Python")
    require_executable(bricknet_python, "BrickNet Python")

    if supplied_results is None:
        if checkpoint is None:
            raise ValueError("--checkpoint is required unless --inference-results is supplied")
        require_file(
            checkpoint / "args.json" if (checkpoint / "adapter_model.safetensors").is_file()
            else checkpoint / "config.json",
            "checkpoint metadata",
        )
        require_executable(swift_bin, "swift CLI")
        model_kind, model = resolve_checkpoint_model(checkpoint, base_model)
    else:
        require_file(supplied_results, "inference results")
        model_kind = "external-results"
        model = base_model or Path("<not-applicable>")

    dataset_rows = load_jsonl(dataset)
    caption_rows = load_jsonl(captions)
    validate_inputs(dataset_rows, caption_rows)

    run_manifest = output_dir / "run_manifest.json"
    identity = manifest_identity(args, model_kind, model)
    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
        if args.force:
            remove_wrapper_outputs(output_dir)
        validate_run_manifest(run_manifest, identity, force=args.force)

    inference_results = supplied_results or output_dir / "swift_inference.jsonl"
    inference_seconds = None
    if supplied_results is None and not inference_results.is_file():
        image_root = infer_image_root(dataset, dataset_rows)
        command = [
            str(swift_bin),
            "infer",
            "--val_dataset",
            str(dataset),
            "--model",
            str(model),
        ]
        if model_kind == "adapter":
            command.extend(["--adapters", str(checkpoint)])
        command.extend(
            [
                "--infer_backend",
                "transformers",
                "--stream",
                "false",
                "--val_dataset_shuffle",
                "false",
                "--result_path",
                str(inference_results),
                "--write_batch_size",
                str(args.write_batch_size),
                "--max_batch_size",
                str(args.max_batch_size),
                "--max_length",
                str(args.max_length),
                "--max_new_tokens",
                str(args.max_new_tokens),
                "--temperature",
                str(args.temperature),
                "--top_p",
                str(args.top_p),
                "--top_k",
                str(args.top_k),
                "--seed",
                str(args.seed),
                "--enable_thinking",
                "false",
                "--response_prefix",
                "",
                "--add_non_thinking_prefix",
                "false",
                "--max_pixels",
                str(args.max_pixels),
                "--truncation_strategy",
                "delete",
            ]
        )
        env = os.environ.copy()
        env.update(
            {
                "ROOT_IMAGE_DIR": str(image_root),
                "BRICKNET_ROOT": str(bricknet_root),
                "BRICKNET_DATA": str(resolve(args.bricknet_data)),
                "MAX_PIXELS": str(args.max_pixels),
                "MIN_PIXELS": str(args.min_pixels),
                "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
                "TOKENIZERS_PARALLELISM": "false",
            }
        )
        inference_seconds = run_command(
            "ms-swift batch inference",
            command,
            cwd=REPO_ROOT,
            env=env,
            dry_run=args.dry_run,
        )
    elif supplied_results is None:
        print(f"[reuse] inference results: {inference_results}", flush=True)

    if args.dry_run:
        return 0

    result_rows = load_jsonl(inference_results)
    validate_inputs(dataset_rows, caption_rows, result_rows)

    predictions = output_dir / "predictions.jsonl"
    converted = convert_results(result_rows, caption_rows)
    write_jsonl(predictions, converted)

    text_metrics = output_dir / "text_metrics.json"
    run_command(
        "LlamaFactory-compatible BLEU/ROUGE",
        [
            str(llamafactory_python),
            str(SCRIPT_PATH),
            "text-worker",
            "--results",
            str(inference_results),
            "--output",
            str(text_metrics),
        ],
        cwd=REPO_ROOT,
    )

    bricknet_command = [
        str(bricknet_python),
        str(bricknet_root / "scripts/evaluate_experiment.py"),
        "--predictions",
        str(predictions),
        "--input-format",
        "bricknet",
        "--text-metrics",
        str(text_metrics),
        "--output-dir",
        str(output_dir),
        "--prompts-file",
        str(captions),
        "--score-workers",
        str(args.score_workers),
        "--render-jobs",
        str(args.render_jobs),
        "--eval-workers",
        str(args.eval_workers),
        "--eval-batch-size",
        str(args.eval_batch_size),
    ]
    if args.skip_image_metrics:
        bricknet_command.append("--skip-image-metrics")
    if args.force:
        bricknet_command.append("--force")
    bricknet_env = os.environ.copy()
    bricknet_env.update(
        {
            "BRICKNET_DATA": str(resolve(args.bricknet_data)),
            "PATH": f"/home/jiahao/.local/bin:{bricknet_env.get('PATH', '')}",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    bricknet_seconds = run_command(
        "upstream BrickNet evaluation",
        bricknet_command,
        cwd=bricknet_root,
        env=bricknet_env,
    )

    run_command(
        "BrickNet-MM alignment metrics",
        [
            str(bricknet_python),
            str(SCRIPT_PATH),
            "alignment-worker",
            "--results",
            str(inference_results),
            "--dataset",
            str(dataset),
            "--scored",
            str(output_dir / "scored.jsonl"),
            "--metrics-json",
            str(output_dir / "metrics.json"),
            "--metrics-md",
            str(output_dir / "metrics.md"),
            "--output",
            str(output_dir / "alignment.jsonl"),
            "--bricknet-root",
            str(bricknet_root),
            "--translation-tolerance",
            str(args.pose_translation_tolerance),
            "--rotation-tolerance",
            str(args.pose_rotation_tolerance),
            "--pose-success-threshold",
            str(args.pose_success_threshold),
        ],
        cwd=bricknet_root,
        env=bricknet_env,
    )

    metrics = load_json(output_dir / "metrics.json")
    write_json(
        run_manifest,
        {
            **identity,
            "status": "complete",
            "updated_at": utc_now(),
            "artifacts": {
                "inference_results": str(inference_results),
                "predictions": str(predictions),
                "metrics_json": str(output_dir / "metrics.json"),
                "metrics_md": str(output_dir / "metrics.md"),
                "alignment": str(output_dir / "alignment.jsonl"),
            },
            "runtime_seconds": {
                "inference": inference_seconds,
                "bricknet_pipeline": bricknet_seconds,
            },
            "headline": {
                "samples": metrics["structure"]["samples"],
                "bleu_4": metrics["text_metrics"].get("predict_bleu-4"),
                "rouge_l": metrics["text_metrics"].get("predict_rouge-l"),
                "parsable_rate": metrics["structure"]["fully_parsable_rate"],
                "clean_rate": metrics["structure"][
                    "parsable_and_collision_free_rate"
                ],
                "dense_reward": metrics["task_alignment"]["dense_reward_mean"],
                "strict_success_rate": metrics["task_alignment"][
                    "strict_success_rate"
                ],
            },
        },
    )
    print(f"\nEvaluation complete: {output_dir / 'metrics.md'}", flush=True)
    return 0


def run_text_worker(args: argparse.Namespace) -> int:
    try:
        import jieba
        import numpy as np
        from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
        from rouge_chinese import Rouge
    except ImportError as exc:
        raise RuntimeError(
            "text-worker requires the LlamaFactory metric dependencies "
            "(jieba, nltk, rouge_chinese)"
        ) from exc

    rows = load_jsonl(resolve(args.results))
    score_dict: dict[str, list[float]] = {
        "rouge-1": [],
        "rouge-2": [],
        "rouge-l": [],
        "bleu-4": [],
    }
    for index, row in enumerate(rows):
        pred = str(row.get("response", ""))
        label = result_label(row, row_index=index)
        hypothesis = list(jieba.cut(pred))
        reference = list(jieba.cut(label))
        if len(" ".join(hypothesis).split()) == 0 or len(
            " ".join(reference).split()
        ) == 0:
            rouge_result = {
                "rouge-1": {"f": 0.0},
                "rouge-2": {"f": 0.0},
                "rouge-l": {"f": 0.0},
            }
        else:
            rouge_result = Rouge().get_scores(
                " ".join(hypothesis), " ".join(reference)
            )[0]
        for key, value in rouge_result.items():
            score_dict[key].append(round(float(value["f"]) * 100, 4))
        bleu = sentence_bleu(
            [list(label)],
            list(pred),
            smoothing_function=SmoothingFunction().method3,
        )
        score_dict["bleu-4"].append(round(float(bleu) * 100, 4))

    metrics = {
        f"predict_{key}": float(np.mean(values))
        for key, values in score_dict.items()
    }
    metrics["predict_samples"] = len(rows)
    write_json(resolve(args.output), metrics)
    print(json.dumps(metrics, indent=2))
    return 0


def load_rl_utils(bricknet_root: Path):
    path = bricknet_root / "data_preprocess/rl_utils.py"
    require_file(path, "BrickNet rl_utils")
    spec = importlib.util.spec_from_file_location("_bricknet_eval_rl_utils", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.setup_bricknet_import()
    return module


def safe_mean(rows: list[dict[str, Any]], key: str) -> float:
    return fmean(float(row[key]) for row in rows) if rows else 0.0


def format_optional_float(value: Any) -> str:
    return "-" if value is None else f"{float(value):.6f}"


def summarize_alignment(rows: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(rows)
    return {
        "samples": count,
        "parse_prefix_mean": safe_mean(rows, "parse_prefix"),
        "inventory_precision_mean": safe_mean(rows, "inventory_precision"),
        "inventory_recall_mean": safe_mean(rows, "inventory_recall"),
        "inventory_f1_mean": safe_mean(rows, "inventory_f1"),
        "exact_inventory": sum(bool(row["exact_inventory"]) for row in rows),
        "exact_inventory_rate": (
            sum(bool(row["exact_inventory"]) for row in rows) / count
            if count
            else 0.0
        ),
        "length_score_mean": safe_mean(rows, "length_score"),
        "exact_length": sum(bool(row["exact_length"]) for row in rows),
        "exact_length_rate": (
            sum(bool(row["exact_length"]) for row in rows) / count
            if count
            else 0.0
        ),
        "collision_prefix_mean": safe_mean(rows, "collision_prefix"),
        "pose_match_mean": safe_mean(rows, "pose_match"),
        "pose_success": sum(bool(row["pose_success"]) for row in rows),
        "pose_success_rate": (
            sum(bool(row["pose_success"]) for row in rows) / count
            if count
            else 0.0
        ),
        "dense_reward_mean": safe_mean(rows, "dense_reward"),
        "strict_success": sum(bool(row["strict_success"]) for row in rows),
        "strict_success_rate": (
            sum(bool(row["strict_success"]) for row in rows) / count
            if count
            else 0.0
        ),
        "exact_path": sum(bool(row["exact_path"]) for row in rows),
        "exact_path_rate": (
            sum(bool(row["exact_path"]) for row in rows) / count
            if count
            else 0.0
        ),
        "predicted_parts_mean": safe_mean(rows, "predicted_parts"),
        "target_parts_mean": safe_mean(rows, "target_parts"),
    }


def alignment_breakdown(
    rows: list[dict[str, Any]], group_key: str
) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row[group_key])].append(row)
    keep = (
        "samples",
        "inventory_f1_mean",
        "length_score_mean",
        "collision_prefix_mean",
        "pose_match_mean",
        "dense_reward_mean",
        "strict_success_rate",
    )
    return {
        group: {
            key: value
            for key, value in summarize_alignment(group_rows).items()
            if key in keep
        }
        for group, group_rows in sorted(grouped.items())
    }


def run_alignment_worker(args: argparse.Namespace) -> int:
    import numpy as np

    bricknet_root = resolve(args.bricknet_root)
    utils = load_rl_utils(bricknet_root)
    from bricknet.tree import parse_sample

    results = load_jsonl(resolve(args.results))
    dataset = load_jsonl(resolve(args.dataset))
    scored = load_jsonl(resolve(args.scored))
    if not (len(results) == len(dataset) == len(scored)):
        raise ValueError(
            "alignment worker requires equal result/dataset/scored row counts: "
            f"{len(results)}/{len(dataset)}/{len(scored)}"
        )

    detail_rows: list[dict[str, Any]] = []
    for index, (result, dataset_row, scored_row) in enumerate(
        zip(results, dataset, scored)
    ):
        prediction_text = normalize_path_text(result["response"])
        target_text = normalize_path_text(
            final_assistant_content(dataset_row, row_index=index)
        )
        predicted = parse_sample(prediction_text)
        target = parse_sample(target_text)
        if target.error is not None:
            raise ValueError(f"target row {index} is not parseable: {target.error}")

        predicted_tree = predicted.tree
        target_tree = target.tree
        predicted_parts = len(predicted_tree.parts)
        target_parts = len(target_tree.parts)
        predicted_inventory: Counter = utils.inventory_counter_from_tree(
            predicted_tree
        )
        target_inventory: Counter = utils.inventory_counter_from_tree(target_tree)
        predicted_n = sum(predicted_inventory.values())
        target_n = sum(target_inventory.values())
        overlap = sum((predicted_inventory & target_inventory).values())
        inventory_precision = overlap / predicted_n if predicted_n else 0.0
        inventory_recall = overlap / target_n if target_n else 0.0
        inventory_f1 = utils.multiset_f1(
            predicted_inventory, target_inventory
        )
        exact_inventory = predicted_inventory == target_inventory
        length_score = (
            min(predicted_parts, target_parts) / max(predicted_parts, target_parts)
            if predicted_parts or target_parts
            else 1.0
        )
        parse_prefix = (
            1.0
            if predicted.error is None
            else min(predicted_parts / max(target_parts, 1), 1.0)
        )
        collisions = scored_row.get("collisions")
        if not isinstance(collisions, list):
            raise ValueError(f"scored row {index}: collisions must be a list")
        collision_prefix = (
            min(min(collisions) / max(target_parts, 1), 1.0)
            if collisions
            else 1.0
        )
        try:
            pose_match = utils.pose_match_score(
                predicted_tree,
                target_tree,
                translation_tolerance=args.translation_tolerance,
                rotation_tolerance_degrees=args.rotation_tolerance,
            )
        except (ValueError, np.linalg.LinAlgError):
            pose_match = 0.0
        pose_success = pose_match >= args.pose_success_threshold
        strict_success = bool(
            predicted.error is None
            and predicted_parts == target_parts
            and exact_inventory
            and not collisions
            and pose_success
        )
        dense_reward = (
            0.20 * parse_prefix
            + 0.20 * inventory_f1
            + 0.10 * length_score
            + 0.20 * collision_prefix
            + 0.30 * pose_match
        )
        _, connector_group = utils.connector_features(target_tree)
        detail_rows.append(
            {
                "index": index,
                "id": dataset_row.get("id", index),
                "parse_error": predicted.error,
                "predicted_parts": predicted_parts,
                "target_parts": target_parts,
                "length_bin": utils.length_bin(target_parts),
                "connector_group": connector_group,
                "source_group": utils.source_group(
                    (dataset_row.get("meta") or {}).get("source_model_id")
                ),
                "parse_prefix": float(parse_prefix),
                "inventory_precision": float(inventory_precision),
                "inventory_recall": float(inventory_recall),
                "inventory_f1": float(inventory_f1),
                "exact_inventory": exact_inventory,
                "length_score": float(length_score),
                "exact_length": predicted_parts == target_parts,
                "collisions": collisions,
                "collision_prefix": float(collision_prefix),
                "pose_match": float(pose_match),
                "pose_success": pose_success,
                "dense_reward": float(dense_reward),
                "strict_success": strict_success,
                "exact_path": prediction_text == target_text,
            }
        )

    output = resolve(args.output)
    write_jsonl(output, detail_rows)
    summary = summarize_alignment(detail_rows)
    summary["weights"] = {
        "parse_prefix": 0.20,
        "inventory_f1": 0.20,
        "length_score": 0.10,
        "collision_prefix": 0.20,
        "pose_match": 0.30,
    }
    summary["pose_tolerances"] = {
        "translation": args.translation_tolerance,
        "rotation_degrees": args.rotation_tolerance,
        "success_threshold": args.pose_success_threshold,
    }
    summary["breakdown"] = {
        "length_bin": alignment_breakdown(detail_rows, "length_bin"),
        "connector_group": alignment_breakdown(detail_rows, "connector_group"),
        "source_group": alignment_breakdown(detail_rows, "source_group"),
    }

    metrics_json = resolve(args.metrics_json)
    metrics = load_json(metrics_json)
    metrics["task_alignment"] = summary
    structure = metrics["structure"]
    text_metrics = metrics["text_metrics"]
    image_metrics = metrics["image_metrics"]
    condition_generation = {
        "samples": structure["samples"],
        "connectivity_num": structure["fully_parsable"],
        "connectivity_rate": structure["fully_parsable_rate"],
        # BrickNet's condition-generation Collision metric: mean number of
        # actions before the first parse failure or geometric collision.
        "collision": structure.get(
            "mean_actions_before_first_failure", structure.get("collision")
        ),
        "clean_num": structure["parsable_and_collision_free"],
        "clean_rate": structure["parsable_and_collision_free_rate"],
        "pe": image_metrics.get("pe", {}).get("mean_max_score"),
        "pe_coverage_adjusted": image_metrics.get("pe", {}).get(
            "coverage_adjusted_mean_max_score"
        ),
        "siglip2": image_metrics.get("siglip2", {}).get("mean_max_score"),
        "siglip2_coverage_adjusted": image_metrics.get("siglip2", {}).get(
            "coverage_adjusted_mean_max_score"
        ),
        "vqa_score": image_metrics.get("vqa", {}).get("mean_max_score"),
        "vqa_score_coverage_adjusted": image_metrics.get("vqa", {}).get(
            "coverage_adjusted_mean_max_score"
        ),
        "bleu_4": text_metrics.get("predict_bleu-4"),
        "rouge_1": text_metrics.get("predict_rouge-1"),
        "rouge_2": text_metrics.get("predict_rouge-2"),
        "rouge_l": text_metrics.get("predict_rouge-l"),
        "parse_prefix": summary["parse_prefix_mean"],
        "inventory_f1": summary["inventory_f1_mean"],
        "length_score": summary["length_score_mean"],
        "collision_prefix": summary["collision_prefix_mean"],
        "pose_match": summary["pose_match_mean"],
        "dense_reward": summary["dense_reward_mean"],
        "strict_success_num": summary["strict_success"],
        "strict_success_rate": summary["strict_success_rate"],
    }
    metrics["condition_generation"] = condition_generation
    write_json(metrics_json, metrics)

    metrics_md = resolve(args.metrics_md)
    base = metrics_md.read_text(encoding="utf-8")
    for section in (CONDITION_SECTION, ALIGNMENT_SECTION):
        if section in base:
            base = base.split(section, 1)[0].rstrip() + "\n"
    lines = [
        "",
        CONDITION_SECTION,
        "",
        "| Connectivity (Num, %) | Collision | PE | SigLIP 2 | VQAScore | Clean (Num, %) | BLEU-4 | ROUGE-L |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        (
            f"| {condition_generation['connectivity_num']} "
            f"({condition_generation['connectivity_rate']:.2%}) "
            f"| {condition_generation['collision']:.6f} "
            f"| {format_optional_float(condition_generation['pe'])} "
            f"| {format_optional_float(condition_generation['siglip2'])} "
            f"| {format_optional_float(condition_generation['vqa_score'])} "
            f"| {condition_generation['clean_num']} "
            f"({condition_generation['clean_rate']:.2%}) "
            f"| {format_optional_float(condition_generation['bleu_4'])} "
            f"| {format_optional_float(condition_generation['rouge_l'])} |"
        ),
        "",
        ALIGNMENT_SECTION,
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Parse prefix mean | {summary['parse_prefix_mean']:.6f} |",
        f"| Inventory precision mean | {summary['inventory_precision_mean']:.6f} |",
        f"| Inventory recall mean | {summary['inventory_recall_mean']:.6f} |",
        f"| Inventory F1 mean | {summary['inventory_f1_mean']:.6f} |",
        f"| Exact inventory rate | {summary['exact_inventory_rate']:.2%} |",
        f"| Length score mean | {summary['length_score_mean']:.6f} |",
        f"| Exact length rate | {summary['exact_length_rate']:.2%} |",
        f"| Collision prefix mean | {summary['collision_prefix_mean']:.6f} |",
        f"| Pose match mean | {summary['pose_match_mean']:.6f} |",
        f"| Pose success rate | {summary['pose_success_rate']:.2%} |",
        f"| Dense reward mean | {summary['dense_reward_mean']:.6f} |",
        f"| Strict success rate | {summary['strict_success_rate']:.2%} |",
        f"| Exact path rate | {summary['exact_path_rate']:.2%} |",
        "",
        "Dense reward weights: ParsePrefix/InventoryF1/Length/CollisionPrefix/PoseMatch "
        "= 0.20/0.20/0.10/0.20/0.30.",
        "",
    ]
    metrics_md.write_text(base + "\n".join(lines), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="Run inference and the complete evaluation pipeline")
    run.add_argument("--experiment-name", required=True)
    run.add_argument("--checkpoint", type=Path)
    run.add_argument("--base-model", type=Path)
    run.add_argument(
        "--inference-results",
        type=Path,
        help="Reuse an existing ms-swift inference JSONL instead of running inference.",
    )
    run.add_argument("--dataset", type=Path, default=DEFAULT_VAL_DATASET)
    run.add_argument("--captions", type=Path, default=DEFAULT_CAPTIONS)
    run.add_argument("--output-dir", required=True, type=Path)
    run.add_argument("--bricknet-root", type=Path, default=DEFAULT_BRICKNET_ROOT)
    run.add_argument(
        "--bricknet-data",
        type=Path,
        default=Path("/home/jiahao/.local/share/bricknet"),
    )
    run.add_argument("--swift-bin", type=Path, default=DEFAULT_SWIFT_BIN)
    run.add_argument(
        "--llamafactory-python", type=Path, default=DEFAULT_LLAMFACTORY_PYTHON
    )
    run.add_argument("--bricknet-python", type=Path, default=DEFAULT_BRICKNET_PYTHON)
    run.add_argument("--max-length", type=int, default=4096)
    run.add_argument("--max-new-tokens", type=int, default=4096)
    run.add_argument("--max-pixels", type=int, default=589824)
    run.add_argument("--min-pixels", type=int, default=1024)
    run.add_argument("--temperature", type=float, default=1.0)
    run.add_argument("--top-p", type=float, default=0.95)
    run.add_argument("--top-k", type=int, default=20)
    run.add_argument("--seed", type=int, default=42)
    run.add_argument("--max-batch-size", type=int, default=8)
    run.add_argument("--write-batch-size", type=int, default=64)
    run.add_argument("--score-workers", type=int, default=8)
    run.add_argument("--render-jobs", type=int, default=8)
    run.add_argument("--eval-workers", type=int, default=8)
    run.add_argument("--eval-batch-size", type=int, default=8)
    run.add_argument("--pose-translation-tolerance", type=float, default=0.5)
    run.add_argument("--pose-rotation-tolerance", type=float, default=5.0)
    run.add_argument("--pose-success-threshold", type=float, default=1.0)
    run.add_argument("--skip-image-metrics", action="store_true")
    run.add_argument("--force", action="store_true")
    run.add_argument("--dry-run", action="store_true")

    text = subparsers.add_parser("text-worker", help=argparse.SUPPRESS)
    text.add_argument("--results", required=True, type=Path)
    text.add_argument("--output", required=True, type=Path)

    alignment = subparsers.add_parser("alignment-worker", help=argparse.SUPPRESS)
    alignment.add_argument("--results", required=True, type=Path)
    alignment.add_argument("--dataset", required=True, type=Path)
    alignment.add_argument("--scored", required=True, type=Path)
    alignment.add_argument("--metrics-json", required=True, type=Path)
    alignment.add_argument("--metrics-md", required=True, type=Path)
    alignment.add_argument("--output", required=True, type=Path)
    alignment.add_argument("--bricknet-root", required=True, type=Path)
    alignment.add_argument("--translation-tolerance", type=float, default=0.5)
    alignment.add_argument("--rotation-tolerance", type=float, default=5.0)
    alignment.add_argument("--pose-success-threshold", type=float, default=1.0)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "run":
        positive = (
            "max_length",
            "max_new_tokens",
            "max_pixels",
            "min_pixels",
            "max_batch_size",
            "write_batch_size",
            "score_workers",
            "render_jobs",
            "eval_workers",
            "eval_batch_size",
        )
        for name in positive:
            if getattr(args, name) <= 0:
                raise ValueError(f"--{name.replace('_', '-')} must be positive")
        return run_pipeline(args)
    if args.command == "text-worker":
        return run_text_worker(args)
    if args.command == "alignment-worker":
        return run_alignment_worker(args)
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
