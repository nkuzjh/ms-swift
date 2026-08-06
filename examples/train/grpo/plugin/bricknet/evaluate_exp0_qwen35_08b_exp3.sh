#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../../.." && pwd)"
PYTHON="${SWIFT_PYTHON:-/home/jiahao/miniconda3/envs/swift/bin/python}"

cd "${REPO_ROOT}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

exec "${PYTHON}" "${SCRIPT_DIR}/evaluate_experiment.py" run \
    --experiment-name grpo_exp0_qwen35_08b_exp3_val_t1_p095_k20 \
    --checkpoint output/bricknet_grpo/exp0_qwen35_08b_exp3_rl_n2000_g8/checkpoint-1000 \
    --base-model models/Qwen3.5-0.8B-PT-exp0-merged \
    --dataset /home/jiahao/task/BrickNet/outputs_preprocess/BrickNet-MM/sharegpt/BrickNet-MM_VAL.jsonl \
    --captions /home/jiahao/task/BrickNet/data/bricknet_datasets/captions_val.jsonl \
    --output-dir output/bricknet_eval/grpo_exp0_qwen35_08b_exp3_val_t1_p095_k20 \
    "$@"
