#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../../.." && pwd)"
EXP_DIR="${REPO_ROOT}/output/bricknet_grpo/exp0_qwen35_08b_exp3_rl_n2000_g8"
BASE_MODEL="${REPO_ROOT}/models/Qwen3.5-0.8B-PT-exp0-merged"
CHECKPOINT="${EXP0_CHECKPOINT:-}"
SWIFT_BIN="${SWIFT_BIN:-}"

if [[ -z "${SWIFT_BIN}" ]]; then
    if command -v swift >/dev/null 2>&1; then
        SWIFT_BIN="$(command -v swift)"
    elif [[ -x /home/jiahao/miniconda3/envs/swift/bin/swift ]]; then
        SWIFT_BIN=/home/jiahao/miniconda3/envs/swift/bin/swift
    else
        echo "swift executable not found." >&2
        echo "Activate the ms-swift conda environment or set SWIFT_BIN explicitly." >&2
        exit 1
    fi
fi

if [[ -z "${CHECKPOINT}" ]]; then
    CHECKPOINT="$(
        find "${EXP_DIR}" -maxdepth 1 -type d -name 'checkpoint-*' -print |
            sort -V |
            tail -n 1
    )"
fi

if [[ -z "${CHECKPOINT}" ]] || [[ ! -f "${CHECKPOINT}/adapter_model.safetensors" ]]; then
    echo "No valid exp0 checkpoint found under: ${EXP_DIR}" >&2
    echo "Set EXP0_CHECKPOINT to a checkpoint directory if it is stored elsewhere." >&2
    exit 1
fi

if [[ ! -f "${BASE_MODEL}/config.json" ]]; then
    echo "Prepared PT-exp0 merged base model not found: ${BASE_MODEL}" >&2
    echo "Run: bash ${SCRIPT_DIR}/prepare_exp3_base.sh" >&2
    exit 1
fi

cd "${REPO_ROOT}"
echo "Using GRPO checkpoint: ${CHECKPOINT}" >&2

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
MAX_PIXELS="${MAX_PIXELS:-589824}" \
MIN_PIXELS="${MIN_PIXELS:-1024}" \
"${SWIFT_BIN}" infer \
    --model "${BASE_MODEL}" \
    --adapters "${CHECKPOINT}" \
    --infer_backend transformers \
    --stream true \
    --temperature 0 \
    --max_new_tokens 4096 \
    --enable_thinking false \
    --response_prefix "" \
    --add_non_thinking_prefix false \
    "$@"
