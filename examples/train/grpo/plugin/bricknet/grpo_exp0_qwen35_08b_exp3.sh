#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../../.." && pwd)"
CONFIG="${SCRIPT_DIR}/grpo_exp0_qwen35_08b_exp3.yaml"

cd "${REPO_ROOT}"

if [[ ! -f models/Qwen3.5-0.8B-PT-exp0-merged/config.json ]] ||
   [[ ! -f models/Qwen3.5-0.8B-exp3-adapter/adapter_config.json ]]; then
    echo "Prepared exp3 base not found; run:" >&2
    echo "  bash ${SCRIPT_DIR}/prepare_exp3_base.sh" >&2
    exit 1
fi

if [[ ! -e data/BrickNet-MM-RL_n2000_seed42.jsonl ]]; then
    echo "Dataset symlink is missing: data/BrickNet-MM-RL_n2000_seed42.jsonl" >&2
    exit 1
fi

if ! python -c "import meshlib" >/dev/null 2>&1; then
    echo "meshlib is missing in the active ms-swift environment." >&2
    echo "Install BrickNet and its dependencies with:" >&2
    echo "  python -m pip install -e /home/jiahao/task/BrickNet" >&2
    exit 1
fi

swift rlhf "${CONFIG}" "$@"
