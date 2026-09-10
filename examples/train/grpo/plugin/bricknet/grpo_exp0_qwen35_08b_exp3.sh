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

# Keep this preflight tied to the dataset that ms-swift will actually receive.
# `swift rlhf <yaml> --dataset ...` applies the command-line value after the
# YAML value, so inspect the same precedence here instead of unconditionally
# requiring the historical n2000 symlink below.
DATASET_OUTPUT="$(python - "${CONFIG}" "$@" <<'PY'
import sys

import yaml


config_path = sys.argv[1]
cli_args = sys.argv[2:]


def cli_dataset(args):
    """Return the last --dataset value, matching argparse override order."""
    selected = None
    index = 0
    while index < len(args):
        argument = args[index]
        if argument == '--dataset':
            index += 1
            values = []
            while index < len(args) and not args[index].startswith('--'):
                values.append(args[index])
                index += 1
            if not values:
                raise SystemExit('--dataset requires at least one path')
            selected = values
            continue
        if argument.startswith('--dataset='):
            value = argument.split('=', 1)[1]
            if not value:
                raise SystemExit('--dataset requires at least one path')
            selected = [value]
        index += 1
    return selected


with open(config_path, encoding='utf-8') as config_file:
    config = yaml.safe_load(config_file) or {}

selected = cli_dataset(cli_args)
if selected is None:
    selected = config.get('dataset', [])

if isinstance(selected, str):
    selected = [selected]
elif isinstance(selected, (list, tuple)):
    selected = list(selected)
else:
    raise SystemExit('dataset must be a string or a YAML list of strings')

if not selected:
    raise SystemExit('no dataset is configured in the YAML or command line')
for dataset in selected:
    if not isinstance(dataset, str) or not dataset:
        raise SystemExit('dataset entries must be non-empty strings')
    print(dataset)
PY
  )"
DATASET_SPECS=()
while IFS= read -r dataset_spec; do
    DATASET_SPECS+=("${dataset_spec}")
done <<< "${DATASET_OUTPUT}"

if [[ "${#DATASET_SPECS[@]}" -eq 0 ]]; then
    echo "No dataset was selected in ${CONFIG} or the command line." >&2
    exit 1
fi

for dataset_spec in "${DATASET_SPECS[@]}"; do
    # ms-swift accepts local paths with optional :subset and #count suffixes.
    # Only the path portion needs to exist for this wrapper preflight.
    dataset_path="${dataset_spec%%#*}"
    if [[ ! -e "${dataset_path}" && "${dataset_path}" == *:* ]]; then
        dataset_path="${dataset_path%%:*}"
    fi
    if [[ "${dataset_path}" != /* ]]; then
        dataset_path="${REPO_ROOT}/${dataset_path}"
    fi
    if [[ ! -f "${dataset_path}" ]]; then
        if [[ "${dataset_spec}" == "data/BrickNet-MM-RL_n2000_seed42.jsonl" ]]; then
            echo "Dataset symlink is missing: data/BrickNet-MM-RL_n2000_seed42.jsonl" >&2
        else
            echo "Dataset file is missing: ${dataset_spec}" >&2
        fi
        exit 1
    fi
done

if ! python -c "import meshlib" >/dev/null 2>&1; then
    echo "meshlib is missing in the active ms-swift environment." >&2
    echo "Install BrickNet and its dependencies with:" >&2
    echo "  python -m pip install -e /home/jiahao/task/BrickNet" >&2
    exit 1
fi

swift rlhf "${CONFIG}" "$@"
