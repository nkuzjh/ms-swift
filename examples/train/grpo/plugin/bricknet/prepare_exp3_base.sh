#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../../.." && pwd)"

BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3.5-0.8B}"
PT_ADAPTER="${PT_ADAPTER:-/home/jiahao/task/LlamaFactory/saves/Qwen3.5-0.8B-Thinking/lora/train_PT_exp0_qwen35_08b_ep3_bs2_ga8_lora64}"
EXP3_ADAPTER="${EXP3_ADAPTER:-/home/jiahao/task/LlamaFactory/saves/Qwen3.5-0.8B-Thinking/lora/train_exp3_qwen35_08b_pt_sft1w_ep3_bs2_ga8_lora64}"
PT_ADAPTER_VIEW="${PT_ADAPTER_VIEW:-${REPO_ROOT}/models/Qwen3.5-0.8B-PT-exp0-adapter}"
PT_MERGED_MODEL="${PT_MERGED_MODEL:-${REPO_ROOT}/models/Qwen3.5-0.8B-PT-exp0-merged}"
EXP3_ADAPTER_VIEW="${EXP3_ADAPTER_VIEW:-${REPO_ROOT}/models/Qwen3.5-0.8B-exp3-adapter}"

for adapter in "${PT_ADAPTER}" "${EXP3_ADAPTER}"; do
    if [[ ! -f "${adapter}/adapter_config.json" ]]; then
        echo "Missing PEFT adapter: ${adapter}" >&2
        exit 1
    fi
done

mkdir -p "${REPO_ROOT}/models"

prepare_adapter_view() {
    local source_dir="$1"
    local view_dir="$2"

    if [[ -L "${view_dir}" ]]; then
        if [[ "$(realpath "${view_dir}")" != "$(realpath "${source_dir}")" ]]; then
            echo "Existing adapter symlink points elsewhere: ${view_dir}" >&2
            exit 1
        fi
        # Replace the directory symlink created by an older version of this script. A
        # clean view avoids loading LlamaFactory checkpoint-* subdirectories as extra adapters.
        unlink "${view_dir}"
    elif [[ -e "${view_dir}" ]] && [[ ! -d "${view_dir}" ]]; then
        echo "Adapter view path is not a directory: ${view_dir}" >&2
        exit 1
    fi

    mkdir -p "${view_dir}"
    for filename in adapter_config.json adapter_model.safetensors; do
        local source_file="${source_dir}/${filename}"
        local view_file="${view_dir}/${filename}"
        if [[ ! -f "${source_file}" ]]; then
            echo "Missing adapter file: ${source_file}" >&2
            exit 1
        fi
        if [[ -L "${view_file}" ]]; then
            if [[ "$(realpath "${view_file}")" != "$(realpath "${source_file}")" ]]; then
                echo "Existing adapter file symlink points elsewhere: ${view_file}" >&2
                exit 1
            fi
        elif [[ -e "${view_file}" ]]; then
            echo "Cannot create adapter file symlink; path already exists: ${view_file}" >&2
            exit 1
        else
            ln -s "${source_file}" "${view_file}"
        fi
    done
}

prepare_adapter_view "${PT_ADAPTER}" "${PT_ADAPTER_VIEW}"
prepare_adapter_view "${EXP3_ADAPTER}" "${EXP3_ADAPTER_VIEW}"

if [[ ! -f "${PT_MERGED_MODEL}/config.json" ]]; then
    swift export \
        --model "${BASE_MODEL}" \
        --adapters "${PT_ADAPTER_VIEW}" \
        --merge_lora true \
        --torch_dtype bfloat16 \
        --use_hf true \
        --output_dir "${PT_MERGED_MODEL}"
else
    echo "PT-merged base already exists: ${PT_MERGED_MODEL}"
fi

echo "PT-merged model: ${PT_MERGED_MODEL}"
echo "exp3 adapter:    ${EXP3_ADAPTER_VIEW}"
