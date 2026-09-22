#!/usr/bin/env bash
# ESM2-650M / Normal=5000 / batch=32 ablation:
# final-3 Q/K/V LoRA and final-5 Q/K/V LoRA for baseline and BCL.
# BCL inference is run immediately after each BCL training task.
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
NORMAL_SIZE="${NORMAL_SIZE:-5000}"
BATCH_SIZE="${BATCH_SIZE:-32}"
EPOCHS="${EPOCHS:-20}"
NUM_WORKERS="${NUM_WORKERS:-4}"
DEVICE="${DEVICE:-cuda}"
SEED="${SEED:-42}"
EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-${PROJECT_DIR}/outputs/normal_5000_esm2_650m_experiments/batch_32_ablation}"
TASKS=(MF BP CC)

run() {
    printf '\n$ '
    printf '%q ' "$@"
    printf '\n'
    "$@"
}

run_baseline() {
    local configuration="$1"
    local lora_layers="$2"
    run "${PYTHON_BIN}" main.py \
        --models ESM2_650M \
        --tasks "${TASKS[@]}" \
        --normal-sizes "${NORMAL_SIZE}" \
        --epochs "${EPOCHS}" \
        --batch-size "${BATCH_SIZE}" \
        --lora-last-n-layers "${lora_layers}" \
        --num-workers "${NUM_WORKERS}" \
        --device "${DEVICE}" \
        --seed "${SEED}" \
        --output-root "${EXPERIMENT_ROOT}/baseline_${configuration}"
}

run_bcl() {
    local configuration="$1"
    local lora_layers="$2"
    local task
    local output_dir="${EXPERIMENT_ROOT}/bcl_${configuration}"

    for task in "${TASKS[@]}"; do
        run "${PYTHON_BIN}" BCL_main.py \
            --task "${task}" \
            --model ESM2_650M \
            --normal-size "${NORMAL_SIZE}" \
            --epochs "${EPOCHS}" \
            --batch-size "${BATCH_SIZE}" \
            --lora-last-n-layers "${lora_layers}" \
            --num-workers "${NUM_WORKERS}" \
            --device "${DEVICE}" \
            --seed "${SEED}" \
            --output-dir "${output_dir}"

        run "${PYTHON_BIN}" BCL_inference.py \
            --checkpoint "${output_dir}/${task}/ESM2_650M/best_bcl_weights.pt" \
            --task "${task}" \
            --model ESM2_650M \
            --normal-size "${NORMAL_SIZE}" \
            --batch-size "${BATCH_SIZE}" \
            --num-workers 0 \
            --device "${DEVICE}" \
            --output-dir "${output_dir}/${task}/ESM2_650M/inference"
    done
}

cd "${PROJECT_DIR}"

# BCL training and BCL_inference.py for every task/configuration.
run_bcl lora_last_5_layers 5
run_bcl lora_last_3_layers 3

# Baseline main.py already performs SwissProt inference after each task.
run_baseline lora_last_3_layers 3
run_baseline lora_last_5_layers 5
