#!/usr/bin/env bash
# BCL ESM2-650M: Normal=5000, Q/K/V LoRA on the final five layers, batch size 64.
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
EPOCHS="${EPOCHS:-20}"
NUM_WORKERS="${NUM_WORKERS:-4}"
DEVICE="${DEVICE:-cuda}"
SEED="${SEED:-42}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_DIR}/outputs/normal_5000_esm2_650m_experiments/bcl_lora_last_5_layers_batch_64}"
TASKS=(MF BP CC)

run() {
    printf '\n$ '
    printf '%q ' "$@"
    printf '\n'
    "$@"
}

cd "${PROJECT_DIR}"

for TASK in "${TASKS[@]}"; do
    run "${PYTHON_BIN}" BCL_main.py \
        --task "${TASK}" \
        --model ESM2_650M \
        --normal-size 5000 \
        --epochs "${EPOCHS}" \
        --batch-size 64 \
        --lora-last-n-layers 5 \
        --num-workers "${NUM_WORKERS}" \
        --device "${DEVICE}" \
        --seed "${SEED}" \
        --output-dir "${OUTPUT_ROOT}"

    run "${PYTHON_BIN}" BCL_inference.py \
        --checkpoint "${OUTPUT_ROOT}/${TASK}/ESM2_650M/best_bcl_weights.pt" \
        --task "${TASK}" \
        --model ESM2_650M \
        --normal-size 5000 \
        --batch-size 64 \
        --num-workers 0 \
        --device "${DEVICE}" \
        --output-dir "${OUTPUT_ROOT}/${TASK}/ESM2_650M/inference"
done
