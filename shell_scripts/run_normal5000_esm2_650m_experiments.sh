#!/usr/bin/env bash
# Run all requested ESM2-650M Normal-5000 baseline and BCL experiments.
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
NORMAL_SIZE="${NORMAL_SIZE:-5000}"
EPOCHS="${EPOCHS:-20}"
NUM_WORKERS="${NUM_WORKERS:-4}"
DEVICE="${DEVICE:-cuda}"
SEED="${SEED:-42}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_DIR}/outputs/normal_5000_esm2_650m_experiments}"
TASKS=(MF BP CC)

run() {
    printf '\n$ '
    printf '%q ' "$@"
    printf '\n'
    "$@"
}

cd "${PROJECT_DIR}"

# 1. Baseline: LoRA on all 33 ESM2-650M layers, batch size 8.
run "${PYTHON_BIN}" main.py \
    --models ESM2_650M --tasks "${TASKS[@]}" \
    --normal-sizes "${NORMAL_SIZE}" --epochs "${EPOCHS}" \
    --batch-size 8 --lora-last-n-layers 33 \
    --num-workers "${NUM_WORKERS}" --device "${DEVICE}" --seed "${SEED}" \
    --output-root "${OUTPUT_ROOT}/baseline_lora_all_layers"

# 2. Baseline: complete encoder frozen, batch size 64.
run "${PYTHON_BIN}" main.py \
    --models ESM2_650M --tasks "${TASKS[@]}" \
    --normal-sizes "${NORMAL_SIZE}" --epochs "${EPOCHS}" \
    --batch-size 64 --lora-last-n-layers 0 \
    --num-workers "${NUM_WORKERS}" --device "${DEVICE}" --seed "${SEED}" \
    --output-root "${OUTPUT_ROOT}/baseline_encoder_frozen"

# 3. BCL: complete encoder frozen, batch size 64.
for TASK in "${TASKS[@]}"; do
    run "${PYTHON_BIN}" BCL_main.py \
        --task "${TASK}" --model ESM2_650M \
        --normal-size "${NORMAL_SIZE}" --epochs "${EPOCHS}" \
        --batch-size 64 --lora-last-n-layers 0 \
        --num-workers "${NUM_WORKERS}" --device "${DEVICE}" --seed "${SEED}" \
        --output-dir "${OUTPUT_ROOT}/bcl_encoder_frozen"
done

# 4. BCL: Q/K/V LoRA on the final three ESM2-650M layers, batch size 64.
for TASK in "${TASKS[@]}"; do
    run "${PYTHON_BIN}" BCL_main.py \
        --task "${TASK}" --model ESM2_650M \
        --normal-size "${NORMAL_SIZE}" --epochs "${EPOCHS}" \
        --batch-size 64 --lora-last-n-layers 3 \
        --num-workers "${NUM_WORKERS}" --device "${DEVICE}" --seed "${SEED}" \
        --output-dir "${OUTPUT_ROOT}/bcl_lora_last_3_layers"
done
