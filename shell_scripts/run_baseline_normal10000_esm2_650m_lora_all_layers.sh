#!/usr/bin/env bash
# Baseline ESM2-650M: Normal=10000, Q/K/V LoRA on all 33 Transformer layers.
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
EPOCHS="${EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-4}"
DEVICE="${DEVICE:-cuda}"
SEED="${SEED:-42}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_DIR}/outputs/normal_10000_esm2_650m_experiments/baseline_lora_all_layers}"

cd "${PROJECT_DIR}"

"${PYTHON_BIN}" main.py \
    --models ESM2_650M \
    --tasks CC BP MF \
    --normal-sizes 10000 \
    --epochs "${EPOCHS}" \
    --batch-size "${BATCH_SIZE}" \
    --lora-last-n-layers 33 \
    --num-workers "${NUM_WORKERS}" \
    --device "${DEVICE}" \
    --seed "${SEED}" \
    --output-root "${OUTPUT_ROOT}"
