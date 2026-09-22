#!/usr/bin/env bash
# Baseline ESM2-650M: Normal=5000, Q/K/V LoRA on the final three layers.
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
EPOCHS="${EPOCHS:-20}"
NUM_WORKERS="${NUM_WORKERS:-4}"
DEVICE="${DEVICE:-cuda}"
SEED="${SEED:-42}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_DIR}/outputs/normal_5000_esm2_650m_experiments/baseline_lora_last_3_layers}"

cd "${PROJECT_DIR}"

"${PYTHON_BIN}" main.py \
    --models ESM2_650M \
    --tasks MF BP CC \
    --normal-sizes 5000 \
    --epochs "${EPOCHS}" \
    --batch-size 8 \
    --lora-last-n-layers 3 \
    --num-workers "${NUM_WORKERS}" \
    --device "${DEVICE}" \
    --seed "${SEED}" \
    --output-root "${OUTPUT_ROOT}"
