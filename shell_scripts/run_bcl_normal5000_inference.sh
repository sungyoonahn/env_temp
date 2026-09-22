#!/usr/bin/env bash
# Run SwissProt inference for all completed Normal-5000 ESM2-650M BCL checkpoints.
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-cuda}"
BATCH_SIZE="${BATCH_SIZE:-64}"
EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-${PROJECT_DIR}/outputs/normal_5000_esm2_650m_experiments}"
CONFIGURATIONS=(bcl_encoder_frozen bcl_lora_last_3_layers)
TASKS=(MF BP CC)

run() {
    printf '\n$ '
    printf '%q ' "$@"
    printf '\n'
    "$@"
}

cd "${PROJECT_DIR}"

for CONFIGURATION in "${CONFIGURATIONS[@]}"; do
    for TASK in "${TASKS[@]}"; do
        CHECKPOINT="${EXPERIMENT_ROOT}/${CONFIGURATION}/${TASK}/ESM2_650M/best_bcl_weights.pt"
        OUTPUT_DIR="${EXPERIMENT_ROOT}/${CONFIGURATION}/${TASK}/ESM2_650M/inference"
        if [[ ! -f "${CHECKPOINT}" ]]; then
            echo "Missing checkpoint: ${CHECKPOINT}" >&2
            exit 1
        fi
        run "${PYTHON_BIN}" BCL_inference.py \
            --checkpoint "${CHECKPOINT}" \
            --task "${TASK}" \
            --model ESM2_650M \
            --normal-size 5000 \
            --batch-size "${BATCH_SIZE}" \
            --device "${DEVICE}" \
            --output-dir "${OUTPUT_DIR}"
    done
done
