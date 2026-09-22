#!/usr/bin/env bash
# Run selected SwissProt inference jobs one at a time on the current GPU.
# Usage: bash run_inference_sequential.sh MODEL TASK NORMAL_SIZE [NORMAL_SIZE ...]
# Example: bash run_inference_sequential.sh ESM2_650M BP 1000 2000 3000

set -euo pipefail

if (( $# < 3 )); then
    echo "Usage: bash $0 MODEL TASK NORMAL_SIZE [NORMAL_SIZE ...]" >&2
    echo "Example: bash $0 ESM2_650M BP 1000 2000 3000" >&2
    exit 2
fi

model="$1"
task="$2"
shift 2

for normal_size in "$@"; do
    echo "[starting] model=${model} task=${task} normal_size=${normal_size}"
    python -u inference_only.py \
        --model "$model" \
        --task "$task" \
        --normal-size "$normal_size" \
        --experiment-type baseline
    echo "[completed] model=${model} task=${task} normal_size=${normal_size}"
done
