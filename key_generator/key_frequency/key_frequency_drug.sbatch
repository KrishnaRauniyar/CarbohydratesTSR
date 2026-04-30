#!/bin/bash
#SBATCH -p checkpt
#SBATCH -n 64
#SBATCH -t 00:15:00
#SBATCH -A loni_tsr_9
#SBATCH -N 1
#SBATCH -J key_frequency_drug
#SBATCH -o output_key_frequency_drug.out
#SBATCH -e error_key_frequency_drug.err

set -euo pipefail

PY_SCRIPT="/ddnB/work/rauniyar/Spring_2024/cancer_drug_fingerprint/src/key_triplets_generator/key_frequency/key_frequency_drug.py"

# Directory containing the *.keys_Freq_theta29_dist18 files.
# The output CSV will be written to this directory's parent.
INPUT_DIR="/ddnB/work/rauniyar/Spring_2024/cancer_drug_fingerprint/src/key_triplets_generator/key_triplets/results/proteins"
OUTPUT_DIR="/ddnB/work/rauniyar/Spring_2024/cancer_drug_fingerprint/data/processed/ai_ml"

HEADER_BOOL="yes"

CMD=(
    python3 "${PY_SCRIPT}"
    --input-dir "${INPUT_DIR}"
    --output-dir "${OUTPUT_DIR}"
    --header_bool "${HEADER_BOOL}"
)

echo "[INFO] Running: ${CMD[*]}"
"${CMD[@]}"

echo "[INFO] Job finished successfully"
