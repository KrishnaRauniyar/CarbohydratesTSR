#!/bin/bash
#SBATCH -p workq
#SBATCH -N 1
#SBATCH -n 64
#SBATCH -t 00:10:00
#SBATCH -A loni_tsr_9
#SBATCH -J balance_sample_detail_with_min
#SBATCH -o slurm_logs/%x_%j.out
#SBATCH -e slurm_logs/%x_%j.err

set -euo pipefail

module load python/3.11.5-anaconda

cd "${SLURM_SUBMIT_DIR}"
mkdir -p slurm_logs

echo "[INFO] Starting sample-detail balancing"

PROJECT_DIR="${PROJECT_DIR:-/ddnB/work/rauniyar/Spring_2024/carbohydrates}"
PY_SCRIPT="${PROJECT_DIR}/src/data_generation/balance_sample_detail_with_min/balance_sample_detail_with_min.py"

INPUT_CSV="${INPUT_CSV:-${PROJECT_DIR}/data/glycoprotein_samples/generated/sample_details_carbohydrates_merged_8_carbs.csv}"
OUTPUT_CSV="${OUTPUT_CSV:-${PROJECT_DIR}/data/glycoprotein_samples/generated/sample_details_carbohydrates_merged_8_carbs_balanced_min.csv}"
CARB_COLUMN="${CARB_COLUMN:-carb_name}"
SEED="${SEED:-42}"
SORT_BY_CARB="${SORT_BY_CARB:-0}"

echo "[INFO] Python script: ${PY_SCRIPT}"
echo "[INFO] Input CSV: ${INPUT_CSV}"
echo "[INFO] Output CSV: ${OUTPUT_CSV}"
echo "[INFO] Carb column: ${CARB_COLUMN}"
echo "[INFO] Seed: ${SEED}"
echo "[INFO] Sort by carb: ${SORT_BY_CARB}"

if [[ ! -f "${PY_SCRIPT}" ]]; then
    echo "[ERROR] Python script does not exist: ${PY_SCRIPT}" >&2
    exit 1
fi

if [[ ! -f "${INPUT_CSV}" ]]; then
    echo "[ERROR] Input CSV does not exist: ${INPUT_CSV}" >&2
    exit 1
fi

PY_ARGS=(
    --input-csv "${INPUT_CSV}"
    --output-csv "${OUTPUT_CSV}"
    --carb-column "${CARB_COLUMN}"
    --seed "${SEED}"
)

if [[ "${SORT_BY_CARB}" == "1" ]]; then
    PY_ARGS+=(--sort-by-carb)
fi

python "${PY_SCRIPT}" "${PY_ARGS[@]}"

echo "[INFO] Job finished successfully"
