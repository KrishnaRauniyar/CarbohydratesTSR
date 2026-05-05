#!/bin/bash
#SBATCH -p workq
#SBATCH -N 1
#SBATCH -n 64
#SBATCH -t 00:30:00
#SBATCH -A loni_tsr_9
#SBATCH -J filterout_nolegacypdb_files
#SBATCH -o slurm_logs/%x_%j.out
#SBATCH -e slurm_logs/%x_%j.err

set -euo pipefail

module load python/3.11.5-anaconda

cd "${SLURM_SUBMIT_DIR}"
mkdir -p slurm_logs

echo "[INFO] Starting no-legacy-PDB filter"

PROJECT_DIR="${PROJECT_DIR:-/ddnB/work/rauniyar/Spring_2024/carbohydrates}"
PY_SCRIPT="${PROJECT_DIR}/src/data_generation/filterout_nolegacypdb_files/filterout_nolegacypdb_files.py"

INPUT_CSV="${INPUT_CSV:-${PROJECT_DIR}/data/glycoprotein_samples/generated/sample_details_carbohydrates_merged_8_carbs.csv}"
OUTPUT_CSV="${OUTPUT_CSV:-${PROJECT_DIR}/data/glycoprotein_samples/generated/sample_details_carbohydrates_merged_8_carbs_legacy_pdb_only.csv}"
REMOVED_CSV="${REMOVED_CSV:-${PROJECT_DIR}/data/glycoprotein_samples/generated/sample_details_carbohydrates_merged_8_carbs_no_legacy_pdb_removed.csv}"
REPORT_CSV="${REPORT_CSV:-${PROJECT_DIR}/data/glycoprotein_samples/generated/sample_details_carbohydrates_merged_8_carbs_legacy_pdb_report.csv}"
DOWNLOAD_DIR="${DOWNLOAD_DIR:-${PROJECT_DIR}/data/glycoprotein_samples/pdb_cache}"

PROTEIN_COLUMN="${PROTEIN_COLUMN:-protein}"
WORKERS="${WORKERS:-8}"
TIMEOUT="${TIMEOUT:-60}"
FORCE_DOWNLOAD="${FORCE_DOWNLOAD:-0}"
NO_DOWNLOAD="${NO_DOWNLOAD:-0}"

echo "[INFO] Python script: ${PY_SCRIPT}"
echo "[INFO] Input CSV: ${INPUT_CSV}"
echo "[INFO] Output CSV: ${OUTPUT_CSV}"
echo "[INFO] Removed CSV: ${REMOVED_CSV}"
echo "[INFO] Report CSV: ${REPORT_CSV}"
echo "[INFO] Download/cache dir: ${DOWNLOAD_DIR}"
echo "[INFO] Protein column: ${PROTEIN_COLUMN}"
echo "[INFO] Workers: ${WORKERS}"
echo "[INFO] Timeout: ${TIMEOUT}"
echo "[INFO] Force download: ${FORCE_DOWNLOAD}"
echo "[INFO] No download: ${NO_DOWNLOAD}"

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
    --removed-csv "${REMOVED_CSV}"
    --report-csv "${REPORT_CSV}"
    --download-dir "${DOWNLOAD_DIR}"
    --protein-column "${PROTEIN_COLUMN}"
    --workers "${WORKERS}"
    --timeout "${TIMEOUT}"
)

if [[ "${FORCE_DOWNLOAD}" == "1" ]]; then
    PY_ARGS+=(--force-download)
fi

if [[ "${NO_DOWNLOAD}" == "1" ]]; then
    PY_ARGS+=(--no-download)
fi

python "${PY_SCRIPT}" "${PY_ARGS[@]}"

echo "[INFO] Job finished successfully"
