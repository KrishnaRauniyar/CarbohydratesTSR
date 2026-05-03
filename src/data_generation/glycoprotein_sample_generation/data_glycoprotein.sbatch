#!/bin/bash
#SBATCH -p workq
#SBATCH -N 1
#SBATCH -n 64
#SBATCH -t 04:00:00
#SBATCH -A loni_tsr_9
#SBATCH -J data_glycoprotein
#SBATCH -o slurm_logs/%x_%j.out
#SBATCH -e slurm_logs/%x_%j.err

set -euo pipefail

module load python/3.11.5-anaconda

cd "${SLURM_SUBMIT_DIR}"
mkdir -p slurm_logs

PROJECT_DIR="${PROJECT_DIR:-/ddnB/work/rauniyar/Spring_2024/carbohydrates}"
PY_SCRIPT="${PROJECT_DIR}/src/data_generation/glycoprotein_sample_generation/data_glycoprotein.py"

OUTPUT_CSV="${OUTPUT_CSV:-${PROJECT_DIR}/data/glycoprotein_samples/sample_details_carbohydrates_generated.csv}"
DOWNLOAD_DIR="${DOWNLOAD_DIR:-${PROJECT_DIR}/data/glycoprotein_samples/pdb_cache}"
SAVE_PDB_IDS="${SAVE_PDB_IDS:-${PROJECT_DIR}/data/glycoprotein_samples/glycoprotein_pdb_ids.txt}"

GROUP="${GROUP:-PD1_PDL1}"
WORKERS="${WORKERS:-8}"
TIMEOUT="${TIMEOUT:-60}"
PAGE_SIZE="${PAGE_SIZE:-500}"
MAX_RESULTS="${MAX_RESULTS:-}"
PDB_IDS_FILE="${PDB_IDS_FILE:-}"

echo "[INFO] Python script: ${PY_SCRIPT}"
echo "[INFO] Output CSV: ${OUTPUT_CSV}"
echo "[INFO] Download/cache dir: ${DOWNLOAD_DIR}"
echo "[INFO] Group: ${GROUP}"
echo "[INFO] Workers: ${WORKERS}"

if [[ ! -f "${PY_SCRIPT}" ]]; then
    echo "[ERROR] Python script does not exist: ${PY_SCRIPT}" >&2
    exit 1
fi

PY_ARGS=(
    --output-csv "${OUTPUT_CSV}"
    --download-dir "${DOWNLOAD_DIR}"
    --group "${GROUP}"
    --selection-mode all
    --workers "${WORKERS}"
    --timeout "${TIMEOUT}"
    --page-size "${PAGE_SIZE}"
    --save-pdb-ids "${SAVE_PDB_IDS}"
)

if [[ -n "${MAX_RESULTS}" ]]; then
    PY_ARGS+=(--max-results "${MAX_RESULTS}")
fi

if [[ -n "${PDB_IDS_FILE}" ]]; then
    if [[ ! -f "${PDB_IDS_FILE}" ]]; then
        echo "[ERROR] PDB IDs file does not exist: ${PDB_IDS_FILE}" >&2
        exit 1
    fi
    PY_ARGS+=(--pdb-ids-file "${PDB_IDS_FILE}")
else
    PY_ARGS+=(--search-text Glycoprotein --require-glycosylation-site)
fi

python "${PY_SCRIPT}" "${PY_ARGS[@]}"

echo "[INFO] Job finished successfully"
