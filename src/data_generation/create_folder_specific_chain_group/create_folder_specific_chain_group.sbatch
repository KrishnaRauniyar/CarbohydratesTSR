#!/bin/bash
#SBATCH -p workq
#SBATCH -N 1
#SBATCH -n 64
#SBATCH -t 00:20:00
#SBATCH -A loni_tsr_9
#SBATCH -J create_folder_specific_chain_group
#SBATCH -o slurm_logs/%x_%j.out
#SBATCH -e slurm_logs/%x_%j.err

set -euo pipefail

module load python/3.11.5-anaconda

cd "${SLURM_SUBMIT_DIR}"
mkdir -p slurm_logs

echo "[INFO] Starting chain-group file copy"

# ---------------------------------------------------------------------------
# Edit these paths for your run.
# ---------------------------------------------------------------------------
PROJECT_DIR="/ddnB/work/rauniyar/Spring_2024/carbohydrates"
PY_SCRIPT="${PROJECT_DIR}/src/data_generation/create_folder_specific_chain_group/create_folder_specific_chain_group.py"

# Folder containing files like:
#   10MU_J_NN.keys_theta29_dist18
#   10MU_J_NN.keys_Freq_theta29_dist18
INPUT_FOLDER="${PROJECT_DIR}/key_generator/key_generation_using_id_chain_res_seq/results_id_chain/proteins"

# New folder where matching files will be copied.
OUTPUT_FOLDER="${PROJECT_DIR}/key_generator/key_generation_using_id_chain_res_seq/results_id_chain/proteins_selected_groups"

# Chain groups to copy. Examples:
#   CHAIN_GROUPS=("NN")
#   CHAIN_GROUPS=("NN" "N" "A")
#
# Do not name this variable GROUPS. Bash uses GROUPS internally for Unix group
# IDs, which would turn your chain labels into numbers.
CHAIN_GROUPS=("NN")

# Safety/options:
#   DRY_RUN=1 previews copies without writing files.
#   DRY_RUN=0 performs the copy.
#   RECURSIVE=1 searches subfolders too.
#   OVERWRITE=1 replaces existing output files.
#   MATCH_MODE=exact matches the group token exactly; contains uses substring matching.
DRY_RUN="${DRY_RUN:-0}"
RECURSIVE="${RECURSIVE:-0}"
OVERWRITE="${OVERWRITE:-0}"
MATCH_MODE="${MATCH_MODE:-exact}"

echo "[INFO] Python script: ${PY_SCRIPT}"
echo "[INFO] Input folder: ${INPUT_FOLDER}"
echo "[INFO] Output folder: ${OUTPUT_FOLDER}"
echo "[INFO] Chain groups: ${CHAIN_GROUPS[*]}"
echo "[INFO] Dry run: ${DRY_RUN}"
echo "[INFO] Recursive: ${RECURSIVE}"
echo "[INFO] Overwrite: ${OVERWRITE}"
echo "[INFO] Match mode: ${MATCH_MODE}"

if [[ ! -f "${PY_SCRIPT}" ]]; then
    echo "[ERROR] Python script does not exist: ${PY_SCRIPT}" >&2
    exit 1
fi

if [[ ! -d "${INPUT_FOLDER}" ]]; then
    echo "[ERROR] Input folder does not exist: ${INPUT_FOLDER}" >&2
    exit 1
fi

PY_ARGS=(
    --input_folder "${INPUT_FOLDER}"
    --output_folder "${OUTPUT_FOLDER}"
    --groups "${CHAIN_GROUPS[@]}"
    --match_mode "${MATCH_MODE}"
)

if [[ "${DRY_RUN}" == "1" ]]; then
    PY_ARGS+=(--dry_run)
fi

if [[ "${RECURSIVE}" == "1" ]]; then
    PY_ARGS+=(--recursive)
fi

if [[ "${OVERWRITE}" == "1" ]]; then
    PY_ARGS+=(--overwrite)
fi

python "${PY_SCRIPT}" "${PY_ARGS[@]}"

echo "[INFO] Job finished successfully"
