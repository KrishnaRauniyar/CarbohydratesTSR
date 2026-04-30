#!/bin/bash
#SBATCH -p workq
#SBATCH -N 1
#SBATCH -n 64
#SBATCH -t 00:30:00
#SBATCH -A loni_tsr_9
#SBATCH -J add_ligands_to_chain
#SBATCH -o slurm_logs/%x_%j.out
#SBATCH -e slurm_logs/%x_%j.err

set -euo pipefail

module load python/3.11.5-anaconda

cd "${SLURM_SUBMIT_DIR}"
mkdir -p slurm_logs

echo "[INFO] Starting ligand suffix generation"

# ---------------------------------------------------------------------------
# Script and files
# ---------------------------------------------------------------------------
PROJECT_DIR="/ddnB/work/rauniyar/Spring_2024/carbohydrates"

PY_SCRIPT="${PROJECT_DIR}/src/data_generation/add_ligands_tochain/add_ligands_to_chain.py"
INPUT_CSV="${PROJECT_DIR}/key_generator/key_generation_using_id_chain_res_seq/results_id_chain/proteinNumKeysDist.csv"
OUTPUT_CSV="${PROJECT_DIR}/key_generator/key_generation_using_id_chain_res_seq/results_id_chain/proteinNumKeysDist_ligand_codes.csv"

# Folder containing files like:
#   10MU_C.keys_theta29_dist18
#   10MU_C.keys_Freq_theta29_dist18
#
# The Python script will rename matching files by adding ligand labels from
# INPUT_CSV:
#   10MU_C.keys_theta29_dist18      -> 10MU_C_NN.keys_theta29_dist18
#   10MU_C.keys_Freq_theta29_dist18 -> 10MU_C_NN.keys_Freq_theta29_dist18
#
# Change this path if your .keys files are stored in a different folder.
KEYS_FOLDER="${PROJECT_DIR}/key_generator/key_generation_using_id_chain_res_seq/results_id_chain/proteins"

# Safety controls:
#   DRY_RUN=0 performs the renames.
#   DRY_RUN=1 previews file renames without changing filenames.
#   RECURSIVE=1 also searches subfolders under KEYS_FOLDER.
#   RECURSIVE=0 only searches files directly inside KEYS_FOLDER.
DRY_RUN="${DRY_RUN:-0}"
RECURSIVE="${RECURSIVE:-0}"

# ---------------------------------------------------------------------------
# Column names
# ---------------------------------------------------------------------------
PROTEIN_COLUMN="Protein"
LIGANDS_COLUMN="ligands"

echo "[INFO] Python script: ${PY_SCRIPT}"
echo "[INFO] Input CSV: ${INPUT_CSV}"
echo "[INFO] Output CSV: ${OUTPUT_CSV}"
echo "[INFO] Keys folder: ${KEYS_FOLDER}"
echo "[INFO] Dry run: ${DRY_RUN}"
echo "[INFO] Recursive: ${RECURSIVE}"

if [[ ! -f "${PY_SCRIPT}" ]]; then
    echo "[ERROR] Python script does not exist: ${PY_SCRIPT}" >&2
    exit 1
fi

if [[ ! -f "${INPUT_CSV}" ]]; then
    echo "[ERROR] Input CSV does not exist: ${INPUT_CSV}" >&2
    exit 1
fi

if [[ ! -d "${KEYS_FOLDER}" ]]; then
    echo "[ERROR] Keys folder does not exist: ${KEYS_FOLDER}" >&2
    exit 1
fi

PY_ARGS=(
    -i "${INPUT_CSV}"
    -o "${OUTPUT_CSV}"
    --protein_column "${PROTEIN_COLUMN}"
    --ligands_column "${LIGANDS_COLUMN}"
    --keys_folder "${KEYS_FOLDER}"
)

if [[ "${DRY_RUN}" == "1" ]]; then
    PY_ARGS+=(--dry_run)
fi

if [[ "${RECURSIVE}" == "1" ]]; then
    PY_ARGS+=(--recursive)
fi

python "${PY_SCRIPT}" "${PY_ARGS[@]}"

echo "[INFO] Job finished successfully"
