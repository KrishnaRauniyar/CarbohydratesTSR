#!/bin/bash
#SBATCH -p workq
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 64
#SBATCH -t 10:00:00
#SBATCH -A loni_tsr_9
#SBATCH -J key_triplets_generation
#SBATCH -o slurm_logs/%x_%j.out
#SBATCH -e slurm_logs/%x_%j.err

set -euo pipefail

module load python/3.11.5-anaconda

cd "${SLURM_SUBMIT_DIR}"

mkdir -p slurm_logs
ENV_DIR="${SLURM_SUBMIT_DIR}/.conda_keytriplets"

source "$(conda info --base)/etc/profile.d/conda.sh"

if [ ! -d "${ENV_DIR}" ]; then
    echo "[INFO] Creating conda environment at ${ENV_DIR}"
    conda create -y -p "${ENV_DIR}" python=3.11
fi

conda activate "${ENV_DIR}"

echo "[INFO] Installing dependencies"
pip install --upgrade pip

REQS="requirements.txt"
if [ -f "${REQS}" ]; then
    pip install -r "${REQS}"
else
    pip install pandas requests joblib
fi

echo "[INFO] Starting key-triplets generation pipeline"

RESULTS_DIR="results"
PDB_DIR="pdb_files"
TARGET_CSV="/ddnB/work/rauniyar/Spring_2024/carbohydrates/data/glycoprotein_samples/sample_details_carbohydrates_merged.csv"
LEXICAL_CSV="/ddnB/work/rauniyar/Spring_2024/carbohydrates/data/lexical_file/drug_atom_lexical_txt_11_4_2025.csv"
PROTEIN_DIR="proteins"
SUMMARY_CSV="proteinNumKeysDist.csv"
PY_SCRIPT="key_triplets_generation.py"

# Key-generation parameters
DTHETA="29"
DLEN="18"
NUM_LABELS="112"
NUM_CORES="${SLURM_CPUS_PER_TASK:-64}"

mkdir -p "${RESULTS_DIR}"

python "${PY_SCRIPT}" \
  -p "${PDB_DIR}" \
  -c "${TARGET_CSV}" \
  -l "${LEXICAL_CSV}" \
  -o "${PROTEIN_DIR}" \
  -s "${SUMMARY_CSV}" \
  -r "${RESULTS_DIR}" \
  --dtheta "${DTHETA}" \
  --dlen "${DLEN}" \
  --num_labels "${NUM_LABELS}" \
  --num_cores "${NUM_CORES}"

echo "[INFO] Job finished successfully"
