#!/bin/bash
#SBATCH -p workq
#SBATCH -N 1
#SBATCH -n 64
#SBATCH -t 01:00:00
#SBATCH -A loni_tsr_9
#SBATCH -J drug_protein_cross
#SBATCH -o outputs/slurm_logs/%x_%j.out
#SBATCH -e outputs/slurm_logs/%x_%j.err

###############################################################################
# Safety & environment
set -euo pipefail

module load python/3.11.5-anaconda

# Going to submission directory
cd "${SLURM_SUBMIT_DIR}"

###############################################################################
# Outputs + logs
mkdir -p outputs/slurm_logs
ENV_DIR="${SLURM_SUBMIT_DIR}/.conda_crosskey"

###############################################################################
# Conda environment (created once, reused later)
source "$(conda info --base)/etc/profile.d/conda.sh"

if [ ! -d "${ENV_DIR}" ]; then
    echo "[INFO] Creating conda environment at ${ENV_DIR}"
    conda create -y -p "${ENV_DIR}" python=3.11
fi

conda activate "${ENV_DIR}"

###############################################################################
# Install dependencies
echo "[INFO] Installing dependencies"
pip install --upgrade pip

# installing from requirements.txt if you have it
REQS="requirements.txt"
if [ -f "${REQS}" ]; then
    pip install -r "${REQS}"
else
    # minimal deps for your cross-key script
    pip install numpy pandas requests joblib
fi

###############################################################################
# Running cross-key pipeline
echo "[INFO] Starting drug-protein cross-key pipeline"

# -----------------------------
# Required inputs
# -----------------------------
# Where PDBs will be downloaded AND/OR where existing .pdb files already live
PDB_DIR="pdb_files"

# CSV that contains a column named `protein` this is for the cross key cutoff calculation data
PROTEIN_CSV="sample_details_carb_protein.csv"

# Lexical mapping CSV: columns like atom/ATOM and seq
LEXICAL_CSV="drug_atom_lexical_txt_11_4_2025.csv"

# carb CSV that contains `carb_name` column (3rd column)
CARB_CSV="sample_details_carbohydrates_small.csv"

# Output
OUTCSV="outputs/carbsdrug_protein_cross.csv"

# -----------------------------
# Tunable parameter
# -----------------------------
CROSS_KEY_CUT="3.0"   # angstrom cutoff

# Script path (edit if your file name differs)
PY_SCRIPT="drug_protein_cutArmst.py"

# Make sure output folder exists
mkdir -p "$(dirname "${OUTCSV}")"
mkdir -p "${PDB_DIR}"

python "${PY_SCRIPT}" \
  -p "${PDB_DIR}" \
  -c "${PROTEIN_CSV}" \
  -l "${LEXICAL_CSV}" \
  --carb_csv "${CARB_CSV}" \
  -o "${OUTCSV}" \
  --cross_key_cut "${CROSS_KEY_CUT}"

echo "[INFO] Job finished successfully"
