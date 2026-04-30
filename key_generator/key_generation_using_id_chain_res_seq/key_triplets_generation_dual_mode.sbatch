#!/bin/bash
#SBATCH -p workq
#SBATCH -N 1
#SBATCH -n 64
#SBATCH -t 10:00:00
#SBATCH -A loni_tsr_9
#SBATCH -J key_triplets_generation_dual_mode
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

# ---------------------------------------------------------------------------
# Core script and outputs
# ---------------------------------------------------------------------------
PY_SCRIPT="/ddnB/work/rauniyar/Spring_2024/carbohydrates/key_generator/key_generation_using_id_chain_res_seq/key_triplets_generation_dual_mode.py"
RESULTS_DIR="results"
PDB_DIR="pdb_files"
PROTEIN_DIR="proteins"
SUMMARY_CSV="proteinNumKeysDist.csv"

# ---------------------------------------------------------------------------
# Input files
# ---------------------------------------------------------------------------
# Residue-oriented example CSV (protein, chain, residue, seqnum, ...):
TARGET_CSV="/ddnB/work/rauniyar/Spring_2024/carbohydrates/data/glycoprotein_samples/sample_details_carbohydrates_merged_eight_residues.csv"
LEXICAL_CSV="/ddnB/work/rauniyar/Spring_2024/carbohydrates/data/lexical_file/drug_atom_lexical_txt_11_4_2025.csv"

# ---------------------------------------------------------------------------
# Mode selection
# ---------------------------------------------------------------------------
# MODE controls how output targets are identified:
#   all_4_id_chain_res_seqnum -> one target per protein/chain/entity_name/seqnum
#   id_chain                  -> one output per protein+chain using only those
#                                two CSV columns
#MODE="all_4_id_chain_res_seqnum"
MODE="id_chain"

# ENTITY_TYPE controls only atom selection:
#   ligand  -> heavy atoms from HETATM records
#   residue -> matching ATOM records (including C-alpha) from the specified residue/seqnum
#ENTITY_TYPE="residue"
ENTITY_TYPE="ligand"

# Use these overrides only when the CSV headers differ from the defaults.
PROTEIN_COLUMN=""
CHAIN_COLUMN=""
ENTITY_NAME_COLUMN="carb_name"
SEQNUM_COLUMN="carb_id"

# Leave blank to use expected_atom_counts.json next to the Python script.
ATOM_COUNT_CONFIG=""

# ---------------------------------------------------------------------------
# Key-generation parameters
# ---------------------------------------------------------------------------
DTHETA="29"
DLEN="18"
NUM_LABELS="112"
NUM_CORES="${SLURM_CPUS_PER_TASK:-64}"

mkdir -p "${RESULTS_DIR}"

echo "[INFO] Using mode: ${MODE}"
echo "[INFO] Using entity type: ${ENTITY_TYPE}"
echo "[INFO] Python script: ${PY_SCRIPT}"
echo "[INFO] Target CSV: ${TARGET_CSV}"

CMD=(
    python "${PY_SCRIPT}"
    -p "${PDB_DIR}"
    -c "${TARGET_CSV}"
    -l "${LEXICAL_CSV}"
    -o "${PROTEIN_DIR}"
    -s "${SUMMARY_CSV}"
    -r "${RESULTS_DIR}"
    --mode "${MODE}"
    --entity_type "${ENTITY_TYPE}"
    --dtheta "${DTHETA}"
    --dlen "${DLEN}"
    --num_labels "${NUM_LABELS}"
    --num_cores "${NUM_CORES}"
)

if [ -n "${PROTEIN_COLUMN}" ]; then
    CMD+=(--protein_column "${PROTEIN_COLUMN}")
fi
if [ -n "${CHAIN_COLUMN}" ]; then
    CMD+=(--chain_column "${CHAIN_COLUMN}")
fi
if [ -n "${ENTITY_NAME_COLUMN}" ]; then
    CMD+=(--entity_name_column "${ENTITY_NAME_COLUMN}")
fi
if [ -n "${SEQNUM_COLUMN}" ]; then
    CMD+=(--seqnum_column "${SEQNUM_COLUMN}")
fi
if [ -n "${ATOM_COUNT_CONFIG}" ]; then
    CMD+=(--atom_count_config "${ATOM_COUNT_CONFIG}")
fi

"${CMD[@]}"

echo "[INFO] Job finished successfully"
