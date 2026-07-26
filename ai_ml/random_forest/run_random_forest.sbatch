#!/bin/bash
#SBATCH -p workq
#SBATCH -N 1
#SBATCH -n 64
#SBATCH -t 24:00:00
#SBATCH -A loni_tsr_9
#SBATCH -J tsr_rf
#SBATCH -o outputs/slurm_logs/%x_%j.out
#SBATCH -e outputs/slurm_logs/%x_%j.err

###############################################################################
# Safety & environment
set -euo pipefail

module load python/3.11.5-anaconda

# Submit from:
# /ddnB/work/rauniyar/Spring_2024/carbohydrates/ai_ml/random_forest
cd "${SLURM_SUBMIT_DIR}"

###############################################################################
# Outputs + logs
mkdir -p outputs/slurm_logs
# OUTPUT_DIR="${SLURM_SUBMIT_DIR}/outputs_chain"
OUTPUT_DIR="${SLURM_SUBMIT_DIR}/outputs_individual"
ENV_DIR="${SLURM_SUBMIT_DIR}/.conda_tsr_rf"

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
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

###############################################################################
# Running Random Forest pipeline
echo "[INFO] Starting TSR Random Forest training"

INPUT_CSV="/ddnB/work/rauniyar/Spring_2024/carbohydrates/data/ai_ml/all_4_id_chain_res_seq/localFeatureVect_theta29_dist18_NoFeatureSelection_keyCombine0_header.csv"
# INPUT_CSV="/ddnB/work/rauniyar/Spring_2024/carbohydrates/data/ai_ml/chain/localFeatureVect_theta29_dist18_NoFeatureSelection_keyCombine0_header.csv"

python train_random_forest_tsr.py \
  --input_csv "${INPUT_CSV}" \
  --output_dir "${OUTPUT_DIR}" \
  --label_from "Protein Name" \
  --split_strategy pdb_grouped \
  --balance_method class_weight \
  --use_log1p \
  --scaler none \
  --n_estimators 500 \
  --class_weight_mode balanced_subsample \
  --n_jobs 64 \
  --random_state 42 \
  --backend sklearn

echo "[INFO] Job finished successfully"
