#!/bin/bash
#SBATCH -p gpu4
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -n 16
#SBATCH -t 24:00:00
#SBATCH -A loni_tsr_9
#SBATCH -J tsr_xgb_gpu
#SBATCH -o outputs/slurm_logs/%x_%j.out
#SBATCH -e outputs/slurm_logs/%x_%j.err

###############################################################################
# Safety & environment
set -euo pipefail

module load python/3.11.5-anaconda

# Submit from:
# /ddnB/work/rauniyar/Spring_2024/carbohydrates/ai_ml/xgboost
cd "${SLURM_SUBMIT_DIR}"

###############################################################################
# Outputs + logs
mkdir -p outputs/slurm_logs
OUTPUT_DIR="${SLURM_SUBMIT_DIR}/outputs"
ENV_DIR="${SLURM_SUBMIT_DIR}/.conda_tsr_xgb"

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
# Running XGBoost pipeline
echo "[INFO] Starting TSR XGBoost GPU training"

INPUT_CSV="/ddnB/work/rauniyar/Spring_2024/carbohydrates/data/ai_ml/all_4_id_chain_res_seq/localFeatureVect_theta29_dist18_NoFeatureSelection_keyCombine0_header.csv"

python train_xgboost_tsr.py \
  --input_csv "${INPUT_CSV}" \
  --output_dir "${OUTPUT_DIR}" \
  --label_from "Protein Name" \
  --split_strategy pdb_grouped \
  --balance_method class_weight \
  --use_log1p \
  --scaler none \
  --n_estimators 1000 \
  --learning_rate 0.05 \
  --max_depth 6 \
  --subsample 0.8 \
  --colsample_bytree 0.8 \
  --early_stopping_rounds 50 \
  --n_jobs 16 \
  --device cuda \
  --random_state 42

echo "[INFO] Job finished successfully"
