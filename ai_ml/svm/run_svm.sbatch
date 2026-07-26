#!/bin/bash
#SBATCH -p workq
#SBATCH -N 1
#SBATCH -n 64
#SBATCH -t 72:00:00
#SBATCH -A loni_tsr_9
#SBATCH -J tsr_svm
#SBATCH -o outputs/logs/%x_%j.out
#SBATCH -e outputs/logs/%x_%j.err

###############################################################################
# Safety & environment
set -euo pipefail

module load python/3.11.5-anaconda

# Submit from:
# /ddnB/work/rauniyar/Spring_2024/carbohydrates/ai_ml/svm
cd "${SLURM_SUBMIT_DIR}"

###############################################################################
# Outputs + logs
mkdir -p outputs/logs
OUTPUT_DIR="${SLURM_SUBMIT_DIR}/outputs_chain"
# OUTPUT_DIR="${SLURM_SUBMIT_DIR}/outputs_individual"
ENV_DIR="${SLURM_SUBMIT_DIR}/.conda_tsr_svm"

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
# Running SVM pipeline
echo "[INFO] Starting TSR SVM CPU training"

# INPUT_CSV="/ddnB/work/rauniyar/Spring_2024/carbohydrates/data/ai_ml/all_4_id_chain_res_seq/localFeatureVect_theta29_dist18_NoFeatureSelection_keyCombine0_header.csv"
INPUT_CSV="/ddnB/work/rauniyar/Spring_2024/carbohydrates/data/ai_ml/chain/localFeatureVect_theta29_dist18_NoFeatureSelection_keyCombine0_header.csv"

python train_svm_tsr.py \
  --input_csv "${INPUT_CSV}" \
  --output_dir "${OUTPUT_DIR}" \
  --label_from "Protein Name" \
  --split_strategy pdb_grouped \
  --balance_method class_weight \
  --use_log1p \
  --scaler standard \
  --svm_type linear_svc \
  --C 1.0 \
  --max_iter 5000 \
  --backend sklearn \
  --random_state 42

echo "[INFO] Job finished successfully"
