#!/bin/bash
#SBATCH -p bigmem
#SBATCH -N 1
#SBATCH -n 64
#SBATCH --mem=1800G
#SBATCH -t 24:00:00
#SBATCH -A loni_tsr_9
#SBATCH -J tsr_clustermap
#SBATCH -o outputs/slurm_logs/%x_%j.out
#SBATCH -e outputs/slurm_logs/%x_%j.err

###############################################################################
# Safety & environment
set -euo pipefail

module load python/3.11.5-anaconda

# Submit from:
cd "${SLURM_SUBMIT_DIR}"

###############################################################################
# CPU threading
# Hierarchical clustering itself is mostly single-node CPU work. MiniBatchKMeans
# and numeric libraries can use OpenMP/BLAS threads, so bind them to Slurm cores.
export OMP_NUM_THREADS="${SLURM_NTASKS:-64}"
export OPENBLAS_NUM_THREADS="${SLURM_NTASKS:-64}"
export MKL_NUM_THREADS="${SLURM_NTASKS:-64}"
export NUMEXPR_NUM_THREADS="${SLURM_NTASKS:-64}"

###############################################################################
# Outputs + logs
mkdir -p outputs/slurm_logs
OUTPUT_DIR="${SLURM_SUBMIT_DIR}/outputs"
ENV_DIR="${SLURM_SUBMIT_DIR}/.conda_tsr_clustermap"

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
python -m pip install --upgrade -r requirements.txt

###############################################################################
# Running clustermap pipeline
#
# Modes:
#   auto  - recommended default. Runs exact mode when the matrix has
#           MAX_EXACT_ITEMS rows or fewer; otherwise switches to large mode.
#   exact - keeps the original scientific behavior: full average-linkage
#           hierarchical clustering and an exact row/column order. This can still
#           be slow or memory-heavy because it needs O(n^2) memory.
#   large - designed for very large matrices. It assigns every row to a cluster
#           with mini-batch k-means, then draws a representative hierarchical
#           clustermap from SAMPLE_SIZE rows. This is the minutes-oriented mode.
#
# Large-mode assignment backend:
#   numpy   - default. Avoids a known scikit-learn/threadpoolctl/OpenBLAS issue
#             seen on some Anaconda HPC modules.
#   sklearn - uses sklearn.cluster.MiniBatchKMeans. Try only after threadpoolctl
#             upgrades cleanly in the conda environment, and install
#             scikit-learn in the environment first.
#
# Common overrides at submit time:
#   INPUT_CSV=/path/to/generalised.csv sbatch clustermap.sbatch
#   CLUSTERMAP_MODE=large SAMPLE_SIZE=3000 sbatch clustermap.sbatch
#   CLUSTERMAP_MODE=exact MAX_EXACT_ITEMS=20000 sbatch clustermap.sbatch
#   FEATURE_SAMPLE_SIZE=4096 sbatch clustermap.sbatch
#   ASSIGNMENT_BACKEND=sklearn sbatch clustermap.sbatch
#
# Input name rule:
#   The residue is read from the last underscore-separated part of the row name:
#   "anything_anything_RESIDUE" -> RESIDUE
###############################################################################
echo "[INFO] Starting TSR clustermap job"
echo "[INFO] CLUSTERMAP_MODE=auto  : exact up to MAX_EXACT_ITEMS rows, else large mode."
echo "[INFO] CLUSTERMAP_MODE=exact : full hierarchical clustering; highest memory/time cost."
echo "[INFO] CLUSTERMAP_MODE=large : full assignments plus representative clustermap."
echo "[INFO] ASSIGNMENT_BACKEND=numpy avoids sklearn/threadpoolctl OpenBLAS errors."
echo "[INFO] Residue parsing rule   : use the final token after the last underscore."

INPUT_CSV="${INPUT_CSV:-/ddnB/work/rauniyar/Spring_2024/carbohydrates/clusters/clustermap/generalised.csv}"
CLUSTERMAP_MODE="${CLUSTERMAP_MODE:-auto}"
MAX_EXACT_ITEMS="${MAX_EXACT_ITEMS:-5000}" # rows <= 5000  -> exact, rows > 5000 -> large
SAMPLE_SIZE="${SAMPLE_SIZE:-2500}"         # only draw a clustermap for 2500 representative rows.
FEATURE_SAMPLE_SIZE="${FEATURE_SAMPLE_SIZE:-4096}"
ASSIGNMENT_BACKEND="${ASSIGNMENT_BACKEND:-numpy}"
KMEANS_MAX_ITER="${KMEANS_MAX_ITER:-20}"

echo "[INFO] INPUT_CSV=${INPUT_CSV}"
echo "[INFO] CLUSTERMAP_MODE=${CLUSTERMAP_MODE}"
echo "[INFO] MAX_EXACT_ITEMS=${MAX_EXACT_ITEMS}"
echo "[INFO] SAMPLE_SIZE=${SAMPLE_SIZE}"
echo "[INFO] FEATURE_SAMPLE_SIZE=${FEATURE_SAMPLE_SIZE}"
echo "[INFO] ASSIGNMENT_BACKEND=${ASSIGNMENT_BACKEND}"
echo "[INFO] KMEANS_MAX_ITER=${KMEANS_MAX_ITER}"

python clustermap.py \
  --input_path "${INPUT_CSV}" \
  --output_dir "${OUTPUT_DIR}" \
  --mode "${CLUSTERMAP_MODE}" \
  --max_exact_items "${MAX_EXACT_ITEMS}" \
  --sample_size "${SAMPLE_SIZE}" \
  --feature_sample_size "${FEATURE_SAMPLE_SIZE}" \
  --batch_size 4096 \
  --assignment_backend "${ASSIGNMENT_BACKEND}" \
  --kmeans_max_iter "${KMEANS_MAX_ITER}" \
  --seaborn_max_items 1500 \
  --max_plot_items 3000 \
  --random_state 42

echo "[INFO] Job finished successfully"
