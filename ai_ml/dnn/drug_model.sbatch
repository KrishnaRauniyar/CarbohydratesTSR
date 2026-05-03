#!/bin/bash
#SBATCH -p workq
#SBATCH -n 64
#SBATCH -t 72:00:00
#SBATCH -A loni_tsr_3
#SBATCH -N 1
#SBATCH -J drug_model
#SBATCH -o output_drug_model.out
#SBATCH -e error_drug_model.err

python3 -m venv myenv
source myenv/bin/activate
pip install --upgrade pip
pip install -r /ddnB/work/rauniyar/Spring_2024/drugs/dnn/requirements.txt

python3 /ddnB/work/rauniyar/Spring_2024/drugs/dnn/drug_model.py -p /ddnB/work/rauniyar/Spring_2024/drugs/dnn/localFeatureVect_theta29_dist18_NoFeatureSelection_keyCombine0_header.csv
