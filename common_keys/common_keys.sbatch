#!/bin/bash
#SBATCH -p workq
#SBATCH -n 64
#SBATCH -t 01:00:00
#SBATCH -A loni_tsr_9
#SBATCH -N 1
#SBATCH -J common_keys
#SBATCH -o output_common_keys.out
#SBATCH -e error_common_keys.err

python3 -m venv myenv
source myenv/bin/activate
pip install --upgrade pip
pip install -r /ddnB/work/rauniyar/Spring_2024/carbohydrates/common_keys/requirements.txt

python3 /ddnB/work/rauniyar/Spring_2024/carbohydrates/common_keys/common_keys.py --path /ddnB/work/rauniyar/Spring_2024/carbohydrates/key_generator/proteins 
