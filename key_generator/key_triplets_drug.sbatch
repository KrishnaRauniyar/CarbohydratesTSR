#!/bin/bash
#SBATCH -p workq
#SBATCH -n 64
#SBATCH -t 01:00:00
#SBATCH -A loni_tsr_9
#SBATCH -N 1
#SBATCH -J key_triplets_drug
#SBATCH -o output_key_triplets_drug.out
#SBATCH -e error_key_triplets_drug.err

python3 -m venv myenv
source myenv/bin/activate
pip install --upgrade pip
pip install -r /ddnB/work/rauniyar/Spring_2024/carbohydrates/key_generator/requirements.txt

python3 /ddnB/work/rauniyar/Spring_2024/carbohydrates/key_generator/key_triplets_drug.py /ddnB/work/rauniyar/Spring_2024/carbohydrates/key_generator/ drug_csv/sample_details_carbohydrates_small.csv drug_csv/drug_atom_lexical_txt_11_4_2025.csv pdb_files proteins


