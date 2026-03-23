#!/bin/bash
#SBATCH -p workq
#SBATCH -n 64
#SBATCH -t 01:00:00
#SBATCH -A loni_tsr_9
#SBATCH -N 1
#SBATCH -J key_frequency_drug
#SBATCH -o output_key_frequency_drug.out
#SBATCH -e error_key_frequency_drug.err

python3 /ddnB/work/rauniyar/Spring_2024/carbohydrates/key_generator/key_frequency_drug.py -r /ddnB/work/rauniyar/Spring_2024/carbohydrates/key_generator/ -p proteins -H no