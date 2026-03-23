#!/bin/bash
#SBATCH -p workq
#SBATCH -n 64
#SBATCH -t 01:00:00
#SBATCH -A loni_tsr_9
#SBATCH -N 1
#SBATCH -J clustermap
#SBATCH -o output_clustermap.out
#SBATCH -e error_clustermap.err

module load python/3.11.5-anaconda

python3 -m venv myenv
source myenv/bin/activate
pip3 install --upgrade pip
pip3 install -r /ddnB/work/rauniyar/Spring_2024/carbohydrates/clusters/clustermap/requirements.txt

python3 /ddnB/work/rauniyar/Spring_2024/carbohydrates/clusters/clustermap/clustermap.py -p /ddnB/work/rauniyar/Spring_2024/carbohydrates/clusters/clustermap/generalised.csv

