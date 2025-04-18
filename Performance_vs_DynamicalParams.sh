#!/bin/bash
#SBATCH -t 1000:00
#SBATCH -n 20
#SBATCH --mem-per-cpu=6144
#SBATCH --mail-type=ALL

source /home/user/miniconda3/etc/profile.d/conda.sh #Activate conda from appropriate directory location
conda activate metafors_env

python3 -u Performance_vs_DynamicalParams.py
