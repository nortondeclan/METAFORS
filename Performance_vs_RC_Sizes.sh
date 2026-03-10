#!/bin/bash
#SBATCH -t 3600:00
#SBATCH -n 130 #90 #180
#SBATCH --mail-type=ALL

source /home/user/miniconda3/etc/profile.d/conda.sh #Activate conda from appropriate directory location
conda activate metafors_env

python3 -u Performance_vs_RC_Sizes.py
