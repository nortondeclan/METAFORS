#!/bin/bash
#SBATCH -t 600:00		# Time limit
#SBATCH -n 20			# One core per sample test signal length
#SBATCH --mail-type=ALL		# Receive email notifications

source /home/user/miniconda3/etc/profile.d/conda.sh #Activate conda from appropriate directory location
conda activate metafors_env

python3 -u Performance_vs_TShort.py
