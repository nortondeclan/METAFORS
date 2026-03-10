#!/bin/bash
#SBATCH -t 1000:00		# Time limit
#SBATCH -n 143			# Request one CPU core per sample combination of signal mapper and forecaster RC regularization
#SBATCH --mail-type=ALL		# Receive email notification about run status

source /home/user/miniconda3/etc/profile.d/conda.sh #Activate conda from appropriate directory location
conda activate metafors_env

python3 -u Performance_vs_Regularization.py
