#!/bin/bash
#SBATCH -t 1000:00		# Time limit
#SBATCH -n 187			# Request CPU cores, so that all sample combinations of training noise and test lengths run in parallel
#SBATCH --mail-type=ALL		# Receive email notifications of run status

source /home/user/miniconda3/etc/profile.d/conda.sh #Activate conda from appropriate directory location
conda activate metafors_env

python3 -u Performance_vs_Test_Length_w_Noise.py