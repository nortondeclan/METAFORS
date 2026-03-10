#!/bin/bash
#SBATCH -t 1000:00		# Time limit
#SBATCH --mem-per-cpu=12288	# Memory per cpu core
#SBATCH -n 9 			# One core per sample number of library members
#SBATCH --mail-type=ALL		# Send email notifications

source /home/user/miniconda3/etc/profile.d/conda.sh #Activate conda from appropriate directory location
conda activate metafors_env

python3 -u Performance_vs_NLib.py
