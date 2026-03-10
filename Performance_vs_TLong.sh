#!/bin/bash
#SBATCH -t 600:00		# Time limit
#SBATCH -n 17			# Memory per cpu core
#SBATCH --mem-per-cpu=6144	# One core per sample library signal length
#SBATCH --mail-type=ALL		# Send email notifications

source /home/user/miniconda3/etc/profile.d/conda.sh #Activate conda from appropriate directory location
conda activate metafors_env

python3 -u Performance_vs_TLong.py
