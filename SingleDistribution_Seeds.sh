#!/bin/bash
#SBATCH -t 600:00

while getopts t:m:s: flag
do
	case "${flag}" in
		t) test_length=${OPTARG};;
		m) forecast_method=${OPTARG};;
		s) seed=${OPTARG};;
	esac
done

source /home/user/miniconda3/etc/profile.d/conda.sh #Activate conda from appropriate directory location
conda activate metafors_env

python3 -u SingleDistribution_Seeds.py $test_length $forecast_method $seed
