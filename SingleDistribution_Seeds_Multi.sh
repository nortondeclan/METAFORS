#!/bin/bash
#SBATCH --mail-type=ALL

for t in {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20}
do
	for m in {"async_sm","library_interpolation","async_sm_ri","vanilla","multitask"}
	do
		for s in {1,2,3,4,5,6,7,8,9,10}
		do
			sbatch --job-name "T$t $m" SingleDistribution_Seeds.sh -t $t -m $m -s $s
			echo "Running Method $m with Test Length $t and Seed $s."
		done
	done
done
