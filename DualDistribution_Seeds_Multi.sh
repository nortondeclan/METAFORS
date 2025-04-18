#!/bin/bash

for t in {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40}
do
	for m in {"async_sm_ri","vanilla","multitask"}
	do
		for s in {1,2,3,4,5,6,7,8,9,10}
		do
			sbatch --job-name "T$t $m" DualDistribution_Seeds.sh -t $t -m $m -s $s
			echo "Running Method $m with test length $t with seed $s."
		done
	done
done
