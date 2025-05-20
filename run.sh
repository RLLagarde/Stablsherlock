#!/usr/bin/bash
#SBATCH --job-name=sr5
#SBATCH --error=./logs/sr5_e_%a.err
#SBATCH --output=./logs/sr5_e_%a.out
#SBATCH --array=0-20:10
#SBATCH --time=48:00:00
#SBATCH -p normal
#SBATCH -c 8
#SBATCH --mem=8GB

ml python/3.12.1

for ((i=0; i<10; i++)); do
    CURRENT_INDEX=$((SLURM_ARRAY_TASK_ID + i))
    
    echo "Processing index $CURRENT_INDEX"
    time python3 ./stablExperiment.py ${CURRENT_INDEX}
done


