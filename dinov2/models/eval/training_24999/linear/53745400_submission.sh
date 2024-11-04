#!/bin/bash

# Parameters
#SBATCH --cpus-per-task=16
#SBATCH --error=/gpfs/data/shenlab/wz1492/HCC/dinov2/models/eval/training_24999/linear/%j_0_log.err
#SBATCH --gpus-per-node=2
#SBATCH --job-name=dinov2:linear
#SBATCH --mem=100GB
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --open-mode=append
#SBATCH --output=/gpfs/data/shenlab/wz1492/HCC/dinov2/models/eval/training_24999/linear/%j_0_log.out
#SBATCH --partition=a100_short
#SBATCH --signal=USR2@120
#SBATCH --time=4319
#SBATCH --wckey=submitit

# command
export SUBMITIT_EXECUTOR=slurm
srun --unbuffered --output /gpfs/data/shenlab/wz1492/HCC/dinov2/models/eval/training_24999/linear/%j_%t_log.out --error /gpfs/data/shenlab/wz1492/HCC/dinov2/models/eval/training_24999/linear/%j_%t_log.err /gpfs/scratch/wz1492/miniconda3/envs/dinov2/bin/python -u -m submitit.core._submit /gpfs/data/shenlab/wz1492/HCC/dinov2/models/eval/training_24999/linear
