#!/bin/bash
#SBATCH -p a100_short,a100_long
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120GB
#SBATCH --time=12:00:00
#SBATCH --job-name=dino_eval
#SBATCH --output=/gpfs/data/shenlab/wz1492/HCC/dinov2/logs/eval-%J.log
#SBATCH --exclude=a100-4020

# Define model checkpoints
MODELS=(12499 24999 37499 49999 62499 74999 87499 99999 112499 124999 137499 149999)

# Load necessary modules
module load gcc/8.1.0
source ~/.bashrc 
conda activate dinov2

# Loop through models and submit a job for each
for MODEL in "${MODELS[@]}"; do
    OUTPUT_DIR="/gpfs/data/shenlab/wz1492/HCC/dinov2/experiments_superlarge2/eval/models/superlarge_${MODEL}/linear-$SLURM_JOB_ID"
    PRETRAINED_WEIGHTS="/gpfs/data/shenlab/wz1492/HCC/dinov2/experiments_superlarge2/eval/training_${MODEL}/teacher_checkpoint.pth"

    echo -e "Running evaluation for model checkpoint: $MODEL\n"
    echo -e "GPUS = $CUDA_VISIBLE_DEVICES\n"

    # Run Python script
    python dinov2/run/eval/linear.py \
        --config-file /gpfs/data/shenlab/wz1492/HCC/dinov2/experiments_superlarge2/config.yaml \
        --pretrained-weights $PRETRAINED_WEIGHTS \
        --output-dir $OUTPUT_DIR \
        --train-dataset ImageNet:split=TRAIN:root=/gpfs/data/mankowskilab/HCC/data/Series_Classification:extra=/gpfs/data/mankowskilab/HCC/data/Series_Classification \
        --val-dataset ImageNet:split=VAL:root=/gpfs/data/mankowskilab/HCC/data/Series_Classification:extra=/gpfs/data/mankowskilab/HCC/data/Series_Classification
done
