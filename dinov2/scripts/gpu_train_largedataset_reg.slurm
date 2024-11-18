#!/bin/bash
#SBATCH -p a100_short,a100_dev,a100_long
#SBATCH --gres=gpu:2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120GB
#SBATCH --time=12:00:00
#SBATCH --job-name=dino_reg
#SBATCH --output=/gpfs/data/shenlab/wz1492/HCC/dinov2/logs/train-%J.log
#SBATCH --exclude=a100-4020

echo -e "GPUS = $CUDA_VISIBLE_DEVICES\n"
nvidia-smi
nvcc --version

# activate conda environment
module load gcc/8.1.0
source ~/.bashrc 
conda activate dinov2

python dinov2/run/eval/linear.py \
    --config-file /gpfs/data/shenlab/wz1492/HCC/dinov2/experiments_large_dataset_reg/config.yaml \
    --pretrained-weights /gpfs/data/shenlab/wz1492/HCC/dinov2/experiments_large_dataset_reg/eval/training_112499/teacher_checkpoint.pth \
    --output-dir /gpfs/data/shenlab/wz1492/HCC/dinov2/experiments_large_dataset_reg/eval/models/linear-$SLURM_JOB_ID \
    --train-dataset ImageNet:split=TRAIN:root=/gpfs/data/mankowskilab/HCC/data/Series_Classification:extra=/gpfs/data/mankowskilab/HCC/data/Series_Classification \
    --val-dataset ImageNet:split=VAL:root=/gpfs/data/mankowskilab/HCC/data/Series_Classification:extra=/gpfs/data/mankowskilab/HCC/data/Series_Classification \


