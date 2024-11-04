#!/bin/bash
#SBATCH -p a100_short,radiology
#SBATCH --gres=gpu:2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120GB
#SBATCH --time=12:00:00
#SBATCH --job-name=dino_pretrained
#SBATCH --output=/gpfs/data/shenlab/wz1492/HCC/dinov2/logs/train-%J.log
#SBATCH --exclude=a100-4020

echo -e "GPUS = $CUDA_VISIBLE_DEVICES\n"
nvidia-smi

# activate conda environment
module load gcc/8.1.0
source ~/.bashrc 
conda activate dinov2

python dinov2/run/eval/linear.py \
    --config-file dinov2/configs/eval/vitl14_pretrain.yaml \
    --output-dir /gpfs/data/shenlab/wz1492/HCC/dinov2/models/eval/training_24999/linear_untrained-$SLURM_JOB_ID \
    --train-dataset ImageNet:split=TRAIN:root=/gpfs/data/mankowskilab/HCC/data/Series_Classification:extra=/gpfs/data/mankowskilab/HCC/data/Series_Classification \
    --val-dataset ImageNet:split=VAL:root=/gpfs/data/mankowskilab/HCC/data/Series_Classification:extra=/gpfs/data/mankowskilab/HCC/data/Series_Classification \


