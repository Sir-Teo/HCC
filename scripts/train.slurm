#!/bin/bash
#SBATCH -p a100_short,radiology
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120GB
#SBATCH --time=3-00:00:00
#SBATCH --job-name=hcc
#SBATCH --output=/gpfs/data/shenlab/wz1492/HCC/logs/train-%J.log
#SBATCH --exclude=a100-4020

echo -e "GPUS = $CUDA_VISIBLE_DEVICES\n"
nvidia-smi
nvcc --version

# activate conda environment
module load gcc/8.1.0
source ~/.bashrc 
conda activate dinov2

srun python train.py \
    --batch_size 64 \
    --num_slices  64\
    --preprocessed_root /gpfs/data/mankowskilab/HCC_Recurrence/preprocessed/ \
    --learning_rate 1e-5 \
    --num_samples_per_patient 1\
    --gradient_clip 0.05\
    --center_risk \
    --coxph_net mlp \
    --upsampling \
    --dinov2_weights /gpfs/data/shenlab/wz1492/HCC/dinov2/experiments_large_dataset/eval/training_124999/teacher_checkpoint.pth \
    --epochs 1000 \
    --alpha 0.5 \
    --gamma 0.5 \
    --early_stopping \
    --cross_validation \









