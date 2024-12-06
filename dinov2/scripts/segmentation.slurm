#!/bin/bash
#SBATCH -p a100_short,radiology,a100_long
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80GB
#SBATCH --time=12:00:00
#SBATCH --job-name=segmentation
#SBATCH --output=/gpfs/data/shenlab/wz1492/HCC/dinov2/logs/seg-%J.log
#SBATCH --exclude=a100-4020

echo -e "GPUS = $CUDA_VISIBLE_DEVICES\n"
nvidia-smi

# activate conda environment
module load gcc/8.1.0
source ~/.bashrc 
conda activate dinov2

python segmentation.py
