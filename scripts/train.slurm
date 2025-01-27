#!/bin/bash
#SBATCH -p a100_short,a100_dev,a100_long
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120GB
#SBATCH --time=12:00:00
#SBATCH --job-name=dino_ssl
#SBATCH --output=/gpfs/data/shenlab/wz1492/HCC/logs/train-%J.log
#SBATCH --exclude=a100-4020

echo -e "GPUS = $CUDA_VISIBLE_DEVICES\n"
nvidia-smi
nvcc --version

# activate conda environment
module load gcc/8.1.0
source ~/.bashrc 
conda activate dinov2

python train.py \
    --dicom_root /gpfs/data/mankowskilab/HCC_Recurrence/dicom \
    --csv_file patient_labels.csv \
    --lr 1e-4 \
    --batch_size 2 \
    --max_epochs 20 \
    --seed 42 \
    --project_name "HCC-Recurrence" \
    --run_name "test-run" \
    --backbone resnet \
    --model_type linear \
    --num_slices 20

