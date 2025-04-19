#!/bin/bash
#SBATCH -p a100_short,radiology,a100_long,a100_dev
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120GB
#SBATCH --time=3:00:00
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

srun python train_binary_nested.py \
    --nyu_dicom_root  /gpfs/data/mankowskilab/HCC_Recurrence/dicom \
    --tcga_dicom_root /gpfs/data/mankowskilab/HCC/data/TCGA/manifest-4lZjKqlp5793424834/TCGA-LIHC \
    --tcga_csv_file /gpfs/data/shenlab/wz1492/HCC/spreadsheets/tcga.csv \
    --nyu_csv_file /gpfs/data/shenlab/wz1492/HCC/spreadsheets/processed_patient_labels_nyu.csv \
    --batch_size 32 \
    --num_slices 0 \
    --learning_rate 1e-5 \
    --inner_folds 6 \
    --learning_rates 1e-8 5e-8 1e-7 5e-6 1e-6 5e-5 1e-5 1e-4 1e-3 \
    --num_samples_per_patient 1 \
    --preprocessed_root /gpfs/data/mankowskilab/HCC_Recurrence/preprocessed/ \
    --upsampling \
    --upsampling_method smote \
    --dinov2_weights /gpfs/data/shenlab/wz1492/HCC/dinov2/experiments_large_dataset/eval/training_124999/teacher_checkpoint.pth \
    --epochs 5000 \
    --early_stopping \
    --cv_folds 7












