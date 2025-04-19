#!/bin/bash
#SBATCH -p radiology,a100_short,a100_long
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120GB
#SBATCH --time=12:00:00
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

srun python train_nested.py \
    --nyu_dicom_root  /gpfs/data/mankowskilab/HCC_Recurrence/dicom \
    --tcga_dicom_root /gpfs/data/mankowskilab/HCC/data/TCGA/manifest-4lZjKqlp5793424834/TCGA-LIHC \
    --tcga_csv_file /gpfs/data/shenlab/wz1492/HCC/spreadsheets/tcga.csv \
    --nyu_csv_file /gpfs/data/shenlab/wz1492/HCC/spreadsheets/processed_patient_labels_nyu.csv \
    --batch_size 32 \
    --num_slices 32 \
    --preprocessed_root /gpfs/data/mankowskilab/HCC_Recurrence/preprocessed/ \
    --learning_rate 5e-7 \
    --num_samples_per_patient 1 \
    --center_risk \
    --coxph_net mlp \
    --upsampling \
    --upsampling_method smote \
    --dinov2_weights /gpfs/data/shenlab/wz1492/HCC/dinov2/experiments_large_dataset/eval/training_124999/teacher_checkpoint.pth \
    --epochs 5000 \
    --alpha 0.5 \
    --gamma 0.5 \
    --early_stopping \
    --inner_folds 6 \
    --learning_rates 1e-7 1e-6 1e-5 1e-4 \
    --cv_folds 7












