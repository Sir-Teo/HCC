#!/bin/bash
#SBATCH -p a100_short,radiology,a100_long,a100_dev
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120GB
#SBATCH --time=6:00:00
#SBATCH --job-name=hcc_precision_boost
#SBATCH --output=/gpfs/data/shenlab/wz1492/HCC/logs/batch_submission/hcc_precision_boost_%J.log
#SBATCH --exclude=a100-4020

echo "=== ULTRA-PRECISION OPTIMIZATION RUN ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo -e "GPUS = $CUDA_VISIBLE_DEVICES\n"
nvidia-smi
nvcc --version

# activate conda environment
module load gcc/8.1.0
source ~/.bashrc 
conda activate dinov2

echo "=== Starting Ultra-Precision Training with Advanced Architectures ==="

srun python train_binary.py \
    --nyu_dicom_root  /gpfs/data/mankowskilab/HCC_Recurrence/dicom \
    --nyu_csv_file /gpfs/data/shenlab/wz1492/HCC/spreadsheets/processed_patient_labels_nyu.csv \
    --batch_size 8 \
    --num_slices 32 \
    --learning_rate 1e-5 \
    --num_samples_per_patient 1 \
    --preprocessed_root /gpfs/data/mankowskilab/HCC_Recurrence/preprocessed/ \
    --upsampling \
    --upsampling_method adasyn \
    --dinov2_weights /gpfs/data/shenlab/wz1492/HCC/dinov2/experiments_large_dataset/eval/training_124999/teacher_checkpoint.pth \
    --epochs 1000 \
    --early_stopping \
    --early_stopping_patience 50 \
    --cv_folds 7 \
    --hyper_search_iters 100 \
    --model_arch hyper_precision \
    --dropout 0.2 \
    --adaptive_focal \
    --threshold_metric auprc \
    --output_dir checkpoints_binary_cv

echo "=== Training completed at $(date) ==="
echo "=== Final Results Summary ==="
# Find the best run directory and show results
find checkpoints_binary_cv -name "best_run" -type d | head -1 | xargs -I {} cat {}/run_summary.json | jq '.metrics.overall'












