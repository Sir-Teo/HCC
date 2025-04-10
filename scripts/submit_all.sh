#!/bin/bash

# --- Configuration ---
LOG_DIR="/gpfs/data/shenlab/wz1492/HCC/logs/batch_submission" # Directory for SLURM logs
BASE_OUTPUT_DIR_SURVIVAL="checkpoints_survival_cv" # Base output for train.py results
BASE_OUTPUT_DIR_BINARY="checkpoints_binary_cv"   # Base output for train_binary.py results
CONDA_ENV_NAME="dinov2"

# --- Base SLURM Options ---
# These will be included in every submitted job script
SBATCH_OPTS=(
"#SBATCH -p radiology,a100_short"      # Partitions
"#SBATCH --gres=gpu:1"                # Request 1 GPU
"#SBATCH --nodes=1"                   # Request 1 node
"#SBATCH --ntasks=1"                  # Request 1 task
"#SBATCH --cpus-per-task=16"          # CPUs per task
"#SBATCH --mem=120GB"                 # Memory
"#SBATCH --time=12:00:00"             # Max walltime
"#SBATCH --exclude=a100-4020"         # Exclude specific nodes if needed
)

# --- Common Python Script Arguments ---
# Arguments shared between train.py and train_binary.py
COMMON_ARGS=(
"--nyu_dicom_root /gpfs/data/mankowskilab/HCC_Recurrence/dicom"
"--tcga_dicom_root /gpfs/data/mankowskilab/HCC/data/TCGA/manifest-4lZjKqlp5793425118292424834/TCGA-LIHC"
"--tcga_csv_file /gpfs/data/shenlab/wz1492/HCC/spreadsheets/tcga.csv"
"--nyu_csv_file /gpfs/data/shenlab/wz1492/HCC/spreadsheets/processed_patient_labels_nyu.csv"
"--batch_size 32"
"--num_slices 32"
"--preprocessed_root /gpfs/data/mankowskilab/HCC_Recurrence/preprocessed/"
"--learning_rate 1e-5"
"--num_samples_per_patient 1"
"--gradient_clip 1.0" # Using 1.0 as per last script update
"--upsampling"
"--dinov2_weights /gpfs/data/shenlab/wz1492/HCC/dinov2/experiments_large_dataset/eval/training_124999/teacher_checkpoint.pth"
"--epochs 1000" # Using 1000 as per slurm scripts
"--early_stopping"
"--early_stopping_patience 10"
# --cross_validation is implicitly handled by running these scripts
)

# --- Specific Python Script Arguments ---
SURVIVAL_ARGS=(
"--center_risk"
"--coxph_net mlp"
"--alpha 0.5"
"--gamma 0.5"
)
# train_binary.py doesn't have extra specific args listed here currently

# --- Create Log Directory ---
mkdir -p "$LOG_DIR"

# --- Loop and Submit ---
echo "Starting SLURM job submission loop..."

# --- Submit Cross-dataset Evaluation Jobs ---
echo "Submitting cross-dataset evaluation jobs (train on X, test on Y)..."

# Configurations for cross-dataset evaluation:
# 1. Train on NYU, predict on TCGA
# 2. Train on TCGA, predict on NYU
# Both for binary and time-to-event models
for script_name in "train.py" "train_binary.py"; do
    echo "  Cross-Dataset Evaluation for Script: $script_name"
    
    specific_args=()
    base_output_dir=""
    if [ "$script_name" == "train.py" ]; then
        specific_args=("${SURVIVAL_ARGS[@]}")
        base_output_dir="$BASE_OUTPUT_DIR_SURVIVAL"
    else # train_binary.py
        base_output_dir="$BASE_OUTPUT_DIR_BINARY"
    fi

    # Cross-dataset scenarios
    for cross_mode in "nyu_to_tcga" "tcga_to_nyu"; do
        echo "    Mode: $cross_mode"
        
        # Parse train and test sources
        if [ "$cross_mode" == "nyu_to_tcga" ]; then
            train_source="nyu"
            test_source="tcga"
        else
            train_source="tcga"
            test_source="nyu"
        fi
        
        # --- Construct Job Name and Log File Path ---
        script_prefix="${script_name%.py}" # train or train_binary
        JOB_NAME="hcc_${script_prefix}_${cross_mode}"
        LOG_FILE="${LOG_DIR}/${JOB_NAME}_%J.log" # %J is the SLURM job ID

        # --- Construct Output Directory for Python Script ---
        PYTHON_OUTPUT_BASE="${base_output_dir}/cross_${cross_mode}"

        # --- Prepare Python Command ---
        PYTHON_CMD="python $script_name"
        PYTHON_CMD="$PYTHON_CMD ${COMMON_ARGS[*]}"
        PYTHON_CMD="$PYTHON_CMD ${specific_args[*]}"
        PYTHON_CMD="$PYTHON_CMD --cross_validation" # Still use CV infrastructure
        PYTHON_CMD="$PYTHON_CMD --cv_mode $train_source" # Train on this dataset
        PYTHON_CMD="$PYTHON_CMD --cv_folds 1" # Special case - one fold for cross-dataset prediction
        PYTHON_CMD="$PYTHON_CMD --output_dir $PYTHON_OUTPUT_BASE"
        # Add argument to switch test dataset
        PYTHON_CMD="$PYTHON_CMD --cross_predict $test_source"

        # --- Create Temporary Job Script ---
        TEMP_SLURM_SCRIPT=$(mktemp /tmp/hcc_job_XXXXXX.slurm)
        
        echo "#!/bin/bash" > "$TEMP_SLURM_SCRIPT"
        # Add SBATCH options
        for opt in "${SBATCH_OPTS[@]}"; do
            echo "$opt" >> "$TEMP_SLURM_SCRIPT"
        done
        # Add job-specific name and output
        echo "#SBATCH --job-name=$JOB_NAME" >> "$TEMP_SLURM_SCRIPT"
        echo "#SBATCH --output=$LOG_FILE" >> "$TEMP_SLURM_SCRIPT"
        echo "" >> "$TEMP_SLURM_SCRIPT"
        
        # Add commands to run
        echo "echo \"--- Starting SLURM Job --- \"" >> "$TEMP_SLURM_SCRIPT"
        echo "echo \"Job Name: \$SLURM_JOB_NAME (\$SLURM_JOB_ID)\"" >> "$TEMP_SLURM_SCRIPT"
        echo "echo \"Running on: \$SLURMD_NODENAME\"" >> "$TEMP_SLURM_SCRIPT"
        echo "echo \"GPU: \$CUDA_VISIBLE_DEVICES\"" >> "$TEMP_SLURM_SCRIPT"
        echo "echo \"Conda Env: $CONDA_ENV_NAME\"" >> "$TEMP_SLURM_SCRIPT"
        echo "echo \"Python Script: $script_name\"" >> "$TEMP_SLURM_SCRIPT"
        echo "echo \"Cross-Dataset Mode: Train on $train_source, Test on $test_source\"" >> "$TEMP_SLURM_SCRIPT"
        echo "echo \"--------------------------\"" >> "$TEMP_SLURM_SCRIPT"
        echo "" >> "$TEMP_SLURM_SCRIPT"
        
        echo "nvidia-smi" >> "$TEMP_SLURM_SCRIPT"
        echo "nvcc --version" >> "$TEMP_SLURM_SCRIPT"
        echo "" >> "$TEMP_SLURM_SCRIPT"

        echo "# Load modules and activate environment" >> "$TEMP_SLURM_SCRIPT"
        echo "module load gcc/8.1.0" >> "$TEMP_SLURM_SCRIPT"
        echo "source ~/.bashrc" >> "$TEMP_SLURM_SCRIPT"
        echo "conda activate $CONDA_ENV_NAME" >> "$TEMP_SLURM_SCRIPT"
        echo "" >> "$TEMP_SLURM_SCRIPT"

        echo "echo \"Executing command:\"" >> "$TEMP_SLURM_SCRIPT"
        echo "echo \"srun $PYTHON_CMD\"" >> "$TEMP_SLURM_SCRIPT"
        echo "" >> "$TEMP_SLURM_SCRIPT"
        
        echo "srun $PYTHON_CMD" >> "$TEMP_SLURM_SCRIPT"
        echo "" >> "$TEMP_SLURM_SCRIPT"
        echo "echo \"--- SLURM Job Finished --- \"" >> "$TEMP_SLURM_SCRIPT"

        # --- Submit Job ---
        echo "  Submitting: $JOB_NAME (Log: ${LOG_FILE/\%J/<job_id>})"
        sbatch "$TEMP_SLURM_SCRIPT"
        
        # Clean up temporary script
        rm "$TEMP_SLURM_SCRIPT"
    done # cross_mode
done # script_name

# Original CV mode jobs
for script_name in "train.py" "train_binary.py"; do
    echo "  Script: $script_name"
    
    specific_args=()
    base_output_dir=""
    if [ "$script_name" == "train.py" ]; then
        specific_args=("${SURVIVAL_ARGS[@]}")
        base_output_dir="$BASE_OUTPUT_DIR_SURVIVAL"
    else # train_binary.py
        base_output_dir="$BASE_OUTPUT_DIR_BINARY"
        # Add any binary-specific args here if needed in the future
    fi

    for mode in "combined" "tcga" "nyu"; do
        echo "    Mode: $mode"
        for fold_strat in "10fold" "loocv"; do
            echo "      Fold Strategy: $fold_strat"

            # --- Determine Fold Args ---
            fold_args_list=()
            fold_suffix=""
            if [ "$fold_strat" == "loocv" ]; then
                fold_args_list+=("--leave_one_out")
                fold_suffix="loocv"
            else
                fold_args_list+=("--cv_folds 10") # Explicitly set 10 folds
                fold_suffix="10fold"
            fi

            # --- Construct Job Name and Log File Path ---
            script_prefix="${script_name%.py}" # train or train_binary
            JOB_NAME="hcc_${script_prefix}_${mode}_${fold_suffix}"
            LOG_FILE="${LOG_DIR}/${JOB_NAME}_%J.log" # %J is the SLURM job ID

            # --- Construct Output Directory for Python Script ---
            # This uses the run_name structure defined within the Python scripts
            # We just provide the base directory
            PYTHON_OUTPUT_BASE="${base_output_dir}" # Python script will add its own subfolder

            # --- Prepare Python Command ---
            PYTHON_CMD="python $script_name"
            PYTHON_CMD="$PYTHON_CMD ${COMMON_ARGS[*]}"
            PYTHON_CMD="$PYTHON_CMD ${specific_args[*]}"
            PYTHON_CMD="$PYTHON_CMD --cv_mode $mode"
            PYTHON_CMD="$PYTHON_CMD ${fold_args_list[*]}"
            PYTHON_CMD="$PYTHON_CMD --output_dir $PYTHON_OUTPUT_BASE" # Pass base output dir

            # --- Create Temporary Job Script ---
            TEMP_SLURM_SCRIPT=$(mktemp /tmp/hcc_job_XXXXXX.slurm)
            
            echo "#!/bin/bash" > "$TEMP_SLURM_SCRIPT"
            # Add SBATCH options
            for opt in "${SBATCH_OPTS[@]}"; do
                echo "$opt" >> "$TEMP_SLURM_SCRIPT"
            done
            # Add job-specific name and output
            echo "#SBATCH --job-name=$JOB_NAME" >> "$TEMP_SLURM_SCRIPT"
            echo "#SBATCH --output=$LOG_FILE" >> "$TEMP_SLURM_SCRIPT"
            echo "" >> "$TEMP_SLURM_SCRIPT"
            
            # Add commands to run
            echo "echo \"--- Starting SLURM Job --- \"" >> "$TEMP_SLURM_SCRIPT"
            echo "echo \"Job Name: \$SLURM_JOB_NAME (\$SLURM_JOB_ID)\"" >> "$TEMP_SLURM_SCRIPT"
            echo "echo \"Running on: \$SLURMD_NODENAME\"" >> "$TEMP_SLURM_SCRIPT"
            echo "echo \"GPU: \$CUDA_VISIBLE_DEVICES\"" >> "$TEMP_SLURM_SCRIPT"
            echo "echo \"Conda Env: $CONDA_ENV_NAME\"" >> "$TEMP_SLURM_SCRIPT"
            echo "echo \"Python Script: $script_name\"" >> "$TEMP_SLURM_SCRIPT"
            echo "echo \"CV Mode: $mode\"" >> "$TEMP_SLURM_SCRIPT"
            echo "echo \"Fold Strategy: $fold_strat\"" >> "$TEMP_SLURM_SCRIPT"
            echo "echo \"--------------------------\"" >> "$TEMP_SLURM_SCRIPT"
            echo "" >> "$TEMP_SLURM_SCRIPT"
            
            echo "nvidia-smi" >> "$TEMP_SLURM_SCRIPT"
            echo "nvcc --version" >> "$TEMP_SLURM_SCRIPT"
            echo "" >> "$TEMP_SLURM_SCRIPT"

            echo "# Load modules and activate environment" >> "$TEMP_SLURM_SCRIPT"
            echo "module load gcc/8.1.0" >> "$TEMP_SLURM_SCRIPT"
            echo "source ~/.bashrc" >> "$TEMP_SLURM_SCRIPT"
            echo "conda activate $CONDA_ENV_NAME" >> "$TEMP_SLURM_SCRIPT"
            echo "" >> "$TEMP_SLURM_SCRIPT"

            echo "echo \"Executing command:\"" >> "$TEMP_SLURM_SCRIPT"
            echo "echo \"srun $PYTHON_CMD\"" >> "$TEMP_SLURM_SCRIPT"
            echo "" >> "$TEMP_SLURM_SCRIPT"
            
            echo "srun $PYTHON_CMD" >> "$TEMP_SLURM_SCRIPT"
            echo "" >> "$TEMP_SLURM_SCRIPT"
            echo "echo \"--- SLURM Job Finished --- \"" >> "$TEMP_SLURM_SCRIPT"

            # --- Submit Job ---
            echo "  Submitting: $JOB_NAME (Log: ${LOG_FILE/\%J/<job_id>})"
            sbatch "$TEMP_SLURM_SCRIPT"
            
            # Clean up temporary script
            rm "$TEMP_SLURM_SCRIPT"

        done # fold_strat
    done # mode
done # script_name

echo "--- All jobs submitted ---"