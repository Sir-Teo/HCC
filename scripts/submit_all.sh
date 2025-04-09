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

# --- Helper function to create and submit SLURM script ---
submit_job() {
    local job_name="$1"
    local log_file="$2"
    local script_name="$3"
    local run_details="$4"
    local python_cmd="$5"

    local temp_slurm_script=$(mktemp /tmp/hcc_job_XXXXXX.slurm)
    
    echo "#!/bin/bash" > "$temp_slurm_script"
    # Add SBATCH options
    for opt in "${SBATCH_OPTS[@]}"; do echo "$opt" >> "$temp_slurm_script"; done
    # Add job-specific name and output
    echo "#SBATCH --job-name=$job_name" >> "$temp_slurm_script"
    echo "#SBATCH --output=$log_file" >> "$temp_slurm_script"
    echo "" >> "$temp_slurm_script"
    
    # Add commands to run
    echo "echo \"--- Starting SLURM Job --- \"" >> "$temp_slurm_script"
    echo "echo \"Job Name: \$SLURM_JOB_NAME (\$SLURM_JOB_ID)\"" >> "$temp_slurm_script"
    echo "echo \"Run Details: $run_details\"" >> "$temp_slurm_script"
    echo "echo \"Running on: \$SLURMD_NODENAME\"" >> "$temp_slurm_script"
    echo "echo \"GPU: \$CUDA_VISIBLE_DEVICES\"" >> "$temp_slurm_script"
    echo "echo \"Conda Env: $CONDA_ENV_NAME\"" >> "$temp_slurm_script"
    echo "echo \"--------------------------\"" >> "$temp_slurm_script"
    echo "" >> "$temp_slurm_script"
    
    echo "nvidia-smi" >> "$temp_slurm_script"
    echo "nvcc --version" >> "$temp_slurm_script"
    echo "" >> "$temp_slurm_script"

    echo "# Load modules and activate environment" >> "$temp_slurm_script"
    echo "module load gcc/8.1.0" >> "$temp_slurm_script"
    echo "source ~/.bashrc" >> "$temp_slurm_script"
    echo "conda activate $CONDA_ENV_NAME" >> "$temp_slurm_script"
    echo "" >> "$temp_slurm_script"

    echo "echo \"Executing command:\"" >> "$temp_slurm_script"
    # Escape special characters in python_cmd for safe echoing if needed, but printing directly is fine
    echo "echo \"srun $python_cmd\"" >> "$temp_slurm_script"
    echo "" >> "$temp_slurm_script"
    
    echo "srun $python_cmd" >> "$temp_slurm_script"
    echo "" >> "$temp_slurm_script"
    echo "echo \"--- SLURM Job Finished --- \"" >> "$temp_slurm_script"

    sbatch "$temp_slurm_script"
    rm "$temp_slurm_script"
}

# --- Loop and Submit ---
echo "Starting SLURM job submission loop..."

# Define the different modes and strategies
RUN_MODES=("cv" "cp_nyu_tcga" "cp_tcga_nyu")
CV_SUB_MODES=("combined" "tcga" "nyu")
FOLD_STRATS=("10fold" "loocv")
SCRIPTS=("train.py" "train_binary.py")

for script_name in "${SCRIPTS[@]}"; do
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

    for run_mode in "${RUN_MODES[@]}"; do
        echo "    Run Mode: $run_mode"

        if [ "$run_mode" == "cv" ]; then
            # --- Cross-Validation Sub-Loop ---
            for cv_mode in "${CV_SUB_MODES[@]}"; do
                echo "      CV Dataset Mode: $cv_mode"
                for fold_strat in "${FOLD_STRATS[@]}"; do
                    echo "        Fold Strategy: $fold_strat"

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

                    # --- Construct Job Name & Log ---
                    script_prefix="${script_name%.py}" # train or train_binary
                    JOB_NAME="hcc_${script_prefix}_cv-${cv_mode}_${fold_suffix}"
                    LOG_FILE="${LOG_DIR}/${JOB_NAME}_%J.log" # %J is the SLURM job ID
                    PYTHON_OUTPUT_BASE="${base_output_dir}" # Python script will add its own subfolder

                    # --- Prepare Python Command ---
                    PYTHON_CMD="python $script_name"
                    PYTHON_CMD="$PYTHON_CMD --run_mode cv" # Explicitly set run_mode
                    PYTHON_CMD="$PYTHON_CMD ${COMMON_ARGS[*]}"
                    PYTHON_CMD="$PYTHON_CMD ${specific_args[*]}"
                    PYTHON_CMD="$PYTHON_CMD --cv_mode $cv_mode" # Pass the CV sub-mode
                    PYTHON_CMD="$PYTHON_CMD ${fold_args_list[*]}"
                    PYTHON_CMD="$PYTHON_CMD --output_dir $PYTHON_OUTPUT_BASE" # Pass base output dir

                    # --- Submit Job Script (using function) ---
                    echo "        Submitting: $JOB_NAME"
                    submit_job "$JOB_NAME" "$LOG_FILE" "$script_name" "CV ($cv_mode, $fold_strat)" "$PYTHON_CMD"
                    sleep 1
                done # fold_strat
            done # cv_mode
        else
            # --- Cross-Prediction Mode ---
            # No inner loops for cv_mode or fold_strat needed here
            
            # --- Construct Job Name & Log ---
            script_prefix="${script_name%.py}" # train or train_binary
            # Example: hcc_train_cp-nyu-on-tcga
            JOB_NAME="hcc_${script_prefix}_${run_mode}"
            LOG_FILE="${LOG_DIR}/${JOB_NAME}_%J.log" # %J is the SLURM job ID
            PYTHON_OUTPUT_BASE="${base_output_dir}" # Python script will add its own subfolder

            # --- Prepare Python Command ---
            PYTHON_CMD="python $script_name"
            PYTHON_CMD="$PYTHON_CMD --run_mode $run_mode" # Set cross-prediction run_mode
            PYTHON_CMD="$PYTHON_CMD ${COMMON_ARGS[*]}"
            PYTHON_CMD="$PYTHON_CMD ${specific_args[*]}"
            # cv_mode, fold args are not needed for cross-prediction
            PYTHON_CMD="$PYTHON_CMD --output_dir $PYTHON_OUTPUT_BASE" # Pass base output dir

            # --- Submit Job Script (using function) ---
            echo "      Submitting: $JOB_NAME"
            submit_job "$JOB_NAME" "$LOG_FILE" "$script_name" "Cross-Prediction ($run_mode)" "$PYTHON_CMD"
            sleep 1
        fi
    done # run_mode
done # script_name

echo "--- All jobs submitted ---"