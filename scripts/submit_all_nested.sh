#!/bin/bash

# --- Configuration ---
LOG_DIR="/gpfs/data/shenlab/wz1492/HCC/logs/batch_submission_nested" # Directory for SLURM logs
BASE_OUTPUT_DIR_SURVIVAL="checkpoints_survival_nested_cv" # Base output for train_nested.py results
BASE_OUTPUT_DIR_BINARY="checkpoints_binary_nested_cv"   # Base output for train_binary_nested.py results
CONDA_ENV_NAME="dinov2"

# --- Base SLURM Options ---
# These will be included in every submitted job script
SBATCH_OPTS=(
"#SBATCH -p radiology,a100_short,a100_long,gpu4_medium,gpu4_long,gpu4_short,gpu8_short,gpu8_medium,gpu8_long"      # Partitions
"#SBATCH --gres=gpu:1"                # Request 1 GPU
"#SBATCH --nodes=1"                   # Request 1 node
"#SBATCH --ntasks=1"                  # Request 1 task
"#SBATCH --cpus-per-task=16"          # CPUs per task
"#SBATCH --mem=100GB"                 # Memory
"#SBATCH --time=12:00:00"             # Max walltime
"#SBATCH --exclude=a100-4020"         # Exclude specific nodes if needed
)

# --- Common Python Script Arguments ---
# Arguments shared between train_nested.py and train_binary_nested.py
COMMON_ARGS=(
"--nyu_dicom_root /gpfs/data/mankowskilab/HCC_Recurrence/dicom"
"--tcga_dicom_root /gpfs/data/mankowskilab/HCC/data/TCGA/manifest-4lZjKqlp5793425118292424834/TCGA-LIHC"
"--tcga_csv_file /gpfs/data/shenlab/wz1492/HCC/spreadsheets/tcga.csv"
"--nyu_csv_file /gpfs/data/shenlab/wz1492/HCC/spreadsheets/processed_patient_labels_nyu.csv"
"--batch_size 32"
"--num_slices 0" 
"--preprocessed_root /gpfs/data/mankowskilab/HCC_Recurrence/preprocessed/"
"--num_samples_per_patient 1"
"--dinov2_weights /gpfs/data/shenlab/wz1492/HCC/dinov2/experiments/eval/training_112499/teacher_checkpoint.pth"
"--upsampling"
"--upsampling_method smote"
"--epochs 5000"
"--early_stopping"
"--early_stopping_patience 5"
# Default learning rate for outer loop, can be overridden
"--learning_rate 1e-5"
# Nested CV specific arguments (defaults from python script used here)
"--inner_folds 6"
# --learning_rates is specific to each script type
# --cv_folds will be set later based on fold_strat
)

# --- Specific Python Script Arguments ---
SURVIVAL_ARGS=(
"--center_risk"
"--coxph_net mlp"
"--alpha 0.3"
"--gamma 0.3"
# Default LR grid for survival
"--learning_rates 1e-8 5e-8 1e-7 5e-6 1e-6 5e-5 1e-5 1e-4 1e-3 1e-2"
)

BINARY_ARGS=(
# Default LR grid for binary
"--learning_rates 1e-8 5e-8 1e-7 5e-6 1e-6 5e-5 1e-5 1e-4 1e-3 1e-2"
)


# --- Create Log Directory ---
mkdir -p "$LOG_DIR"

# --- Loop and Submit ---
echo "Starting NESTED CV SLURM job submission loop..."

# --- Submit Cross-dataset Evaluation Jobs ---
echo "Submitting NESTED cross-dataset evaluation jobs (train on X, test on Y)..."

# Configurations for cross-dataset evaluation:
# Train on NYU, predict on TCGA
# Train on TCGA, predict on NYU
for script_name in "train_nested.py" "train_binary_nested.py"; do
    echo "  Cross-Dataset Evaluation for Script: $script_name"

    specific_args=()
    base_output_dir=""
    if [ "$script_name" == "train_nested.py" ]; then
        specific_args=("${SURVIVAL_ARGS[@]}")
        base_output_dir="$BASE_OUTPUT_DIR_SURVIVAL"
    else # train_binary_nested.py
        specific_args=("${BINARY_ARGS[@]}")
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
        script_prefix="${script_name%.py}" # train_nested or train_binary_nested
        JOB_NAME="hcc_${script_prefix}_${cross_mode}"
        LOG_FILE="${LOG_DIR}/${JOB_NAME}_%J.log" # %J is the SLURM job ID

        # --- Construct Output Directory for Python Script ---
        PYTHON_OUTPUT_BASE="${base_output_dir}/nested_cross_${cross_mode}"

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
        TEMP_SLURM_SCRIPT=$(mktemp /tmp/hcc_nested_job_XXXXXX.slurm)

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
        echo "echo \"--- Starting NESTED SLURM Job --- \"" >> "$TEMP_SLURM_SCRIPT"
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
        echo "echo \"srun --cpu-bind=none $PYTHON_CMD\"" >> "$TEMP_SLURM_SCRIPT"
        echo "" >> "$TEMP_SLURM_SCRIPT"

        echo "srun --cpu-bind=none $PYTHON_CMD" >> "$TEMP_SLURM_SCRIPT"
        echo "" >> "$TEMP_SLURM_SCRIPT"
        echo "echo \"--- SLURM Job Finished --- \"" >> "$TEMP_SLURM_SCRIPT"

        # --- Submit Job ---
        echo "  Submitting: $JOB_NAME (Log: ${LOG_FILE/\%J/<job_id>})"
        sbatch "$TEMP_SLURM_SCRIPT"

        # Clean up temporary script
        rm "$TEMP_SLURM_SCRIPT"
    done # cross_mode
done # script_name

# Nested CV mode jobs
for script_name in "train_nested.py" "train_binary_nested.py"; do
    echo "  Script: $script_name"

    specific_args=()
    base_output_dir=""
    if [ "$script_name" == "train_nested.py" ]; then
        specific_args=("${SURVIVAL_ARGS[@]}")
        base_output_dir="$BASE_OUTPUT_DIR_SURVIVAL"
    else # train_binary_nested.py
        specific_args=("${BINARY_ARGS[@]}")
        base_output_dir="$BASE_OUTPUT_DIR_BINARY"
    fi

    for mode in "combined" "tcga" "nyu"; do
        echo "    Mode: $mode"
        for fold_strat in "7fold" "loocv"; do
            echo "      Fold Strategy: $fold_strat"

            # --- Determine Fold Args ---
            fold_args_list=()
            fold_suffix=""
            if [ "$fold_strat" == "loocv" ]; then
                fold_args_list+=("--leave_one_out")
                fold_suffix="loocv"
            else
                fold_args_list+=("--cv_folds 7") # Explicitly set 7 folds
                fold_suffix="7fold"
            fi

            # --- Construct Job Name and Log File Path ---
            script_prefix="${script_name%.py}" # train_nested or train_binary_nested
            JOB_NAME="hcc_${script_prefix}_${mode}_${fold_suffix}"
            LOG_FILE="${LOG_DIR}/${JOB_NAME}_%J.log" # %J is the SLURM job ID

            # --- Construct Output Directory for Python Script ---
            PYTHON_OUTPUT_BASE="${base_output_dir}" # Python script adds its own subfolder

            # --- Prepare Python Command ---
            PYTHON_CMD="python $script_name"
            PYTHON_CMD="$PYTHON_CMD ${COMMON_ARGS[*]}"
            PYTHON_CMD="$PYTHON_CMD ${specific_args[*]}"
            PYTHON_CMD="$PYTHON_CMD --cv_mode $mode"
            PYTHON_CMD="$PYTHON_CMD ${fold_args_list[*]}"
            PYTHON_CMD="$PYTHON_CMD --output_dir $PYTHON_OUTPUT_BASE" # Pass base output dir

            # Override default base learning rate for survival CV: combined vs single-dataset
            if [ "$script_name" == "train_nested.py" ]; then
                if [ "$mode" == "combined" ]; then
                    lr=5e-6
                else
                    lr=1e-7
                fi
                # Remove existing --learning_rate from COMMON_ARGS if present
                # (This part is tricky, easier to handle it by placement/overriding in Python script)
                # We will rely on the --learning_rates being passed and the default --learning_rate potentially
                # being used if the list is empty or tuning fails. The nested CV logic handles picking the best LR.
                # Instead of overriding here, we ensure --learning_rate is set in COMMON_ARGS as a fallback.
                 # Check if --learning_rate exists and replace it
                temp_cmd=""
                lr_set=false
                for arg in $PYTHON_CMD; do
                    if [[ "$arg" == "--learning_rate" ]]; then
                        temp_cmd="$temp_cmd --learning_rate $lr"
                        lr_set=true
                        # Skip the next argument (the old LR value)
                        skip_next=true
                    elif [[ "$skip_next" == true ]]; then
                        skip_next=false
                    else
                        temp_cmd="$temp_cmd $arg"
                    fi
                done
                 if [[ "$lr_set" == false ]]; then
                     # If --learning_rate wasn't in the command initially, add it
                     temp_cmd="$temp_cmd --learning_rate $lr"
                 fi
                PYTHON_CMD="$temp_cmd"

            fi

            # --- Create Temporary Job Script ---
            TEMP_SLURM_SCRIPT=$(mktemp /tmp/hcc_nested_job_XXXXXX.slurm)

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
            echo "echo \"--- Starting NESTED SLURM Job --- \"" >> "$TEMP_SLURM_SCRIPT"
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
            # Use echo -e to handle potential newlines if PYTHON_CMD gets complex
            echo "echo -e \"srun --cpu-bind=none $PYTHON_CMD\"" >> "$TEMP_SLURM_SCRIPT"
            echo "" >> "$TEMP_SLURM_SCRIPT"

            echo "srun --cpu-bind=none $PYTHON_CMD" >> "$TEMP_SLURM_SCRIPT"
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

echo "--- All NESTED jobs submitted ---" 