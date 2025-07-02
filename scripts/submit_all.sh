#!/bin/bash

# --- Configuration ---
LOG_DIR="/gpfs/data/shenlab/wz1492/HCC/logs/batch_submission" # Directory for SLURM logs
BASE_OUTPUT_DIR_SURVIVAL="checkpoints_survival_cv" # Base output for train.py results
BASE_OUTPUT_DIR_BINARY="checkpoints_binary_cv"   # Base output for train_binary.py results
CONDA_ENV_NAME="dinov2"

# --- Weight Configuration ---
# Parse command line argument for weight selection
WEIGHT_TYPE="$1"

# Weight paths
LOCAL_WEIGHTS="/gpfs/data/shenlab/wz1492/HCC/dinov2/experiments/eval/training_112499/teacher_checkpoint.pth"
PRETRAINED_WEIGHTS="./dinov2/pretrained_weights/dinov2_vitl14_pretrain.pth"

# Select weights based on command line argument
if [ "$WEIGHT_TYPE" == "pretrain" ]; then
    DINOV2_WEIGHTS="$PRETRAINED_WEIGHTS"
    WEIGHT_SUFFIX="pretrain"
    echo "Using pretrained weights: $DINOV2_WEIGHTS"
elif [ "$WEIGHT_TYPE" == "local" ]; then
    DINOV2_WEIGHTS="$LOCAL_WEIGHTS"
    WEIGHT_SUFFIX="local"
    echo "Using local weights: $DINOV2_WEIGHTS"
else
    echo "Usage: $0 [pretrain|local]"
    echo "  pretrain - Use pretrained DinoV2 weights"
    echo "  local    - Use local trained weights"
    exit 1
fi

# --- Base SLURM Options ---
# These will be included in every submitted job script
SBATCH_OPTS=(
"#SBATCH -p radiology,a100_short,a100_long,gpu4_short,gpu4_medium,gpu4_long"      # Partitions
"#SBATCH --gres=gpu:1"                # Request 1 GPU
"#SBATCH --nodes=1"                   # Request 1 node
"#SBATCH --ntasks=1"                  # Request 1 task
"#SBATCH --cpus-per-task=16"          # CPUs per task
"#SBATCH --mem=100GB"                 # Memory
"#SBATCH --time=12:00:00"             # Max walltime
"#SBATCH --exclude=a100-4020"         # Exclude specific nodes if needed
)

# --- Common Python Script Arguments ---
# Arguments shared between train.py and train_binary.py
COMMON_ARGS=(
"--nyu_dicom_root /gpfs/data/mankowskilab/HCC_Recurrence/dicom"
"--nyu_csv_file /gpfs/data/shenlab/wz1492/HCC/spreadsheets/processed_patient_labels_nyu.csv"
"--batch_size 8"  # Smaller batch size performed better
"--num_slices 32"  # Optimal slice count from successful runs
"--preprocessed_root /gpfs/data/mankowskilab/HCC_Recurrence/preprocessed/"
"--learning_rate 1e-5"
"--dinov2_weights $DINOV2_WEIGHTS"
"--num_samples_per_patient 1"
"--upsampling"
"--upsampling_method adasyn"  # ADASYN performed best
"--epochs 1000" # Using 1000 as per slurm scripts
"--early_stopping"
"--early_stopping_patience 15"  # Slightly more patience for complex models
# --cross_validation is implicitly handled by running these scripts
)

# --- Specific Python Script Arguments ---
SURVIVAL_ARGS=(
"--center_risk"
"--coxph_net mlp"
"--alpha 0.1"  # Reduced from 0.5 for less aggressive regularization
"--gamma 0.3"  # Reduced from 0.5 for more L2 regularization
)
# train_binary.py doesn't have extra specific args listed here currently

# Default configuration
num_trials=50  # Increased to 50 for better hyperparameter search coverage
model_arch="precision_weighted_ensemble"  # Use improved ensemble with shape fix
loss_type="precision_recall_focal"  # Use precision-recall focal loss

# Add model architecture and loss type to arguments
specific_args+=("--model_arch $model_arch")  # Use ensemble architecture with shape fix
specific_args+=("--precision_recall_focal")  # Enable precision-recall focal loss
specific_args+=("--hyper_search_iters $num_trials")  # Increased search coverage

# --- Create Log Directory ---
mkdir -p "$LOG_DIR"

# --- Loop and Submit ---
echo "Starting SLURM job submission loop..."

# NYU-only jobs
for script_name in "train.py" "train_binary.py"; do
    echo "  Script: $script_name"
    
    specific_args=()
    base_output_dir=""
    if [ "$script_name" == "train.py" ]; then
        specific_args=("${SURVIVAL_ARGS[@]}")
        base_output_dir="$BASE_OUTPUT_DIR_SURVIVAL"
    else # train_binary.py
        base_output_dir="$BASE_OUTPUT_DIR_BINARY"
        # Binary-specific args (optimized for extreme imbalance and precision)
        specific_args+=("--dropout 0.2")  # Reduced from 0.3 based on best results
        specific_args+=("--precision_recall_focal")  # Use new precision-recall focal loss
        specific_args+=("--focal_gamma 2.0")
        specific_args+=("--upsampling_method adasyn")  # Best method from recent testing
        specific_args+=("--cv_folds 7")
        specific_args+=("--hyper_search_iters $num_trials")
        specific_args+=("--model_arch optimized_simple")  # Use best performing architecture
    fi

    for fold_strat in "7fold"; do
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
        script_prefix="${script_name%.py}" # train or train_binary
        JOB_NAME="hcc_${script_prefix}_nyu_${fold_suffix}_${WEIGHT_SUFFIX}"
        LOG_FILE="${LOG_DIR}/${JOB_NAME}_%J.log" # %J is the SLURM job ID

        # --- Construct Output Directory for Python Script ---
        # This uses the run_name structure defined within the Python scripts
        # We just provide the base directory
        PYTHON_OUTPUT_BASE="${base_output_dir}" # Python script will add its own subfolder

        # --- Prepare Python Command ---
        PYTHON_CMD="python $script_name"
        PYTHON_CMD="$PYTHON_CMD ${COMMON_ARGS[*]}"
        PYTHON_CMD="$PYTHON_CMD ${specific_args[*]}"
        PYTHON_CMD="$PYTHON_CMD ${fold_args_list[*]}"
        PYTHON_CMD="$PYTHON_CMD --output_dir $PYTHON_OUTPUT_BASE" # Pass base output dir

        # Override learning rate for survival CV
        if [ "$script_name" == "train.py" ]; then
            lr=1e-4  # Use improved default instead of 1e-7
            PYTHON_CMD="$PYTHON_CMD --learning_rate $lr"
        fi

        # Enable internal hyper-parameter search for binary classification jobs
        if [ "$script_name" == "train_binary.py" ]; then
            PYTHON_CMD="$PYTHON_CMD --hyper_search_iters 50"  # More trials for precision optimization
        fi

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

    done # fold_strat
done # script_name

echo "--- All jobs submitted ---"