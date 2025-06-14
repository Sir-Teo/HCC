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
"--tcga_dicom_root /gpfs/data/mankowskilab/HCC/data/TCGA/manifest-4lZjKqlp5793425118292424834/TCGA-LIHC"
"--tcga_csv_file /gpfs/data/shenlab/wz1492/HCC/spreadsheets/tcga.csv"
"--nyu_csv_file /gpfs/data/shenlab/wz1492/HCC/spreadsheets/processed_patient_labels_nyu.csv"
"--batch_size 32"
"--num_slices 0"
"--preprocessed_root /gpfs/data/mankowskilab/HCC_Recurrence/preprocessed/"
"--learning_rate 1e-5"
"--dinov2_weights $DINOV2_WEIGHTS"
"--num_samples_per_patient 1"
"--upsampling"
"--upsampling_method smote"
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
            script_prefix="${script_name%.py}" # train or train_binary
            JOB_NAME="hcc_${script_prefix}_${mode}_${fold_suffix}_${WEIGHT_SUFFIX}"
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

            # Override learning rate for survival CV: combined vs single-dataset
            if [ "$script_name" == "train.py" ]; then
                if [ "$mode" == "combined" ]; then
                    lr=5e-6
                else
                    lr=1e-7
                fi
                PYTHON_CMD="$PYTHON_CMD --learning_rate $lr"
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
    done # mode
done # script_name

echo "--- All jobs submitted ---"