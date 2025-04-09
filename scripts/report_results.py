#!/usr/bin/env python3

import os
import re
import glob
import pandas as pd
from collections import defaultdict

# --- Configuration ---
# Match these with your submit_all.sh script
LOG_DIR = "/gpfs/data/shenlab/wz1492/HCC/logs/batch_submission"
BASE_OUTPUT_DIR_SURVIVAL = "checkpoints_survival_cv"
BASE_OUTPUT_DIR_BINARY = "checkpoints_binary_cv"

SCRIPTS = ["train.py", "train_binary.py"]
RUN_MODES = ["cv", "cp_nyu_tcga", "cp_tcga_nyu"]
CV_SUB_MODES = ["combined", "tcga", "nyu"]
FOLD_STRATS = ["10fold", "loocv"]

# --- Regex patterns to find metrics in log files ---
# Adjust these patterns based on the exact output format in your log files
METRIC_PATTERNS = {
    # For train.py (Survival) - Focus on C-index primarily
    "train.py": {
        "Combined C-Index": r"Combined Final Test C-Index: (\d+\.\d+)",
        "TCGA Test C-Index": r"TCGA Final Test C-Index: (\d+\.\d+)",
        "NYU Test C-Index": r"NYU Final Test C-Index: (\d+\.\d+)",
        "Test C-Index": r"Final Test C-Index: (\d+\.\d+)", # Generic for single dataset CV or CP
        # Add patterns for train metrics if needed
    },
    # For train_binary.py (Binary Classification)
    "train_binary.py": {
        "Combined Test Acc": r"Combined Final Test Accuracy: (\d+\.\d+)",
        "Combined Test AUC": r"Combined Final Test ROC AUC: (\d+\.\d+)",
        "Combined Test Precision": r"Combined Final Test Precision: (\d+\.\d+)",
        "Combined Test Recall": r"Combined Final Test Recall: (\d+\.\d+)",
        "Combined Test F1": r"Combined Final Test F1 Score: (\d+\.\d+)",
        "TCGA Test Acc": r"TCGA Final Test Accuracy: (\d+\.\d+)",
        "TCGA Test AUC": r"TCGA Final Test ROC AUC: (\d+\.\d+)",
        "NYU Test Acc": r"NYU Final Test Accuracy: (\d+\.\d+)",
        "NYU Test AUC": r"NYU Final Test ROC AUC: (\d+\.\d+)",
        "Test Acc": r"Final Test Accuracy: (\d+\.\d+)", # Generic for single dataset CV or CP
        "Test AUC": r"Final Test ROC AUC: (\d+\.\d+)",
        "Test Precision": r"Final Test Precision: (\d+\.\d+)",
        "Test Recall": r"Final Test Recall: (\d+\.\d+)",
        "Test F1": r"Final Test F1 Score: (\d+\.\d+)",
        # Add patterns for train metrics if needed
    }
}

def parse_log_file(log_path, patterns):
    """Parses a single log file for metrics defined by regex patterns."""
    results = {}
    try:
        with open(log_path, 'r') as f:
            log_content = f.read()
            for metric_name, pattern in patterns.items():
                match = re.search(pattern, log_content)
                if match:
                    try:
                        results[metric_name] = float(match.group(1))
                    except ValueError:
                        results[metric_name] = "ErrorParsing" # Or None
                else:
                    # Check if pattern should have matched (e.g. specific dataset not present)
                    # For simplicity, we mark as NaN if not found. Could add more logic here.
                    pass # Let missing keys be handled later

    except FileNotFoundError:
        print(f"Warning: Log file not found: {log_path}")
        return None
    except Exception as e:
        print(f"Error reading or parsing log file {log_path}: {e}")
        return None
    return results


def find_latest_log(job_name_pattern):
    """Finds the most recent log file matching a base job name pattern."""
    search_pattern = os.path.join(LOG_DIR, f"{job_name_pattern}_*.log")
    log_files = glob.glob(search_pattern)
    if not log_files:
        return None
    # Find the latest log file based on modification time or job ID (assuming higher ID is later)
    latest_log = max(log_files, key=os.path.getmtime)
    # Alternative: parse job ID from filename if consistent
    # latest_log = max(log_files, key=lambda f: int(re.search(r'_(\d+)\.log$', f).group(1)))
    return latest_log

# --- Main Reporting Logic ---
all_results = []

print(f"Scanning log directory: {LOG_DIR}")

for script_name in SCRIPTS:
    script_prefix = script_name.replace(".py", "")
    patterns = METRIC_PATTERNS.get(script_name, {})
    if not patterns:
        print(f"Warning: No metric patterns defined for {script_name}")
        continue

    for run_mode in RUN_MODES:
        base_info = {"Script": script_name, "Run Mode": run_mode}

        if run_mode == "cv":
            for cv_mode in CV_SUB_MODES:
                for fold_strat in FOLD_STRATS:
                    fold_suffix = "loocv" if fold_strat == "loocv" else "10fold"
                    job_name_pattern = f"hcc_{script_prefix}_cv-{cv_mode}_{fold_suffix}"

                    log_file = find_latest_log(job_name_pattern)
                    run_info = {**base_info, "CV Mode": cv_mode, "Fold Strategy": fold_strat}

                    if log_file:
                        print(f"  Parsing {os.path.basename(log_file)}...")
                        metrics = parse_log_file(log_file, patterns)
                        if metrics is not None:
                             run_info.update(metrics)
                        else:
                            run_info["Status"] = "LogReadError"
                    else:
                        print(f"  Log file not found for pattern: {job_name_pattern}")
                        run_info["Status"] = "LogNotFound"

                    all_results.append(run_info)

        else: # Cross-Prediction modes
            job_name_pattern = f"hcc_{script_prefix}_{run_mode}"
            run_info = {**base_info, "CV Mode": "-", "Fold Strategy": "-"} # Indicate N/A

            log_file = find_latest_log(job_name_pattern)

            if log_file:
                print(f"  Parsing {os.path.basename(log_file)}...")
                metrics = parse_log_file(log_file, patterns)
                if metrics is not None:
                     run_info.update(metrics)
                else:
                    run_info["Status"] = "LogReadError"
            else:
                print(f"  Log file not found for pattern: {job_name_pattern}")
                run_info["Status"] = "LogNotFound"

            all_results.append(run_info)


# --- Display Results ---
if not all_results:
    print("\nNo results found or parsed.")
else:
    df = pd.DataFrame(all_results)

    # Reorder columns for clarity
    cols_order = ["Script", "Run Mode", "CV Mode", "Fold Strategy"]
    # Dynamically add metric columns found, preserving order somewhat
    metric_cols = sorted([col for col in df.columns if col not in cols_order + ["Status"]])
    final_cols = cols_order + metric_cols + (["Status"] if "Status" in df.columns else [])

    # Ensure all columns exist before reordering
    df = df.reindex(columns=[col for col in final_cols if col in df.columns])


    print("\n--- Experiment Results Summary ---")
    # Configure pandas display options for better viewing
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 200) # Adjust width as needed

    # Format floating point numbers
    float_cols = df.select_dtypes(include=['float']).columns
    format_dict = {col: '{:.4f}' for col in float_cols}

    print(df.to_string(index=False, na_rep='-', formatters=format_dict))
    print("-------------------------------\n")

    # Optionally save to CSV
    # csv_output_path = "experiment_summary.csv"
    # df.to_csv(csv_output_path, index=False, float_format='%.4f')
    # print(f"Results also saved to {csv_output_path}") 