#!/usr/bin/env python3
"""
Advanced monitoring script for precision boost experiments.
Tracks hyperparameter search progress and identifies best precision results.
"""

import os
import json
import glob
import subprocess
import time
from datetime import datetime
import pandas as pd
import numpy as np

def get_job_status():
    """Check SLURM job status."""
    try:
        result = subprocess.run(['squeue', '-u', 'wz1492'], capture_output=True, text=True)
        return result.stdout
    except:
        return "Could not check job status"

def find_precision_boost_logs():
    """Find all precision boost experiment logs."""
    log_pattern = "/gpfs/data/shenlab/wz1492/HCC/logs/batch_submission/hcc_precision_boost_*.log"
    return glob.glob(log_pattern)

def parse_trial_results(log_file):
    """Parse hyperparameter search results from log file."""
    trials = []
    
    try:
        with open(log_file, 'r') as f:
            lines = f.readlines()
        
        for line in lines:
            # Look for trial completion lines
            if "Trial" in line and "finished" in line and "AUC=" in line:
                try:
                    # Extract trial number, AUC, and Precision
                    parts = line.split()
                    trial_info = [p for p in parts if "Trial" in p][0]
                    trial_num = int(trial_info.split('/')[-1])
                    
                    auc_part = [p for p in parts if "AUC=" in p][0]
                    auc = float(auc_part.split('=')[1].rstrip(','))
                    
                    prec_part = [p for p in parts if "Prec_Ov=" in p][0]
                    precision = float(prec_part.split('=')[1])
                    
                    trials.append({
                        'trial': trial_num,
                        'auc': auc,
                        'precision': precision,
                        'score': 0.3 * auc + 0.7 * precision  # Precision-weighted score
                    })
                except Exception as e:
                    continue
    except Exception as e:
        print(f"Error parsing log file {log_file}: {e}")
    
    return trials

def get_best_results(log_file):
    """Get best results from hyperparameter search."""
    best_info = {}
    
    try:
        with open(log_file, 'r') as f:
            lines = f.readlines()
        
        for line in lines:
            if "Best trial directory:" in line:
                best_info['directory'] = line.split("Best trial directory:")[1].strip()
            elif "Best ROC-AUC:" in line:
                best_info['auc'] = float(line.split("Best ROC-AUC:")[1].strip())
            elif "Best Precision:" in line:
                best_info['precision'] = float(line.split("Best Precision:")[1].strip())
    except Exception as e:
        print(f"Error getting best results: {e}")
    
    return best_info

def analyze_trial_progression(trials):
    """Analyze progression of trials to identify trends."""
    if not trials:
        return {}
    
    df = pd.DataFrame(trials)
    
    return {
        'total_trials': len(df),
        'best_auc': df['auc'].max(),
        'best_precision': df['precision'].max(),
        'best_combined_score': df['score'].max(),
        'avg_auc': df['auc'].mean(),
        'avg_precision': df['precision'].mean(),
        'precision_improvement': df['precision'].max() - 0.0789,  # Improvement over baseline
        'auc_improvement': df['auc'].max() - 0.5520,  # Improvement over baseline
    }

def check_final_results_summary(best_dir):
    """Check final cross-validation results from best run."""
    summary_file = os.path.join(best_dir, "run_summary.json")
    
    if os.path.exists(summary_file):
        try:
            with open(summary_file, 'r') as f:
                data = json.load(f)
            return data.get('metrics', {}).get('overall', {})
        except Exception as e:
            print(f"Error reading summary: {e}")
    
    return {}

def main():
    print(f"=== Precision Boost Monitoring Report ===")
    print(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Check job status
    print("Current Job Status:")
    print(get_job_status())
    print()
    
    # Find and analyze logs
    log_files = find_precision_boost_logs()
    
    if not log_files:
        print("No precision boost experiment logs found.")
        return
    
    # Analyze latest log
    latest_log = max(log_files, key=os.path.getctime)
    print(f"Analyzing log: {latest_log}")
    print()
    
    # Parse trial results
    trials = parse_trial_results(latest_log)
    
    if trials:
        print("=== Hyperparameter Search Progress ===")
        analysis = analyze_trial_progression(trials)
        
        print(f"Trials completed: {analysis['total_trials']}")
        print(f"Best AUC so far: {analysis['best_auc']:.4f} (improvement: +{analysis['auc_improvement']:.4f})")
        print(f"Best Precision so far: {analysis['best_precision']:.4f} (improvement: +{analysis['precision_improvement']:.4f})")
        print(f"Best Combined Score: {analysis['best_combined_score']:.4f}")
        print(f"Average AUC: {analysis['avg_auc']:.4f}")
        print(f"Average Precision: {analysis['avg_precision']:.4f}")
        print()
        
        # Show top 5 trials
        df = pd.DataFrame(trials).sort_values('score', ascending=False)
        print("=== Top 5 Trials by Precision-Weighted Score ===")
        for i, (_, row) in enumerate(df.head().iterrows()):
            print(f"{i+1}. Trial {row['trial']}: AUC={row['auc']:.4f}, Precision={row['precision']:.4f}, Score={row['score']:.4f}")
        print()
    
    # Check for final results
    best_results = get_best_results(latest_log)
    if best_results:
        print("=== Best Trial Results ===")
        if 'directory' in best_results:
            print(f"Best directory: {os.path.basename(best_results['directory'])}")
        if 'auc' in best_results:
            print(f"Best AUC: {best_results['auc']:.4f}")
        if 'precision' in best_results:
            print(f"Best Precision: {best_results['precision']:.4f}")
        
        # Check final cross-validation summary
        if 'directory' in best_results:
            final_results = check_final_results_summary(best_results['directory'])
            if final_results:
                print("\n=== Final Cross-Validation Results ===")
                for metric, value in final_results.items():
                    if isinstance(value, (int, float)):
                        print(f"{metric}: {value:.4f}")
        print()
    
    # Progress estimation
    if trials:
        total_expected = 100  # From SLURM script
        completed = len(trials)
        if completed < total_expected:
            remaining = total_expected - completed
            if completed > 0:
                # Estimate time per trial from log timestamps (rough estimate)
                print(f"Progress: {completed}/{total_expected} trials ({completed/total_expected*100:.1f}%)")
                print(f"Estimated remaining trials: {remaining}")
            else:
                print(f"Trials in progress...")
    
    print("=== Key Improvements in This Run ===")
    print("1. New HyperPrecisionMLP architecture with multi-path processing")
    print("2. Advanced ensemble methods with meta-learning")
    print("3. Adaptive focal loss that adjusts based on precision")
    print("4. Precision-constrained loss with minimum precision penalty")
    print("5. Ultra-sophisticated threshold tuning")
    print("6. 100 hyperparameter trials (vs previous 50)")
    print("7. Optimized sampling weights favoring precision architectures")

if __name__ == "__main__":
    main() 