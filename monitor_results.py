#!/usr/bin/env python3
"""
Monitor and compare binary classification results across different runs.
"""

import glob
import json
import math
import pandas as pd
from datetime import datetime
import numpy as np

def load_all_results():
    """Load all run_summary.json files and extract metrics."""
    paths = glob.glob('checkpoints_binary_cv/**/run_summary.json', recursive=True)
    results = []
    
    for path in paths:
        try:
            with open(path) as f:
                data = json.load(f)
            
            metrics = data.get('metrics', {}).get('overall', {})
            hyperparams = data.get('hyperparameters', {})
            
            # Extract timestamp from path
            timestamp_str = path.split('/')[1].split('_')[1] + '_' + path.split('/')[1].split('_')[2]
            timestamp = datetime.strptime(timestamp_str, '%Y%m%d_%H%M%S')
            
            result = {
                'path': path,
                'timestamp': timestamp,
                'roc_auc': metrics.get('roc_auc'),
                'precision': metrics.get('precision'),
                'recall': metrics.get('recall'),
                'f1': metrics.get('f1'),
                'accuracy': metrics.get('accuracy'),
                'learning_rate': hyperparams.get('learning_rate'),
                'batch_size': hyperparams.get('batch_size'),
                'num_slices': hyperparams.get('num_slices'),
                'model_arch': hyperparams.get('model_arch', hyperparams.get('model_type', 'unknown')),
                'dropout': hyperparams.get('dropout'),
                'focal_loss': hyperparams.get('focal_loss', False),
                'focal_gamma': hyperparams.get('focal_gamma'),
                'upsampling_method': hyperparams.get('upsampling_method'),
                'cv_folds': hyperparams.get('cv_folds'),
                'leave_one_out': hyperparams.get('leave_one_out', False)
            }
            
            # Filter out invalid results
            if result['roc_auc'] is not None and not math.isnan(result['roc_auc']):
                results.append(result)
                
        except Exception as e:
            print(f"Error processing {path}: {e}")
            continue
    
    return results

def analyze_improvements():
    """Analyze improvements over time."""
    results = load_all_results()
    df = pd.DataFrame(results)
    
    if df.empty:
        print("No valid results found!")
        return
    
    # Sort by timestamp
    df = df.sort_values('timestamp')
    
    print("=== BINARY CLASSIFICATION RESULTS ANALYSIS ===\n")
    
    # Overall statistics
    print(f"Total completed runs: {len(df)}")
    print(f"Time span: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print()
    
    # Best overall results
    print("=== TOP 10 BEST PERFORMING RUNS (by ROC-AUC) ===")
    top_runs = df.nlargest(10, 'roc_auc')
    
    for i, (_, row) in enumerate(top_runs.iterrows()):
        print(f"{i+1:2d}. AUC: {row['roc_auc']:.3f} | Prec: {row['precision']:.3f} | "
              f"Rec: {row['recall']:.3f} | F1: {row['f1']:.3f}")
        print(f"     LR: {row['learning_rate']:.2e} | BS: {row['batch_size']} | "
              f"Slices: {row['num_slices']} | Arch: {row['model_arch']}")
        print(f"     Dropout: {row['dropout']} | Focal: {row['focal_loss']} | "
              f"Ups: {row['upsampling_method']}")
        print()
    
    # Time-based analysis
    print("=== PERFORMANCE TRENDS ===")
    
    # Define time periods
    cutoff_time = datetime(2025, 6, 27, 14, 0, 0)  # 2 PM cutoff
    baseline = df[df['timestamp'] < cutoff_time]
    improved = df[df['timestamp'] >= cutoff_time]
    
    if not baseline.empty and not improved.empty:
        print(f"Baseline runs (before 14:00): {len(baseline)}")
        print(f"  Best ROC-AUC: {baseline['roc_auc'].max():.3f}")
        print(f"  Mean ROC-AUC: {baseline['roc_auc'].mean():.3f}")
        print(f"  Mean Precision: {baseline['precision'].mean():.3f}")
        print(f"  Mean Recall: {baseline['recall'].mean():.3f}")
        print()
        
        print(f"Improved runs (after 14:00): {len(improved)}")
        print(f"  Best ROC-AUC: {improved['roc_auc'].max():.3f}")
        print(f"  Mean ROC-AUC: {improved['roc_auc'].mean():.3f}")
        print(f"  Mean Precision: {improved['precision'].mean():.3f}")
        print(f"  Mean Recall: {improved['recall'].mean():.3f}")
        print()
        
        # Calculate improvements
        auc_improvement = improved['roc_auc'].max() - baseline['roc_auc'].max()
        prec_improvement = improved['precision'].mean() - baseline['precision'].mean()
        
        print(f"IMPROVEMENTS:")
        print(f"  Best ROC-AUC improvement: {auc_improvement:+.3f}")
        print(f"  Mean Precision improvement: {prec_improvement:+.3f}")
        print()
    
    # Architecture analysis
    print("=== ARCHITECTURE PERFORMANCE ===")
    arch_stats = df.groupby('model_arch').agg({
        'roc_auc': ['count', 'mean', 'max'],
        'precision': 'mean',
        'recall': 'mean'
    }).round(3)
    
    print(arch_stats)
    print()
    
    # Recent best results
    print("=== RECENT BEST RESULTS (Last 6 hours) ===")
    recent_cutoff = datetime.now() - pd.Timedelta(hours=6)
    recent = df[df['timestamp'] > recent_cutoff]
    
    if not recent.empty:
        recent_best = recent.nlargest(5, 'roc_auc')
        for i, (_, row) in enumerate(recent_best.iterrows()):
            print(f"{i+1}. AUC: {row['roc_auc']:.3f} | Prec: {row['precision']:.3f} | "
                  f"Time: {row['timestamp'].strftime('%H:%M')}")
    else:
        print("No recent results found.")
    
    print("\n=== SUMMARY ===")
    print(f"Best ROC-AUC achieved: {df['roc_auc'].max():.3f}")
    print(f"Best Precision achieved: {df['precision'].max():.3f}")
    print(f"Best F1 achieved: {df['f1'].max():.3f}")
    
    # Count high-performing runs
    high_auc = len(df[df['roc_auc'] > 0.8])
    high_prec = len(df[df['precision'] > 0.5])
    
    print(f"Runs with ROC-AUC > 0.8: {high_auc}")
    print(f"Runs with Precision > 0.5: {high_prec}")

if __name__ == "__main__":
    analyze_improvements() 