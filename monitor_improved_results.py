#!/usr/bin/env python3
"""
Enhanced results monitoring script for HCC training improvements
Focus on tracking AUC and precision improvements from new architectures
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import time
import glob
import argparse

def monitor_job_progress(log_dir="/gpfs/data/shenlab/wz1492/HCC/logs/batch_submission"):
    """Monitor SLURM job progress and extract key metrics"""
    
    print("=== HCC Training Progress Monitor ===")
    print(f"Monitoring logs in: {log_dir}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("-" * 50)
    
    # Find recent log files
    log_pattern = os.path.join(log_dir, "hcc_train_binary_nyu_*_local_84*.log")
    recent_logs = glob.glob(log_pattern)
    recent_logs.sort(key=os.path.getmtime, reverse=True)
    
    if not recent_logs:
        print("No recent binary training logs found")
        return
    
    for log_file in recent_logs[:4]:  # Check last 4 jobs
        job_id = os.path.basename(log_file).split('_')[-1].split('.')[0]
        print(f"\nJob ID {job_id}: {os.path.basename(log_file)}")
        
        try:
            with open(log_file, 'r') as f:
                content = f.read()
            
            # Check job status
            if "Early stopping triggered" in content:
                print("  Status: Completed with early stopping")
            elif "Final aggregated predictions" in content:
                print("  Status: Completed successfully")
            elif "SLURM Job Finished" in content:
                print("  Status: Finished (check for errors)")
            elif len(content) > 1000:
                print("  Status: Running...")
            else:
                print("  Status: Starting/Initializing")
            
            # Extract key performance metrics
            lines = content.split('\n')
            
            # Look for architecture being used
            for line in lines:
                if "model_arch" in line and ("ultra_precision" in line or "precision_weighted_ensemble" in line):
                    arch = line.split("--model_arch")[1].split()[0] if "--model_arch" in line else "unknown"
                    print(f"  Architecture: {arch}")
                    break
            
            # Look for final metrics
            for line in lines:
                if "ROC AUC:" in line:
                    auc = line.split("ROC AUC:")[1].strip()
                    print(f"  AUC: {auc}")
                if "Precision:" in line:
                    precision = line.split("Precision:")[1].strip()
                    print(f"  Precision: {precision}")
                if "Best trial directory:" in line:
                    best_dir = line.split("Best trial directory:")[1].strip()
                    print(f"  Best trial: {os.path.basename(best_dir)}")
            
            # Look for hyperparameter search progress
            search_lines = [line for line in lines if "Trial" in line and "finished" in line]
            if search_lines:
                print(f"  Hyperparameter search: {len(search_lines)} trials completed")
                # Show best trial so far
                for line in search_lines[-3:]:
                    if "AUC=" in line and "Prec_Ov=" in line:
                        trial_info = line.split("Trial")[1].split("finished:")[1].strip()
                        print(f"    Latest trial: {trial_info}")
        
        except Exception as e:
            print(f"  Error reading log: {e}")

def analyze_results_improvements():
    """Analyze completed results and compare with previous runs"""
    
    print("\n=== Results Analysis ===")
    
    # Check for completed results
    binary_results_dirs = [
        "checkpoints_binary_cv",
        "checkpoints_survival_cv"
    ]
    
    improvements_found = False
    
    for results_dir in binary_results_dirs:
        if not os.path.exists(results_dir):
            continue
            
        print(f"\nAnalyzing {results_dir}:")
        
        # Find all run directories
        run_dirs = glob.glob(os.path.join(results_dir, "run_*"))
        run_dirs.sort(key=os.path.getmtime, reverse=True)
        
        if not run_dirs:
            print("  No completed runs found")
            continue
        
        metrics_data = []
        
        for run_dir in run_dirs[:10]:  # Analyze last 10 runs
            summary_file = os.path.join(run_dir, "run_summary.json")
            best_run_summary = os.path.join(run_dir, "best_run", "run_summary.json")
            
            # Try best_run first, then regular summary
            summary_path = best_run_summary if os.path.exists(best_run_summary) else summary_file
            
            if os.path.exists(summary_path):
                try:
                    with open(summary_path, 'r') as f:
                        data = json.load(f)
                    
                    run_name = os.path.basename(run_dir)
                    hyperparams = data.get('hyperparameters', {})
                    metrics = data.get('metrics', {})
                    overall = metrics.get('overall', {})
                    
                    if overall:
                        metrics_data.append({
                            'run_name': run_name,
                            'timestamp': run_name.split('_')[1] if '_' in run_name else 'unknown',
                            'model_arch': hyperparams.get('model_arch', 'unknown'),
                            'learning_rate': hyperparams.get('learning_rate', 0),
                            'dropout': hyperparams.get('dropout', 0),
                            'auc': overall.get('roc_auc', 0),
                            'precision': overall.get('precision', 0),
                            'recall': overall.get('recall', 0),
                            'f1': overall.get('f1', 0),
                            'accuracy': overall.get('accuracy', 0)
                        })
                
                except Exception as e:
                    print(f"    Error reading {summary_path}: {e}")
        
        if metrics_data:
            improvements_found = True
            df = pd.DataFrame(metrics_data)
            
            print(f"  Found {len(df)} completed runs")
            
            # Show top performers
            print("\n  Top 5 runs by AUC:")
            top_auc = df.nlargest(5, 'auc')[['run_name', 'model_arch', 'auc', 'precision', 'f1']]
            for _, row in top_auc.iterrows():
                print(f"    {row['model_arch']}: AUC={row['auc']:.4f}, Prec={row['precision']:.4f}, F1={row['f1']:.4f}")
            
            print("\n  Top 5 runs by Precision:")
            top_precision = df.nlargest(5, 'precision')[['run_name', 'model_arch', 'auc', 'precision', 'f1']]
            for _, row in top_precision.iterrows():
                print(f"    {row['model_arch']}: AUC={row['auc']:.4f}, Prec={row['precision']:.4f}, F1={row['f1']:.4f}")
            
            # Check for improvements with new architectures
            new_archs = ['ultra_precision', 'precision_weighted_ensemble']
            new_arch_runs = df[df['model_arch'].isin(new_archs)]
            
            if not new_arch_runs.empty:
                print(f"\n  New Architecture Performance:")
                print(f"    Ultra-precision runs: {len(new_arch_runs[new_arch_runs['model_arch'] == 'ultra_precision'])}")
                print(f"    Ensemble runs: {len(new_arch_runs[new_arch_runs['model_arch'] == 'precision_weighted_ensemble'])}")
                
                best_new = new_arch_runs.loc[new_arch_runs['auc'].idxmax()]
                print(f"    Best new arch: {best_new['model_arch']} - AUC: {best_new['auc']:.4f}, Prec: {best_new['precision']:.4f}")
                
                # Compare with best previous arch
                other_runs = df[~df['model_arch'].isin(new_archs)]
                if not other_runs.empty:
                    best_previous = other_runs.loc[other_runs['auc'].idxmax()]
                    auc_improvement = best_new['auc'] - best_previous['auc']
                    prec_improvement = best_new['precision'] - best_previous['precision']
                    
                    print(f"    Improvement over previous best:")
                    print(f"      AUC: {auc_improvement:+.4f} ({auc_improvement/best_previous['auc']*100:+.2f}%)")
                    print(f"      Precision: {prec_improvement:+.4f} ({prec_improvement/best_previous['precision']*100:+.2f}%)")
    
    if not improvements_found:
        print("  No completed results found yet")

def create_performance_plot():
    """Create visualization of performance improvements"""
    
    try:
        # Find results data
        results_dirs = glob.glob("checkpoints_binary_cv/run_*")
        
        if not results_dirs:
            print("No results available for plotting")
            return
        
        metrics_data = []
        
        for run_dir in results_dirs:
            summary_file = os.path.join(run_dir, "run_summary.json")
            best_run_summary = os.path.join(run_dir, "best_run", "run_summary.json")
            
            summary_path = best_run_summary if os.path.exists(best_run_summary) else summary_file
            
            if os.path.exists(summary_path):
                try:
                    with open(summary_path, 'r') as f:
                        data = json.load(f)
                    
                    hyperparams = data.get('hyperparameters', {})
                    metrics = data.get('metrics', {})
                    overall = metrics.get('overall', {})
                    
                    if overall and overall.get('roc_auc') is not None:
                        # Extract timestamp for ordering
                        run_name = os.path.basename(run_dir)
                        timestamp = run_name.split('_')[1] if '_' in run_name else '20250101_000000'
                        
                        metrics_data.append({
                            'timestamp': timestamp,
                            'model_arch': hyperparams.get('model_arch', 'unknown'),
                            'auc': overall.get('roc_auc', 0),
                            'precision': overall.get('precision', 0),
                            'recall': overall.get('recall', 0),
                            'f1': overall.get('f1', 0)
                        })
                        
                except Exception as e:
                    continue
        
        if len(metrics_data) < 2:
            print("Not enough data for plotting")
            return
        
        df = pd.DataFrame(metrics_data)
        df = df.sort_values('timestamp')
        
        # Create the plot
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # AUC over time
        for arch in df['model_arch'].unique():
            arch_data = df[df['model_arch'] == arch]
            ax1.plot(range(len(arch_data)), arch_data['auc'], marker='o', label=arch)
        ax1.set_title('AUC Performance by Architecture')
        ax1.set_ylabel('AUC')
        ax1.legend()
        ax1.grid(True)
        
        # Precision over time
        for arch in df['model_arch'].unique():
            arch_data = df[df['model_arch'] == arch]
            ax2.plot(range(len(arch_data)), arch_data['precision'], marker='s', label=arch)
        ax2.set_title('Precision Performance by Architecture')
        ax2.set_ylabel('Precision')
        ax2.legend()
        ax2.grid(True)
        
        # AUC vs Precision scatter
        arch_colors = plt.cm.Set3(np.linspace(0, 1, len(df['model_arch'].unique())))
        for i, arch in enumerate(df['model_arch'].unique()):
            arch_data = df[df['model_arch'] == arch]
            ax3.scatter(arch_data['auc'], arch_data['precision'], 
                       c=[arch_colors[i]], label=arch, s=60, alpha=0.7)
        ax3.set_xlabel('AUC')
        ax3.set_ylabel('Precision')
        ax3.set_title('AUC vs Precision Trade-off')
        ax3.legend()
        ax3.grid(True)
        
        # Architecture performance summary
        arch_summary = df.groupby('model_arch')[['auc', 'precision']].agg(['mean', 'max']).round(4)
        arch_summary.plot(kind='bar', ax=ax4)
        ax4.set_title('Architecture Performance Summary')
        ax4.set_ylabel('Score')
        ax4.legend()
        ax4.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = "performance_improvements.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"\nPerformance plot saved to: {plot_path}")
        
        plt.show()
        
    except Exception as e:
        print(f"Error creating plot: {e}")

def main():
    parser = argparse.ArgumentParser(description="Monitor HCC training improvements")
    parser.add_argument('--plot', action='store_true', help='Create performance plots')
    parser.add_argument('--continuous', action='store_true', help='Monitor continuously')
    parser.add_argument('--interval', type=int, default=60, help='Monitoring interval in seconds')
    
    args = parser.parse_args()
    
    if args.continuous:
        print("Starting continuous monitoring (Ctrl+C to stop)...")
        try:
            while True:
                os.system('clear')  # Clear screen
                monitor_job_progress()
                analyze_results_improvements()
                print(f"\nNext update in {args.interval} seconds...")
                time.sleep(args.interval)
        except KeyboardInterrupt:
            print("\nMonitoring stopped")
    else:
        monitor_job_progress()
        analyze_results_improvements()
        
        if args.plot:
            create_performance_plot()

if __name__ == "__main__":
    main() 