import json
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter
import seaborn as sns
from matplotlib.gridspec import GridSpec
from datetime import datetime, timedelta
from scipy.stats import gaussian_kde

def read_metrics(filename):
    """Read JSON lines file and convert to lists of metrics."""
    iterations = []
    metrics = {}
    
    with open(filename, 'r') as f:
        for line in f:
            data = json.loads(line)
            iterations.append(data['iteration'])
            
            # Initialize dictionary keys on first iteration
            if not metrics:
                metrics = {key: [] for key in data.keys()}
            
            # Append values
            for key, value in data.items():
                metrics[key].append(value)
                
    return iterations, metrics

def smooth_curve(y, window_length=101, polyorder=3):
    """Apply Savitzky-Golay filter to smooth the curve."""
    if len(y) < window_length:
        window_length = len(y) - 1 if len(y) % 2 == 0 else len(y) - 2
    if window_length < polyorder:
        return y
    return savgol_filter(y, window_length, polyorder)

def calculate_confidence_interval(y, window=100):
    """Calculate moving standard deviation for confidence intervals."""
    rolling_std = np.array([np.std(y[max(0, i-window):min(len(y), i+window)])
                           for i in range(len(y))])
    rolling_mean = smooth_curve(y, window_length=window)
    return rolling_mean - 2*rolling_std, rolling_mean + 2*rolling_std

def set_style():
    """Set the style for all plots."""
    sns.set_palette("deep")
    plt.rcParams.update({
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'grid.alpha': 0.2,
        'axes.grid': True,
        'font.size': 10,
        'axes.labelsize': 12,
        'axes.titlesize': 14,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'legend.title_fontsize': 12
    })

def format_time(seconds):
    """Convert seconds to human-readable format."""
    return str(timedelta(seconds=int(seconds)))

def plot_training_metrics(iterations, metrics, smooth=True, save_dir='./'):
    """Create comprehensive training visualization with emphasized raw data."""
    set_style()
    
    # Create figure with custom layout
    fig = plt.figure(figsize=(20, 15))
    gs = GridSpec(3, 2, figure=fig)
    
    # 1. Total Loss Plot (larger, spanning two columns)
    ax1 = fig.add_subplot(gs[0, :])
    x = np.array(iterations)
    y = np.array(metrics['total_loss'])
    
    # Plot raw data with density-based transparency
    xy = np.vstack([x, y])
    z = gaussian_kde(xy)(xy)
    scatter = ax1.scatter(x, y, c=z, s=20, alpha=0.6, cmap='viridis', 
                         label='Raw Data Points')
    plt.colorbar(scatter, ax=ax1, label='Point Density')
    
    if smooth:
        y_smooth = smooth_curve(y)
        conf_low, conf_high = calculate_confidence_interval(y)
        
        # Plot confidence interval
        ax1.fill_between(x, conf_low, conf_high, alpha=0.2, color='blue',
                        label='95% Confidence Interval')
        
        # Plot trend line
        ax1.plot(x, y_smooth, 'r-', linewidth=2, label='Trend Line')
        
        # Add min/max annotations
        min_idx = np.argmin(y_smooth)
        ax1.scatter(x[min_idx], y_smooth[min_idx], color='red', s=100, zorder=5)
        ax1.annotate(f'Min: {y_smooth[min_idx]:.4f}', 
                    (x[min_idx], y_smooth[min_idx]),
                    xytext=(10, 10), textcoords='offset points')
    
    ax1.set_title('Total Training Loss Evolution', fontweight='bold', pad=15)
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Loss Value')
    ax1.legend(loc='upper right')
    
    # 2. Component Losses with Raw Data
    ax2 = fig.add_subplot(gs[1, 0])
    loss_components = ['dino_local_crops_loss', 'dino_global_crops_loss', 
                      'koleo_loss', 'ibot_loss']
    
    for loss in loss_components:
        y = np.array(metrics[loss])
        # Plot raw data with low alpha
        ax2.scatter(x[::5], y[::5], alpha=0.2, s=10, label=f'{loss} (Raw)')
        if smooth:
            y_smooth = smooth_curve(y)
            ax2.plot(x, y_smooth, linewidth=2, 
                    label=f'{loss} (Trend)', alpha=0.8)
    
    ax2.set_title('Component Losses Comparison', fontweight='bold', pad=15)
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Loss Value')
    ax2.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    
    # 3. Learning Parameters with Step Detection
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.set_yscale('log')
    
    # Detect and highlight learning rate steps
    lr = np.array(metrics['lr'])
    lr_changes = np.where(np.diff(lr) != 0)[0]
    
    # Plot raw learning rates with step highlights
    ax3.plot(x, lr, 'b-', label='Base LR', linewidth=2)
    ax3.scatter(x[lr_changes], lr[lr_changes], color='red', s=50, 
                label='LR Changes', zorder=5)
    
    ax3.plot(x, metrics['last_layer_lr'], 'g-', label='Last Layer LR', linewidth=2)
    ax3.plot(x, metrics['wd'], 'r-', label='Weight Decay', linewidth=2)
    
    ax3.set_title('Training Parameters (Log Scale)', fontweight='bold', pad=15)
    ax3.set_xlabel('Iteration')
    ax3.set_ylabel('Value')
    ax3.legend()
    
    # 4. Timing Analysis with Raw Data Points
    ax4 = fig.add_subplot(gs[2, 0])
    window = min(50, len(iterations) // 10)
    
    # Plot raw timing data
    ax4.scatter(x[::10], metrics['iter_time'][::10], alpha=0.2, s=10, 
                color='blue', label='Raw Iteration Time')
    ax4.scatter(x[::10], metrics['data_time'][::10], alpha=0.2, s=10, 
                color='green', label='Raw Data Loading Time')
    
    # Calculate and plot moving averages
    iter_time_ma = np.convolve(metrics['iter_time'], 
                              np.ones(window)/window, 
                              mode='valid')
    data_time_ma = np.convolve(metrics['data_time'], 
                              np.ones(window)/window, 
                              mode='valid')
    x_ma = x[window-1:]
    
    ax4.plot(x_ma, iter_time_ma, 'b-', label='Iteration Time (MA)', linewidth=2)
    ax4.plot(x_ma, data_time_ma, 'g-', label='Data Loading Time (MA)', linewidth=2)
    
    # Add timing statistics
    avg_iter_time = np.mean(iter_time_ma)
    ax4.axhline(y=avg_iter_time, color='r', linestyle='--', alpha=0.5)
    
    ax4.set_title('Training Speed Analysis', fontweight='bold', pad=15)
    ax4.set_xlabel('Iteration')
    ax4.set_ylabel('Time (seconds)')
    ax4.legend()
    
    # 5. Training Efficiency Distribution
    ax5 = fig.add_subplot(gs[2, 1])
    efficiency = (np.array(metrics['data_time']) / 
                 np.array(metrics['iter_time'])) * 100
    
    # Create violin plot with individual points
    parts = ax5.violinplot(efficiency, points=100, showmeans=True, 
                          showextrema=True)
    ax5.scatter(np.ones_like(efficiency), efficiency, alpha=0.1, 
                color='blue', s=10)
    
    # Customize violin plot colors
    for pc in parts['bodies']:
        pc.set_facecolor('lightblue')
        pc.set_alpha(0.7)
    
    ax5.set_title('Data Loading Efficiency Distribution', fontweight='bold', pad=15)
    ax5.set_ylabel('Data Loading Time / Total Iteration Time (%)')
    ax5.set_xticks([])
    
    # Add statistics annotations
    stats_text = (f'Mean: {np.mean(efficiency):.1f}%\n'
                 f'Median: {np.median(efficiency):.1f}%\n'
                 f'Std Dev: {np.std(efficiency):.1f}%')
    ax5.text(0.95, 0.95, stats_text, transform=ax5.transAxes,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(facecolor='white', alpha=0.8))
    
    # Overall layout adjustments
    plt.suptitle('Comprehensive Training Analysis Dashboard', 
                fontsize=16, fontweight='bold', y=0.95)
    plt.tight_layout()
    
    # Save high-quality figures
    plt.savefig(f'{save_dir}/training_dashboard.png', 
                dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    filename = "/gpfs/data/shenlab/wz1492/HCC/dinov2/experiments_large_dataset_reg/training_metrics.json"
    iterations, metrics = read_metrics(filename)
    plot_training_metrics(iterations, metrics, smooth=True)