import json
import matplotlib.pyplot as plt
import numpy as np

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

def plot_training_metrics(iterations, metrics):
    """Create a combined figure with enhanced styling for training metrics visualization."""
    
    # Create figure with a grid of subplots
    fig = plt.figure(figsize=(20, 15))
    
    # Color schemes
    colors = plt.cm.Set2(np.linspace(0, 1, 8))
    
    # Plot 1: Total Loss
    ax1 = plt.subplot(2, 2, 1)
    ax1.plot(iterations, metrics['total_loss'], color=colors[0], linewidth=2, label='Total Loss')
    ax1.set_title('Total Training Loss', fontsize=14, pad=15)
    ax1.set_xlabel('Iteration', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend(fontsize=10)
    
    # Plot 2: Component Losses
    ax2 = plt.subplot(2, 2, 2)
    loss_components = ['dino_local_crops_loss', 'dino_global_crops_loss', 
                      'koleo_loss', 'ibot_loss']
    for idx, loss in enumerate(loss_components):
        ax2.plot(iterations, metrics[loss], color=colors[idx+1], 
                linewidth=2, label=loss.replace('_', ' ').title())
    ax2.set_title('Component Losses', fontsize=14, pad=15)
    ax2.set_xlabel('Iteration', fontsize=12)
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend(fontsize=10, bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Plot 3: Training Parameters
    ax3 = plt.subplot(2, 2, 3)
    param_lines = []
    param_lines.append(ax3.plot(iterations, metrics['lr'], 
                              color=colors[5], linewidth=2, label='Learning Rate')[0])
    param_lines.append(ax3.plot(iterations, metrics['last_layer_lr'], 
                              color=colors[6], linewidth=2, label='Last Layer LR')[0])
    
    # Create twin axis for weight decay
    ax3_twin = ax3.twinx()
    param_lines.append(ax3_twin.plot(iterations, metrics['wd'], 
                                   color=colors[7], linewidth=2, 
                                   linestyle='--', label='Weight Decay')[0])
    
    ax3.set_title('Training Parameters', fontsize=14, pad=15)
    ax3.set_xlabel('Iteration', fontsize=12)
    ax3.set_ylabel('Learning Rate', fontsize=12)
    ax3_twin.set_ylabel('Weight Decay', fontsize=12)
    ax3.grid(True, linestyle='--', alpha=0.7)
    
    # Combined legend for both axes
    labs = [l.get_label() for l in param_lines]
    ax3.legend(param_lines, labs, fontsize=10, loc='upper right')
    
    # Plot 4: Timing Metrics
    ax4 = plt.subplot(2, 2, 4)
    ax4.plot(iterations, metrics['iter_time'], color=colors[3], 
             linewidth=2, label='Iteration Time')
    ax4.plot(iterations, metrics['data_time'], color=colors[4], 
             linewidth=2, label='Data Loading Time')
    ax4.set_title('Timing Metrics', fontsize=14, pad=15)
    ax4.set_xlabel('Iteration', fontsize=12)
    ax4.set_ylabel('Time (seconds)', fontsize=12)
    ax4.grid(True, linestyle='--', alpha=0.7)
    ax4.legend(fontsize=10)
    
    # Overall title
    plt.suptitle('Training Progress Overview', fontsize=16, y=0.95)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save figure
    plt.savefig('../img/training_overview_dinov2_reg.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    # Replace with your actual filename
    filename = "/gpfs/data/shenlab/wz1492/HCC/dinov2/experiments_large_dataset_reg/training_metrics.json"
    # Read and plot the metrics
    iterations, metrics = read_metrics(filename)
    plot_training_metrics(iterations, metrics)