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
    """Create multiple subplot figures to visualize different metric groups."""
    
    # Create figure for losses
    plt.figure(figsize=(15, 10))
    
    # Plot total loss
    plt.subplot(2, 1, 1)
    plt.plot(iterations, metrics['total_loss'], 'b-', label='Total Loss')
    plt.title('Total Training Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    
    # Plot component losses
    plt.subplot(2, 1, 2)
    loss_components = ['dino_local_crops_loss', 'dino_global_crops_loss', 
                      'koleo_loss', 'ibot_loss']
    for loss in loss_components:
        plt.plot(iterations, metrics[loss], label=loss)
    plt.title('Component Losses')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('training_loss.png')
    
    # Create figure for training parameters
    plt.figure(figsize=(15, 10))
    
    # Plot learning rates and weight decay
    plt.subplot(2, 1, 1)
    plt.plot(iterations, metrics['lr'], label='Learning Rate')
    plt.plot(iterations, metrics['last_layer_lr'], label='Last Layer LR')
    plt.plot(iterations, metrics['wd'], label='Weight Decay')
    plt.title('Training Parameters')
    plt.xlabel('Iteration')
    plt.ylabel('Value')
    plt.grid(True)
    plt.legend()
    
    # Plot timing metrics
    plt.subplot(2, 1, 2)
    plt.plot(iterations, metrics['iter_time'], label='Iteration Time')
    plt.plot(iterations, metrics['data_time'], label='Data Loading Time')
    plt.title('Timing Metrics')
    plt.xlabel('Iteration')
    plt.ylabel('Time (seconds)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    
    plt.savefig('training_metrics.png')

if __name__ == "__main__":
    # Replace with your actual filename
    filename = "/gpfs/data/shenlab/wz1492/HCC/dinov2/experiments/training_metrics.json"
    # Read and plot the metrics
    iterations, metrics = read_metrics(filename)
    plot_training_metrics(iterations, metrics)