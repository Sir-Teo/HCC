#!/usr/bin/env python3
"""
Visualize DINOv2 embeddings for HCC binary classification.
Uses the same embedding extraction pipeline as train_binary.py to ensure consistency.
"""

import argparse
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Import modules from the project
from data.dataset import HCCDataModule
from models.dino import load_dinov2_model

# Set style for better plots
plt.style.use('default')
sns.set_palette("husl")

def extract_features(data_loader, model, device):
    """
    Extract features using the DINOv2 model.
    This is the same function as in train_binary.py to ensure consistency.
    """
    model.eval()
    features = []
    events = []
    patient_info = []

    if data_loader is None:
        return np.array([]).reshape(0, 1, 1, 768), np.array([]), []

    desc = "Extracting Features"
    try: 
        if hasattr(data_loader.dataset, 'patient_data') and len(data_loader.dataset.patient_data) > 0:
            source_example = data_loader.dataset.patient_data[0].get('source', 'NYU')
            desc = f"Extracting Features ({source_example} set)" 
    except Exception:
        pass

    with torch.no_grad():
        for batch in tqdm(data_loader, desc=desc):
            if len(batch) == 2:
                images, e = batch
            elif len(batch) == 3:
                images, _, e = batch
            else:
                raise ValueError(f"Unexpected batch structure length: {len(batch)}")

            images = images.to(device)
            batch_size, num_samples, num_slices, C, H, W = images.size()
            images_reshaped = images.view(batch_size * num_samples * num_slices, C, H, W)
            
            feats = model.forward_features(images_reshaped)
            feature_dim = feats.size(-1)
            feats = feats.view(batch_size, num_samples * num_slices, feature_dim)
            feats_reshaped = feats.view(batch_size, num_samples, num_slices, feature_dim)

            features.append(feats_reshaped.cpu().numpy())
            events.append(e.cpu().numpy())

    if not features:
        feature_dim_expected = 768 
        return np.empty((0, 1, 1, feature_dim_expected), dtype=np.float32), np.empty((0,), dtype=np.float32), []

    features = np.concatenate(features, axis=0)
    events = np.concatenate(events, axis=0)
    
    if hasattr(data_loader.dataset, 'patient_data'):
        patient_info = data_loader.dataset.patient_data[:len(features)]
    else:
        patient_info = [{'patient_id': f'unknown_{i}', 'dataset_type': 'NYU'} for i in range(len(features))]
        
    return features, events, patient_info

def preprocess_features(features, events):
    """
    Apply the same preprocessing pipeline as in train_binary.py
    """
    # Average across samples dimension -> [patients, slices, features]
    features = features.mean(axis=1)
    
    # Remove zero-variance features
    n_patients, n_slices, n_features = features.shape
    features_flat = features.reshape(-1, n_features)
    variances = np.var(features_flat, axis=0)
    zero_var_indices = np.where(variances == 0)[0]
    
    if len(zero_var_indices) > 0:
        print(f"Removing {len(zero_var_indices)} features with zero variance")
        non_zero_var_indices = np.where(variances != 0)[0]
        features = features[:, :, non_zero_var_indices]
        n_features = features.shape[2]
    
    # Standardize features
    scaler = StandardScaler()
    features_reshaped = features.reshape(-1, n_features)
    features_scaled = scaler.fit_transform(features_reshaped).astype('float32')
    features_scaled = features_scaled.reshape(n_patients, n_slices, n_features)
    
    # Collapse slice dimension using adaptive average pooling
    slice_pool = nn.AdaptiveAvgPool1d(1)
    features_tensor = torch.from_numpy(features_scaled.transpose(0, 2, 1))  # [patients, features, slices]
    features_final = slice_pool(features_tensor).squeeze(-1).numpy()  # [patients, features]
    
    return features_final

def create_visualizations(features, events, patient_info, output_dir):
    """
    Create PCA, t-SNE, and UMAP visualizations
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create labels for visualization
    labels = ['Negative' if e == 0 else 'Positive' for e in events]
    colors = ['#1f77b4' if e == 0 else '#ff7f0e' for e in events]  # Blue for negative, orange for positive
    
    n_positive = np.sum(events == 1)
    n_negative = np.sum(events == 0)
    
    print(f"Dataset composition: {n_positive} positive, {n_negative} negative samples")
    
    # Set up the figure
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('DINOv2 Embeddings Visualization for HCC Binary Classification', fontsize=16)
    
    # 1. PCA
    print("Computing PCA...")
    pca = PCA(n_components=2, random_state=42)
    pca_features = pca.fit_transform(features)
    
    axes[0, 0].scatter(pca_features[:, 0], pca_features[:, 1], c=colors, alpha=0.7, s=50)
    axes[0, 0].set_title(f'PCA (Explained Variance: {pca.explained_variance_ratio_.sum():.2%})')
    axes[0, 0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
    axes[0, 0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. t-SNE
    print("Computing t-SNE...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(features)-1))
    tsne_features = tsne.fit_transform(features)
    
    axes[0, 1].scatter(tsne_features[:, 0], tsne_features[:, 1], c=colors, alpha=0.7, s=50)
    axes[0, 1].set_title('t-SNE')
    axes[0, 1].set_xlabel('t-SNE 1')
    axes[0, 1].set_ylabel('t-SNE 2')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. UMAP
    print("Computing UMAP...")
    umap_model = umap.UMAP(n_components=2, random_state=42, n_neighbors=min(15, len(features)-1))
    umap_features = umap_model.fit_transform(features)
    
    axes[1, 0].scatter(umap_features[:, 0], umap_features[:, 1], c=colors, alpha=0.7, s=50)
    axes[1, 0].set_title('UMAP')
    axes[1, 0].set_xlabel('UMAP 1')
    axes[1, 0].set_ylabel('UMAP 2')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Feature statistics
    axes[1, 1].hist([features[events == 0].mean(axis=1), features[events == 1].mean(axis=1)], 
                    bins=20, alpha=0.7, label=['Negative', 'Positive'], color=['#1f77b4', '#ff7f0e'])
    axes[1, 1].set_title('Feature Mean Distribution')
    axes[1, 1].set_xlabel('Mean Feature Value')
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # Add legend to the entire figure
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='#1f77b4', label=f'Negative (n={n_negative})'),
                      Patch(facecolor='#ff7f0e', label=f'Positive (n={n_positive})')]
    fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.95))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'embeddings_visualization.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'embeddings_visualization.pdf'), bbox_inches='tight')
    print(f"Visualization saved to {output_dir}")
    
    # Save individual plots for better inspection
    create_individual_plots(pca_features, tsne_features, umap_features, events, output_dir)
    
    # Save embedding data
    save_embedding_data(pca_features, tsne_features, umap_features, events, patient_info, output_dir)

def create_individual_plots(pca_features, tsne_features, umap_features, events, output_dir):
    """
    Create individual high-resolution plots for each visualization method
    """
    colors = ['#1f77b4' if e == 0 else '#ff7f0e' for e in events]
    n_positive = np.sum(events == 1)
    n_negative = np.sum(events == 0)
    
    # Individual PCA plot
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(pca_features[:, 0], pca_features[:, 1], c=colors, alpha=0.7, s=80)
    plt.title('PCA Visualization of DINOv2 Embeddings', fontsize=14)
    plt.xlabel('Principal Component 1', fontsize=12)
    plt.ylabel('Principal Component 2', fontsize=12)
    plt.grid(True, alpha=0.3)
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='#1f77b4', label=f'Negative (n={n_negative})'),
                      Patch(facecolor='#ff7f0e', label=f'Positive (n={n_positive})')]
    plt.legend(handles=legend_elements, loc='best')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'pca_individual.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Individual t-SNE plot
    plt.figure(figsize=(10, 8))
    plt.scatter(tsne_features[:, 0], tsne_features[:, 1], c=colors, alpha=0.7, s=80)
    plt.title('t-SNE Visualization of DINOv2 Embeddings', fontsize=14)
    plt.xlabel('t-SNE Dimension 1', fontsize=12)
    plt.ylabel('t-SNE Dimension 2', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(handles=legend_elements, loc='best')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'tsne_individual.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Individual UMAP plot
    plt.figure(figsize=(10, 8))
    plt.scatter(umap_features[:, 0], umap_features[:, 1], c=colors, alpha=0.7, s=80)
    plt.title('UMAP Visualization of DINOv2 Embeddings', fontsize=14)
    plt.xlabel('UMAP Dimension 1', fontsize=12)
    plt.ylabel('UMAP Dimension 2', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(handles=legend_elements, loc='best')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'umap_individual.png'), dpi=300, bbox_inches='tight')
    plt.close()

def save_embedding_data(pca_features, tsne_features, umap_features, events, patient_info, output_dir):
    """
    Save the embedding coordinates and metadata to CSV files
    """
    # Create DataFrame with all embedding coordinates
    embedding_df = pd.DataFrame({
        'patient_id': [p['patient_id'] for p in patient_info],
        'event': events,
        'pca_1': pca_features[:, 0],
        'pca_2': pca_features[:, 1],
        'tsne_1': tsne_features[:, 0],
        'tsne_2': tsne_features[:, 1],
        'umap_1': umap_features[:, 0],
        'umap_2': umap_features[:, 1]
    })
    
    embedding_df.to_csv(os.path.join(output_dir, 'embedding_coordinates.csv'), index=False)
    print(f"Embedding coordinates saved to {os.path.join(output_dir, 'embedding_coordinates.csv')}")

def main():
    parser = argparse.ArgumentParser(description="Visualize DINOv2 embeddings for HCC binary classification")
    
    # Use the same default arguments as train_binary.py
    parser.add_argument("--nyu_dicom_root", type=str, default="/gpfs/data/mankowskilab/HCC_Recurrence/dicom",
                        help="Path to the NYU DICOM root directory.")
    parser.add_argument("--nyu_csv_file", type=str, default="/gpfs/data/shenlab/wz1492/HCC/spreadsheets/processed_patient_labels_nyu.csv",
                        help="Path to the NYU CSV metadata file.")
    parser.add_argument('--preprocessed_root', type=str, default='/gpfs/data/mankowskilab/HCC_Recurrence/preprocessed/', 
                        help='Base directory to store/load preprocessed image tensors.')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for data loaders')
    parser.add_argument('--num_slices', type=int, default=32,
                        help='Number of slices per patient sample')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of workers for data loaders')
    parser.add_argument('--num_samples_per_patient', type=int, default=1,
                        help='Number of times to sample slices from each patient series')
    parser.add_argument('--dinov2_weights', type=str, default=None,
                        help="Path to your local DINOv2 state dict file (.pth or .pt). If not provided, uses pretrained ImageNet DINO weights.")
    parser.add_argument('--output_dir', type=str, default='embedding_visualizations',
                        help='Directory to save visualization outputs')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set up data module (without cross-validation to get all data)
    print("Setting up data module...")
    data_module = HCCDataModule(
        csv_file=args.nyu_csv_file,
        dicom_root=args.nyu_dicom_root,
        model_type="linear",
        batch_size=args.batch_size,
        num_slices=args.num_slices,
        num_samples=args.num_samples_per_patient,
        num_workers=args.num_workers,
        preprocessed_root=args.preprocessed_root,
        cross_validation=False,  # Get all data for visualization
        random_state=42
    )
    data_module.setup()
    
    # Get data loader for all data
    train_loader = data_module.train_dataloader()
    
    # Set up device and model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    print("Loading DINOv2 model...")
    if args.dinov2_weights is not None:
        dino_model = load_dinov2_model(args.dinov2_weights)
    else:
        # Use pretrained hub model if no weights path provided
        print("No custom weights provided, using pretrained DINOv2 ViT-B/14 model...")
        from models.dino import DinoV2Wrapper
        base_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
        dino_model = DinoV2Wrapper(base_model)
    
    dino_model = dino_model.to(device)
    dino_model.eval()
    
    # Prevent DINOv2 from tracking gradients
    for param in dino_model.parameters():
        param.requires_grad = False
    
    # Extract features
    print("Extracting features...")
    features, events, patient_info = extract_features(train_loader, dino_model, device)
    
    if features.size == 0:
        print("No features extracted. Exiting.")
        return
    
    print(f"Extracted features shape: {features.shape}")
    print(f"Number of positive samples: {np.sum(events == 1)}")
    print(f"Number of negative samples: {np.sum(events == 0)}")
    
    # Preprocess features using the same pipeline as train_binary.py
    print("Preprocessing features...")
    features_processed = preprocess_features(features, events)
    
    print(f"Processed features shape: {features_processed.shape}")
    
    # Create visualizations
    print("Creating visualizations...")
    create_visualizations(features_processed, events, patient_info, args.output_dir)
    
    print("Visualization complete!")

if __name__ == "__main__":
    main() 