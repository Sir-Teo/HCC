#!/usr/bin/env python3
"""
Simple embeddings visualization script for local trained DINOv2 model
Creates PCA, t-SNE, and UMAP plots showing positive and negative cases
"""

import argparse
import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap.umap_ as umap
from tqdm import tqdm

# Custom module imports
from data.dataset import HCCDataModule
from models.dino import load_dinov2_model

def extract_features_simple(data_loader, model, device):
    """Extract features using the local trained DINOv2 model."""
    model.eval()
    all_features = []
    all_events = []
    all_patient_ids = []

    print("Extracting features from local trained model...")
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Extracting Features"):
            if len(batch) == 2:
                images, events = batch
            elif len(batch) == 3:
                images, _, events = batch
            else:
                raise ValueError(f"Unexpected batch structure length: {len(batch)}")

            images = images.to(device)
            batch_size, num_samples, num_slices, C, H, W = images.size()
            images_reshaped = images.view(batch_size * num_samples * num_slices, C, H, W)
            
            # Extract features using forward_features (local model)
            feats = model.forward_features(images_reshaped)
            feature_dim = feats.size(-1)
            
            # Reshape and average
            feats = feats.view(batch_size, num_samples * num_slices, feature_dim)
            feats_avg = feats.mean(dim=1)  # Average across slices
            
            all_features.append(feats_avg.cpu().numpy())
            all_events.append(events.cpu().numpy())

    features = np.concatenate(all_features, axis=0)
    events = np.concatenate(all_events, axis=0)
    
    # Get patient IDs
    if hasattr(data_loader.dataset, 'patient_data'):
        patient_ids = [p['patient_id'] for p in data_loader.dataset.patient_data[:len(features)]]
    else:
        patient_ids = [f'patient_{i}' for i in range(len(features))]
    
    print(f"Extracted features for {len(features)} patients")
    print(f"Feature shape: {features.shape}")
    print(f"Positive cases: {int(events.sum())}, Negative cases: {int(len(events) - events.sum())}")
    
    return features, events, patient_ids

def create_simple_visualizations(features, events, patient_ids, output_dir):
    """Create PCA, t-SNE, and UMAP visualizations."""
    os.makedirs(output_dir, exist_ok=True)
    
    colors = ['#1f77b4', '#ff7f0e']  # Blue for negative, Orange for positive
    labels = ['Negative (Event=0)', 'Positive (Event=1)']
    
    # Create figure with 1 row x 3 columns
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Embedding Visualizations: Local Trained DINOv2 Model', fontsize=16, fontweight='bold')
    
    print("Creating visualizations...")
    
    # PCA
    print("Computing PCA...")
    pca = PCA(n_components=2, random_state=42)
    features_pca = pca.fit_transform(features)
    
    ax = axes[0]
    for event_val in [0, 1]:
        mask = events == event_val
        if np.any(mask):
            ax.scatter(features_pca[mask, 0], features_pca[mask, 1], 
                      c=colors[event_val], label=labels[event_val], alpha=0.7, s=60)
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
    ax.set_title('PCA')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # t-SNE
    print("Computing t-SNE...")
    perplexity = min(30, len(features)//4)
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
    features_tsne = tsne.fit_transform(features)
    
    ax = axes[1]
    for event_val in [0, 1]:
        mask = events == event_val
        if np.any(mask):
            ax.scatter(features_tsne[mask, 0], features_tsne[mask, 1], 
                      c=colors[event_val], label=labels[event_val], alpha=0.7, s=60)
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    ax.set_title('t-SNE')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # UMAP
    print("Computing UMAP...")
    n_neighbors = min(15, len(features)//3)
    reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=n_neighbors)
    features_umap = reducer.fit_transform(features)
    
    ax = axes[2]
    for event_val in [0, 1]:
        mask = events == event_val
        if np.any(mask):
            ax.scatter(features_umap[mask, 0], features_umap[mask, 1], 
                      c=colors[event_val], label=labels[event_val], alpha=0.7, s=60)
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    ax.set_title('UMAP')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Save
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'embeddings_local_visualization.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to: {output_path}")
    
    # Save results
    results_df = pd.DataFrame({
        'patient_id': patient_ids,
        'event': events
    })
    results_path = os.path.join(output_dir, 'embedding_results.csv')
    results_df.to_csv(results_path, index=False)
    print(f"Results saved to: {results_path}")

def main():
    # Simple argument setup
    args = argparse.Namespace(
        nyu_dicom_root="/gpfs/data/mankowskilab/HCC_Recurrence/dicom",
        nyu_csv_file="/gpfs/data/shenlab/wz1492/HCC/spreadsheets/processed_patient_labels_nyu.csv",
        preprocessed_root="/gpfs/data/mankowskilab/HCC_Recurrence/preprocessed/",
        batch_size=8,
        num_slices=32,
        num_workers=4,
        num_samples_per_patient=1,
        local_weights="/gpfs/data/shenlab/wz1492/HCC/dinov2/experiments/eval/training_112499/teacher_checkpoint.pth",
        output_dir="./embedding_visualizations_all_patients"
    )
    
    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Set up data
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
        cross_validation=False,
        random_state=42,
        use_validation=False
    )
    
    data_module.setup()
    
    # Create dataloader
    from torch.utils.data import DataLoader
    from data.dataset import HCCDicomDataset
    
    all_dataset = HCCDicomDataset(
        patient_data_list=data_module.all_patients,
        model_type="linear",
        transform=data_module.transform,
        num_slices=args.num_slices,
        num_samples=args.num_samples_per_patient,
        preprocessed_root=args.preprocessed_root
    )
    
    all_dataloader = DataLoader(
        all_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=data_module.collate_fn
    )
    
    print(f"Total patients: {len(all_dataset)}")
    
    # Load model
    print("Loading local trained model...")
    model = load_dinov2_model(args.local_weights)
    model = model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    print("Model loaded successfully")
    
    # Extract features
    features, events, patient_ids = extract_features_simple(all_dataloader, model, device)
    
    # Create visualizations
    create_simple_visualizations(features, events, patient_ids, args.output_dir)
    
    print("Visualization complete!")

if __name__ == "__main__":
    main() 