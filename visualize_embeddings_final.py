#!/usr/bin/env python3
"""
Final embeddings visualization script comparing local trained and pretrained DINOv2 models
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

def load_pretrained_model():
    """Load pretrained DINOv2 model using local weights"""
    pretrained_path = "./dinov2/pretrained_weights/dinov2_vitl14_pretrain.pth"
    if os.path.exists(pretrained_path):
        print(f"Loading pretrained model from: {pretrained_path}")
        
        # Load pretrained weights
        checkpoint = torch.load(pretrained_path, map_location='cpu')
        
        # Create model
        model = torch.hub.load('./dinov2', 'dinov2_vitl14', source='local', pretrained=False)
        
        # Load state dict
        model.load_state_dict(checkpoint, strict=False)
        print("Pretrained model loaded successfully")
        return model
    else:
        raise FileNotFoundError(f"Pretrained weights not found at {pretrained_path}")

def extract_features_unified(data_loader, model, device, model_name, is_local=True):
    """Extract features from either local or pretrained model."""
    model.eval()
    all_features = []
    all_events = []

    print(f"Extracting features from {model_name}...")
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc=f"Extracting Features ({model_name})"):
            if len(batch) == 2:
                images, events = batch
            elif len(batch) == 3:
                images, _, events = batch
            else:
                raise ValueError(f"Unexpected batch structure length: {len(batch)}")

            images = images.to(device)
            batch_size, num_samples, num_slices, C, H, W = images.size()
            images_reshaped = images.view(batch_size * num_samples * num_slices, C, H, W)
            
            if is_local:
                # Local model with forward_features method
                feats = model.forward_features(images_reshaped)
            else:
                # Pretrained model with standard forward
                feats = model(images_reshaped)
                
                # Handle different output formats for pretrained
                if isinstance(feats, dict):
                    # Take CLS token if available
                    if 'x_norm_clstoken' in feats:
                        feats = feats['x_norm_clstoken']
                    else:
                        # Use the first tensor we find
                        tensor_keys = [k for k, v in feats.items() if hasattr(v, 'shape')]
                        if tensor_keys:
                            feats = feats[tensor_keys[0]]
                            if feats.dim() == 3:  # [batch, seq, features]
                                feats = feats[:, 0]  # Take CLS token
                        else:
                            raise ValueError(f"No valid tensor found in model output")
            
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
    
    print(f"Extracted features for {len(features)} patients from {model_name}")
    print(f"Feature shape: {features.shape}")
    print(f"Positive cases: {int(events.sum())}, Negative cases: {int(len(events) - events.sum())}")
    
    return features, events, patient_ids

def create_comparison_visualizations(features_local, features_pretrained, events, patient_ids, output_dir):
    """Create comparison visualizations for both models."""
    os.makedirs(output_dir, exist_ok=True)
    
    colors = ['#1f77b4', '#ff7f0e']  # Blue for negative, Orange for positive
    labels = ['Negative (Event=0)', 'Positive (Event=1)']
    
    # Check if we have both models
    has_pretrained = features_pretrained is not None and len(features_pretrained) > 0
    
    if has_pretrained:
        # Create comparison plot (2 rows x 3 columns)
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Embedding Visualizations: Local Trained vs Pretrained DINOv2 Models', fontsize=16, fontweight='bold')
        
        models_data = [
            ("Local Trained Model", features_local),
            ("Pretrained Model", features_pretrained)
        ]
    else:
        # Only local model (1 row x 3 columns)
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle('Embedding Visualizations: Local Trained DINOv2 Model', fontsize=16, fontweight='bold')
        
        models_data = [("Local Trained Model", features_local)]
        axes = axes.reshape(1, -1)  # Ensure 2D array for consistent indexing
    
    for row_idx, (model_name, features) in enumerate(models_data):
        print(f"\nCreating visualizations for {model_name}...")
        
        # PCA
        print("  Computing PCA...")
        pca = PCA(n_components=2, random_state=42)
        features_pca = pca.fit_transform(features)
        
        ax = axes[row_idx, 0]
        for event_val in [0, 1]:
            mask = events == event_val
            if np.any(mask):
                ax.scatter(features_pca[mask, 0], features_pca[mask, 1], 
                          c=colors[event_val], label=labels[event_val], alpha=0.7, s=60)
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
        ax.set_title(f'PCA - {model_name}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # t-SNE
        print("  Computing t-SNE...")
        perplexity = min(30, len(features)//4)
        tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
        features_tsne = tsne.fit_transform(features)
        
        ax = axes[row_idx, 1]
        for event_val in [0, 1]:
            mask = events == event_val
            if np.any(mask):
                ax.scatter(features_tsne[mask, 0], features_tsne[mask, 1], 
                          c=colors[event_val], label=labels[event_val], alpha=0.7, s=60)
        ax.set_xlabel('t-SNE 1')
        ax.set_ylabel('t-SNE 2')
        ax.set_title(f't-SNE - {model_name}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # UMAP
        print("  Computing UMAP...")
        n_neighbors = min(15, len(features)//3)
        reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=n_neighbors)
        features_umap = reducer.fit_transform(features)
        
        ax = axes[row_idx, 2]
        for event_val in [0, 1]:
            mask = events == event_val
            if np.any(mask):
                ax.scatter(features_umap[mask, 0], features_umap[mask, 1], 
                          c=colors[event_val], label=labels[event_val], alpha=0.7, s=60)
        ax.set_xlabel('UMAP 1')
        ax.set_ylabel('UMAP 2')
        ax.set_title(f'UMAP - {model_name}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Save
    plt.tight_layout()
    if has_pretrained:
        output_path = os.path.join(output_dir, 'embeddings_comparison_final.png')
    else:
        output_path = os.path.join(output_dir, 'embeddings_local_only_final.png')
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nVisualization saved to: {output_path}")
    
    # Save results
    results_df = pd.DataFrame({
        'patient_id': patient_ids,
        'event': events
    })
    results_path = os.path.join(output_dir, 'embedding_results_final.csv')
    results_df.to_csv(results_path, index=False)
    print(f"Results saved to: {results_path}")

def main():
    # Setup arguments
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
    
    # Load local model
    print("\nLoading local trained model...")
    local_model = load_dinov2_model(args.local_weights)
    local_model = local_model.to(device)
    local_model.eval()
    for param in local_model.parameters():
        param.requires_grad = False
    print("Local model loaded successfully")
    
    # Load pretrained model
    print("\nLoading pretrained model...")
    try:
        pretrained_model = load_pretrained_model()
        pretrained_model = pretrained_model.to(device)
        pretrained_model.eval()
        for param in pretrained_model.parameters():
            param.requires_grad = False
        print("Pretrained model loaded successfully")
        has_pretrained = True
    except Exception as e:
        print(f"Failed to load pretrained model: {e}")
        print("Continuing with local model only...")
        pretrained_model = None
        has_pretrained = False
    
    # Extract features from local model
    features_local, events_local, patient_ids_local = extract_features_unified(
        all_dataloader, local_model, device, "Local Trained Model", is_local=True
    )
    
    # Extract features from pretrained model if available
    if has_pretrained:
        features_pretrained, events_pretrained, patient_ids_pretrained = extract_features_unified(
            all_dataloader, pretrained_model, device, "Pretrained Model", is_local=False
        )
        
        # Verify consistency
        if not np.array_equal(events_local, events_pretrained):
            print("[WARN] Event labels don't match between models")
    else:
        features_pretrained = None
        events_pretrained = events_local
        patient_ids_pretrained = patient_ids_local
    
    # Use local model data as reference
    events = events_local
    patient_ids = patient_ids_local
    
    print(f"\nFinal dataset summary:")
    print(f"Total patients: {len(patient_ids)}")
    print(f"Positive cases: {int(events.sum())}")
    print(f"Negative cases: {int(len(events) - events.sum())}")
    
    # Create visualizations
    print("\nCreating visualizations...")
    create_comparison_visualizations(features_local, features_pretrained, events, patient_ids, args.output_dir)
    
    print("\nVisualization complete!")

if __name__ == "__main__":
    main() 