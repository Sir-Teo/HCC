#!/usr/bin/env python3
"""
Visualize embeddings comparison between local trained model and pretrained model
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
import umap
from tqdm import tqdm
import torch.nn as nn

# Custom module imports
from data.dataset import HCCDataModule
from models.dino import load_dinov2_model

def load_pretrained_dinov2_model():
    """Load pretrained DINOv2 model from torch hub"""
    try:
        # Load pretrained model from torch hub
        model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
        print("Pretrained DINOv2 model loaded successfully")
        return model
    except Exception as e:
        print(f"Error loading pretrained model: {e}")
        # Fallback to local pretrained weights
        pretrained_path = "./dinov2/pretrained_weights/dinov2_vitl14_pretrain.pth"
        if os.path.exists(pretrained_path):
            print(f"Loading from local pretrained weights: {pretrained_path}")
            checkpoint = torch.load(pretrained_path, map_location='cpu')
            model = torch.hub.load('./dinov2', 'dinov2_vitl14', source='local', pretrained=False)
            model.load_state_dict(checkpoint, strict=False)
            return model
        else:
            raise Exception("Cannot load pretrained model")

def extract_all_features(data_loader, model, device, model_name="Model"):
    """
    Extract features using the DINOv2 model for all patients.
    """
    model.eval()
    all_features = []
    all_events = []
    all_patient_ids = []

    if data_loader is None:
        print(f"[WARN] No data loader provided for {model_name}")
        return np.array([]), np.array([]), []

    print(f"Extracting features using {model_name}...")
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc=f"Extracting Features ({model_name})"):
            if len(batch) == 2:
                images, events = batch
            elif len(batch) == 3:
                images, _, events = batch  # Ignore time for binary classification
            else:
                raise ValueError(f"Unexpected batch structure length: {len(batch)}")

            images = images.to(device)
            # Unpack the 5 dimensions (batch_size refers to patients)
            batch_size, num_samples, num_slices, C, H, W = images.size()
            # Combine batch, num_samples, and num_slices dimensions for feature extraction
            images_reshaped = images.view(batch_size * num_samples * num_slices, C, H, W)
            
            # Extract features - handle different model types
            if hasattr(model, 'forward_features'):
                # Local trained model (wrapped)
                feats = model.forward_features(images_reshaped)
            else:
                # Pretrained model from torch hub
                feats = model(images_reshaped)
                
                # Debug: print what we got from the pretrained model
                if isinstance(feats, dict):
                    print(f"[DEBUG] Pretrained model returned dict with keys: {list(feats.keys())}")
                    for key, value in feats.items():
                        if hasattr(value, 'shape'):
                            print(f"[DEBUG] {key}: shape {value.shape}")
                        else:
                            print(f"[DEBUG] {key}: {type(value)}")
                else:
                    print(f"[DEBUG] Pretrained model returned {type(feats)} with shape {feats.shape if hasattr(feats, 'shape') else 'unknown'}")
                
                # Handle different output formats
                if isinstance(feats, dict):
                    # Some models return dict, extract the features
                    if 'x_norm_clstoken' in feats:
                        feats = feats['x_norm_clstoken']
                    elif 'x_norm_patchtokens' in feats:
                        # Use patch tokens and average them
                        patch_tokens = feats['x_norm_patchtokens']  # [batch, patches, embed_dim]
                        feats = patch_tokens.mean(dim=1)  # Average over patches
                    elif 'last_hidden_state' in feats:
                        feats = feats['last_hidden_state'][:, 0]  # Use CLS token
                    else:
                        # Try to find the main feature tensor
                        for key in ['features', 'embeddings', 'hidden_states']:
                            if key in feats:
                                feats = feats[key]
                                break
                        else:
                            # If still dict, try the first tensor value
                            tensor_keys = [k for k, v in feats.items() if hasattr(v, 'shape')]
                            if tensor_keys:
                                feats = feats[tensor_keys[0]]
                                print(f"[DEBUG] Using tensor from key: {tensor_keys[0]}")
                            else:
                                raise ValueError(f"Could not find tensor in dict with keys: {list(feats.keys())}")
                             
                # Handle 3D tensors (batch, sequence, features) - take CLS token or average
                if feats.dim() == 3:
                    feats = feats[:, 0]  # Take CLS token (first token)
                elif feats.dim() == 1:
                    feats = feats.unsqueeze(0)  # Add batch dimension if needed
                
            feature_dim = feats.size(-1)
            # Reshape back: each patient now has num_samples * num_slices feature vectors
            feats = feats.view(batch_size, num_samples * num_slices, feature_dim)
            
            # Average across samples and slices to get one vector per patient
            feats_avg = feats.mean(dim=1)  # Shape: (batch_size, feature_dim)
            
            all_features.append(feats_avg.cpu().numpy())
            all_events.append(events.cpu().numpy())

    if not all_features:
        print(f"[WARN] No features extracted for {model_name}")
        return np.array([]), np.array([]), []

    features = np.concatenate(all_features, axis=0)
    events = np.concatenate(all_events, axis=0)
    
    # Get patient IDs from the dataset
    if hasattr(data_loader.dataset, 'patient_data'):
        patient_ids = [p['patient_id'] for p in data_loader.dataset.patient_data[:len(features)]]
    else:
        patient_ids = [f'patient_{i}' for i in range(len(features))]
    
    print(f"Extracted features for {len(features)} patients using {model_name}")
    print(f"Feature shape: {features.shape}")
    print(f"Positive cases: {int(events.sum())}, Negative cases: {int(len(events) - events.sum())}")
    
    return features, events, patient_ids

def create_visualizations(features_local, features_pretrained, events, patient_ids, output_dir):
    """
    Create PCA, t-SNE, and UMAP visualizations for both models
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Define colors for positive/negative cases
    colors = ['#1f77b4', '#ff7f0e']  # Blue for negative, Orange for positive
    labels = ['Negative (Event=0)', 'Positive (Event=1)']
    
    # Set up the plot style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Check if we have pretrained features
    has_pretrained = features_pretrained.size > 0
    
    if has_pretrained:
        # Create figure with subplots (2 rows x 3 columns)
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Embedding Visualizations: Local Trained vs Pretrained DINOv2 Models', fontsize=16, fontweight='bold')
        
        # Row 1: Local trained model
        # Row 2: Pretrained model
        models_data = [
            ("Local Trained Model", features_local),
            ("Pretrained Model", features_pretrained)
        ]
    else:
        # Create figure with single row (only local model)
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle('Embedding Visualizations: Local Trained DINOv2 Model Only', fontsize=16, fontweight='bold')
        
        models_data = [
            ("Local Trained Model", features_local)
        ]
        # Reshape axes for consistent indexing
        axes = axes.reshape(1, -1)
    
    for row_idx, (model_name, features) in enumerate(models_data):
        print(f"\nCreating visualizations for {model_name}...")
        
        if features.size == 0:
            print(f"[WARN] No features available for {model_name}")
            continue
            
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
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(features)//4))
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
        reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=min(15, len(features)//3))
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
    
    # Adjust layout and save
    plt.tight_layout()
    if has_pretrained:
        output_path = os.path.join(output_dir, 'embeddings_comparison_visualization.png')
    else:
        output_path = os.path.join(output_dir, 'embeddings_local_only_visualization.png')
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nVisualization saved to: {output_path}")
    plt.show()
    
    # Save the processed data for later analysis
    results_df = pd.DataFrame({
        'patient_id': patient_ids,
        'event': events
    })
    
    results_path = os.path.join(output_dir, 'embedding_results.csv')
    results_df.to_csv(results_path, index=False)
    print(f"Results data saved to: {results_path}")

def main():
    parser = argparse.ArgumentParser(description="Visualize embeddings comparison between local and pretrained DINOv2 models")
    
    # Data arguments - same as train_binary.py
    parser.add_argument("--nyu_dicom_root", type=str, default="/gpfs/data/mankowskilab/HCC_Recurrence/dicom",
                        help="Path to the NYU DICOM root directory.")
    parser.add_argument("--nyu_csv_file", type=str, default="/gpfs/data/shenlab/wz1492/HCC/spreadsheets/processed_patient_labels_nyu.csv",
                        help="Path to the NYU CSV metadata file.")
    parser.add_argument('--preprocessed_root', type=str, default='/gpfs/data/mankowskilab/HCC_Recurrence/preprocessed/', 
                        help='Base directory to store/load preprocessed image tensors.')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for data loaders')
    parser.add_argument('--num_slices', type=int, default=32,
                        help='Number of slices per patient sample')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of workers for data loaders')
    parser.add_argument('--num_samples_per_patient', type=int, default=1,
                        help='Number of times to sample slices from each patient series')
    
    # Model arguments
    parser.add_argument('--local_weights', type=str, 
                        default="/gpfs/data/shenlab/wz1492/HCC/dinov2/experiments/eval/training_112499/teacher_checkpoint.pth",
                        help="Path to local trained DINOv2 weights")
    
    # Output argument
    parser.add_argument('--output_dir', type=str, default='./embedding_visualizations_all_patients',
                        help='Directory to save visualization outputs')
    
    args = parser.parse_args()
    
    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Set up data module to load ALL patients (no cross-validation)
    print("Setting up data module for all patients...")
    data_module = HCCDataModule(
        csv_file=args.nyu_csv_file,
        dicom_root=args.nyu_dicom_root,
        model_type="linear",
        batch_size=args.batch_size,
        num_slices=args.num_slices,
        num_samples=args.num_samples_per_patient,
        num_workers=args.num_workers,
        preprocessed_root=args.preprocessed_root,
        cross_validation=False,  # Don't use cross-validation to get all patients
        random_state=42,
        use_validation=False
    )
    
    # Setup the data module (this will load all patients)
    data_module.setup()
    
    # Create a single dataloader with all patients
    from torch.utils.data import DataLoader
    from data.dataset import HCCDicomDataset
    
    # Create dataset with all patients
    all_dataset = HCCDicomDataset(
        patient_data_list=data_module.all_patients,
        model_type="linear",
        transform=data_module.transform,
        num_slices=args.num_slices,
        num_samples=args.num_samples_per_patient,
        preprocessed_root=args.preprocessed_root
    )
    
    # Create dataloader
    all_dataloader = DataLoader(
        all_dataset,
        batch_size=args.batch_size,
        shuffle=False,  # Don't shuffle for consistent ordering
        num_workers=args.num_workers,
        collate_fn=data_module.collate_fn
    )
    
    print(f"Total patients loaded: {len(all_dataset)}")
    
    # Load models
    print("\nLoading models...")
    
    # Load local trained model
    print("Loading local trained model...")
    try:
        local_model = load_dinov2_model(args.local_weights)
        local_model = local_model.to(device)
        local_model.eval()
        print("Local trained model loaded successfully")
    except Exception as e:
        print(f"Error loading local model: {e}")
        raise
    
    # Load pretrained model
    print("Loading pretrained model...")
    try:
        pretrained_model = load_pretrained_dinov2_model()
        pretrained_model = pretrained_model.to(device)
        pretrained_model.eval()
        print("Pretrained model loaded successfully")
    except Exception as e:
        print(f"Error loading pretrained model: {e}")
        print("Continuing with local model only...")
        pretrained_model = None
    
    # Prevent gradient computation
    for model in [local_model] + ([pretrained_model] if pretrained_model else []):
        for param in model.parameters():
            param.requires_grad = False
    
    print("Models loaded, starting feature extraction...")
    
    # Extract features from both models
    features_local, events_local, patient_ids_local = extract_all_features(
        all_dataloader, local_model, device, "Local Trained Model"
    )
    
    if pretrained_model is not None:
        features_pretrained, events_pretrained, patient_ids_pretrained = extract_all_features(
            all_dataloader, pretrained_model, device, "Pretrained Model"
        )
    else:
        print("Skipping pretrained model feature extraction due to loading error")
        features_pretrained = np.array([])
        events_pretrained = events_local
        patient_ids_pretrained = patient_ids_local
    
    print("Feature extraction completed!")
    
    # Verify consistency
    if len(events_local) != len(events_pretrained) and pretrained_model is not None:
        print(f"[WARN] Different number of samples: Local={len(events_local)}, Pretrained={len(events_pretrained)}")
    
    if pretrained_model is not None and not np.array_equal(events_local, events_pretrained):
        print(f"[WARN] Event labels don't match between models")
    
    # Use the local model's events and patient IDs as reference
    events = events_local
    patient_ids = patient_ids_local
    
    print(f"\nFinal dataset for visualization:")
    print(f"Total patients: {len(patient_ids)}")
    print(f"Positive cases: {int(events.sum())}")
    print(f"Negative cases: {int(len(events) - events.sum())}")
    
    # Create visualizations
    print("\nCreating visualizations...")
    create_visualizations(features_local, features_pretrained, events, patient_ids, args.output_dir)
    
    print("\nVisualization complete!")

if __name__ == "__main__":
    main() 