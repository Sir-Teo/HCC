#!/usr/bin/env python

import argparse
import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import umap
import datetime
from scipy import ndimage
from skimage import exposure

# Custom module imports (adjust your PYTHONPATH accordingly)
from data.dataset import HCCDataModule
from models.dino import load_dinov2_model

# Outlier patient IDs that need special preprocessing
OUTLIER_PATIENTS = {
    'TCGA-DD-A1EH', '16946681', '14875250', '25170129', '24849898'
}

class OutlierPreprocessor:
    """Simple but effective preprocessing for outlier patients."""
    
    @staticmethod
    def normalize_outlier_image(image):
        """Apply domain adaptation preprocessing for outlier patients."""
        # Step 1: Robust percentile normalization
        p1, p99 = np.percentile(image, [1, 99])
        if p99 > p1:
            image = np.clip((image - p1) / (p99 - p1), 0, 1)
        else:
            image = np.clip(image / (image.max() + 1e-8), 0, 1)
        
        # Step 2: Adaptive histogram equalization for better contrast
        image = exposure.equalize_adapthist(image, clip_limit=0.02)
        
        # Step 3: Domain adaptation - normalize to consistent stats
        current_mean = np.mean(image)
        current_std = np.std(image)
        target_mean, target_std = 0.5, 0.2
        
        if current_std > 1e-8:
            image = (image - current_mean) / current_std
            image = image * target_std + target_mean
            image = np.clip(image, 0, 1)
        
        return image.astype(np.float32)

def extract_features(data_loader, model, device, output_dir='debug_images'):
    """
    Extract features from the data using DINOv2 with improved preprocessing for outliers.
    """
    import torchvision.utils as vutils

    model.eval()
    features = []
    image_save_dir = None
    if output_dir:
        image_save_dir = os.path.join(output_dir, "debug_images")
        os.makedirs(image_save_dir, exist_ok=True)

        # Also create improved images directory
        improved_dir = os.path.join(output_dir, "improved_debug_images")
        os.makedirs(improved_dir, exist_ok=True)

    outlier_preprocessor = OutlierPreprocessor()

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(data_loader, desc="Extracting Features")):
            if len(batch) == 2:
                images, _, = batch
                patient_ids = [f"unknown_{batch_idx}_{i}" for i in range(images.shape[0])]
            else:
                images, _, patient_data = batch
                # Extract patient IDs from patient data
                patient_ids = []
                for i, patient in enumerate(patient_data):
                    if hasattr(patient, 'patient_id'):
                        patient_ids.append(str(patient.patient_id))
                    elif hasattr(patient, 'Pre op MRI Accession number'):
                        patient_ids.append(str(patient['Pre op MRI Accession number']))
                    else:
                        patient_ids.append(f"unknown_{batch_idx}_{i}")

            images = images.to(device)
            batch_size, num_samples, num_slices, C, H, W = images.size()

            # Apply improved preprocessing for outlier patients
            processed_images = []
            for b in range(batch_size):
                patient_id = patient_ids[b] if b < len(patient_ids) else f"unknown_{b}"
                is_outlier = any(outlier in str(patient_id) for outlier in OUTLIER_PATIENTS)
                
                patient_samples = []
                for s in range(num_samples):
                    sample_slices = []
                    for sl in range(num_slices):
                        slice_tensor = images[b, s, sl]  # shape: (C, H, W)
                        
                        if is_outlier:
                            # Apply improved preprocessing for outliers
                            if slice_tensor.shape[0] == 1:  # Grayscale
                                slice_np = slice_tensor.squeeze(0).cpu().numpy()
                            else:
                                slice_np = torch.mean(slice_tensor, dim=0).cpu().numpy()
                            
                            # Apply outlier preprocessing
                            processed_np = outlier_preprocessor.normalize_outlier_image(slice_np)
                            
                            # Convert back to tensor
                            if slice_tensor.shape[0] == 1:
                                processed_tensor = torch.from_numpy(processed_np).unsqueeze(0).to(device)
                            else:
                                # Convert grayscale back to 3-channel if needed
                                processed_tensor = torch.from_numpy(processed_np).unsqueeze(0).repeat(3, 1, 1).to(device)
                        else:
                            # Keep original for non-outliers
                            processed_tensor = slice_tensor
                        
                        sample_slices.append(processed_tensor)
                    
                    sample_tensor = torch.stack(sample_slices, dim=0)
                    patient_samples.append(sample_tensor)
                
                patient_tensor = torch.stack(patient_samples, dim=0)
                processed_images.append(patient_tensor)

            processed_batch = torch.stack(processed_images, dim=0)

            # Save sample images for comparison
            if image_save_dir and batch_idx < 5:
                # Original images
                sample_tensor = images[0, 0]  # First patient, first sample
                num_display = min(5, sample_tensor.shape[0])
                for i in range(num_display):
                    slice_tensor = sample_tensor[i]  # shape: (C, H, W)
                    if slice_tensor.shape[0] == 1:  # Grayscale  
                        img = slice_tensor.squeeze(0).cpu().numpy()
                        plt.imshow(img, cmap="gray")
                    else:
                        img = slice_tensor.permute(1, 2, 0).cpu().numpy()
                        plt.imshow(img)

                    patient_id = patient_ids[0] if len(patient_ids) > 0 else "unknown"
                    is_outlier = any(outlier in str(patient_id) for outlier in OUTLIER_PATIENTS)
                    title_suffix = " (OUTLIER)" if is_outlier else ""
                    
                    plt.axis("off")
                    plt.title(f"Original: {patient_id} - Slice {i}{title_suffix}")
                    plt.savefig(os.path.join(image_save_dir, f"original_batch{batch_idx}_slice{i}.png"))
                    plt.close()

                # Processed images
                processed_sample = processed_batch[0, 0]  # First patient, first sample
                for i in range(num_display):
                    slice_tensor = processed_sample[i]  # shape: (C, H, W)
                    if slice_tensor.shape[0] == 1:  # Grayscale  
                        img = slice_tensor.squeeze(0).cpu().numpy()
                        plt.imshow(img, cmap="gray")
                    else:
                        img = slice_tensor.permute(1, 2, 0).cpu().numpy()
                        plt.imshow(img)

                    plt.axis("off")
                    plt.title(f"Processed: {patient_id} - Slice {i}{title_suffix}")
                    plt.savefig(os.path.join(improved_dir, f"processed_batch{batch_idx}_slice{i}.png"))
                    plt.close()

            # Reshape for feature extraction
            batch_size, num_samples, num_slices, C, H, W = processed_batch.size()
            processed_batch = processed_batch.view(batch_size * num_samples * num_slices, C, H, W)
            feats = model.forward_features(processed_batch)
            feature_dim = feats.size(-1)
            feats = feats.view(batch_size, num_samples, num_slices, feature_dim)
            features.append(feats.cpu().numpy())

    features = np.concatenate(features, axis=0)
    return features

def dimensionality_reduction(features, patient_ids, output_dir, n_components=2):
    """Apply multiple dimensionality reduction techniques and save results."""
    results = {}
    
    # Reshape features for dimensionality reduction
    n_patients, n_samples, n_slices, feature_dim = features.shape
    features_flat = features.reshape(n_patients * n_samples * n_slices, feature_dim)
    
    # Create patient ID labels for each slice
    patient_labels = []
    for i, pid in enumerate(patient_ids):
        for _ in range(n_samples * n_slices):
            patient_labels.append(pid)
    
    # Standardize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features_flat)
    
    # PCA
    print("Applying PCA...")
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(features_scaled)
    results['PCA'] = {
        'embedding': pca_result,
        'explained_variance_ratio': pca.explained_variance_ratio_,
        'outliers': find_outliers(pca_result, patient_labels, method='PCA')
    }
    
    # t-SNE
    print("Applying t-SNE...")
    tsne = TSNE(n_components=n_components, random_state=42, perplexity=30)
    tsne_result = tsne.fit_transform(features_scaled)
    results['t-SNE'] = {
        'embedding': tsne_result,
        'outliers': find_outliers(tsne_result, patient_labels, method='t-SNE')
    }
    
    # UMAP
    print("Applying UMAP...")
    umap_reducer = umap.UMAP(n_components=n_components, random_state=42)
    umap_result = umap_reducer.fit_transform(features_scaled)
    results['UMAP'] = {
        'embedding': umap_result,
        'outliers': find_outliers(umap_result, patient_labels, method='UMAP')
    }
    
    return results, patient_labels

def find_outliers(embedding, patient_labels, method='', top_k=10):
    """Find outliers based on distance from centroid."""
    centroid = np.mean(embedding, axis=0)
    distances = np.sqrt(np.sum((embedding - centroid) ** 2, axis=1))
    
    outlier_indices = np.argsort(distances)[-top_k:]
    outliers = []
    
    for idx in outlier_indices:
        patient_id = patient_labels[idx]
        distance = distances[idx]
        outliers.append((patient_id, distance, idx))
    
    # Print outliers
    print(f"\n{method} - Top {top_k} Outliers:")
    for i, (patient_id, distance, idx) in enumerate(outliers):
        is_known_outlier = any(outlier in str(patient_id) for outlier in OUTLIER_PATIENTS)
        marker = "🎯" if is_known_outlier else "  "
        print(f"{marker} {i+1:2d}. {patient_id:<15} (Distance: {distance:.2f})")
    
    return outliers

def visualize_embeddings(results, patient_labels, output_dir):
    """Create visualization plots for all dimensionality reduction methods."""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    methods = ['PCA', 't-SNE', 'UMAP']
    
    for i, method in enumerate(methods):
        embedding = results[method]['embedding']
        outliers = results[method]['outliers']
        
        # Get unique patient IDs for coloring
        unique_patients = list(set(patient_labels))
        colors = plt.cm.tab20(np.linspace(0, 1, len(unique_patients)))
        
        # Create scatter plot
        for j, patient in enumerate(unique_patients):
            mask = [patient in label for label in patient_labels]
            if np.any(mask):
                is_outlier = any(outlier in str(patient) for outlier in OUTLIER_PATIENTS)
                marker = 'X' if is_outlier else 'o'
                size = 100 if is_outlier else 20
                alpha = 0.8 if is_outlier else 0.6
                
                axes[i].scatter(embedding[mask, 0], embedding[mask, 1], 
                              c=[colors[j]], label=patient, 
                              marker=marker, s=size, alpha=alpha)
        
        # Highlight top outliers
        outlier_patients = set([outlier[0] for outlier in outliers[:5]])
        for patient in outlier_patients:
            mask = [patient in label for label in patient_labels]
            if np.any(mask):
                axes[i].scatter(embedding[mask, 0], embedding[mask, 1], 
                              s=200, facecolors='none', edgecolors='red', 
                              linewidth=3, marker='o')
        
        axes[i].set_title(f'{method} Visualization (Improved Preprocessing)')
        axes[i].set_xlabel(f'{method} 1')
        axes[i].set_ylabel(f'{method} 2')
        
        # Add explained variance for PCA
        if method == 'PCA':
            var_ratio = results[method]['explained_variance_ratio']
            axes[i].set_xlabel(f'{method} 1 ({var_ratio[0]:.1%} variance)')
            axes[i].set_ylabel(f'{method} 2 ({var_ratio[1]:.1%} variance)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'dimensionality_reduction_improved.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    # Save detailed results
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(os.path.join(output_dir, f'outlier_analysis_improved_{timestamp}.txt'), 'w') as f:
        f.write("IMPROVED PREPROCESSING OUTLIER ANALYSIS\n")
        f.write("=" * 50 + "\n\n")
        
        for method in methods:
            f.write(f"{method} OUTLIERS:\n")
            f.write("-" * 20 + "\n")
            for i, (patient_id, distance, idx) in enumerate(results[method]['outliers']):
                is_known_outlier = any(outlier in str(patient_id) for outlier in OUTLIER_PATIENTS)
                marker = "🎯" if is_known_outlier else "  "
                f.write(f"{marker} {i+1:2d}. {patient_id:<15} (Distance: {distance:.2f})\n")
            f.write("\n")

# -----------------------------------------------------------
# Utility to plot embeddings with dataset source + event label
# (adapted from the original visualize.py)
# -----------------------------------------------------------
def plot_embedding(embedding, title, filename, sources, event_labels, patient_ids, output_dir, num_labels=10):
    """Scatter-plot 2-D embedding with colour = source (train/test) and marker = event (+/-)."""
    import seaborn as sns
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 10))

    ax = sns.scatterplot(
        x=embedding[:, 0],
        y=embedding[:, 1],
        hue=sources,
        style=event_labels,
        palette="Set2",
        s=100,
        alpha=0.75,
        edgecolor="k",
    )

    plt.title(title)
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.legend(title="Source / Event", bbox_to_anchor=(1.05, 1), loc="upper left")

    # Highlight the furthest points from the centroid
    if num_labels > 0 and len(patient_ids) == len(embedding):
        center = embedding.mean(axis=0)
        distances = np.linalg.norm(embedding - center, axis=1)
        outlier_idx = np.argsort(distances)[-num_labels:]

        print(f"\n--- Top {num_labels} Outliers for {title} ---")
        for idx in outlier_idx:
            pid = patient_ids[idx]
            dist = distances[idx]
            print(f"  • {pid:<15}  Dist: {dist:.2f}  ({sources[idx]}, {event_labels[idx]})")
            plt.text(
                embedding[idx, 0] + 0.02,
                embedding[idx, 1] + 0.02,
                pid,
                fontsize=9,
            )

    plt.tight_layout(rect=(0, 0, 0.85, 1))
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, filename)
    plt.savefig(plot_path, dpi=300)
    print(f"[PLOT] {title} saved → {plot_path}")
    plt.close()

def main():
    """Main function to run the visualization pipeline with improved preprocessing."""
    
    # Parse arguments
    parser = argparse.ArgumentParser(description='Visualize HCC dataset embeddings with improved preprocessing')
    parser.add_argument("--train_dicom_root", type=str, default="/gpfs/data/mankowskilab/HCC/data/TCGA/manifest-4lZjKqlp5793425118292424834/TCGA-LIHC",
                         help="Path to the training DICOM directory.")
    parser.add_argument("--test_dicom_root", type=str, default="/gpfs/data/mankowskilab/HCC_Recurrence/dicom",
                         help="Path to the testing DICOM directory.")
    parser.add_argument("--train_csv_file", type=str, default="/gpfs/data/shenlab/wz1492/HCC/spreadsheets/tcga.csv",
                         help="Path to the training CSV file.")
    parser.add_argument("--test_csv_file", type=str, default="/gpfs/data/shenlab/wz1492/HCC/spreadsheets/processed_patient_labels_nyu.csv",
                         help="Path to the testing CSV file.")
    parser.add_argument("--dinov2_weights", type=str, default="/gpfs/data/shenlab/wz1492/HCC/dinov2/experiments/eval/training_99999/teacher_checkpoint.pth",
                        help="Path to your local DINOv2 state dict file (.pth or .pt).")
    parser.add_argument('--preprocessed_root', type=str, default='/gpfs/data/mankowskilab/HCC_Recurrence/preprocessed/', 
                        help="Directory to store/load preprocessed image tensors")
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size for data loading')
    parser.add_argument('--num_slices', type=int, default=64, help="Number of slices per patient")
    parser.add_argument('--num_samples_per_patient', type=int, default=1, help="Number of times to sample from each patient per epoch")
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading')
    parser.add_argument('--output_dir', type=str, default='visualization_outputs_improved', help='Output directory')
    
    args = parser.parse_args()
    
    # Create timestamped output directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"run_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    print("🚀 Starting HCC Dataset Visualization with Improved Preprocessing")
    print(f"📁 Output directory: {output_dir}")
    print(f"🎯 Targeting outliers: {OUTLIER_PATIENTS}")
    
    # Initialize data module
    print("\n📊 Loading data...")
    data_module = HCCDataModule(
        train_csv_file=args.train_csv_file,
        test_csv_file=args.test_csv_file,
        train_dicom_root=args.train_dicom_root,
        test_dicom_root=args.test_dicom_root,
        model_type="time_to_event",
        batch_size=args.batch_size,
        num_slices=args.num_slices,
        num_samples=args.num_samples_per_patient,
        num_workers=args.num_workers,
        preprocessed_root=args.preprocessed_root,
        use_validation=False,
    )
    data_module.setup()

    # -------------------------------------------------------------------
    # NEW: Extract features for TRAIN _and_ TEST to allow source/event labels
    # -------------------------------------------------------------------

    print("\n🤖 Loading DINOv2 model…")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_dinov2_model(args.dinov2_weights).to(device)

    # Extract features for each split
    print("🔍 Extracting TRAIN features…")
    train_features = extract_features(data_module.train_dataloader(), model, device, output_dir)
    print("🔍 Extracting TEST features…")
    test_features  = extract_features(data_module.test_dataloader(),  model, device, output_dir)

    # Collapse sample + slice dims → per-patient embedding
    train_emb = train_features.mean(axis=(1, 2))  # (N_train, feat_dim)
    test_emb  = test_features.mean(axis=(1, 2))   # (N_test,  feat_dim)

    # Build metadata arrays
    train_patient_data = data_module.train_dataset.patient_data
    test_patient_data  = data_module.test_dataset.patient_data

    sources = np.array(["train"] * len(train_emb) + ["test"] * len(test_emb))

    events_train = np.array([p.get("event", p.get("label", 0)) for p in train_patient_data])
    events_test  = np.array([p.get("event", p.get("label", 0)) for p in test_patient_data])
    events = np.concatenate([events_train, events_test])
    event_labels = np.where(events == 1, "positive", "negative")

    patient_ids = np.array([p.get("patient_id", "unknown") for p in train_patient_data] +
                           [p.get("patient_id", "unknown") for p in test_patient_data])

    # Combine embeddings
    embeddings = np.vstack([train_emb, test_emb])

    # Standardise then reduce dimensions
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings)

    print("\n📈 Running PCA/t-SNE/UMAP on patient-level embeddings…")

    # PCA
    pca = PCA(n_components=2)
    emb_pca = pca.fit_transform(embeddings_scaled)
    plot_embedding(emb_pca,
                   title=f"PCA – DINOv2 features (var: {pca.explained_variance_ratio_[0]:.1%}/{pca.explained_variance_ratio_[1]:.1%})",
                   filename="pca_patient_level.png",
                   sources=sources,
                   event_labels=event_labels,
                   patient_ids=patient_ids,
                   output_dir=output_dir)

    # t-SNE
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    emb_tsne = tsne.fit_transform(embeddings_scaled)
    plot_embedding(emb_tsne,
                   title="t-SNE – DINOv2 features",
                   filename="tsne_patient_level.png",
                   sources=sources,
                   event_labels=event_labels,
                   patient_ids=patient_ids,
                   output_dir=output_dir)

    # UMAP
    umap_model = umap.UMAP(n_components=2, random_state=42)
    emb_umap = umap_model.fit_transform(embeddings_scaled)
    plot_embedding(emb_umap,
                   title="UMAP – DINOv2 features",
                   filename="umap_patient_level.png",
                   sources=sources,
                   event_labels=event_labels,
                   patient_ids=patient_ids,
                   output_dir=output_dir)

    print(f"\n✅ Analysis complete! Patient-level plots saved to → {output_dir}")
    print("   • Colour = dataset split (train vs test)  |  Marker = positive / negative event")

if __name__ == '__main__':
    main() 