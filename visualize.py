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
from sklearn.preprocessing import StandardScaler
import datetime

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap.umap_ as umap

def plot_embedding(embedding, title, filename, sources, event_labels, patient_ids, output_dir, num_labels=5):
    plt.figure(figsize=(12, 10)) # Increased figure size slightly for labels
    ax = sns.scatterplot(
        x=embedding[:, 0],
        y=embedding[:, 1],
        hue=sources,
        style=event_labels,
        palette="Set2",  # Better color palette
        s=100,
        alpha=0.75,
        edgecolor="k"
    )
    plt.title(title)
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.legend(title="Source / Event", bbox_to_anchor=(1.05, 1), loc='upper left')

    # --- Add labels for outliers ---
    if num_labels > 0 and len(patient_ids) == len(embedding):
        # Calculate center
        center = embedding.mean(axis=0)
        # Calculate distance from center
        distances = np.linalg.norm(embedding - center, axis=1)
        # Get indices of N furthest points
        outlier_indices = np.argsort(distances)[-num_labels:]

        print(f"\n--- Top {num_labels} Outliers for {title} ---")
        # Add text labels and print info
        for i in outlier_indices:
            patient_id = patient_ids[i]
            distance = distances[i]
            print(f"  - Patient ID: {patient_id:<15} Distance: {distance:.4f}")
            plt.text(
                embedding[i, 0] + 0.01, # Slight offset for x
                embedding[i, 1] + 0.01, # Slight offset for y
                patient_id,
                fontsize=9,
                ha='left', # Horizontal alignment
                va='bottom' # Vertical alignment
            )
    # --- End of label addition ---

    plt.tight_layout(rect=(0, 0, 0.85, 1)) # FIX: Use tuple for rect
    plot_path = os.path.join(output_dir, filename)
    plt.savefig(plot_path)
    print(f"{title} saved to {plot_path}")
    plt.show()

# Custom module imports (adjust your PYTHONPATH accordingly)
from data.dataset import HCCDataModule
from models.dino import load_dinov2_model

def extract_features(data_loader, model, device, output_dir='debug_images'):
    """
    Extract features using the DINOv2 model.
    Each slice will have its own feature vector (no averaging on the slice level).
    Saves a few sample images per batch to disk for inspection.
    Returns an array of shape (num_patients, num_samples, num_slices, feature_dim).
    """
    import torchvision.utils as vutils

    model.eval()
    features = []
    image_save_dir = None
    if output_dir:
        image_save_dir = os.path.join(output_dir, "debug_images")
        os.makedirs(image_save_dir, exist_ok=True)

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(data_loader, desc="Extracting Features")):
            images, _, _ = batch  # Only need the images
            images = images.to(device)
            batch_size, num_samples, num_slices, C, H, W = images.size()

            # Save up to 5 slices of the first image in the batch
            if image_save_dir and batch_idx < 5:
                sample_tensor = images[0, 0]  # shape: (num_slices, C, H, W)
                num_display = min(5, sample_tensor.shape[0])
                for i in range(num_display):
                    slice_tensor = sample_tensor[i]  # shape: (C, H, W)
                    # Unnormalize (assuming ImageNet normalization — change if using other stats)
                    mean = torch.tensor([0.485, 0.456, 0.406], device=slice_tensor.device).view(3, 1, 1)
                    std = torch.tensor([0.229, 0.224, 0.225], device=slice_tensor.device).view(3, 1, 1)

                    if slice_tensor.shape[0] == 1:  # Grayscale
                        img = slice_tensor.squeeze(0).cpu().numpy()
                        plt.imshow(img, cmap="gray")
                    else:
                        unnorm = slice_tensor * std + mean
                        img = unnorm.permute(1, 2, 0).cpu().numpy()
                        img = np.clip(img, 0, 1)
                        plt.imshow(img)

                    plt.axis("off")
                    plt.title(f"Batch {batch_idx} - Slice {i}")
                    plt.savefig(os.path.join(image_save_dir, f"batch{batch_idx}_slice{i}.png"))
                    plt.close()

            # Reshape for feature extraction
            images = images.view(batch_size * num_samples * num_slices, C, H, W)
            feats = model.forward_features(images)
            feature_dim = feats.size(-1)
            feats = feats.view(batch_size, num_samples, num_slices, feature_dim)
            features.append(feats.cpu().numpy())

    features = np.concatenate(features, axis=0)
    return features


def main(args):
    # Create the output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Load CSV file for training and add additional columns
    df_train = pd.read_csv(args.train_csv_file)
    df_train['dicom_root'] = args.train_dicom_root
    df_train['source'] = 'train'

    # Check if a test CSV file is provided
    train_csv_path_for_module = args.train_csv_file
    test_csv_path_for_module = args.test_csv_file

    if args.test_csv_file:
        df_test = pd.read_csv(args.test_csv_file)
        df_test['dicom_root'] = args.test_dicom_root
        df_test['source'] = 'test'
        # df_all = pd.concat([df_train, df_test], ignore_index=True) # Keep for now, maybe needed later
    else:
        print("[INFO] No test CSV provided. Performing train-test split on training data.")
        from sklearn.model_selection import train_test_split
        # We split the original df_train before adding dicom_root/source cols
        df_train_orig = pd.read_csv(args.train_csv_file)
        df_train_split, df_test_split = train_test_split(df_train_orig, test_size=0.2, random_state=42)

        # Define paths for temporary CSVs within the output directory
        temp_train_csv = os.path.join(args.output_dir, "temp_train_split.csv")
        temp_test_csv = os.path.join(args.output_dir, "temp_test_split.csv")

        # Save the splits to temporary files
        df_train_split.to_csv(temp_train_csv, index=False)
        df_test_split.to_csv(temp_test_csv, index=False)
        print(f"[INFO] Saved temporary train split to {temp_train_csv}")
        print(f"[INFO] Saved temporary test split to {temp_test_csv}")

        # Update the paths to be passed to the data module
        train_csv_path_for_module = temp_train_csv
        test_csv_path_for_module = temp_test_csv

        # Re-create df_train and df_test for feature extraction/plotting logic below
        # These dataframes need the 'dicom_root' and 'source' columns
        df_train = df_train_split.copy()
        df_train['dicom_root'] = args.train_dicom_root
        df_train['source'] = 'train'

        df_test = df_test_split.copy()
        # For the split test set, use the same dicom root as training by default
        df_test['dicom_root'] = args.train_dicom_root
        df_test['source'] = 'test'

    # Initialize the data module with CSV file PATHS
    data_module = HCCDataModule(
        train_csv_file=train_csv_path_for_module,
        test_csv_file=test_csv_path_for_module,
        train_dicom_root=args.train_dicom_root,
        test_dicom_root=args.test_dicom_root if args.test_csv_file else args.train_dicom_root, # Use train root if split
        model_type="time_to_event",
        batch_size=args.batch_size,
        num_slices=args.num_slices,
        num_samples=args.num_samples_per_patient,
        num_workers=args.num_workers,
        preprocessed_root=args.preprocessed_root,
        use_validation=False,
    )
    data_module.setup()

    # Load the DINOv2 model and move it to the appropriate device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dino_model = load_dinov2_model(args.dinov2_weights)
    dino_model = dino_model.to(device)
    dino_model.eval()

    # Extract features for both datasets using the same extraction logic
    print("Extracting features for training dataset...")
    train_features = extract_features(data_module.train_dataloader(), dino_model, device)
    print("Extracting features for test dataset...")
    test_features = extract_features(data_module.test_dataloader(), dino_model, device)

    # Average across samples (for each patient) to get one embedding per patient
    train_embeddings = train_features.mean(axis=1)  # shape: (num_train_patients, feature_dim)
    test_embeddings = test_features.mean(axis=1)    # shape: (num_test_patients, feature_dim)

    # Combine embeddings and create corresponding source labels
    embeddings = np.concatenate([train_embeddings, test_embeddings], axis=0)
    sources = np.array(['train'] * train_embeddings.shape[0] + ['test'] * test_embeddings.shape[0])
    
    # Instead of using the CSV directly, extract event labels from the dataset's filtered patient_data.
    # For model_type "linear", the label is stored under 'label';
    # for "time_to_event", it is stored under 'event'.
    train_events = np.array([
        patient.get('event', patient.get('label')) 
        for patient in data_module.train_dataset.patient_data
    ])
    test_events = np.array([
        patient.get('event', patient.get('label')) 
        for patient in data_module.test_dataset.patient_data
    ])
    events = np.concatenate([train_events, test_events], axis=0)
    
    # Convert numeric events (0/1) into string labels for better legend readability
    event_labels = np.array(['positive' if e == 1 else 'negative' for e in events])
    
    # Extract patient IDs
    train_ids = np.array([p['patient_id'] for p in data_module.train_dataset.patient_data])
    test_ids = np.array([p['patient_id'] for p in data_module.test_dataset.patient_data])
    patient_ids = np.concatenate([train_ids, test_ids], axis=0)

    # Optionally standardize embeddings before PCA
    embeddings_2d = embeddings.mean(axis=1)  # Collapse the slice dimension by averaging
    print(len(embeddings_2d), len(embeddings_2d[0]))
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings_2d)

    # PCA
    pca = PCA(n_components=2)
    embeddings_pca = pca.fit_transform(embeddings_scaled)
    print(f"PCA explained variance ratios: {pca.explained_variance_ratio_}")
    plot_embedding(embeddings_pca, "PCA of DINOv2 Embeddings", "dino_embeddings_pca.png", sources, event_labels, patient_ids, args.output_dir)

    # t-SNE
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    embeddings_tsne = tsne.fit_transform(embeddings_scaled)
    plot_embedding(embeddings_tsne, "t-SNE of DINOv2 Embeddings", "dino_embeddings_tsne.png", sources, event_labels, patient_ids, args.output_dir)

    # UMAP
    umap_model = umap.UMAP(n_components=2, random_state=42)
    embeddings_umap = umap_model.fit_transform(embeddings_scaled)
    plot_embedding(embeddings_umap, "UMAP of DINOv2 Embeddings", "dino_embeddings_umap.png", sources, event_labels, patient_ids, args.output_dir)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize DINOv2 Embeddings with PCA")
    parser.add_argument("--train_dicom_root", type=str, default="/gpfs/data/mankowskilab/HCC/data/TCGA/manifest-4lZjKqlp5793425118292424834/TCGA-LIHC",
                         help="Path to the training DICOM directory.")
    parser.add_argument("--test_dicom_root", type=str, default="/gpfs/data/mankowskilab/HCC_Recurrence/dicom",
                         help="Path to the testing DICOM directory.")
    parser.add_argument("--train_csv_file", type=str, default="/gpfs/data/shenlab/wz1492/HCC/spreadsheets/tcga.csv",
                         help="Path to the training CSV file.")
    parser.add_argument("--test_csv_file", type=str, default="/gpfs/data/shenlab/wz1492/HCC/spreadsheets/processed_patient_labels_nyu.csv",
                         help="Path to the testing CSV file. If not provided, a train-test split will be performed on the training dataset.")
    parser.add_argument("--dinov2_weights", type=str, default="/gpfs/data/shenlab/wz1492/HCC/dinov2/experiments/eval/training_99999/teacher_checkpoint.pth",
                        help="Path to your local DINOv2 state dict file (.pth or .pt).")
    parser.add_argument('--preprocessed_root', type=str, default='/gpfs/data/mankowskilab/HCC_Recurrence/preprocessed/', 
                        help="Directory to store/load preprocessed image tensors")
    parser.add_argument('--batch_size', type=int, default=32,
                        help="Batch size for data loaders")
    parser.add_argument('--num_slices', type=int, default=64,
                        help="Number of slices per patient")
    parser.add_argument('--num_workers', type=int, default=4,
                        help="Number of workers for data loaders")
    parser.add_argument('--num_samples_per_patient', type=int, default=1,
                        help="Number of times to sample from each patient per epoch")
    parser.add_argument('--output_dir', type=str, default='visualization_outputs',
                        help="Directory to save outputs and plots")
    args = parser.parse_args()

    # Create a unique subdirectory for this run using a timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    args.output_dir = os.path.join(args.output_dir, f"run_{timestamp}")
    os.makedirs(args.output_dir, exist_ok=True)

    sns.set(style="whitegrid")
    main(args)
