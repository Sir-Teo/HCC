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
    if args.test_csv_file:
        df_test = pd.read_csv(args.test_csv_file)
        df_test['dicom_root'] = args.test_dicom_root
        df_test['source'] = 'test'
    else:
        # If no test CSV provided, perform a train-test split on the training data
        from sklearn.model_selection import train_test_split
        # For example, use an 80-20 split
        df_train, df_test = train_test_split(df_train, test_size=0.2, random_state=42)
        # For the split test set, use the same dicom root as training (or adjust if needed)
        df_test['dicom_root'] = args.train_dicom_root
        df_test['source'] = 'test'

    # Initialize the data module with separate CSVs for train and test
    data_module = HCCDataModule(
        train_csv_file=df_train,
        test_csv_file=df_test,
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

    # Optionally standardize embeddings before PCA
    
    embeddings_2d = embeddings.mean(axis=1)  # Collapse the slice dimension by averaging
    print(len(embeddings_2d), len(embeddings_2d[0]))
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings_2d)

    # Perform PCA to reduce to 2 dimensions
    pca = PCA(n_components=2)
    embeddings_pca = pca.fit_transform(embeddings_scaled)
    print(f"PCA explained variance ratios: {pca.explained_variance_ratio_}")

    # Create a scatter plot with points colored by dataset source
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=embeddings_pca[:, 0], y=embeddings_pca[:, 1],
                    hue=sources, palette="viridis", s=100, alpha=0.7)
    plt.title("PCA of DINOv2 Embeddings (per patient)")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend(title="Source")
    plot_file = os.path.join(args.output_dir, "dino_embeddings_pca.png")
    plt.savefig(plot_file)
    print(f"PCA plot saved to {plot_file}")
    plt.show()

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
    parser.add_argument('--preprocessed_root', type=str, default='', 
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
