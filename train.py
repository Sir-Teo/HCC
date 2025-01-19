#!/usr/bin/env python

import os
import argparse
from typing import List, Dict, Tuple
import pandas as pd
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl

# For reading DICOM (you can use pydicom or a specialized library)
import pydicom

# PyTorch transforms
import torchvision.transforms as T
from torchvision import models

# Example: Using DINOV2 (Hugging Face, or from a local checkpoint)
# If you have a local checkpoint or huggingface:
#   pip install transformers timm
#   from transformers import Dinov2Model, Dinov2Config
# 
# We'll do a placeholder import so that you can fill in the actual DINOV2 code
try:
    import timm
except ImportError:
    print("Please install timm or the relevant library to load DINOv2.")


# For survival models
from pycox.models import LogisticHazard
from pycox.evaluation import EvalSurv
import torchtuples as tt


# 1) PARSE ARGUMENTS
def parse_args():
    parser = argparse.ArgumentParser(description="Training script for Time-to-Event with PyTorch Lightning and pycox.")
    parser.add_argument("--dicom_root", type=str, default="/gpfs/data/mankowskilab/HCC_Recurrence/dicom",
                        help="Root directory of dicom images.")
    parser.add_argument("--csv_path", type=str, required=True, 
                        help="Path to the CSV file containing patient_id and event_label (or censor info).")
    parser.add_argument("--output_dir", type=str, default="./output",
                        help="Output directory for logs and checkpoints.")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for training.")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of workers for dataloader.")
    parser.add_argument("--max_epochs", type=int, default=10,
                        help="Number of epochs to train.")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate.")
    parser.add_argument("--gpus", type=int, default=0,
                        help="Number of GPUs.")
    parser.add_argument("--check_val_every_n_epoch", type=int, default=1,
                        help="Frequency of validation steps (in epochs).")
    # Add more arguments if needed (e.g. pretrained checkpoint paths, etc.)
    return parser.parse_args()


# 2) CREATE A CUSTOM DATASET
class HCCTimeToEventDataset(Dataset):
    """
    Custom dataset that:
    - Reads (patient_id, label) from CSV.
    - Looks for axial DICOM images under dicom_root/patient_id/*.dcm
    - Filters out non-axial images, and possible other metadata filtering (e.g. SeriesTime).
    - Loads all images for that patient, applies transformations, and average-pools them.
    """
    def __init__(self, 
                 csv_path: str, 
                 dicom_root: str,
                 transform=None,
                 orientation_axial: Tuple[str, ...] = ('1','0','0','0','1','0')):
        """
        Args:
            csv_path: Path to CSV with columns: [patient_id, label, (optional) time, event].
            dicom_root: Root directory of all DICOMs.
            transform: Optional transforms to apply to the image (e.g. resize, normalization).
            orientation_axial: Tuple representing the orientation for axial images.
        """
        super().__init__()
        self.df = pd.read_csv(csv_path)
        
        # If your CSV has columns: [patient_id, event_label], possibly also [duration/event_time].
        # For demonstration, let's assume we have:
        #   self.df['patient_id'], self.df['event_label']  (0 or 1),
        #   (Optional) self.df['time'] if you have it for survival
        #   (Optional) self.df['event'] or self.df['label'] for event indicator
        # Adjust accordingly for your data.
        
        self.dicom_root = dicom_root
        self.transform = transform
        self.orientation_axial = orientation_axial

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        patient_id = str(row['patient_id'])
        
        # If you have time and event columns:
        # time = row['time']  # or row['duration']
        # event = row['event']  # or row['event_label'] 
        # If you only have a label for now (no actual duration):
        event_label = row['label']
        
        # 1) Collect the DICOM paths for this patient
        patient_path = os.path.join(self.dicom_root, patient_id)
        if not os.path.isdir(patient_path):
            raise FileNotFoundError(f"Patient folder not found: {patient_path}")
        
        dicom_files = [os.path.join(patient_path, f) for f in os.listdir(patient_path) 
                       if f.lower().endswith('.dcm')]
        
        # 2) Filter out non-axial or unneeded images 
        #    (we do an example check with orientation)
        axial_files = []
        for dcm_path in dicom_files:
            ds = pydicom.dcmread(dcm_path, stop_before_pixels=True)
            
            # Example orientation check: 
            # ds.ImageOrientationPatient might be something like [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
            orientation = tuple([str(round(x)) for x in ds.ImageOrientationPatient])
            if orientation == self.orientation_axial:
                # Optionally also filter by SeriesTime if needed
                # series_time = ds.get("SeriesTime", None) 
                # or any other metadata-based filtering
                axial_files.append(dcm_path)
        
        # 3) Load the pixel data from each axial file & transform. (In HPC, you might do caching or efficient I/O)
        #    For demonstration, we'll just do a single pass.
        #    Then we average-pool them into a single representation.
        
        images = []
        for path_ in axial_files:
            ds = pydicom.dcmread(path_)
            pixel_array = ds.pixel_array.astype(np.float32)
            # Convert to 3-channel or keep 1-channel (depends on your model's expected input)
            # For simplicity, let's replicate 1-channel -> 3-channel
            pixel_array_3ch = np.repeat(pixel_array[np.newaxis, :, :], 3, axis=0)  # shape: (3, H, W)
            tensor_img = torch.tensor(pixel_array_3ch)
            
            if self.transform:
                tensor_img = self.transform(tensor_img)  # transform is expected to handle torch tensors
            
            images.append(tensor_img)
        
        if len(images) == 0:
            # If no axial images, handle gracefully (e.g., return dummy data or raise exception)
            # We'll return a zero image
            data = torch.zeros((3, 224, 224))  # example shape
        else:
            # Stack and average
            data = torch.stack(images, dim=0)  # shape: (N, 3, H, W)
            data = data.mean(dim=0)           # shape: (3, H, W)
        
        # For survival analysis with pycox, we typically want:
        #   x: features (our image representation)
        #   y_time: time to event (if available)
        #   y_event: event indicator (0/1)
        # If you do not have a real time, you might do a classification or partial usage.
        
        # Here, we return data, label. You could also store event_time if you have it.
        return data, torch.tensor([event_label], dtype=torch.float)


# 3) MODEL DEFINITION (Example using DINOV2 or a placeholder)
# ------------------------------------------------------------------
# If you have a huggingface or timm-based DINOv2 model:
#    model = timm.create_model("dino_v2_xxx", pretrained=True)
# We'll define a small wrapper that chops off the final layer, or we do entire forward.

class DinoV2SurvivalModel(torch.nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        # Example with timm or huggingface if available. 
        # Below is a placeholder using resnet18 for demonstration.
        self.backbone = models.resnet18(pretrained=True)  # placeholder
        # Replace final layer with identity to get feature vector
        num_features = self.backbone.fc.in_features
        self.backbone.fc = torch.nn.Identity()
        
        # Then an MLP for survival. With pycox, we often pass a final linear 
        # that outputs logits for each discrete time bin (for e.g. LogisticHazard).
        # If you only have binary event/no-event, you might do something simpler.
        # We'll do a 1D logistic output for demonstration, but adapt as needed.
        self.fc = torch.nn.Linear(num_features, 1)
    
    def forward(self, x):
        """
        x shape: (B, 3, H, W)
        """
        features = self.backbone(x)  # shape: (B, num_features)
        out = self.fc(features)      # shape: (B, 1)
        return out


# 4) Pytorch Lightning Module to integrate with pycox
# -----------------------------------------------------
class SurvivalLightningModule(pl.LightningModule):
    def __init__(self, model, lr=1e-4):
        super().__init__()
        self.save_hyperparameters()
        self.model = model
        self.loss_fn = torch.nn.BCEWithLogitsLoss()
        self.lr = lr
    
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, label = batch
        logits = self.forward(x)
        loss = self.loss_fn(logits, label)
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, label = batch
        logits = self.forward(x)
        loss = self.loss_fn(logits, label)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer


# 5) TRAIN FUNCTION
# -----------------
def main():
    args = parse_args()
    
    # Create transforms if desired (Resize, etc.)
    # NOTE: If using pretrained models, ensure you apply the proper normalization
    transform = T.Compose([
        T.Resize((224, 224)),
        # T.Normalize(mean, std) if needed
    ])
    
    # Datasets
    train_dataset = HCCTimeToEventDataset(
        csv_path=args.csv_path,
        dicom_root=args.dicom_root,
        transform=transform
    )
    
    # In real scenario, you'd split into train and val sets. We'll do a naive split for demonstration.
    val_fraction = 0.2
    val_size = int(len(train_dataset) * val_fraction)
    train_size = len(train_dataset) - val_size
    train_ds, val_ds = torch.utils.data.random_split(train_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    
    # Initialize the model (DINOv2 placeholder)
    dino_model = DinoV2SurvivalModel(pretrained=True)
    
    # Lightning module
    lit_model = SurvivalLightningModule(model=dino_model, lr=args.lr)
    
    # Trainer
    trainer = pl.Trainer(
        default_root_dir=args.output_dir,
        max_epochs=args.max_epochs,
        accelerator="gpu" if args.gpus > 0 else "cpu",
        devices=args.gpus if args.gpus > 0 else None,
        check_val_every_n_epoch=args.check_val_every_n_epoch,
        log_every_n_steps=1
    )
    
    # Fit
    trainer.fit(lit_model, train_loader, val_loader)


if __name__ == "__main__":
    main()
