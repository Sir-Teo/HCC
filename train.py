#!/usr/bin/env python

import argparse
import os
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
import pytorch_lightning as pl
from pytorch_lightning import seed_everything, Trainer
from torch.utils.data import Dataset, DataLoader

import wandb
from pytorch_lightning.loggers import WandbLogger

# Example for placeholders:
import torchvision.models as models
# If you want to do proper survival/time-to-event modeling:
# from pycox.models import CoxPH, LogisticHazard, etc.

###############################################################################
# 1) DATASET
###############################################################################

class HCCDicomDataset(Dataset):
    """
    A placeholder dataset that:
    - Reads from CSV with columns [patient_id, label].
    - For each patient, loads all axial images (here we just create random tensors).
    - Returns the 3D stack or a single image representation (placeholder).
    """
    def __init__(self, csv_file, dicom_root, transform=None):
        # Placeholder for reading CSV
        self.samples = [(f"patient_{i}", random.randint(0,1)) for i in range(10)]
        
        self.dicom_root = dicom_root
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        patient_id, label = self.samples[idx]
        
        # Just create a random "stack" of 5 images, 3 channels, 224x224
        stack_of_images = torch.randn(5, 3, 224, 224)
        
        if self.transform:
            stack_of_images = torch.stack([self.transform(img) for img in stack_of_images])
        
        return stack_of_images, torch.tensor(label, dtype=torch.float32)

###############################################################################
# 2) LIGHTNING MODULE
###############################################################################

class HCCLightningModel(pl.LightningModule):
    """
    A LightningModule that:
     - uses either a ResNet backbone or a placeholder DINOv2 backbone
     - can output either a binary classifier head (model_type='linear')
       or a placeholder time-to-event head (model_type='time_to_event').
    """
    def __init__(
        self, 
        backbone="resnet", 
        model_type="linear", 
        lr=1e-4, 
        num_classes=1
    ):
        """
        Args:
            backbone (str): 'resnet' or 'dinov2'
            model_type (str): 'linear' or 'time_to_event'
            lr (float): Learning rate
            num_classes (int): Number of output units for classification; 
                               for time-to-event, this may differ
        """
        super().__init__()
        self.save_hyperparameters()

        # -------------------------------------------------------
        # Build the backbone
        # -------------------------------------------------------
        if backbone == "resnet":
            # Example: ResNet18
            backbone_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            feature_dim = 512  # ResNet18 final embedding dim
        elif backbone == "dinov2":
            # Placeholder: using ResNet50 to stand in for a "DINOv2" style model
            backbone_model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            feature_dim = 2048  # ResNet50 final embedding dim
        else:
            raise ValueError(f"Unknown backbone: {backbone}")

        # Remove the classification layer of the chosen backbone
        self.feature_extractor = nn.Sequential(*list(backbone_model.children())[:-1])

        # -------------------------------------------------------
        # Build the head
        # -------------------------------------------------------
        if model_type == "linear":
            # Simple linear classifier for binary classification
            self.head = nn.Linear(feature_dim, num_classes)
        elif model_type == "time_to_event":
            # Placeholder head for time-to-event. 
            # Could be multi-output for hazard function, etc.
            self.head = nn.Linear(feature_dim, 1)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        self.model_type = model_type
        self.lr = lr

    def forward(self, x):
        """
        x is shape (batch_size, 5, 3, 224, 224) if each sample has 5 slices/images.
        - Flatten the slice dimension
        - Extract features
        - Pool across slices
        - Pass through head
        """
        b, n_slices, c, h, w = x.shape  
        x = x.view(b*n_slices, c, h, w)  # Flatten the slices

        feats = self.feature_extractor(x)  # => (b*n_slices, feature_dim, 1, 1)
        feats = feats.view(feats.size(0), -1)  # => (b*n_slices, feature_dim)

        # Re-group by batch and average pool
        feats = feats.view(b, n_slices, -1)  
        feats_mean = feats.mean(dim=1)  # (b, feature_dim)

        logits = self.head(feats_mean)  # => (b, out_dim)
        return logits

    def training_step(self, batch, batch_idx):
        x, y = batch  # x: (batch_size, 5, 3, 224, 224), y: (batch_size,)

        # Forward
        preds = self.forward(x).squeeze(dim=-1)  # => (batch_size,)

        # -------------------------------------------------------
        # Different losses depending on model_type
        # -------------------------------------------------------
        if self.model_type == "linear":
            # Binary classification loss
            loss_fn = nn.BCEWithLogitsLoss()
            loss = loss_fn(preds, y)
        else:
            # time_to_event (placeholder!)
            # Typically you'd have (time, event) data and use a 
            # partial log-likelihood, e.g., from pycox. Here is a dummy example:
            loss = ((preds - y)**2).mean()  # placeholder for demonstration

        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        preds = self.forward(x).squeeze(dim=-1)

        if self.model_type == "linear":
            loss_fn = nn.BCEWithLogitsLoss()
            loss = loss_fn(preds, y)
        else:
            # Placeholder time-to-event
            loss = ((preds - y)**2).mean()

        self.log("val_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        return optimizer


###############################################################################
# 3) DATA MODULE (OPTIONAL)
###############################################################################

class HCCDataModule(pl.LightningDataModule):
    """
    A LightningDataModule to handle train/val/test splits and loaders.
    """
    def __init__(self, csv_file, dicom_root, batch_size=2):
        super().__init__()
        self.csv_file = csv_file
        self.dicom_root = dicom_root
        self.batch_size = batch_size

        # define transforms if needed
        self.transform = T.Compose([
            T.Resize((224, 224)),  # Just for demonstration
        ])

    def setup(self, stage=None):
        full_dataset = HCCDicomDataset(self.csv_file, self.dicom_root, transform=self.transform)
        
        n_total = len(full_dataset)
        n_train = int(n_total * 0.7)
        n_val = int(n_total * 0.15)
        n_test = n_total - n_train - n_val
        
        self.train_dataset, self.val_dataset, self.test_dataset = torch.utils.data.random_split(
            full_dataset,
            [n_train, n_val, n_test],
            generator=torch.Generator().manual_seed(42)
        )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)


###############################################################################
# 4) TRAIN SCRIPT
###############################################################################

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a model using PyTorch Lightning with either a ResNet or DinoV2 backbone, "
                    "and either a linear or time-to-event head."
    )
    parser.add_argument("--dicom_root", type=str, default="/gpfs/data/mankowskilab/HCC_Recurrence/dicom",
                        help="Path to the root DICOM directory.")
    parser.add_argument("--csv_file", type=str, default="patients.csv",
                        help="Path to CSV file with columns [patient_id, label].")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size.")
    parser.add_argument("--max_epochs", type=int, default=2, help="Max number of epochs to train.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--project_name", type=str, default="HCC-Recurrence", help="WandB project name.")
    parser.add_argument("--run_name", type=str, default="test-run", help="WandB run name.")

    # -------------------------------------------------------------------------
    # NEW ARGUMENTS:
    # -------------------------------------------------------------------------
    parser.add_argument("--backbone", type=str, default="resnet",
                        choices=["resnet", "dinov2"],
                        help="Which model backbone to use.")
    parser.add_argument("--model_type", type=str, default="linear",
                        choices=["linear", "time_to_event"],
                        help="Which type of model head to use (binary linear or time-to-event).")

    return parser.parse_args()


def main():
    args = parse_args()
    
    # Set random seed
    seed_everything(args.seed, workers=True)
    
    # Initialize Weights & Biases
    wandb_logger = WandbLogger(project=args.project_name, name=args.run_name)
    
    # Create DataModule
    data_module = HCCDataModule(args.csv_file, args.dicom_root, batch_size=args.batch_size)
    data_module.setup()
    
    # Create model
    model = HCCLightningModel(
        backbone=args.backbone,
        model_type=args.model_type,
        lr=args.lr
    )
    
    # Create trainer
    trainer = Trainer(
        logger=wandb_logger,
        max_epochs=args.max_epochs,
        accelerator="auto",
        devices="auto"
    )
    
    # Fit
    trainer.fit(model, datamodule=data_module)
    trainer.test(model, datamodule=data_module)


if __name__ == "__main__":
    main()
