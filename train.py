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

# Example: You might import your DinoV2 model or other PyTorch model here:
# from torchvision.models import dino_v2_something as DinoV2
# For simplicity, let's just use resnet18 as a placeholder for the Dino backbone
import torchvision.models as models

# If you want to do proper survival/time-to-event modeling, you can import pycox:
# from pycox.models import CoxPH, LogisticHazard, MTLR, etc.
# from pycox.evaluation import EvalSurv

###############################################################################
# 1) DATASET
###############################################################################

class HCCDicomDataset(Dataset):
    """
    A placeholder dataset that:
    - Reads from CSV with columns [patient_id, label].
    - For each patient, loads all axial images (here we just create random tensors).
    - Returns the 3D stack or a single image representation (placeholder).
    
    Replace this with your logic to:
      - parse the DICOM folder
      - filter images by orientation (axial)
      - possibly group all images belonging to the same patient and series
      - load them as torch Tensors
    """
    def __init__(self, csv_file, dicom_root, transform=None):
        # Placeholder for reading CSV
        # For example: 
        #    patient_id,label
        #    9023679,1
        #    1234567,0
        #
        # We'll fake it here.
        
        # In your real code, parse the CSV to get [patient_id, label]
        # e.g., with pandas:
        # import pandas as pd
        # df = pd.read_csv(csv_file)
        # self.samples = list(df.itertuples(index=False, name=None))
        
        # For demonstration, let's just assume 10 patients
        self.samples = [(f"patient_{i}", random.randint(0,1)) for i in range(10)]
        
        self.dicom_root = dicom_root
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        patient_id, label = self.samples[idx]
        
        # Here you would find all .dcm files for that patient, e.g.:
        # patient_path = os.path.join(self.dicom_root, patient_id)
        # image_paths = [os.path.join(patient_path, f) for f in os.listdir(patient_path) 
        #                if f.endswith(".dcm") and is_axial(...)]
        
        # Then read them (with pydicom, e.g.), convert to array, etc.
        # For simplicity, let's just create a random "stack" of 5 images, 3 channels, 224x224:
        # shape: (5, 3, 224, 224)
        # In practice, you'd likely read each DICOM, do your transforms, etc.
        stack_of_images = torch.randn(5, 3, 224, 224)
        
        if self.transform:
            stack_of_images = torch.stack([self.transform(img) for img in stack_of_images])
        
        # Return the entire stack plus the label
        return stack_of_images, torch.tensor(label, dtype=torch.float32)


###############################################################################
# 2) MODEL (PyTorch Lightning Module)
###############################################################################

class TimeToEventModel(pl.LightningModule):
    """
    A placeholder LightningModule that:
     - uses a pretrained backbone (mock: resnet18 here, or your DinoV2).
     - pools across multiple slices (average).
     - does a final linear head for classification or survival prediction.
    """
    def __init__(self, lr=1e-4, num_classes=1):
        super(TimeToEventModel, self).__init__()
        # Example using ResNet18 as a placeholder for Dino:
        backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        # Remove final layer, get a feature vector:
        self.feature_extractor = nn.Sequential(*list(backbone.children())[:-1])
        
        # A linear head to produce 1 logit (for binary classification).
        # For pycox, you might use different heads (e.g. multi-output for hazard).
        self.classifier = nn.Linear(512, num_classes)  # 512 for resnet18
        
        self.lr = lr
        self.save_hyperparameters()

        # If you want a survival approach, you might store:
        # self.survival_model = CoxPH(...) or LogisticHazard(...)

    def forward(self, x):
        """
        x is shape (batch_size, 5, 3, 224, 224) if each sample has 5 images.
        We want to:
         - Flatten the batch dimension + number of slices
         - Extract features
         - Pool features (average) to get single representation
         - Classify
        """
        b, n_slices, c, h, w = x.shape  # e.g. (batch, 5, 3, 224, 224)
        x = x.view(b*n_slices, c, h, w)  # flatten slices => (b*5, 3, 224, 224)
        
        feats = self.feature_extractor(x)  # => shape (b*5, 512, 1, 1) for ResNet
        feats = feats.view(feats.size(0), -1)  # => (b*5, 512)
        
        # Now group back by patient (batch):
        feats = feats.view(b, n_slices, -1)  # => (b, 5, 512)
        feats_mean = feats.mean(dim=1)       # => (b, 512) average pooling across slices
        
        logits = self.classifier(feats_mean) # => (b, num_classes)
        return logits

    def training_step(self, batch, batch_idx):
        x, y = batch  # x: (batch_size, 5, 3, 224, 224), y: (batch_size,)
        logits = self.forward(x).squeeze(dim=-1)  # => (batch_size,)
        
        # For demonstration: BCE with logits. If your label is 0 or 1:
        loss_fn = nn.BCEWithLogitsLoss()
        loss = loss_fn(logits, y)
        
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x).squeeze(dim=-1)
        loss_fn = nn.BCEWithLogitsLoss()
        loss = loss_fn(logits, y)
        
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
    Optional: A LightningDataModule to handle train/val/test splits and loaders.
    """
    def __init__(self, csv_file, dicom_root, batch_size=2):
        super().__init__()
        self.csv_file = csv_file
        self.dicom_root = dicom_root
        self.batch_size = batch_size

        # define transforms if needed
        self.transform = T.Compose([
            # For each 2D slice, apply some transform
            # In real code, you'd do more sophisticated transforms
            T.Resize((224, 224)),  # just in case
        ])

    def setup(self, stage=None):
        # Typically you read the entire dataset once, then split into train/val/test
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
    parser = argparse.ArgumentParser(description="Train a Time-to-Event model using PyTorch Lightning + DinoV2 + pycox (placeholder).")
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
    model = TimeToEventModel(lr=args.lr)
    
    # Create trainer
    trainer = Trainer(
        logger=wandb_logger,
        max_epochs=args.max_epochs,
        accelerator="auto",  # Let PL figure out if there's a GPU
        devices="auto"
    )
    
    # Fit
    trainer.fit(model, datamodule=data_module)
    trainer.test(model, datamodule=data_module)


if __name__ == "__main__":
    main()
