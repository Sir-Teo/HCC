#!/usr/bin/env python

import argparse
import os
import random
from collections import defaultdict
import pandas as pd
import numpy as np
import pydicom

import torch
import torch.nn as nn
import torch.optim as optim
import torchmetrics
import torchvision.transforms as T
import pytorch_lightning as pl
from pytorch_lightning import seed_everything, Trainer
from torch.utils.data import Dataset, DataLoader

import wandb
from pytorch_lightning.loggers import WandbLogger

import torchvision.models as models

# For survival metrics:
from pycox.evaluation import EvalSurv
import lifelines

###############################################################################
# 1) DATASET
###############################################################################

class HCCDicomDataset(Dataset):
    def __init__(self, csv_file, dicom_root, model_type="linear", transform=None,num_slices=10):
        self.dicom_root = dicom_root
        self.transform = transform
        self.model_type = model_type
        self.num_slices = num_slices
        self.patient_data = []
        
        # Read CSV and prepare patient data
        df = pd.read_csv(csv_file)
        for _, row in df.iterrows():
            patient_id = str(row['Patient_id'])
            dicom_dir = os.path.join(dicom_root, patient_id)
            
            if os.path.exists(dicom_dir):
                self.patient_data.append({
                    'patient_id': patient_id,
                    'label': row['Label'],
                    'time': row.get('time', 0),  # Default values if not in CSV
                    'event': row.get('event', 0),
                    'dicom_dir': dicom_dir
                })

    def __len__(self):
        return len(self.patient_data)

    def __getitem__(self, idx):
        patient = self.patient_data[idx]
        dicom_dir = patient['dicom_dir']
        
        # Load and process DICOM images
        image_stack = self.load_axial_series(dicom_dir)
        
        # Return appropriate targets based on model type
        if self.model_type == "linear":
            return image_stack, torch.tensor(patient['label'], dtype=torch.float32)
        else:
            return image_stack, torch.tensor(patient['time'], dtype=torch.float32), \
                   torch.tensor(patient['event'], dtype=torch.float32)

    def load_axial_series(self, dicom_dir):
        """Load axial DICOM series with proper orientation"""
        # 1. Group files by series time
        series_dict = defaultdict(list)
        
        # List all DICOM files in directory
        dcm_files = [f for f in os.listdir(dicom_dir) if f.endswith('.dcm')]
        
        # Read basic metadata from all files
        for fname in dcm_files:
            try:
                dcm = pydicom.dcmread(os.path.join(dicom_dir, fname), stop_before_pixels=True)
                series_time = dcm.get('SeriesTime', '000000.000')
                orientation = tuple(map(float, dcm.ImageOrientationPatient))
                series_dict[(series_time, orientation)].append(dcm)
            except Exception as e:
                print(f"Error reading {fname}: {str(e)}")
                continue

        # 2. Select axial series with proper orientation
        axial_series = []
        for (stime, orient), dcms in series_dict.items():
            # Check if orientation is axial (approximate check)
            if self.is_axial(orient):
                axial_series.extend(dcms)

        if not axial_series:
            raise ValueError(f"No axial series found in {dicom_dir}")

        # 3. Sort slices by slice location (z-position)
        axial_series.sort(key=lambda d: float(d.SliceLocation))

        # 4. Load and preprocess images
        image_stack = []
        for dcm in axial_series:
            dcm = pydicom.dcmread(dcm.filename)
            img = self.dicom_to_tensor(dcm)
            
            if self.transform:
                img = self.transform(img)
            
            image_stack.append(img)

        # Handle variable slice counts
        current_slices = len(image_stack)
        
        if current_slices < self.num_slices:
            # Pad with empty slices
            pad_size = self.num_slices - current_slices
            slice_shape = image_stack[0].shape if image_stack else (3, 224, 224)
            for _ in range(pad_size):
                image_stack.append(torch.zeros(slice_shape))
        else:
            # Take center slices
            start_idx = (current_slices - self.num_slices) // 2
            image_stack = image_stack[start_idx:start_idx+self.num_slices]

        return torch.stack(image_stack)

    def is_axial(self, orientation):
        """Check if orientation is approximately axial (within tolerance)"""
        # Expected axial orientation: (1, 0, 0, 0, 1, 0)
        ref_orientation = (1, 0, 0, 0, 1, 0)
        tolerance = 1e-3
        
        return all(abs(a - b) < tolerance 
                   for a, b in zip(orientation, ref_orientation))

    def dicom_to_tensor(self, dcm):
        """Convert DICOM pixel data to normalized tensor"""
        img = dcm.pixel_array.astype(np.float32)
        
        # Apply modality-specific preprocessing
        if dcm.Modality == 'CT':
            # Apply Hounsfield Units scaling
            intercept = float(dcm.RescaleIntercept)
            slope = float(dcm.RescaleSlope)
            img = slope * img + intercept
            
            # Clip to typical soft tissue range
            img = np.clip(img, -100, 400)
            
            # Normalize to [0, 1]
            img = (img + 100) / 500.0
        else:
            # Handle other modalities (MR, etc)
            img = (img - img.min()) / (img.max() - img.min())
        
        # Convert to RGB if needed (repeat grayscale to 3 channels)
        if img.ndim == 2:
            img = np.repeat(img[..., np.newaxis], 3, axis=-1)
            
        return torch.from_numpy(img).permute(2, 0, 1)  # CHW format

###############################################################################
# 2) LIGHTNING MODULE
###############################################################################

class HCCLightningModel(pl.LightningModule):
    """
    A LightningModule that can do:
      - 'linear' (binary classification) or
      - 'time_to_event' (survival).
    It uses either a ResNet or Dino-like backbone (placeholder).
    """
    def __init__(
        self,
        backbone="resnet",       # "resnet" or "dinov2" (placeholder)
        model_type="linear",     # "linear" or "time_to_event"
        lr=1e-4,
        num_classes=1
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model_type = model_type
        self.lr = lr

        # ---------------------------------------------------------------------
        # 2.1) Build Backbone
        # ---------------------------------------------------------------------
        if backbone == "resnet":
            backbone_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            feature_dim = 512
        elif backbone == "dinov2":
            # Placeholder: using resnet50 as a stand-in for "dinov2"
            # In real code, you would load your DINOv2 model.
            backbone_model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            feature_dim = 2048
        else:
            raise ValueError(f"Unknown backbone: {backbone}")

        # Remove final classification layer of the chosen backbone
        self.feature_extractor = nn.Sequential(*list(backbone_model.children())[:-1])

        # ---------------------------------------------------------------------
        # 2.2) Build Head
        # ---------------------------------------------------------------------
        if model_type == "linear":
            # Simple binary classifier
            self.head = nn.Linear(feature_dim, num_classes)
        else:
            # Placeholder for time-to-event
            # In a real model, you might do multiple outputs or something
            self.head = nn.Linear(feature_dim, 1)

        # ---------------------------------------------------------------------
        # 2.3) Define metrics for classification
        # ---------------------------------------------------------------------
        if self.model_type == "linear":
            # We'll track these metrics over test set
            self.test_acc = torchmetrics.Accuracy(task="binary")
            self.test_prec = torchmetrics.Precision(task="binary")
            self.test_rec = torchmetrics.Recall(task="binary")
            self.test_f1 = torchmetrics.F1Score(task="binary")
            self.test_auroc = torchmetrics.AUROC(task="binary")

        # For time-to-event, we will rely on storing predictions and computing c-index externally.

    def forward(self, x):
        """
        x shape: (batch_size, 5, 3, 224, 224)
        Steps:
         - Flatten slices
         - Extract features
         - Average pool across slices
         - Pass through head
        """
        b, n_slices, c, h, w = x.shape
        x = x.view(b*n_slices, c, h, w)  # flatten slices

        feats = self.feature_extractor(x)   # (b*n_slices, feat_dim, 1, 1)
        feats = feats.view(feats.size(0), -1)  # (b*n_slices, feat_dim)

        # regroup by batch, average over slices
        feats = feats.view(b, n_slices, -1)
        feats_mean = feats.mean(dim=1)

        logits = self.head(feats_mean)  # (b, out_dim)
        return logits

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)

    # -------------------------------------------------------------------------
    # 2.4) Training / Validation Steps
    # -------------------------------------------------------------------------
    def training_step(self, batch, batch_idx):
        if self.model_type == "linear":
            x, y = batch
            logits = self.forward(x).squeeze(-1)
            loss = nn.BCEWithLogitsLoss()(logits, y)
        else:
            x, t, e = batch  # time, event
            # Placeholder risk score
            risk_score = self.forward(x).squeeze(-1)
            # Example "dummy" survival loss: MSE with time (NOT a real survival loss!)
            # Real code might do partial log-likelihood with pycox etc.
            loss = ((risk_score - t)**2).mean()

        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        if self.model_type == "linear":
            x, y = batch
            logits = self.forward(x).squeeze(-1)
            loss = nn.BCEWithLogitsLoss()(logits, y)
        else:
            x, t, e = batch
            risk_score = self.forward(x).squeeze(-1)
            # same dummy survival "loss"
            loss = ((risk_score - t)**2).mean()

        self.log("val_loss", loss, prog_bar=True)
        return loss

    # -------------------------------------------------------------------------
    # 2.5) Testing Steps
    # -------------------------------------------------------------------------
    def on_test_epoch_start(self):
        # We'll store predictions so we can compute metrics for classification
        # (like AUROC), and also c-index for survival
        if self.model_type == "linear":
            self.test_acc.reset()
            self.test_prec.reset()
            self.test_rec.reset()
            self.test_f1.reset()
            self.test_auroc.reset()

            self.all_probs = []
            self.all_labels = []
        else:
            self.all_risks = []
            self.all_times = []
            self.all_events = []

    def test_step(self, batch, batch_idx):
        if self.model_type == "linear":
            x, y = batch
            logits = self.forward(x).squeeze(-1)
            loss = nn.BCEWithLogitsLoss()(logits, y)

            # Compute probabilities
            probs = torch.sigmoid(logits)

            # Update metrics incrementally
            self.test_acc.update(probs, y.long())
            self.test_prec.update(probs, y.long())
            self.test_rec.update(probs, y.long())
            self.test_f1.update(probs, y.long())
            self.test_auroc.update(probs, y.long())

            self.all_probs.append(probs.detach().cpu())
            self.all_labels.append(y.detach().cpu())

            self.log("test_loss", loss)
            return {"test_loss": loss}

        else:
            # time to event
            x, t, e = batch
            risk_score = self.forward(x).squeeze(-1)
            # A dummy survival "loss" just to log
            loss = ((risk_score - t)**2).mean()

            # We'll store risk_score, time, event for c-index
            self.all_risks.append(risk_score.detach().cpu())
            self.all_times.append(t.detach().cpu())
            self.all_events.append(e.detach().cpu())

            self.log("test_loss", loss)
            return {"test_loss": loss}

    def on_test_epoch_end(self):
        if self.model_type == "linear":
            # Compute final aggregated metrics from torchmetrics
            acc = self.test_acc.compute()
            prec = self.test_prec.compute()
            rec = self.test_rec.compute()
            f1 = self.test_f1.compute()
            auroc = self.test_auroc.compute()

            self.log("test_acc", acc)
            self.log("test_precision", prec)
            self.log("test_recall", rec)
            self.log("test_f1", f1)
            self.log("test_auroc", auroc)

        else:
            # -------------------------------------------------
            # Compute c-index for survival
            # -------------------------------------------------
            # Concatenate everything
            all_risks = torch.cat(self.all_risks).numpy()
            all_times = torch.cat(self.all_times).numpy()
            all_events = torch.cat(self.all_events).numpy()

            # "all_risks" is your predicted risk (higher => earlier event).
            # We'll use lifelines or pycox to compute c-index.

            # Option 1) lifelines:
            # from lifelines.utils import concordance_index
            c_index = lifelines.utils.concordance_index(
                event_times=all_times,
                predicted_scores=all_risks,  # risk scores
                event_observed=all_events
            )
            self.log("test_c_index", c_index)

            # Option 2) pycox (EvalSurv) typically expects survival predictions
            # across multiple time points. For a pure risk score, you can do:
            # eval_surv = EvalSurv(pd.Series(-all_risks), all_times, all_events, censor_seq="end")
            # c_index_pycox = eval_surv.concordance_td()
            # self.log("test_c_index_pycox", c_index_pycox)


###############################################################################
# 3) DATA MODULE
###############################################################################

class HCCDataModule(pl.LightningDataModule):
    def __init__(self, csv_file, dicom_root, model_type="linear", batch_size=2, num_slices=100):
        super().__init__()
        self.csv_file = csv_file
        self.dicom_root = dicom_root
        self.batch_size = batch_size
        self.model_type = model_type
        self.num_slices = num_slices  # Add num_slices parameter

        self.transform = T.Compose([
            T.Resize((224, 224))
        ])

    def setup(self, stage=None):
        full_dataset = HCCDicomDataset(
            csv_file=self.csv_file,
            dicom_root=self.dicom_root,
            model_type=self.model_type,
            transform=self.transform,
            num_slices=self.num_slices  # Pass parameter here
        )
        
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
        description="Train + test a model for either binary classification or time-to-event, "
                    "and evaluate many metrics."
    )
    parser.add_argument("--dicom_root", type=str, default="/gpfs/data/mankowskilab/HCC_Recurrence/dicom",
                        help="Path to the root DICOM directory.")
    parser.add_argument("--csv_file", type=str, default="patient_labels.csv",
                        help="Path to CSV with columns [patient_id, label, time, event].")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size.")
    parser.add_argument("--max_epochs", type=int, default=50, help="Max number of epochs.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--project_name", type=str, default="HCC-Recurrence", help="WandB project name.")
    parser.add_argument("--run_name", type=str, default="test-run", help="WandB run name.")
    
    parser.add_argument("--backbone", type=str, default="resnet",
                        choices=["resnet", "dinov2"],
                        help="Which model backbone to use.")
    parser.add_argument("--model_type", type=str, default="linear",
                        choices=["linear", "time_to_event"],
                        help="Which type of head to use (binary classification vs survival).")
    parser.add_argument("--num_slices", type=int, default=100,
                        help="Fixed number of slices to use per patient (pad or crop)")

    return parser.parse_args()


def main():
    args = parse_args()
    
    # Set random seed
    seed_everything(args.seed, workers=True)
    
    # Initialize Weights & Biases logger
    wandb_logger = WandbLogger(project=args.project_name, name=args.run_name)
    
    # Create DataModule
    data_module = HCCDataModule(
        csv_file=args.csv_file,
        dicom_root=args.dicom_root,
        model_type=args.model_type,
        batch_size=args.batch_size,
        num_slices=args.num_slices  # Add this line
    )
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

    # Test
    trainer.test(model, datamodule=data_module)


if __name__ == "__main__":
    main()
