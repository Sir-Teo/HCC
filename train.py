#!/usr/bin/env python
import timm
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
from lifelines.utils import concordance_index

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
        """Load axial DICOM series with proper orientation, then randomly sample num_slices."""
        series_dict = defaultdict(list)
        dcm_files = [f for f in os.listdir(dicom_dir) if f.endswith('.dcm')]
        
        # 1. Read each DICOM and group by orientation
        for fname in dcm_files:
            try:
                dcm = pydicom.dcmread(os.path.join(dicom_dir, fname), stop_before_pixels=True)
                if not hasattr(dcm, 'ImageOrientationPatient') or not hasattr(dcm, 'ImagePositionPatient'):
                    continue
                
                # Round orientation for grouping
                orientation = np.array(dcm.ImageOrientationPatient, dtype=np.float32).round(4)
                orientation_tuple = tuple(orientation.flatten())
                series_dict[orientation_tuple].append(dcm)
            except Exception as e:
                continue

        # 2. Filter for axial series
        axial_series = []
        for orient, dcms in series_dict.items():
            if self.is_axial(orient):
                axial_series.extend(dcms)

        if not axial_series:
            raise ValueError(f"No axial series found in {dicom_dir}")

        # 3. Sort slices by z-position
        axial_series.sort(key=lambda d: (
            float(d.SliceLocation) 
            if hasattr(d, 'SliceLocation') 
            else float(d.ImagePositionPatient[2])
        ))

        # 4. Load pixel data and apply transforms
        image_stack = []
        for dcm in axial_series:
            dcm = pydicom.dcmread(dcm.filename)
            img = self.dicom_to_tensor(dcm)
            if self.transform:
                img = self.transform(img)
            image_stack.append(img)

        current_slices = len(image_stack)

        # 5. Randomly sample or pad
        if current_slices < self.num_slices:
            # Pad with empty slices
            pad_size = self.num_slices - current_slices
            slice_shape = image_stack[0].shape if image_stack else (3, 224, 224)
            for _ in range(pad_size):
                image_stack.append(torch.zeros(slice_shape))
        else:
            # Randomly choose `self.num_slices` from the total slices
            selected_indices = np.random.choice(current_slices, self.num_slices, replace=False)
            # Sort them so the final stack preserves the natural top-to-bottom order
            selected_indices.sort()
            image_stack = [image_stack[i] for i in selected_indices]

        return torch.stack(image_stack)


    def is_axial(self, orientation):
        """Check if orientation is axial by verifying cross product aligns with z-axis."""
        # Extract row and column direction vectors
        row_dir = np.array(orientation[:3], dtype=np.float32)
        col_dir = np.array(orientation[3:], dtype=np.float32)
        
        # Compute cross product of row and column directions
        cross = np.cross(row_dir, col_dir)
        
        # Normalize the cross product to get the direction
        norm = np.linalg.norm(cross)
        if norm < 1e-6:  # Avoid division by zero for invalid orientations
            return False
        cross_normalized = cross / norm
        
        # Check if the cross product is aligned with the z-axis (positive or negative)
        return (np.abs(cross_normalized[0]) < 5e-2 and 
                np.abs(cross_normalized[1]) < 5e-2 and 
                np.abs(np.abs(cross_normalized[2]) - 1.0) < 5e-2)

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

class CoxPHLoss(nn.Module):
    """
    Cox Proportional Hazards Loss Function.
    """
    def __init__(self):
        super(CoxPHLoss, self).__init__()

    def forward(self, preds, durations, events):
        preds = preds.squeeze()
        durations = durations.squeeze()
        events = events.squeeze()

        # Sort by descending durations
        order = torch.argsort(-durations)
        preds = preds[order]
        durations = durations[order]
        events = events[order]

        # Calculate hazard ratios
        hazard_ratio = torch.exp(preds)
        log_cum_hazard = torch.log(torch.cumsum(hazard_ratio, dim=0))

        # Partial log-likelihood
        log_likelihood = preds - log_cum_hazard
        partial_log_likelihood = log_likelihood * events

        # CoxPH loss
        loss = -torch.sum(partial_log_likelihood) / torch.sum(events)
        return loss


class HCCLightningModel(pl.LightningModule):
    def __init__(
        self,
        backbone="resnet",       # "resnet", "dinov1", or "dinov2"
        model_type="linear",     # "linear" or "time_to_event"
        lr=1e-4,
        num_classes=1
    ):
        super(HCCLightningModel, self).__init__()
        self.save_hyperparameters()

        self.model_type = model_type
        self.lr = lr

        # ---------------------------------------------------------------------
        # 1) Build Backbone
        # ---------------------------------------------------------------------
        if backbone == "resnet":
            backbone_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            feature_dim = backbone_model.fc.in_features
            self.feature_extractor = nn.Sequential(*list(backbone_model.children())[:-1])
            self.is_vit = False

        elif backbone == "dinov1":
            backbone_model = timm.create_model("vit_small_patch16_224_dino", pretrained=True)
            feature_dim = backbone_model.embed_dim
            self.feature_extractor = backbone_model
            self.is_vit = True

        elif backbone == "dinov2":
            backbone_model = timm.create_model("dinov2_vitb14", pretrained=True)
            feature_dim = backbone_model.embed_dim
            self.feature_extractor = backbone_model
            self.is_vit = True

        else:
            raise ValueError(f"Unknown backbone: {backbone}")
        
        # ---------------------------------------------------------------------
        # 2.2) Build Head
        # ---------------------------------------------------------------------
        if model_type == "linear":
            self.head = nn.Linear(feature_dim, num_classes)
            self.criterion = nn.BCEWithLogitsLoss()
        elif model_type == "time_to_event":
            self.head = nn.Linear(feature_dim, 1)
            self.criterion = CoxPHLoss()
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        # ---------------------------------------------------------------------
        # 2.3) Define Metrics
        # ---------------------------------------------------------------------
        if self.model_type == "linear":
            self.test_acc = torchmetrics.Accuracy(task="binary")
            self.test_prec = torchmetrics.Precision(task="binary")
            self.test_rec = torchmetrics.Recall(task="binary")
            self.test_f1 = torchmetrics.F1Score(task="binary")
            self.test_auroc = torchmetrics.AUROC(task="binary")
        elif self.model_type == "time_to_event":
            pass  # Metrics handled separately

    def forward(self, x):
        b, n_slices, c, h, w = x.shape
        x = x.view(b * n_slices, c, h, w)  # Flatten slices into a single batch

        if not self.is_vit:
            feats = self.feature_extractor(x)  # (B*n_slices, feature_dim, 1, 1)
            feats = feats.view(feats.size(0), -1)  # Flatten to (B*n_slices, feature_dim)
        else:
            feats_vit = self.feature_extractor.forward_features(x)
            if isinstance(feats_vit, dict):
                feats_vit = feats_vit["x_norm_clstoken"]  # Adjust based on model
            feats = feats_vit[:, 0, :]  # CLS token

        feats = feats.view(b, n_slices, -1)  # (B, n_slices, feature_dim)
        feats_mean = feats.mean(dim=1)       # Average over slices => (B, feature_dim)

        logits = self.head(feats_mean)        # (B, out_dim)
        return logits

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def training_step(self, batch, batch_idx):
        if self.model_type == "linear":
            x, y = batch
            logits = self.forward(x).squeeze(-1)
            loss = self.criterion(logits, y)
        elif self.model_type == "time_to_event":
            x, t, e = batch
            risk_score = self.forward(x).squeeze(-1)
            loss = self.criterion(risk_score, t, e)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        if self.model_type == "linear":
            x, y = batch
            logits = self.forward(x).squeeze(-1)
            loss = self.criterion(logits, y)
        elif self.model_type == "time_to_event":
            x, t, e = batch
            risk_score = self.forward(x).squeeze(-1)
            loss = self.criterion(risk_score, t, e)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

        self.log("val_loss", loss, prog_bar=True)
        return loss

    def on_test_epoch_start(self):
        if self.model_type == "linear":
            self.test_acc.reset()
            self.test_prec.reset()
            self.test_rec.reset()
            self.test_f1.reset()
            self.test_auroc.reset()

            self.all_probs = []
            self.all_labels = []
        elif self.model_type == "time_to_event":
            self.all_risks = []
            self.all_times = []
            self.all_events = []

    def test_step(self, batch, batch_idx):
        if self.model_type == "linear":
            x, y = batch
            logits = self.forward(x).squeeze(-1)
            loss = self.criterion(logits, y)

            probs = torch.sigmoid(logits)

            self.test_acc.update(probs, y.long())
            self.test_prec.update(probs, y.long())
            self.test_rec.update(probs, y.long())
            self.test_f1.update(probs, y.long())
            self.test_auroc.update(probs, y.long())

            self.all_probs.append(probs.detach().cpu())
            self.all_labels.append(y.detach().cpu())

            self.log("test_loss", loss)
            return {"test_loss": loss}

        elif self.model_type == "time_to_event":
            x, t, e = batch
            risk_score = self.forward(x).squeeze(-1)
            loss = self.criterion(risk_score, t, e)

            self.all_risks.append(risk_score.detach().cpu())
            self.all_times.append(t.detach().cpu())
            self.all_events.append(e.detach().cpu())

            self.log("test_loss", loss)
            return {"test_loss": loss}
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def on_test_epoch_end(self):
        if self.model_type == "linear":
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

        elif self.model_type == "time_to_event":
            all_risks = torch.cat(self.all_risks).numpy()
            all_times = torch.cat(self.all_times).numpy()
            all_events = torch.cat(self.all_events).numpy()

            # Compute Concordance Index (c-index)
            c_index = concordance_index(
                event_times=all_times,
                predicted_scores=-all_risks,  # Negative because higher risk = shorter survival
                event_observed=all_events
            )
            self.log("test_c_index", c_index)

            # Option 2) pycox (EvalSurv) typically expects survival predictions
            # across multiple time points. For a pure risk score, you can do:
            eval_surv = EvalSurv(pd.Series(-all_risks), all_times, all_events, censor_seq="end")
            c_index_pycox = eval_surv.concordance_td()
            self.log("test_c_index_pycox", c_index_pycox)


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
    parser.add_argument("--csv_file", type=str, default="processed_patient_labels.csv",
                        help="Path to processed CSV with columns [Deidentified ID, recurrence post tx, time, event].")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size.")
    parser.add_argument("--max_epochs", type=int, default=50, help="Max number of epochs.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--project_name", type=str, default="HCC-Recurrence", help="WandB project name.")
    parser.add_argument("--run_name", type=str, default="time_to_event-run", help="WandB run name.")
    
    parser.add_argument("--backbone", type=str, default="resnet",
                        choices=["resnet", "dinov2", "dinov1"],
                        help="Which model backbone to use.")
    parser.add_argument("--model_type", type=str, default="time_to_event",
                        choices=["linear", "time_to_event"],
                        help="Which type of head to use (binary classification vs survival).")
    parser.add_argument("--num_slices", type=int, default=20,
                        help="Fixed number of slices to use per patient (pad or crop)")

    return parser.parse_args()


def main():
    args = parse_args()
    
    # Set random seed
    pl.seed_everything(args.seed, workers=True)
    
    # Initialize Weights & Biases logger
    wandb_logger = WandbLogger(project=args.project_name, name=args.run_name)
    
    # Create DataModule
    data_module = HCCDataModule(
        csv_file=args.csv_file,
        dicom_root=args.dicom_root,
        model_type=args.model_type,
        batch_size=args.batch_size,
        num_slices=args.num_slices
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
