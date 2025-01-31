import timm
import argparse
import os
import random
from collections import defaultdict
import pandas as pd
import numpy as np
import pydicom
from pycox.models import CoxPH as PyCoxCoxPH
from pycox.evaluation import EvalSurv
from pycox.models.loss import CoxPHLoss
import torch
import torch.nn as nn
import torch.optim as optim
import torchmetrics
import torchvision.transforms as T
import pytorch_lightning as pl
from pytorch_lightning import seed_everything, Trainer
from torch.utils.data import Dataset, DataLoader, default_collate
from sklearn.model_selection import train_test_split
import wandb
from pytorch_lightning.loggers import WandbLogger
import torchvision.models as models
from lifelines.utils import concordance_index
from data.dataset import *

class HCCLightningModel(pl.LightningModule):
    def __init__(
        self,
        backbone="resnet",       # "resnet", "dinov1", or "dinov2"
        model_type="linear",     # "linear" or "time_to_event"
        lr=1e-6,                 # Learning rate
        num_classes=1,
        pretrained=True          # Use pretrained weights
    ):
        super(HCCLightningModel, self).__init__()
        self.save_hyperparameters()

        self.model_type = model_type
        self.lr = lr

        # ---------------------------------------------------------------------
        # 1) Build Backbone
        # ---------------------------------------------------------------------
        if backbone == "resnet":
            backbone_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT) if pretrained else models.resnet18(weights=None)
            feature_dim = backbone_model.fc.in_features
            self.feature_extractor = nn.Sequential(*list(backbone_model.children())[:-1])
            self.is_vit = False

        elif backbone == "dinov1":
            backbone_model = timm.create_model("vit_small_patch16_224_dino", pretrained=pretrained)
            feature_dim = backbone_model.embed_dim
            self.feature_extractor = backbone_model
            self.is_vit = True

        elif backbone == "dinov2":
            backbone_model = timm.create_model("dinov2_vitb14", pretrained=pretrained)
            feature_dim = backbone_model.embed_dim
            self.feature_extractor = backbone_model
            self.is_vit = True

        else:
            raise ValueError(f"Unknown backbone: {backbone}")

        if pretrained:
            for param in self.feature_extractor.parameters():
                param.requires_grad = False  # Freeze pretrained weights

        # ---------------------------------------------------------------------
        # 2.2) Build Head
        # ---------------------------------------------------------------------
        if model_type == "linear":
            self.head = nn.Linear(feature_dim, num_classes)
            nn.init.normal_(self.head.weight, mean=0.0, std=0.01)
            nn.init.constant_(self.head.bias, 0.0)
            self.criterion = nn.BCEWithLogitsLoss()
        elif model_type == "time_to_event":
            # Use a linear layer for risk scores
            self.head = nn.Linear(feature_dim, 1)
            nn.init.normal_(self.head.weight, mean=0.0, std=0.01)
            nn.init.constant_(self.head.bias, 0.0)
            self.criterion = CoxPHLoss()
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        # ---------------------------------------------------------------------
        # 2.3) Define Metrics
        # ---------------------------------------------------------------------
        if self.model_type == "linear":
            # Validation metrics
            self.val_acc = torchmetrics.Accuracy(task="binary")
            self.val_prec = torchmetrics.Precision(task="binary")
            self.val_rec = torchmetrics.Recall(task="binary")
            self.val_f1 = torchmetrics.F1Score(task="binary")
            self.val_auroc = torchmetrics.AUROC(task="binary")
            self.val_probs = []
            self.val_labels = []
            
            self.test_acc = torchmetrics.Accuracy(task="binary")
            self.test_prec = torchmetrics.Precision(task="binary")
            self.test_rec = torchmetrics.Recall(task="binary")
            self.test_f1 = torchmetrics.F1Score(task="binary")
            self.test_auroc = torchmetrics.AUROC(task="binary")
            self.all_probs = []
            self.all_labels = []
        # ---------------------------------------------------------------------
        # 2.4) Initialize Baseline Hazards
        # ---------------------------------------------------------------------
        if self.model_type == "time_to_event":
            self.val_risks = []
            self.val_times = []
            self.val_events = []
            
            # Test tracking (ADDED if needed)
            self.all_risks = []
            self.all_times = []
            self.all_events = []
            self.baseline_hazards = None
            self.surv_funcs = None

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

            # Data validation
            if (t <= 0).any():
                raise ValueError("All durations must be positive.")
            if not torch.all((e == 0) | (e == 1)):
                raise ValueError("Event indicators must be binary (0 or 1).")

            risk_score = self.forward(x).squeeze(-1)
            loss = self.criterion(risk_score, t, e)  # PyCox's CoxPHLoss expects (risk_score, durations, events)

            # Check for NaNs and Infs in loss
            if torch.isnan(loss):
                self.log("NaN_loss", True)
                raise ValueError("Loss is NaN")
            else:
                self.log("NaN_loss", False)

            # Log risk score statistics
            self.log("risk_score_mean", torch.mean(risk_score), prog_bar=True)
            self.log("risk_score_std", torch.std(risk_score), prog_bar=True)
            self.log("risk_score_min", torch.min(risk_score), prog_bar=True)
            self.log("risk_score_max", torch.max(risk_score), prog_bar=True)
            self.log("risk_score_nan", torch.isnan(risk_score).any(), prog_bar=True)
            self.log("risk_score_inf", torch.isinf(risk_score).any(), prog_bar=True)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        if self.model_type == "linear":
            x, y = batch
            logits = self.forward(x).squeeze(-1)
            loss = self.criterion(logits, y)

            probs = torch.sigmoid(logits)
            self.val_acc.update(probs, y)
            self.val_prec.update(probs, y)
            self.val_rec.update(probs, y)
            self.val_f1.update(probs, y)
            self.val_auroc.update(probs, y)
            self.val_probs.append(probs.detach().cpu())
            self.val_labels.append(y.detach().cpu())

            self.log("val_loss", loss, prog_bar=True)
            return loss

        elif self.model_type == "time_to_event":
            x, t, e = batch
            risk_score = self.forward(x).squeeze(-1)
            loss = self.criterion(risk_score, t, e)

            self.val_risks.append(risk_score.detach().cpu())
            self.val_times.append(t.detach().cpu())
            self.val_events.append(e.detach().cpu())
            self.log("val_loss", loss, prog_bar=True)
            return loss
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def on_validation_epoch_start(self):
        """Reset validation metrics at the start of each epoch."""
        if self.model_type == "linear":
            self.val_acc.reset()
            self.val_prec.reset()
            self.val_rec.reset()
            self.val_f1.reset()
            self.val_auroc.reset()
            self.val_probs.clear()
            self.val_labels.clear()
        elif self.model_type == "time_to_event":
            self.val_risks.clear()
            self.val_times.clear()
            self.val_events.clear()

    def on_validation_epoch_end(self):
        """Compute and log validation metrics."""
        if self.model_type == "linear":
            # Log classification metrics
            self.log("val_acc", self.val_acc.compute(), prog_bar=True)
            self.log("val_precision", self.val_prec.compute())
            self.log("val_recall", self.val_rec.compute())
            self.log("val_f1", self.val_f1.compute())
            self.log("val_auroc", self.val_auroc.compute())

            # Log class distribution
            if self.val_labels:
                all_labels = torch.cat(self.val_labels)
                num_pos = all_labels.sum().item()
                total = len(all_labels)
                self.log("val_ratio_pos", num_pos / total)
                self.log("val_num_pos", num_pos)
                self.log("val_num_neg", total - num_pos)

        elif self.model_type == "time_to_event":
            # Compute C-index for survival
            if self.val_risks and self.val_times and self.val_events:
                risks = torch.cat(self.val_risks).numpy()
                times = torch.cat(self.val_times).numpy()
                events = torch.cat(self.val_events).numpy().astype(bool)

                try:
                    c_index = concordance_index(times, -risks, events)
                    self.log("val_c_index", c_index, prog_bar=True)
                except Exception as e:
                    print(f"Validation C-index error: {e}")

                # Log event distribution
                num_events = events.sum()
                self.log("val_num_events", num_events)
                self.log("val_num_censored", len(events) - num_events)

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
            self.surv_funcs = None  # To store survival functions

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

            # Compute Concordance Index (c-index) using lifelines
            try:
                c_index = concordance_index(all_times, -all_risks, all_events)
                self.log("test_c_index_lifelines", c_index)
            except Exception as e:
                print(f"Lifelines C-index error: {e}")
                self.log("test_c_index_lifelines", float('nan'))

            # Compute C-index using PyCox's EvalSurv
            try:
                surv = self.predict_surv_func(all_risks, all_times, all_events)
                eval_surv = EvalSurv(surv, all_times, all_events, censor_surv='km')
                self.log("test_c_index_pycox", eval_surv.concordance_td())
            except Exception as e:
                print(f"PyCox C-index error: {e}")
                self.log("test_c_index_pycox", float('nan'))

    def compute_baseline_hazards(self, dataloader):
        """Compute baseline hazards using PyCox's CoxPH model."""
        # Collect all risks, durations, and events
        risks = []
        durations = []
        events = []
        self.eval()
        with torch.no_grad():
            for batch in dataloader:
                x, t, e = batch
                risk_score = self.forward(x).squeeze(-1)
                risks.append(risk_score.cpu().numpy())
                durations.append(t.cpu().numpy())
                events.append(e.cpu().numpy())

        risks = np.concatenate(risks)
        print(risk)
        durations = np.concatenate(durations)
        events = np.concatenate(events).astype(bool)

        # Initialize PyCox CoxPH model
        cph = PyCoxCoxPH()
        cph.fit(risks, durations, events)

        # Store baseline hazards
        self.baseline_hazards = pd.Series(cph.baseline_hazard_, index=cph.unique_times_)
        self.baseline_hazards.index.name = 'time'


    def predict_surv_func(self, risks, durations, events):
        if self.baseline_hazards is None:
            raise ValueError("Baseline hazards not computed. Call compute_baseline_hazards first.")

        # Sort the baseline hazards
        sorted_times = np.sort(self.baseline_hazards.index.values)
        cum_baseline = self.baseline_hazards.values

        # Compute cumulative hazard
        cumulative_hazard = np.cumsum(cum_baseline)

        # Compute survival functions
        S = np.exp(-np.outer(risks, cumulative_hazard))

        return pd.DataFrame(S, columns=sorted_times, dtype=np.float32)




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
                        help="Path to processed CSV with columns [Pre op MRI Accession number, recurrence post tx, time, event].")
    parser.add_argument("--lr", type=float, default=1e-4, help="Fixed Learning rate.")  # Adjusted default
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size.")
    parser.add_argument("--max_epochs", type=int, default=1, help="Max number of epochs.")
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
    parser.add_argument("--num_workers", type=int, default=2)
    
    parser.add_argument("--preprocessed_root", type=str, default=None,
                        help="Directory to save/load preprocessed DICOM tensors for faster loading.")
    parser.add_argument("--pretrained", action='store_true', default=True,
                        help="Use pretrained weights for the backbone model. Default is True.")
    
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
        num_slices=args.num_slices,
        num_workers=args.num_workers,
        preprocessed_root=args.preprocessed_root
    )
    data_module.setup()

    # Create model
    model = HCCLightningModel(
        backbone=args.backbone,
        model_type=args.model_type,
        lr=args.lr,
        pretrained=args.pretrained  # Pass the pretrained argument
    )

    # -------------------------------------------------------------------------
    # 5) Callbacks: Early Stopping and Model Checkpointing
    # -------------------------------------------------------------------------
    from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        min_delta=0.00,
        patience=10,
        verbose=True,
        mode="min"
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath="checkpoints",
        filename="best-checkpoint",
        save_top_k=1,
        mode="min"
    )

    # -------------------------------------------------------------------------
    # 6) Create Trainer without Learning Rate Finder
    # -------------------------------------------------------------------------
    trainer = Trainer(
        logger=wandb_logger,
        max_epochs=args.max_epochs,
        accelerator="auto",
        devices="auto",
        gradient_clip_val=1.0,  # Clip gradients with norm > 1.0
        gradient_clip_algorithm='norm',
        precision=16,  # Enable mixed precision
        log_every_n_steps=10,  # Adjust as needed
        callbacks=[early_stop_callback, checkpoint_callback],
        enable_progress_bar=True
    )

    # -------------------------------------------------------------------------
    # 7) Fit the model
    # -------------------------------------------------------------------------
    trainer.fit(model, datamodule=data_module)
    if args.model_type == "time_to_event":
        trainer.model.compute_baseline_hazards(data_module.train_dataloader())

    # -------------------------------------------------------------------------
    # 8) Test the model
    # -------------------------------------------------------------------------
    trainer.test(model, datamodule=data_module)

    # -------------------------------------------------------------------------
    # 9) Save the Model (Optional)
    # -------------------------------------------------------------------------
    # Save the best checkpoint
    best_model_path = checkpoint_callback.best_model_path
    print(f"Best model saved at: {best_model_path}")
    wandb.save(best_model_path)

    # -------------------------------------------------------------------------
    # 10) Plot Survival Functions (Only for time_to_event)
    # -------------------------------------------------------------------------
    if args.model_type == "time_to_event":
        import matplotlib.pyplot as plt

        # Access the survival functions from the model
        surv = model.surv_funcs
        if surv is not None:
            # Plot the first 5 survival functions
            surv.iloc[:5].plot()
            plt.ylabel('S(t | x)')
            plt.xlabel('Time')
            plt.title('Survival Functions for First 5 Test Samples')
            plt.savefig("survival_functions.png")
            plt.close()
        else:
            print("Survival functions were not computed.")

if __name__ == "__main__":
    main()
