import argparse
import os
import random
from collections import defaultdict

import pandas as pd
import numpy as np

import pydicom
import torch
import torch.nn as nn
import torchvision.transforms as T

import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from torch.utils.data import Dataset, DataLoader, default_collate

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import torchtuples as tt
from pycox.models import CoxPH
from pycox.evaluation import EvalSurv

import matplotlib.pyplot as plt
from tqdm import tqdm

# If you plan to compare with lifelines (traditional CoxPH):
try:
    from lifelines import CoxPHFitter
    from lifelines.utils import concordance_index
    HAS_LIFELINES = True
except ImportError:
    HAS_LIFELINES = False

# Global debug flag
DEBUG = True

# ======================================================
# Debug hook: Print a message if a module outputs NaNs.
# ======================================================
def nan_hook(module, input, output):
    if torch.isnan(output).any() and DEBUG:
        print(f"[DEBUG] NaN detected in output of {module}")

# -----------------------------
# Data Modules and Dataset
# -----------------------------
class HCCDataModule(pl.LightningDataModule):
    def __init__(
        self,
        csv_file,
        dicom_root,
        model_type="linear",
        batch_size=2,
        num_slices=20,
        num_workers=2,
        preprocessed_root=None,
        num_samples_per_patient=1
    ):
        super().__init__()
        self.csv_file = csv_file
        self.dicom_root = dicom_root
        self.batch_size = batch_size
        self.model_type = model_type
        self.num_slices = num_slices
        self.num_workers = num_workers
        self.preprocessed_root = preprocessed_root
        self.num_samples_per_patient = num_samples_per_patient

        # Basic preprocessing transforms (resize + normalization) for 224x224
        self.preprocess_transform = T.Compose([
            T.Resize((224, 224), antialias=True),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.transform = self.preprocess_transform

    def setup(self, stage=None):
        df = pd.read_csv(self.csv_file)
        stratify_col = df['recurrence post tx'] if self.model_type == "linear" else df['event']

        train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42, stratify=stratify_col)
        val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df['event'])

        print(f"Train: {len(train_df)} patients, Val: {len(val_df)} patients, Test: {len(test_df)} patients")
        print(f"Label distribution in Train:\n{train_df['recurrence post tx'].value_counts()}")
        print(f"Label distribution in Val:\n{val_df['recurrence post tx'].value_counts()}")
        print(f"Label distribution in Test:\n{test_df['recurrence post tx'].value_counts()}")

        self.train_dataset = HCCDicomDataset(
            csv_file=train_df,
            dicom_root=self.dicom_root,
            model_type=self.model_type,
            transform=self.preprocess_transform,
            num_slices=self.num_slices,
            preprocessed_root=self.preprocessed_root,
            num_samples_per_patient=self.num_samples_per_patient
        )
        self.val_dataset = HCCDicomDataset(
            csv_file=val_df,
            dicom_root=self.dicom_root,
            model_type=self.model_type,
            transform=self.preprocess_transform,
            num_slices=self.num_slices,
            preprocessed_root=self.preprocessed_root,
            num_samples_per_patient=self.num_samples_per_patient
        )
        self.test_dataset = HCCDicomDataset(
            csv_file=test_df,
            dicom_root=self.dicom_root,
            model_type=self.model_type,
            transform=self.preprocess_transform,
            num_slices=self.num_slices,
            preprocessed_root=self.preprocessed_root,
            num_samples_per_patient=self.num_samples_per_patient
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True, 
            num_workers=self.num_workers,
            collate_fn=self.train_collate_fn
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=self.num_workers,
            collate_fn=self.collate_fn
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=self.num_workers,
            collate_fn=self.collate_fn
        )

    def train_collate_fn(self, batch):
        # This ensures some events exist in each training batch (for time_to_event).
        if self.model_type == "time_to_event":
            events = [sample[2].item() for sample in batch]
            # Force at least 2 events in the batch if possible.
            if sum(events) < 2:
                indices = random.sample(range(len(batch)), k=2)
                for idx in indices:
                    x, t, e = batch[idx]
                    batch[idx] = (x, t, torch.tensor(1.0, dtype=torch.float32))
            if DEBUG:
                print(f"[DEBUG] train_collate_fn - Batch size: {len(batch)}, events sum: {sum([s[2].item() for s in batch])}")
            return default_collate(batch)
        else:
            return default_collate(batch)

    def collate_fn(self, batch):
        # Similar to train_collate_fn but for val/test sets
        if self.model_type == "time_to_event":
            events = [sample[2].item() for sample in batch]
            if sum(events) == 0 and len(batch) > 0:
                idx = random.randint(0, len(batch)-1)
                x, t, e = batch[idx]
                batch[idx] = (x, t, torch.tensor(1.0, dtype=torch.float32))
            if DEBUG:
                print(f"[DEBUG] collate_fn - Batch size: {len(batch)}, events sum: {sum([s[2].item() for s in batch])}")
            return default_collate(batch)
        else:
            return default_collate(batch)

class HCCDicomDataset(Dataset):
    def __init__(
        self,
        csv_file,
        dicom_root,
        model_type="linear",
        transform=None,
        num_slices=10,
        preprocessed_root=None,
        num_samples_per_patient=1
    ):
        self.dicom_root = dicom_root
        self.transform = transform
        self.model_type = model_type
        self.num_slices = num_slices
        self.preprocessed_root = preprocessed_root
        self.num_samples_per_patient = num_samples_per_patient
        self.patient_data = []
        
        if isinstance(csv_file, str):
            df = pd.read_csv(csv_file)
        else:
            df = csv_file.copy()
        for _, row in df.iterrows():
            patient_id = str(row['Pre op MRI Accession number'])
            dicom_dir = os.path.join(dicom_root, patient_id)
            if os.path.exists(dicom_dir):
                data_entry = {'patient_id': patient_id}
                if self.model_type == "linear":
                    data_entry['label'] = row['recurrence post tx']
                elif self.model_type == "time_to_event":
                    data_entry['time'] = row['time']
                    data_entry['event'] = row['event']
                data_entry['dicom_dir'] = dicom_dir
                self.patient_data.append(data_entry)
        if DEBUG:
            print(f"[DEBUG] HCCDicomDataset initialized with {len(self.patient_data)} patients, "
                  f"each yielding {self.num_samples_per_patient} sample(s).")

    def __len__(self):
        return len(self.patient_data) * self.num_samples_per_patient

    def __getitem__(self, idx):
        patient_idx = idx // self.num_samples_per_patient
        patient = self.patient_data[patient_idx]
        dicom_dir = patient['dicom_dir']
        image_stack = self.load_axial_series(dicom_dir)

        if DEBUG:
            print(f"[DEBUG] __getitem__ patient {patient['patient_id']} - image_stack shape after loading: {image_stack.shape}")

        # If we do not have preprocessed images, apply transformation slice by slice.
        if self.preprocessed_root is None and self.transform:
            transformed_images = []
            for img in image_stack:
                img = self.transform(img)
                transformed_images.append(img)
            image_stack = torch.stack(transformed_images)
        else:
            if DEBUG:
                print(f"[DEBUG] __getitem__ patient {patient['patient_id']} - using preprocessed images (no re-transform).")

        if self.model_type == "linear":
            return image_stack, torch.tensor(patient['label'], dtype=torch.float32)
        elif self.model_type == "time_to_event":
            return (
                image_stack,
                torch.tensor(patient['time'], dtype=torch.float32),
                torch.tensor(patient['event'], dtype=torch.float32)
            )
        
    def load_axial_series(self, dicom_dir):
        patient_id = os.path.basename(dicom_dir)
        if self.preprocessed_root:
            preprocessed_path = os.path.join(self.preprocessed_root, f"{patient_id}.pt")
            if os.path.exists(preprocessed_path):
                if DEBUG:
                    print(f"[DEBUG] Loading preprocessed tensor for patient {patient_id} from {preprocessed_path}")
                image_stack = torch.load(preprocessed_path)
                return self.process_loaded_stack(image_stack)
        
        series_dict = defaultdict(list)
        dcm_files = [f for f in os.listdir(dicom_dir) if f.endswith('.dcm')]
        
        for fname in dcm_files:
            try:
                dcm_path = os.path.join(dicom_dir, fname)
                dcm = pydicom.dcmread(dcm_path, stop_before_pixels=True)
                if not hasattr(dcm, 'ImageOrientationPatient') or not hasattr(dcm, 'ImagePositionPatient'):
                    continue
                orientation = np.array(dcm.ImageOrientationPatient, dtype=np.float32).round(4)
                orientation_tuple = tuple(orientation.flatten())
                series_dict[orientation_tuple].append(dcm_path)
            except Exception as e:
                print(f"Error reading DICOM file {fname}: {e}")
                continue

        # Select the axial series
        axial_series = []
        for orient, dcm_paths in series_dict.items():
            if self.is_axial(orient):
                axial_series.extend(dcm_paths)

        if not axial_series:
            raise ValueError(f"No axial series found in {dicom_dir}")

        def get_slice_position(dcm_path):
            try:
                dcm = pydicom.dcmread(dcm_path, stop_before_pixels=True)
                return float(dcm.SliceLocation) if hasattr(dcm, 'SliceLocation') else float(dcm.ImagePositionPatient[2])
            except Exception:
                return 0.0

        # Sort by slice position
        axial_series.sort(key=get_slice_position)

        if DEBUG:
            print(f"[DEBUG] load_axial_series patient {patient_id}: {len(axial_series)} axial slices found.")

        image_stack = []
        for dcm_path in axial_series:
            try:
                dcm = pydicom.dcmread(dcm_path)
                img = self.dicom_to_tensor(dcm)
                # If we do NOT have a preprocessed root, we might do transform later in __getitem__.
                image_stack.append(img)
            except Exception as e:
                print(f"Error processing DICOM file {dcm_path}: {e}")
                continue

        if image_stack:
            image_stack = torch.stack(image_stack)
        else:
            image_stack = torch.zeros((0, 3, 224, 224))

        if DEBUG:
            print(f"[DEBUG] load_axial_series patient {patient_id} - image_stack shape before saving: {image_stack.shape}")

        if self.preprocessed_root:
            os.makedirs(self.preprocessed_root, exist_ok=True)
            preprocessed_path = os.path.join(self.preprocessed_root, f"{patient_id}.pt")
            torch.save(image_stack, preprocessed_path)
            if DEBUG:
                print(f"[DEBUG] Saved preprocessed tensor for patient {patient_id} to {preprocessed_path}")

        return self.process_loaded_stack(image_stack)

    def process_loaded_stack(self, image_stack):
        current_slices = image_stack.size(0)
        if DEBUG:
            print(f"[DEBUG] process_loaded_stack - current slices: {current_slices}, target: {self.num_slices}")
        if current_slices < self.num_slices:
            # Pad with zeros
            pad_size = self.num_slices - current_slices
            padding = torch.zeros((pad_size, *image_stack.shape[1:]), dtype=image_stack.dtype)
            image_stack = torch.cat([image_stack, padding], dim=0)
            if DEBUG:
                print(f"[DEBUG] Padded image_stack to shape: {image_stack.shape}")
        else:
            # Randomly select self.num_slices slices
            selected_indices = torch.randperm(current_slices)[:self.num_slices]
            selected_indices, _ = torch.sort(selected_indices)
            image_stack = image_stack[selected_indices]
            if DEBUG:
                print(f"[DEBUG] Randomly selected slices. New shape: {image_stack.shape}")
        return image_stack

    def is_axial(self, orientation):
        row_dir = np.array(orientation[:3], dtype=np.float32)
        col_dir = np.array(orientation[3:], dtype=np.float32)
        cross = np.cross(row_dir, col_dir)
        norm = np.linalg.norm(cross)
        if norm < 1e-6:
            return False
        cross_normalized = cross / norm
        # Check if the cross product is close to the Z-axis
        return (
            (abs(cross_normalized[0]) < 5e-2) and
            (abs(cross_normalized[1]) < 5e-2) and
            (abs(abs(cross_normalized[2]) - 1.0) < 5e-2)
        )

    def dicom_to_tensor(self, dcm):
        img = dcm.pixel_array.astype(np.float32)
        if dcm.Modality == 'CT':
            intercept = float(dcm.RescaleIntercept)
            slope = float(dcm.RescaleSlope)
            img = slope * img + intercept
            img = np.clip(img, -100, 400)
            img = (img + 100) / 500.0
        else:
            # Simple min-max normalization for MRI (for example)
            img = (img - img.min()) / (img.max() - img.min() + 1e-6)
            img = img * 255.0
        # Convert to 3-channel
        if img.ndim == 2:
            img = np.repeat(img[..., np.newaxis], 3, axis=-1)
        tensor_img = torch.from_numpy(img).permute(2, 0, 1)  # CHW format
        if DEBUG:
            print(f"[DEBUG] dicom_to_tensor - tensor shape: {tensor_img.shape}, min={tensor_img.min()}, max={tensor_img.max()}")
        return tensor_img

# -----------------------------
# DINOv2 Model Loader & Wrapper
# -----------------------------
class DinoV2Wrapper(nn.Module):
    """
    A wrapper around the DINOv2 model that returns one feature vector per image.
    Here we average the patch tokens (excluding the [CLS] token) to obtain the image feature.
    """
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model

    def forward(self, x):
        return self.base_model(x)

    def forward_features(self, x):
        token_embeddings = self.forward(x)
        if token_embeddings.ndim == 3 and token_embeddings.size(1) > 1:
            # Exclude the [CLS] token and average the remaining patch tokens.
            patch_tokens = token_embeddings[:, 1:, :]
            features = patch_tokens.mean(dim=1)  # shape: (B, embed_dim)
            return features
        elif token_embeddings.ndim == 2:
            return token_embeddings
        else:
            raise ValueError(f"Unexpected token_embeddings shape: {token_embeddings.shape}")


def load_dinov2_model(local_weights_path):
    """
    Loads DINOv2 with a local state dict. 
    You must have a local copy of the DINOv2 repo so that the torch.hub call 
    can load the architecture from 'facebookresearch/dinov2' with source='local'.
    If you have the entire .pth or .pt file as a state_dict, load it with the model.
    """
    # Example: loading the ViT-Base/14 architecture from local hub
    model_arch = "dinov2_vitb14"  # or whichever variant you have weights for
    print(f"Loading {model_arch} from local hub...")

    # 1. Create the model structure
    base_model = torch.hub.load(
        './dinov2',
        model_arch,
        source='local'  # or 'github' if you have a local GitHub clone
    )

    # 2. Load the local weights
    checkpoint = torch.load(local_weights_path, map_location='cpu')
    if isinstance(checkpoint, dict) and 'model' in checkpoint:
        # Some weights are saved in a dict with 'model' key
        base_model.load_state_dict(checkpoint['teacher']['model'])

    print("Local DINOv2 weights loaded successfully.")
    # Wrap it so we have .forward_features method
    return DinoV2Wrapper(base_model)

# -----------------------------
# Custom MLP with L1 Regularization
# -----------------------------
class CustomMLP(nn.Module):
    def __init__(self, in_features, num_nodes, out_features, dropout=0.0, l1_lambda=0.0):
        """
        Args:
            in_features (int): Number of input features.
            num_nodes (list of int): List with the number of nodes in each hidden layer.
            out_features (int): Number of output features.
            dropout (float): Dropout rate.
            l1_lambda (float): Coefficient for L1 regularization.
        """
        super().__init__()
        self.l1_lambda = l1_lambda

        layers = []
        prev_features = in_features
        for nodes in num_nodes:
            layers.append(nn.Linear(prev_features, nodes))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_features = nodes

        layers.append(nn.Linear(prev_features, out_features))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

    def l1_regularization(self):
        """
        Computes the L1 regularization term for all linear layers.
        """
        l1_loss = 0.0
        for module in self.net:
            if isinstance(module, nn.Linear):
                l1_loss += torch.sum(torch.abs(module.weight))
        return self.l1_lambda * l1_loss

# -----------------------------
# Optional Risk Score Centering Wrapper
# -----------------------------
class CenteredModel(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        out = self.net(x)
        if DEBUG:
            print(f"[DEBUG] CenteredModel.forward - raw risk scores: "
                  f"mean={out.mean().item():.4f}, std={out.std().item():.4f}, "
                  f"min={out.min().item():.4f}, max={out.max().item():.4f}")
        # Center risk scores for numerical stability
        return out - out.mean(dim=0, keepdim=True)

# -----------------------------
# Custom Gradient Clipping Callback
# -----------------------------
class GradientClippingCallback(tt.callbacks.Callback):
    def __init__(self, clip_value):
        super().__init__()
        self.clip_value = clip_value

    def on_batch_end(self):
        torch.nn.utils.clip_grad_norm_(self.model.net.parameters(), self.clip_value)
        total_norm = 0.0
        for p in self.model.net.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        if DEBUG:
            print(f"[DEBUG] GradientClippingCallback - Batch gradient norm: {total_norm:.4f}")

# -----------------------------
# Callback to Log Loss and Check Parameters
# -----------------------------
class LossLogger(tt.callbacks.Callback):
    def __init__(self):
        self.epoch_losses = []

    def on_epoch_end(self):
        log_df = self.model.log.to_pandas()
        if not log_df.empty:
            current_loss = log_df.iloc[-1]['train_loss']
            print(f"Epoch {len(self.epoch_losses)} - Train Loss: {current_loss:.4f}")
            self.epoch_losses.append(current_loss)
            # Compute gradient norm
            total_norm = 0.0
            for p in self.model.net.parameters():
                if p.grad is not None:
                    total_norm += p.grad.data.norm(2).item() ** 2
            total_norm = total_norm ** 0.5
            print(f"Epoch {len(self.epoch_losses)} - Gradient Norm: {total_norm:.4f}")

            # Forward pass on a small sample if available
            if hasattr(self.model, "x_train_std") and self.model.x_train_std is not None:
                sample_input = torch.tensor(self.model.x_train_std[:5]).to(next(self.model.net.parameters()).device)
                with torch.no_grad():
                    risk_scores = self.model.net(sample_input)
                print(f"[DEBUG] Sample risk scores: "
                      f"mean={risk_scores.mean().item():.4f}, std={risk_scores.std().item():.4f}, "
                      f"min={risk_scores.min().item():.4f}, max={risk_scores.max().item():.4f}")

# -----------------------------
# Callback to Check Model Parameters for NaNs
# -----------------------------
class ParamCheckerCallback(tt.callbacks.Callback):
    def on_epoch_end(self):
        for name, param in self.model.net.named_parameters():
            if torch.isnan(param).any():
                print(f"[DEBUG] Parameter {name} contains NaNs.")

# -----------------------------
# Subclassed CoxPH Model to Include L1 Regularization in Loss
# -----------------------------
class CoxPHWithL1(CoxPH):
    def loss(self, preds, durations, events):
        base_loss = super().loss(preds, durations, events)
        # If the network has the l1_regularization method, add its value to the loss
        reg_loss = self.net.l1_regularization() if hasattr(self.net, 'l1_regularization') else 0.0
        return base_loss + reg_loss

# -----------------------------
# Feature Extraction
# -----------------------------
def extract_features(data_loader, model, device):
    """
    Extract features using the DINOv2 model which now returns one feature vector per image.
    Patient-level features are computed by averaging the per-slice features.
    """
    model.eval()
    features = []
    durations = []
    events = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Extracting Features"):
            # For time-to-event, batch is (images, t, e)
            images, t, e = batch
            images = images.to(device)
            batch_size, num_slices, C, H, W = images.size()
            # Reshape so that each slice is an independent image.
            images = images.view(batch_size * num_slices, C, H, W)
            
            # Get one feature vector per image.
            feats = model.forward_features(images)  # Expected shape: (batch_size*num_slices, embed_dim)
            feature_dim = feats.size(-1)
            # Reshape back to (batch_size, num_slices, embed_dim)
            feats = feats.view(batch_size, num_slices, feature_dim)
            # Average the features across slices to get a patient-level feature.
            feats = feats.mean(dim=1)
            
            features.append(feats.cpu().numpy())
            durations.append(t.cpu().numpy())
            events.append(e.cpu().numpy())

    features = np.concatenate(features, axis=0)
    durations = np.concatenate(durations, axis=0)
    events = np.concatenate(events, axis=0)
    return features, durations, events

# -----------------------------
# Main Training and Evaluation Function
# -----------------------------
def main(args):
    seed_everything(42, workers=True)

    # Initialize DataModule
    data_module = HCCDataModule(
        csv_file=args.csv_file,
        dicom_root=args.dicom_root,
        model_type="time_to_event",
        batch_size=args.batch_size,
        num_slices=args.num_slices,
        num_workers=args.num_workers,
        preprocessed_root=args.preprocessed_root,
        num_samples_per_patient=args.num_samples_per_patient
    )
    data_module.setup()

    # 1) Load DINOv2 from local weights
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dino_model = load_dinov2_model(args.dinov2_weights)
    dino_model = dino_model.to(device)
    dino_model.eval()

    # Register a forward hook to catch NaNs in the DINOv2 model
    dino_model.register_forward_hook(nan_hook)

    # Create DataLoaders
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()
    test_loader = data_module.test_dataloader()

    # Extract features from DINOv2
    print("Extracting train features...")
    x_train, y_train_durations, y_train_events = extract_features(train_loader, dino_model, device)
    print("Extracting validation features...")
    x_val, y_val_durations, y_val_events = extract_features(val_loader, dino_model, device)
    print("Extracting test features...")
    x_test, y_test_durations, y_test_events = extract_features(test_loader, dino_model, device)

    # Sanity checks
    print("Checking for NaNs in features...")
    print("x_train contains NaNs:", np.isnan(x_train).any())
    print("x_val contains NaNs:", np.isnan(x_val).any())
    print("x_test contains NaNs:", np.isnan(x_test).any())

    # Remove zero-variance features if any
    variances = np.var(x_train, axis=0)
    if np.any(variances == 0):
        zero_var_features = np.where(variances == 0)[0]
        print(f"Warning: Features with zero variance: {zero_var_features}")
        x_train = x_train[:, variances != 0]
        x_val = x_val[:, variances != 0]
        x_test = x_test[:, variances != 0]

    # Prepare survival labels
    y_train = (y_train_durations, y_train_events)
    y_val = (y_val_durations, y_val_events)
    y_test = (y_test_durations, y_test_events)

    # Standardize features
    x_mapper = StandardScaler()
    x_train_std = x_mapper.fit_transform(x_train).astype('float32')
    x_val_std = x_mapper.transform(x_val).astype('float32')
    x_test_std = x_mapper.transform(x_test).astype('float32')

    print("Feature value ranges (train):")
    print(f"  Min: {x_train_std.min()}, Max: {x_train_std.max()}")
    print(f"  Mean abs: {np.mean(np.abs(x_train_std))}, Std: {np.std(x_train_std)}")

    # Quick correlation check
    corr_matrix = pd.DataFrame(x_train_std).corr()
    print("Max absolute correlation pairs (top 10):")
    print(corr_matrix.abs().unstack().sort_values(ascending=False).head(100))

    # Basic validity check on survival data
    def validate_survival_data(durations, events):
        sort_idx = np.argsort(durations)
        sorted_durations = durations[sort_idx]
        sorted_events = events[sort_idx]
        for i in range(len(sorted_durations)):
            if sorted_events[i] == 1:
                current_time = sorted_durations[i]
                num_at_risk = np.sum(sorted_durations >= current_time)
                if num_at_risk == 0:
                    raise ValueError(f"Event at {current_time} has no at-risk individuals.")
                elif num_at_risk == 1:
                    print(f"Warning: Event at {current_time} has only 1 at-risk.")
    validate_survival_data(y_train_durations, y_train_events)

    # If you choose to do "traditional" lifelines CoxPH:
    if args.coxph_method == 'traditional':
        if not HAS_LIFELINES:
            raise ImportError("lifelines is not installed; cannot run traditional CoxPH.")
        feature_names = [f'feat_{i}' for i in range(x_train_std.shape[1])]

        train_df = pd.DataFrame(x_train_std, columns=feature_names)
        train_df['time'] = y_train_durations
        train_df['event'] = y_train_events

        cph = CoxPHFitter(penalizer=0.1)
        cph.fit(train_df, duration_col='time', event_col='event')
        print(cph.summary)

        # Validation
        val_df = pd.DataFrame(x_val_std, columns=feature_names)
        val_df['time'] = y_val_durations
        val_df['event'] = y_val_events
        val_pred = cph.predict_partial_hazard(val_df)
        c_index_val = concordance_index(val_df['time'], -val_pred, val_df['event'])
        print(f"Validation Concordance Index: {c_index_val}")

        # Test
        test_df = pd.DataFrame(x_test_std, columns=feature_names)
        test_df['time'] = y_test_durations
        test_df['event'] = y_test_events
        test_pred = cph.predict_partial_hazard(test_df)
        c_index_test = concordance_index(test_df['time'], -test_pred, test_df['event'])
        print(f"Test Concordance Index: {c_index_test}")

        # Plot baseline survival function
        surv_func = cph.baseline_survival_
        plt.figure()
        surv_func.plot()
        plt.title("Baseline Survival Function")
        plt.savefig(os.path.join(args.output_dir, "baseline_survival.png"))
        plt.close()

        # Plot survival functions for a few test samples
        plt.figure()
        for i in range(min(5, test_df.shape[0])):
            sample = test_df.iloc[[i]]
            surv_pred = cph.predict_survival_function(sample)
            plt.step(surv_pred.index, surv_pred.values.flatten(), where='post', label=f"Sample {i}")
        plt.xlabel("Time")
        plt.ylabel("Survival probability")
        plt.legend()
        plt.title("Survival Functions for Test Samples")
        plt.savefig(os.path.join(args.output_dir, "test_survival_functions.png"))
        plt.close()
        return

    # -----------------------------
    # pycox-based neural CoxPH with L1 Regularization
    # -----------------------------
    in_features = x_train_std.shape[1]
    print("Input feature dimension:", in_features)
    out_features = 1

    # Build the net (either MLP or single linear layer)
    if args.coxph_net == 'mlp':
        # Use the custom MLP with L1 regularization.
        num_nodes = [2048, 2048]
        net = CustomMLP(
            in_features,
            num_nodes,
            out_features,
            dropout=args.dropout,
            l1_lambda=args.l1_lambda  # regularization strength provided via command line
        )
    elif args.coxph_net == 'linear':
        net = nn.Linear(in_features, out_features, bias=False)
    else:
        raise ValueError("Unknown coxph_net option. Choose 'mlp' or 'linear'.")

    if args.center_risk:
        net = CenteredModel(net)

    net.register_forward_hook(nan_hook)

    # Use our subclassed CoxPH model that adds L1 regularization
    model = CoxPHWithL1(net, tt.optim.Adam)
    model.optimizer.set_lr(args.learning_rate)
    model.optimizer.param_groups[0]['weight_decay'] = 1e-4

    # Save standardized train features for debugging/logging
    model.x_train_std = x_train_std

    callbacks = [tt.callbacks.EarlyStopping(), LossLogger(), ParamCheckerCallback()]
    if args.gradient_clip > 0:
        callbacks.insert(1, GradientClippingCallback(args.gradient_clip))
    
    verbose = True
    batch_size = args.batch_size

    print("Training the CoxPH model...")
    log = model.fit(
        x_train_std,
        (y_train_durations, y_train_events),
        batch_size,
        args.epochs,
        callbacks,
        verbose,
        val_data=(x_val_std, (y_val_durations, y_val_events)),
        val_batch_size=batch_size
    )

    # Plot training log
    plt.figure()
    log.plot()
    plt.title("Training Log (Loss)")
    plt.savefig(os.path.join(args.output_dir, "training_log.png"))
    plt.close()

    partial_ll = model.partial_log_likelihood(x_val_std, (y_val_durations, y_val_events)).mean()
    print(f"Partial Log-Likelihood on Validation Set: {partial_ll}")

    # Compute baseline hazards and evaluate
    model.compute_baseline_hazards()
    surv = model.predict_surv_df(x_test_std)
    plt.figure()
    surv.iloc[:, :5].plot()
    plt.ylabel("S(t | x)")
    plt.xlabel("Time")
    plt.title("Survival Functions for Test Samples")
    plt.savefig(os.path.join(args.output_dir, "survival_functions.png"))
    plt.close()

    ev = EvalSurv(surv, y_test_durations, y_test_events, censor_surv='km')
    concordance = ev.concordance_td()
    print(f"Concordance Index: {concordance}")

    time_grid = np.linspace(y_test_durations.min(), y_test_durations.max(), 100)
    brier_score = ev.brier_score(time_grid)
    plt.figure()
    brier_score.plot()
    plt.title("Brier Score Over Time")
    plt.xlabel("Time")
    plt.ylabel("Brier Score")
    plt.savefig(os.path.join(args.output_dir, "brier_score.png"))
    plt.close()

    print(f"Integrated Brier Score: {ev.integrated_brier_score(time_grid)}")
    print(f"Integrated NBLL: {ev.integrated_nbll(time_grid)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CoxPH model with DINOv2 features (local weights) and custom MLP with L1 regularization")

    parser.add_argument("--dicom_root", type=str, default="/gpfs/data/mankowskilab/HCC_Recurrence/dicom",
                        help="Path to the root DICOM directory.")
    parser.add_argument("--csv_file", type=str, default="processed_patient_labels.csv",
                        help="Path to processed CSV with columns [Pre op MRI Accession number, recurrence post tx, time, event].")
    parser.add_argument('--preprocessed_root', type=str, default=None, 
                        help='Directory to store/load preprocessed image tensors')
    parser.add_argument('--batch_size', type=int, default=32, 
                        help='Batch size for data loaders')
    parser.add_argument('--num_slices', type=int, default=2, 
                        help='Number of slices per patient')
    parser.add_argument('--num_workers', type=int, default=4, 
                        help='Number of workers for data loaders')
    parser.add_argument('--epochs', type=int, default=20, 
                        help='Number of training epochs')
    parser.add_argument('--output_dir', type=str, default='checkpoints', 
                        help='Directory to save outputs and models')
    parser.add_argument('--learning_rate', type=float, default=1e-6, 
                        help='Learning rate for the CoxPH model')
    parser.add_argument('--gradient_clip', type=float, default=0, 
                        help='Gradient clipping threshold. Set 0 to disable.')
    parser.add_argument('--center_risk', action='store_true', 
                        help='If set, center risk scores for numerical stability')
    parser.add_argument('--dropout', type=float, default=0.0, 
                        help='Dropout rate for the MLP if used')
    parser.add_argument('--num_samples_per_patient', type=int, default=1,
                        help='Number of times to sample from each patient per epoch')
    parser.add_argument('--coxph_net', type=str, default='mlp', choices=['mlp', 'linear'],
                        help='Type of network for pycox survival regression.')
    parser.add_argument('--coxph_method', type=str, default='pycox', choices=['pycox', 'traditional'],
                        help='Regression method: "pycox" or "traditional" (lifelines) for CoxPH.')
    parser.add_argument('--dinov2_weights', type=str, required=True,
                        help="Path to your local DINOv2 state dict file (.pth or .pt).")
    parser.add_argument('--l1_lambda', type=float, default=1e-5,
                        help="L1 regularization strength for the custom MLP.")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    main(args)
