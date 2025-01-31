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
import timm
from pycox.models import CoxPH
from pycox.evaluation import EvalSurv
import torchtuples as tt
import matplotlib.pyplot as plt
from tqdm import tqdm

# -----------------------------
# Data Modules and Dataset
# -----------------------------
class HCCDataModule(pl.LightningDataModule):
    def __init__(self, csv_file, dicom_root, model_type="linear", batch_size=2, num_slices=20, num_workers=2, preprocessed_root=None):
        super().__init__()
        self.csv_file = csv_file
        self.dicom_root = dicom_root
        self.batch_size = batch_size
        self.model_type = model_type
        self.num_slices = num_slices
        self.num_workers = num_workers
        self.preprocessed_root = preprocessed_root

        # Basic preprocessing transforms (resize + normalization)
        self.preprocess_transform = T.Compose([
            T.Resize((518, 518), antialias=True),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        # Runtime transforms can be extended as needed.
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
            csv_file=train_df, dicom_root=self.dicom_root, model_type=self.model_type,
            transform=self.preprocess_transform, num_slices=self.num_slices,
            preprocessed_root=self.preprocessed_root
        )
        self.val_dataset = HCCDicomDataset(
            csv_file=val_df, dicom_root=self.dicom_root, model_type=self.model_type,
            transform=self.preprocess_transform, num_slices=self.num_slices,
            preprocessed_root=self.preprocessed_root
        )
        self.test_dataset = HCCDicomDataset(
            csv_file=test_df, dicom_root=self.dicom_root, model_type=self.model_type,
            transform=self.preprocess_transform, num_slices=self.num_slices,
            preprocessed_root=self.preprocessed_root
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
        if self.model_type == "time_to_event":
            events = [sample[2].item() for sample in batch]
            if sum(events) < 2:  # Ensure at least 2 events per batch
                indices = random.sample(range(len(batch)), k=2)
                for idx in indices:
                    x, t, e = batch[idx]
                    batch[idx] = (x, t, torch.tensor(1.0, dtype=torch.float32))
            return default_collate(batch)
        else:
            return default_collate(batch)

    def collate_fn(self, batch):
        if self.model_type == "time_to_event":
            events = [sample[2].item() for sample in batch]
            if sum(events) == 0 and len(batch) > 0:
                idx = random.randint(0, len(batch)-1)
                x, t, e = batch[idx]
                batch[idx] = (x, t, torch.tensor(1.0, dtype=torch.float32))
            return default_collate(batch)
        else:
            return default_collate(batch)

class HCCDicomDataset(Dataset):
    def __init__(self, csv_file, dicom_root, model_type="linear", transform=None, num_slices=10, preprocessed_root=None):
        self.dicom_root = dicom_root
        self.transform = transform
        self.model_type = model_type
        self.num_slices = num_slices
        self.preprocessed_root = preprocessed_root
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

    def __len__(self):
        return len(self.patient_data)

    def __getitem__(self, idx):
        patient = self.patient_data[idx]
        dicom_dir = patient['dicom_dir']
        image_stack = self.load_axial_series(dicom_dir)
        
        if self.transform:
            transformed_images = []
            for img in image_stack:
                img = self.transform(img)
                transformed_images.append(img)
            image_stack = torch.stack(transformed_images)
        
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

        axial_series.sort(key=get_slice_position)

        image_stack = []
        for dcm_path in axial_series:
            try:
                dcm = pydicom.dcmread(dcm_path)
                img = self.dicom_to_tensor(dcm)
                if self.transform:
                    img = self.transform(img)
                image_stack.append(img)
            except Exception as e:
                print(f"Error processing DICOM file {dcm_path}: {e}")
                continue

        image_stack = torch.stack(image_stack) if image_stack else torch.zeros((0, 3, 224, 224))
        
        if self.preprocessed_root:
            os.makedirs(self.preprocessed_root, exist_ok=True)
            preprocessed_path = os.path.join(self.preprocessed_root, f"{patient_id}.pt")
            torch.save(image_stack, preprocessed_path)
        
        return self.process_loaded_stack(image_stack)

    def process_loaded_stack(self, image_stack):
        current_slices = image_stack.size(0)
        if current_slices < self.num_slices:
            pad_size = self.num_slices - current_slices
            padding = torch.zeros((pad_size, *image_stack.shape[1:]), dtype=image_stack.dtype)
            image_stack = torch.cat([image_stack, padding], dim=0)
        else:
            selected_indices = torch.randperm(current_slices)[:self.num_slices]
            selected_indices, _ = torch.sort(selected_indices)
            image_stack = image_stack[selected_indices]
        return image_stack

    def is_axial(self, orientation):
        row_dir = np.array(orientation[:3], dtype=np.float32)
        col_dir = np.array(orientation[3:], dtype=np.float32)
        cross = np.cross(row_dir, col_dir)
        norm = np.linalg.norm(cross)
        if norm < 1e-6:
            return False
        cross_normalized = cross / norm
        return (np.abs(cross_normalized[0]) < 5e-2 and 
                np.abs(cross_normalized[1]) < 5e-2 and 
                np.abs(np.abs(cross_normalized[2]) - 1.0) < 5e-2)

    def dicom_to_tensor(self, dcm):
        img = dcm.pixel_array.astype(np.float32)
        if dcm.Modality == 'CT':
            intercept = float(dcm.RescaleIntercept)
            slope = float(dcm.RescaleSlope)
            img = slope * img + intercept
            img = np.clip(img, -100, 400)
            img = (img + 100) / 500.0
        else:
            img = (img - img.min()) / (img.max() - img.min())
            img = img * 255.0
        if img.ndim == 2:
            img = np.repeat(img[..., np.newaxis], 3, axis=-1)
        return torch.from_numpy(img).permute(2, 0, 1)  # CHW format

# -----------------------------
# Feature Extraction
# -----------------------------
def extract_features(data_loader, model, device):
    """
    Extract features using the DINO model and aggregate slice-level features 
    into a patient-level representation.
    """
    model.eval()
    features = []
    durations = []
    events = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Extracting Features"):
            images, t, e = batch  # images: (batch_size, num_slices, 3, 224, 224)
            images = images.to(device)
            batch_size, num_slices, C, H, W = images.size()
            images = images.view(batch_size * num_slices, C, H, W)
            feats = model(images)  # (batch_size*num_slices, feature_dim)
            feats = feats.view(batch_size, num_slices, -1)
            feats = feats.mean(dim=1)  # Average slice features per patient
            features.append(feats.cpu().numpy())
            durations.append(t.cpu().numpy())
            events.append(e.cpu().numpy())
    features = np.concatenate(features, axis=0)
    durations = np.concatenate(durations, axis=0)
    events = np.concatenate(events, axis=0)
    return features, durations, events

# -----------------------------
# Optional Risk Score Centering Wrapper
# -----------------------------
class CenteredModel(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        out = self.net(x)
        # Center risk scores to help numerical stability
        return out - out.mean(dim=0, keepdim=True)

# -----------------------------
# Custom Gradient Clipping Callback
# -----------------------------
class GradientClippingCallback(tt.callbacks.Callback):
    def __init__(self, clip_value):
        super().__init__()
        self.clip_value = clip_value

    def on_batch_end(self):
        # Use clip_grad_norm_ to clip gradients of the network parameters.
        torch.nn.utils.clip_grad_norm_(self.model.net.parameters(), self.clip_value)

# -----------------------------
# Main Training and Evaluation Function
# -----------------------------
def main(args):
    # Set seeds for reproducibility
    seed_everything(42, workers=True)
    
    # Initialize DataModule
    data_module = HCCDataModule(
        csv_file=args.csv_file,
        dicom_root=args.dicom_root,
        model_type="time_to_event",
        batch_size=args.batch_size,
        num_slices=args.num_slices,
        num_workers=args.num_workers,
        preprocessed_root=args.preprocessed_root
    )
    data_module.setup()

    # Load DINO model (ViT-based feature extractor)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dino_model = timm.create_model('timm/vit_small_patch14_dinov2.lvd142m', pretrained=True)
    dino_model.to(device)
    dino_model.eval()
    # Remove the classification head to extract features
    if hasattr(dino_model, 'head'):
        dino_model.head = nn.Identity()

    # Create DataLoaders
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()
    test_loader = data_module.test_dataloader()

    # Extract features
    print("Extracting train features...")
    x_train, y_train_durations, y_train_events = extract_features(train_loader, dino_model, device)
    print("Extracting validation features...")
    x_val, y_val_durations, y_val_events = extract_features(val_loader, dino_model, device)
    print("Extracting test features...")
    x_test, y_test_durations, y_test_events = extract_features(test_loader, dino_model, device)

    # Check for NaNs in features
    print("Checking for NaNs in features...")
    print("x_train contains NaNs:", np.isnan(x_train).any())
    print("x_val contains NaNs:", np.isnan(x_val).any())
    print("x_test contains NaNs:", np.isnan(x_test).any())

    # Remove zero-variance features if any exist
    variances = np.var(x_train, axis=0)
    if np.any(variances == 0):
        zero_var_features = np.where(variances == 0)[0]
        print(f"Warning: Features with zero variance: {zero_var_features}")
        x_train = x_train[:, variances != 0]
        x_val = x_val[:, variances != 0]
        x_test = x_test[:, variances != 0]

    y_train = (y_train_durations, y_train_events)
    y_val = (y_val_durations, y_val_events)
    y_test = (y_test_durations, y_test_events)

    if isinstance(y_train, tuple):
        y_train_durations, y_train_events = y_train
        y_val_durations, y_val_events = y_val
        y_test_durations, y_test_events = y_test
    else:
        raise ValueError("Expected time_to_event labels")

    # Standardize features
    x_mapper = StandardScaler()
    x_train_std = x_mapper.fit_transform(x_train).astype('float32')
    x_val_std = x_mapper.transform(x_val).astype('float32')
    x_test_std = x_mapper.transform(x_test).astype('float32')
    print("Feature value ranges (train):")
    print(f"Min: {x_train_std.min()}, Max: {x_train_std.max()}")
    print(f"Mean abs: {np.mean(np.abs(x_train_std))}, Std: {np.std(x_train_std)}")

    # Validate survival data
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
    print("Minimum duration:", np.min(y_train_durations))
    if np.any(y_train_durations <= 0):
        print("Found non-positive durations. Correcting...")
        y_train_durations[y_train_durations <= 0] = 1e-6

    get_target = lambda durations, events: (durations, events)
    y_train_target = get_target(y_train_durations, y_train_events)
    y_val_target = get_target(y_val_durations, y_val_events)
    y_test_target = get_target(y_test_durations, y_test_events)

    # -----------------------------
    # Define the neural network for CoxPH
    # -----------------------------
    in_features = x_train_std.shape[1]
    print("Input feature dimension:", in_features)
    num_nodes = [32, 32]
    out_features = 1

    net = tt.practical.MLPVanilla(in_features, num_nodes, out_features, batch_norm=True,
                                  dropout=args.dropout, output_bias=False)
    if args.center_risk:
        net = CenteredModel(net)
    
    model = CoxPH(net, tt.optim.Adam)
    model.optimizer.set_lr(args.learning_rate)
    model.optimizer.param_groups[0]['weight_decay'] = 1e-4  # L2 regularization

    # Define a callback to log loss and gradient norms
    class LossLogger(tt.callbacks.Callback):
        def __init__(self):
            self.epoch_losses = []

        def on_epoch_end(self):
            log_df = self.model.log.to_pandas()
            if not log_df.empty:
                current_loss = log_df.iloc[-1]['train_loss']
                print(f"Epoch {len(self.epoch_losses)} - Train Loss: {current_loss:.4f}")
                self.epoch_losses.append(current_loss)
                total_norm = 0.0
                for p in self.model.net.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                total_norm = total_norm ** 0.5
                print(f"Epoch {len(self.epoch_losses)} - Gradient Norm: {total_norm:.4f}")

    # Set up callbacks. Insert our custom gradient clipping callback if enabled.
    callbacks = [tt.callbacks.EarlyStopping(), LossLogger()]
    if args.gradient_clip > 0:
        callbacks.insert(1, GradientClippingCallback(args.gradient_clip))
    
    verbose = True
    batch_size = args.batch_size

    print("Training the CoxPH model...")
    log = model.fit(x_train_std, y_train_target, batch_size, args.epochs, callbacks, verbose,
                    val_data=(x_val_std, y_val_target), val_batch_size=batch_size)
    
    # Plot and save the training log
    plt.figure()
    log.plot()
    plt.title("Training Log")
    plt.savefig(os.path.join(args.output_dir, "training_log.png"))
    plt.close()

    print(f"y_val_target type: {type(y_val_target)}")
    print(f"y_val_target length: {len(y_val_target) if isinstance(y_val_target, tuple) else 'Not a tuple'}")
    print(f"y_val_target first element type: {type(y_val_target[0]) if isinstance(y_val_target, tuple) else 'N/A'}")
    print(f"y_val_target second element type: {type(y_val_target[1]) if isinstance(y_val_target, tuple) else 'N/A'}")

    partial_ll = model.partial_log_likelihood(x_val_std, y_val_target).mean()
    print(f"Partial Log-Likelihood on Validation Set: {partial_ll}")

    model.compute_baseline_hazards()
    surv = model.predict_surv_df(x_test_std)
    plt.figure()
    surv.iloc[:, :5].plot()
    plt.ylabel('S(t | x)')
    plt.xlabel('Time')
    plt.title("Survival Functions for Test Set")
    plt.savefig(os.path.join(args.output_dir, "survival_functions.png"))
    plt.close()

    ev = EvalSurv(surv, y_test_target[0], y_test_target[1], censor_surv='km')
    concordance = ev.concordance_td()
    print(f"Concordance Index: {concordance}")

    time_grid = np.linspace(y_test_target[0].min(), y_test_target[0].max(), 100)
    brier_score = ev.brier_score(time_grid)
    plt.figure()
    brier_score.plot()
    plt.title("Brier Score Over Time")
    plt.xlabel("Time")
    plt.ylabel("Brier Score")
    plt.savefig(os.path.join(args.output_dir, "brier_score.png"))
    plt.close()
    print(f"Integrated Brier Score: {ev.integrated_brier_score(time_grid)}")
    print(f"Integrated Negative Binomial Log-Likelihood: {ev.integrated_nbll(time_grid)}")

# -----------------------------
# Argument Parsing and Entry Point
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CoxPH model with DINO features")
    parser.add_argument("--dicom_root", type=str, default="/gpfs/data/mankowskilab/HCC_Recurrence/dicom",
                        help="Path to the root DICOM directory.")
    parser.add_argument("--csv_file", type=str, default="processed_patient_labels.csv",
                        help="Path to processed CSV with columns [Pre op MRI Accession number, recurrence post tx, time, event].")
    parser.add_argument('--preprocessed_root', type=str, default=None, help='Directory to store preprocessed image tensors')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for data loaders')
    parser.add_argument('--num_slices', type=int, default=2, help='Number of slices per patient')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loaders')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--output_dir', type=str, default='checkpoints', help='Directory to save outputs and models')
    # New flags for improvements:
    parser.add_argument('--learning_rate', type=float, default=1e-6, help='Learning rate for the CoxPH model')
    parser.add_argument('--gradient_clip', type=float, default=1.0, help='Gradient clipping threshold. Set 0 to disable.')
    parser.add_argument('--center_risk', action='store_true', help='If set, center risk scores for numerical stability')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate for the CoxPH network')
    
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    main(args)
