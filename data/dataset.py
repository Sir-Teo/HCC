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
from torch.utils.data import Dataset, DataLoader, default_collate
from sklearn.model_selection import train_test_split

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

        # Use basic transforms for preprocessing (resize)
        self.preprocess_transform = T.Compose([
            T.Resize((224, 224), antialias=True)
        ])
        # Runtime transforms (augmentations) can be added here
        self.transform = self.preprocess_transform  # Default to preprocessing transform

    def setup(self, stage=None):
        df = pd.read_csv(self.csv_file)
        stratify_col = df['recurrence post tx'] if self.model_type == "linear" else df['event']

        train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42, stratify=stratify_col)
        val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df['event'])

        print(f"Train: {len(train_df)} patients, Val: {len(val_df)} patients, Test: {len(test_df)} patients")
        print(f"Label distribution in Train:\n{train_df['recurrence post tx'].value_counts()}")
        print(f"Label distribution in Val:\n{val_df['recurrence post tx'].value_counts()}")
        print(f"Label distribution in Test:\n{test_df['recurrence post tx'].value_counts()}")

        # Create datasets with preprocessed_root
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
        """Ensures at least one event per batch ONLY during training."""
        if self.model_type == "time_to_event":
            events = [sample[2].item() for sample in batch]
            if sum(events) == 0 and len(batch) > 0:
                idx = random.randint(0, len(batch)-1)
                x, t, e = batch[idx]
                batch[idx] = (x, t, torch.tensor(1.0, dtype=torch.float32))
            return default_collate(batch)
        else:
            return default_collate(batch)

    def collate_fn(self, batch):
        if self.model_type == "time_to_event":
            # Ensure at least one event per batch
            events = [sample[2].item() for sample in batch]
            if sum(events) == 0 and len(batch) > 0:
                # Randomly select one to have event (adjust label if necessary)
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
        
        # Read CSV and prepare patient data
        if isinstance(csv_file, str):
            df = pd.read_csv(csv_file)
        else:
            df = csv_file.copy()  # Assume it's a DataFrame
        for _, row in df.iterrows():
            patient_id = str(row['Pre op MRI Accession number'])  # Update to match your CSV
            dicom_dir = os.path.join(dicom_root, patient_id)
            
            if os.path.exists(dicom_dir):
                data_entry = {
                    'patient_id': patient_id,
                }

                if self.model_type == "linear":
                    data_entry['label'] = row['recurrence post tx']  # Binary label
                elif self.model_type == "time_to_event":
                    data_entry['time'] = row['time']    # Duration
                    data_entry['event'] = row['event']  # Event indicator

                data_entry['dicom_dir'] = dicom_dir
                self.patient_data.append(data_entry)

    def __len__(self):
        return len(self.patient_data)

    def __getitem__(self, idx):
        patient = self.patient_data[idx]
        dicom_dir = patient['dicom_dir']
        
        # Load and process DICOM images
        image_stack = self.load_axial_series(dicom_dir)
        
        # Apply runtime transforms (e.g., augmentations)
        if self.transform:
            transformed_images = []
            for img in image_stack:
                img = self.transform(img)
                transformed_images.append(img)
            image_stack = torch.stack(transformed_images)
        
        # Return appropriate targets based on model type
        if self.model_type == "linear":
            return image_stack, torch.tensor(patient['label'], dtype=torch.float32)
        elif self.model_type == "time_to_event":
            return (
                image_stack,
                torch.tensor(patient['time'], dtype=torch.float32),
                torch.tensor(patient['event'], dtype=torch.float32)
            )
        
    def load_axial_series(self, dicom_dir):
        """Load axial DICOM series with proper orientation, then randomly sample num_slices."""
        patient_id = os.path.basename(dicom_dir)
        if self.preprocessed_root:
            preprocessed_path = os.path.join(self.preprocessed_root, f"{patient_id}.pt")
            if os.path.exists(preprocessed_path):
                image_stack = torch.load(preprocessed_path)
                return self.process_loaded_stack(image_stack)
        
        # If preprocessed not found or not enabled, process DICOMs
        series_dict = defaultdict(list)
        dcm_files = [f for f in os.listdir(dicom_dir) if f.endswith('.dcm')]
        
        # 1. Read each DICOM and group by orientation
        for fname in dcm_files:
            try:
                dcm_path = os.path.join(dicom_dir, fname)
                dcm = pydicom.dcmread(dcm_path, stop_before_pixels=True)
                if not hasattr(dcm, 'ImageOrientationPatient') or not hasattr(dcm, 'ImagePositionPatient'):
                    continue
                
                # Round orientation for grouping
                orientation = np.array(dcm.ImageOrientationPatient, dtype=np.float32).round(4)
                orientation_tuple = tuple(orientation.flatten())
                series_dict[orientation_tuple].append(dcm_path)
            except Exception as e:
                print(f"Error reading DICOM file {fname}: {e}")
                continue

        # 2. Filter for axial series
        axial_series = []
        for orient, dcm_paths in series_dict.items():
            if self.is_axial(orient):
                axial_series.extend(dcm_paths)

        if not axial_series:
            raise ValueError(f"No axial series found in {dicom_dir}")

        # 3. Sort slices by z-position
        def get_slice_position(dcm_path):
            try:
                dcm = pydicom.dcmread(dcm_path, stop_before_pixels=True)
                return float(dcm.SliceLocation) if hasattr(dcm, 'SliceLocation') else float(dcm.ImagePositionPatient[2])
            except Exception:
                return 0.0  # Default position if unavailable

        axial_series.sort(key=get_slice_position)

        # 4. Load pixel data and apply transforms
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
        
        # Save preprocessed data if enabled
        if self.preprocessed_root:
            os.makedirs(self.preprocessed_root, exist_ok=True)
            preprocessed_path = os.path.join(self.preprocessed_root, f"{patient_id}.pt")
            torch.save(image_stack, preprocessed_path)
        
        return self.process_loaded_stack(image_stack)

    def process_loaded_stack(self, image_stack):
        """Sample or pad the loaded image stack to num_slices."""
        current_slices = image_stack.size(0)
        if current_slices < self.num_slices:
            pad_size = self.num_slices - current_slices
            padding = torch.zeros((pad_size, *image_stack.shape[1:]), dtype=image_stack.dtype)
            image_stack = torch.cat([image_stack, padding], dim=0)
        else:
            selected_indices = torch.randperm(current_slices)[:self.num_slices]
            selected_indices, _ = torch.sort(selected_indices)  # Maintain order
            image_stack = image_stack[selected_indices]
        return image_stack


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