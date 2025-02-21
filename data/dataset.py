import os
import random
from collections import defaultdict

import pandas as pd
import numpy as np
import torch
import torchvision.transforms as T
import pydicom
from torch.utils.data import Dataset, DataLoader, default_collate
from sklearn.model_selection import train_test_split

DEBUG = False

def nan_hook(module, input, output):
    if torch.isnan(output).any() and DEBUG:
        print(f"[DEBUG] NaN detected in output of {module}")

class HCCDicomDataset(Dataset):
    def __init__(
        self,
        csv_file,
        dicom_root,
        model_type="linear",
        transform=None,
        num_slices=10,
        num_samples=1,
        preprocessed_root=None
    ):
        """
        Args:
            csv_file (str or pd.DataFrame): Path to CSV or already loaded DataFrame.
            dicom_root (str): Root directory where patient DICOM folders are stored.
            model_type (str): Either 'linear' or 'time_to_event'.
            transform (callable): Torchvision transforms to apply on each slice.
            num_slices (int): Number of slices per sub-sample.
            num_samples (int): Number of sub-samples to extract from each patient's stack.
            preprocessed_root (str): Directory for caching/loading preprocessed tensors.
        """
        self.dicom_root = dicom_root
        self.transform = transform
        self.model_type = model_type
        self.num_slices = num_slices
        self.num_samples = num_samples
        self.preprocessed_root = preprocessed_root

        # Build the patient_data list from CSV
        if isinstance(csv_file, str):
            df = pd.read_csv(csv_file)
        else:
            df = csv_file.copy()

        self.patient_data = []
        for _, row in df.iterrows():
            patient_id = str(row['Pre op MRI Accession number'])
            dicom_dir = os.path.join(dicom_root, patient_id)
            if os.path.exists(dicom_dir):
                data_entry = {
                    'patient_id': patient_id,
                    'dicom_dir': dicom_dir
                }
                if self.model_type == "linear":
                    data_entry['label'] = row['recurrence post tx']
                elif self.model_type == "time_to_event":
                    data_entry['time'] = row['time']
                    data_entry['event'] = row['event']
                
                self.patient_data.append(data_entry)

        if DEBUG:
            print(f"[DEBUG] HCCDicomDataset initialized with {len(self.patient_data)} patients.")

    def __len__(self):
        """
        Now, dataset length = number of patients. We do NOT multiply by num_samples;
        instead we return multiple sub-samples from a single patient in __getitem__.
        """
        return len(self.patient_data)

    def __getitem__(self, idx):
        """
        Returns:
            - A 5D tensor of shape [num_samples, num_slices, C, H, W]
            - label (linear) or (time, event) (time_to_event)
        """
        patient = self.patient_data[idx]
        dicom_dir = patient['dicom_dir']
        patient_id = patient['patient_id']

        # Load the entire axial series (as a single 4D tensor: [total_slices, C, H, W]).
        full_stack = self.load_axial_series(dicom_dir)

        if DEBUG:
            print(f"[DEBUG] __getitem__ for patient {patient_id} - full stack shape: {full_stack.shape}")

        # Generate num_samples 3D sub-stacks (each with num_slices).
        sub_stacks = []
        for _ in range(self.num_samples):
            # Sample sub-stack
            sampled_stack = self.sample_sub_stack(full_stack)

            # Apply transform (if any) to each slice in this sub-stack.
            if self.transform is not None:
                transformed_slices = []
                for slice_ in sampled_stack:
                    # slice_ shape: [C, H, W]
                    slice_ = self.transform(slice_)
                    transformed_slices.append(slice_)
                # Re-stack along slice dimension => shape [num_slices, C, H, W]
                sampled_stack = torch.stack(transformed_slices, dim=0)

            sub_stacks.append(sampled_stack)

        # Shape: [num_samples, num_slices, C, H, W]
        sub_stacks = torch.stack(sub_stacks, dim=0)

        # Return according to model type
        if self.model_type == "linear":
            label = torch.tensor(patient['label'], dtype=torch.float32)
            return sub_stacks, label
        elif self.model_type == "time_to_event":
            time_ = torch.tensor(patient['time'], dtype=torch.float32)
            event_ = torch.tensor(patient['event'], dtype=torch.float32)
            return sub_stacks, time_, event_

    def load_axial_series(self, dicom_dir):
        """
        Loads (or retrieves from cache) all axial slices as a 4D tensor:
        [num_total_slices, 3, H, W].

        If preprocessed_root is specified, tries to load from .pt file to speed up.
        """
        patient_id = os.path.basename(dicom_dir)

        # If we have preprocessed data, just load that
        if self.preprocessed_root:
            preprocessed_path = os.path.join(self.preprocessed_root, f"{patient_id}.pt")
            if os.path.exists(preprocessed_path):
                if DEBUG:
                    print(f"[DEBUG] Loading preprocessed tensor for patient {patient_id} from {preprocessed_path}")
                image_stack = torch.load(preprocessed_path)
                return image_stack  # shape: [num_slices, C, H, W]

        # Otherwise, read from raw DICOM files
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

        # Find all axial series
        axial_series = []
        for orient, dcm_paths in series_dict.items():
            if self.is_axial(orient):
                axial_series.extend(dcm_paths)

        if not axial_series:
            raise ValueError(f"No axial series found in {dicom_dir}")

        # Sort slices by z-position
        def get_slice_position(dcm_path):
            try:
                dcm = pydicom.dcmread(dcm_path, stop_before_pixels=True)
                if hasattr(dcm, 'SliceLocation'):
                    return float(dcm.SliceLocation)
                else:
                    return float(dcm.ImagePositionPatient[2])
            except:
                return 0.0

        axial_series.sort(key=get_slice_position)

        if DEBUG:
            print(f"[DEBUG] load_axial_series {patient_id}: {len(axial_series)} axial slices found.")

        image_stack = []
        for dcm_path in axial_series:
            try:
                dcm = pydicom.dcmread(dcm_path)
                tensor_img = self.dicom_to_tensor(dcm)
                image_stack.append(tensor_img)
            except Exception as e:
                print(f"Error processing DICOM file {dcm_path}: {e}")
                continue

        if len(image_stack) > 0:
            # shape: [num_slices, C, H, W]
            image_stack = torch.stack(image_stack, dim=0)
        else:
            # No valid images => return empty
            image_stack = torch.zeros((0, 3, 224, 224))

        # Save preprocessed if requested
        if self.preprocessed_root:
            os.makedirs(self.preprocessed_root, exist_ok=True)
            preprocessed_path = os.path.join(self.preprocessed_root, f"{patient_id}.pt")
            torch.save(image_stack, preprocessed_path)
            if DEBUG:
                print(f"[DEBUG] Saved preprocessed tensor for patient {patient_id} to {preprocessed_path}")

        return image_stack

    def sample_sub_stack(self, full_stack):
        """
        Randomly selects `self.num_slices` slices from `full_stack`.
        If `full_stack` has fewer than `self.num_slices`, it will pad with zeros.
        Returns a 4D tensor: [num_slices, C, H, W].
        """
        total_slices = full_stack.size(0)
        if DEBUG:
            print(f"[DEBUG] sample_sub_stack - total slices: {total_slices}, needed: {self.num_slices}")

        if total_slices == 0:
            # Edge case: no slices at all
            return torch.zeros((self.num_slices, 3, 224, 224), dtype=full_stack.dtype)

        if total_slices < self.num_slices:
            # Pad
            pad_size = self.num_slices - total_slices
            padding = torch.zeros((pad_size, *full_stack.shape[1:]), dtype=full_stack.dtype)
            sub_stack = torch.cat([full_stack, padding], dim=0)
        else:
            # Randomly select self.num_slices among total_slices
            selected_indices = torch.randperm(total_slices)[:self.num_slices]
            selected_indices, _ = torch.sort(selected_indices)
            sub_stack = full_stack[selected_indices]

        return sub_stack

    def is_axial(self, orientation):
        """
        Determines if the orientation is axial based on cross product
        of the row and column direction vectors.
        """
        row_dir = np.array(orientation[:3], dtype=np.float32)
        col_dir = np.array(orientation[3:], dtype=np.float32)
        cross = np.cross(row_dir, col_dir)
        norm = np.linalg.norm(cross)
        if norm < 1e-6:
            return False
        cross_normalized = cross / norm
        # For an axial plane, Z-axis ~ ±1, X & Y ~ 0
        return (
            (abs(cross_normalized[0]) < 5e-2) and
            (abs(cross_normalized[1]) < 5e-2) and
            (abs(abs(cross_normalized[2]) - 1.0) < 5e-2)
        )

    def dicom_to_tensor(self, dcm):
        """
        Convert DICOM pixel data to a FloatTensor of shape [C, H, W].
        We replicate a single channel to 3 channels if needed.
        """
        img = dcm.pixel_array.astype(np.float32)
        # Example windowing for CT images:
        if dcm.Modality == 'CT':
            intercept = float(dcm.RescaleIntercept) if hasattr(dcm, 'RescaleIntercept') else 0.0
            slope = float(dcm.RescaleSlope) if hasattr(dcm, 'RescaleSlope') else 1.0
            img = slope * img + intercept
            # Simple clip (e.g., -100 to 400 HU)
            img = np.clip(img, -100, 400)
            # Then normalize to [0,1]
            img = (img + 100) / 500.0
        else:
            # If MRI or unknown
            img_min, img_max = img.min(), img.max()
            img = (img - img_min) / (img_max - img_min + 1e-6)

        # Expand to 3 channels if grayscale
        if img.ndim == 2:
            img = np.repeat(img[..., np.newaxis], 3, axis=-1)

        tensor_img = torch.from_numpy(img).permute(2, 0, 1)
        return tensor_img


class HCCDataModule:
    def __init__(
        self,
        csv_file,
        dicom_root,
        model_type="linear",
        batch_size=2,
        num_slices=20,
        num_samples=1,
        num_workers=2,
        preprocessed_root=None
    ):
        """
        Args:
            csv_file (str): Path to your CSV file
            dicom_root (str): Root directory containing sub-folders for each patient
            model_type (str): 'linear' or 'time_to_event'
            batch_size (int): Batch size for DataLoader
            num_slices (int): Number of slices per sub-sample
            num_samples (int): Number of sub-samples per patient
            num_workers (int): Number of DataLoader workers
            preprocessed_root (str): Where to save/load preprocessed .pt files
        """
        self.csv_file = csv_file
        self.dicom_root = dicom_root
        self.batch_size = batch_size
        self.model_type = model_type
        self.num_slices = num_slices
        self.num_samples = num_samples
        self.num_workers = num_workers
        self.preprocessed_root = preprocessed_root

        # Basic transform example (resize + normalization)
        self.transform = T.Compose([
            T.Resize((224, 224), antialias=True),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
        ])

    def setup(self):
        df = pd.read_csv(self.csv_file)

        # Stratification logic
        if self.model_type == "linear":
            stratify_col = df['recurrence post tx']
        else:
            stratify_col = df['event']

        train_df, temp_df = train_test_split(
            df,
            test_size=0.35,
            random_state=42,
            stratify=stratify_col
        )


        if self.model_type == "linear":
            val_df, test_df = train_test_split(
                temp_df,
                test_size=0.5,
                random_state=42,
                stratify=temp_df['recurrence post tx']
            )
        else:
            val_df, test_df = train_test_split(
                temp_df,
                test_size=0.5,
                random_state=42,
                stratify=temp_df['event']
            )

        print(f"Train: {len(train_df)} patients, Val: {len(val_df)} patients, Test: {len(test_df)} patients")

        pos_train = (train_df['recurrence post tx'] == 1).sum()
        pos_val = (val_df['recurrence post tx'] == 1).sum()
        pos_test = (test_df['recurrence post tx'] == 1).sum()
        print(f"Positive cases - Train: {pos_train}, Val: {pos_val}, Test: {pos_test}")

        self.train_dataset = HCCDicomDataset(
            csv_file=train_df,
            dicom_root=self.dicom_root,
            model_type=self.model_type,
            transform=self.transform,
            num_slices=self.num_slices,
            num_samples=self.num_samples,
            preprocessed_root=self.preprocessed_root
        )

        self.val_dataset = HCCDicomDataset(
            csv_file=val_df,
            dicom_root=self.dicom_root,
            model_type=self.model_type,
            transform=self.transform,
            num_slices=self.num_slices,
            num_samples=self.num_samples,
            preprocessed_root=self.preprocessed_root
        )

        self.test_dataset = HCCDicomDataset(
            csv_file=test_df,
            dicom_root=self.dicom_root,
            model_type=self.model_type,
            transform=self.transform,
            num_slices=self.num_slices,
            num_samples=self.num_samples,
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
        # Removed random 0 -> 1 flipping logic.
        return default_collate(batch)

    def collate_fn(self, batch):
        # Removed random 0 -> 1 flipping logic.
        return default_collate(batch)
