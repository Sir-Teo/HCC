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

DEBUG = True

def infer_dataset_type(csv_input):
    # If csv_input is a file path string, use the file name to infer dataset type.
    if isinstance(csv_input, pd.DataFrame):
        # Use the second row if available, otherwise fallback to the first row.
        row_idx = 1 if len(csv_input) > 1 else 0
        row_data = csv_input.iloc[row_idx]
        for cell in row_data:
            if isinstance(cell, str):
                cell_lower = cell.lower()
                if "tcga" in cell_lower:
                    return "TCGA"
                elif "nyu" in cell_lower:
                    return "NYU"
    return "NYU"

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
        preprocessed_root=None,
        dataset_type="NYU"
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
            dataset_type (str): Either "TCGA" or "NYU".
        """
        self.dicom_root = dicom_root
        self.transform = transform
        self.model_type = model_type
        self.num_slices = num_slices
        self.num_samples = num_samples
        self.preprocessed_root = preprocessed_root
        self.dataset_type = dataset_type  # store the dataset type

        # Build the patient_data list from CSV
        if isinstance(csv_file, str):
            df = pd.read_csv(csv_file)
        else:
            df = csv_file.copy()

        self.patient_data = []
        for _, row in df.iterrows():
            patient_id = str(row['Pre op MRI Accession number'])
            # Use the dicom root from the CSV row if available; otherwise fall back
            if 'dicom_root' in row:
                dicom_dir = os.path.join(row['dicom_root'], patient_id)
            else:
                dicom_dir = os.path.join(dicom_root, patient_id)
            if os.path.exists(dicom_dir):
                data_entry = {
                    'patient_id': patient_id,
                    'dicom_dir': dicom_dir
                }
                if self.model_type == "linear":
                    data_entry['label'] = row['event']
                elif self.model_type == "time_to_event":
                    data_entry['time'] = row['time']
                    data_entry['event'] = row['event']
                
                self.patient_data.append(data_entry)
        print(f"[DEBUG] Found {len(self.patient_data)} patients in the dataset.")

        # --- Filtering out patients with no valid images ---
        filtered_patient_data = []
        for entry in self.patient_data:
            if self._has_valid_images(entry['dicom_dir']):
                filtered_patient_data.append(entry)
            else:
                if DEBUG:
                    print(f"[DEBUG] Filtering out patient {entry['patient_id']} due to no valid images.")
        self.patient_data = filtered_patient_data

        if DEBUG:
            print(f"[DEBUG] HCCDicomDataset ({self.dataset_type}) initialized with {len(self.patient_data)} patients.")

    def _has_valid_images(self, dicom_dir):
        """
        Checks if the DICOM directory contains at least one valid image.
        For TCGA, traverse the top-level series folders.
        For NYU, look for .dcm files and check header fields.
        """
        if not os.path.exists(dicom_dir):
            return False

        if self.dataset_type == "TCGA":
            for top_folder in os.listdir(dicom_dir):
                top_folder_path = os.path.join(dicom_dir, top_folder)
                if "MR" not in top_folder.upper():
                    continue
                for subfolder in os.listdir(top_folder_path):
                    subfolder_path = os.path.join(top_folder_path, subfolder)
                    for fname in os.listdir(subfolder_path):
                        if not fname.endswith(".dcm"):
                            continue
                        file_path = os.path.join(subfolder_path, fname)
                        try:
                            # Only read header information
                            dcm = pydicom.dcmread(file_path, stop_before_pixels=True)
                            return True  # Valid image found
                        except Exception:
                            continue
            return False
        else:  # NYU
            dcm_files = [os.path.join(dicom_dir, f) for f in os.listdir(dicom_dir) if f.endswith('.dcm')]
            for dcm_path in dcm_files:
                try:
                    dcm = pydicom.dcmread(dcm_path, stop_before_pixels=True)
                    # Skip CT images
                    if hasattr(dcm, 'Modality') and dcm.Modality.upper() == 'CT':
                        continue
                    # Check that necessary fields exist
                    if not (hasattr(dcm, 'ImageOrientationPatient') and hasattr(dcm, 'ImagePositionPatient')):
                        continue
                    return True  # Valid image found
                except Exception:
                    continue
            return False
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

        For the "TCGA" dataset, we ignore axial filtering and load all DICOM slices,
        whereas for "NYU" we only load slices that pass the axial filter.
        
        If preprocessed_root is specified, tries to load from a .pt file to speed up.
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

        # --- File search based on dataset type ---
        if self.dataset_type == "TCGA":
            dcm_files = []

            # Step 1: Traverse only top-level series folders (e.g., MRI-AbdomenPelvis...)
            for top_folder in os.listdir(dicom_dir):
                
                top_folder_path = os.path.join(dicom_dir, top_folder)

                if "MR" not in top_folder.upper():
                    continue
                
                for subfolder in os.listdir(top_folder_path):
                    subfolder_path = os.path.join(top_folder_path, subfolder)

                    # Step 3: Read DICOM files inside these subfolders
                    for fname in os.listdir(subfolder_path):
                        if not fname.endswith(".dcm"):
                            continue
                        file_path = os.path.join(subfolder_path, fname)
                        try:
                            dcm = pydicom.dcmread(file_path, stop_before_pixels=True)
                            dcm_files.append(file_path)
                        except Exception as e:
                            print(f"Error reading DICOM file {file_path}: {e}")
                            continue
            selected_files = dcm_files
        else:  # NYU
            dcm_files = [os.path.join(dicom_dir, f) for f in os.listdir(dicom_dir) if f.endswith('.dcm')]
            series_dict = defaultdict(list)
            for dcm_path in dcm_files:
                try:
                    dcm = pydicom.dcmread(dcm_path, stop_before_pixels=True)
                    # Skip CT images
                    if hasattr(dcm, 'Modality') and dcm.Modality.upper() == 'CT':
                        continue
                    if not hasattr(dcm, 'ImageOrientationPatient') or not hasattr(dcm, 'ImagePositionPatient'):
                        continue
                    orientation = np.array(dcm.ImageOrientationPatient, dtype=np.float32).round(4)
                    orientation_tuple = tuple(orientation.flatten())
                    series_dict[orientation_tuple].append(dcm_path)
                except Exception as e:
                    print(f"Error reading DICOM file {dcm_path}: {e}")
                    continue

            selected_files = []
            # Only include slices from axial series
            for orient, dcm_paths in series_dict.items():
                if self.is_axial(orient):
                    selected_files.extend(dcm_paths)

        if DEBUG:
            print(f"[DEBUG] load_axial_series {patient_id}: {len(selected_files)} slices found.")

        image_stack = []
        for dcm_path in selected_files:
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

        return image_stack  # shape: [num_slices, C, H, W]



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
        img = dcm.pixel_array.astype(np.float32)
        img_min, img_max = img.min(), img.max()
        img = (img - img_min) / (img_max - img_min + 1e-6)

        if img.ndim == 2:
            img = np.repeat(img[..., np.newaxis], 3, axis=-1)

        tensor_img = torch.from_numpy(img).permute(2, 0, 1)
        # Resize tensor_img to (3, 224, 224) using torch.nn.functional.interpolate:
        tensor_img = tensor_img.unsqueeze(0)  # add batch dim for interpolate
        tensor_img = torch.nn.functional.interpolate(tensor_img, size=(224, 224), mode='bilinear', align_corners=False)
        tensor_img = tensor_img.squeeze(0)  # remove batch dim

        return tensor_img

class HCCDataModule:
    def __init__(
        self,
        train_csv_file,
        test_csv_file,
        train_dicom_root,
        test_dicom_root,
        model_type="linear",
        batch_size=2,
        num_slices=20,
        num_samples=1,
        num_workers=2,
        preprocessed_root=None,
        eval_batch_size=None,
        use_validation=True  # New parameter to control validation
    ):
        self.train_csv_file = train_csv_file
        self.test_csv_file = test_csv_file
        self.train_dicom_root = train_dicom_root
        self.test_dicom_root = test_dicom_root
        self.batch_size = batch_size
        self.model_type = model_type
        self.num_slices = num_slices
        self.num_samples = num_samples
        self.num_workers = num_workers
        self.preprocessed_root = preprocessed_root
        self.eval_batch_size = eval_batch_size if eval_batch_size is not None else batch_size
        self.use_validation = use_validation

        self.transform = T.Compose([
            T.Resize((224, 224), antialias=True),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
        ])

    def setup(self):
        if isinstance(self.train_csv_file, pd.DataFrame):
            train_df_full = self.train_csv_file.copy()
        else:
            train_df_full = pd.read_csv(self.train_csv_file)

        if isinstance(self.test_csv_file, pd.DataFrame):
            test_df = self.test_csv_file.copy()
        else:
            test_df = pd.read_csv(self.test_csv_file)

        stratify_col = train_df_full['event']

        if self.use_validation:
            train_df, val_df = train_test_split(
                train_df_full,
                test_size=0.2,
                random_state=42,
                stratify=stratify_col
            )
        else:
            train_df = train_df_full
            val_df = None

        print(f"Train: {len(train_df)} patients, Test: {len(test_df)} patients")
        print(f"Positive cases - Train: {(train_df['event'] == 1).sum()}, Test: {(test_df['event'] == 1).sum()}")
        if self.use_validation:
            print(f"Val: {len(val_df)} patients, Positive Val: {(val_df['event'] == 1).sum()}")

        train_dataset_type = infer_dataset_type(self.train_csv_file)
        test_dataset_type = infer_dataset_type(self.test_csv_file)

        print(f"Train dataset type: {train_dataset_type}")
        print(f"Test dataset type: {test_dataset_type}")

        self.train_dataset = HCCDicomDataset(
            csv_file=train_df,
            dicom_root=self.train_dicom_root,
            model_type=self.model_type,
            transform=self.transform,
            num_slices=self.num_slices,
            num_samples=self.num_samples,
            preprocessed_root=self.preprocessed_root,
            dataset_type=train_dataset_type
        )

        if self.use_validation:
            self.val_dataset = HCCDicomDataset(
                csv_file=val_df,
                dicom_root=self.train_dicom_root,
                model_type=self.model_type,
                transform=self.transform,
                num_slices=self.num_slices,
                num_samples=self.num_samples,
                preprocessed_root=self.preprocessed_root,
                dataset_type=train_dataset_type
            )

        self.test_dataset = HCCDicomDataset(
            csv_file=test_df,
            dicom_root=self.test_dicom_root,
            model_type=self.model_type,
            transform=self.transform,
            num_slices=self.num_slices,
            num_samples=self.num_samples,
            preprocessed_root=self.preprocessed_root,
            dataset_type=test_dataset_type
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
        if not self.use_validation:
            raise ValueError("Validation is disabled in this configuration.")
        return DataLoader(
            self.val_dataset,
            batch_size=self.eval_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            drop_last=False
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.eval_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            drop_last=False
        )

    def train_collate_fn(self, batch):
        return default_collate(batch)

    def collate_fn(self, batch):
        return default_collate(batch)
