# data/dataset.py
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

# Global debug flag
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

        if DEBUG:
            print(f"[DEBUG] load_axial_series patient {patient_id}: {len(axial_series)} axial slices found.")

        image_stack = []
        for dcm_path in axial_series:
            try:
                dcm = pydicom.dcmread(dcm_path)
                img = self.dicom_to_tensor(dcm)
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
            pad_size = self.num_slices - current_slices
            padding = torch.zeros((pad_size, *image_stack.shape[1:]), dtype=image_stack.dtype)
            image_stack = torch.cat([image_stack, padding], dim=0)
            if DEBUG:
                print(f"[DEBUG] Padded image_stack to shape: {image_stack.shape}")
        else:
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
            img = (img - img.min()) / (img.max() - img.min() + 1e-6)
            img = img * 255.0
        if img.ndim == 2:
            img = np.repeat(img[..., np.newaxis], 3, axis=-1)
        tensor_img = torch.from_numpy(img).permute(2, 0, 1)
        if DEBUG:
            print(f"[DEBUG] dicom_to_tensor - tensor shape: {tensor_img.shape}, min={tensor_img.min()}, max={tensor_img.max()}")
        return tensor_img

class HCCDataModule:
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
        self.csv_file = csv_file
        self.dicom_root = dicom_root
        self.batch_size = batch_size
        self.model_type = model_type
        self.num_slices = num_slices
        self.num_workers = num_workers
        self.preprocessed_root = preprocessed_root
        self.num_samples_per_patient = num_samples_per_patient

        self.preprocess_transform = T.Compose([
            T.Resize((224, 224), antialias=True),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.transform = self.preprocess_transform

    def setup(self):
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
        if self.model_type == "time_to_event":
            events = [sample[2].item() for sample in batch]
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
