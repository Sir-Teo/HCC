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
from tqdm import tqdm # Added for progress bar

DEBUG = False 

def nan_hook(module, input, output):
    if torch.isnan(output).any() and DEBUG:
        print(f"[DEBUG] NaN detected in output of {module}")

class HCCDicomDataset(Dataset):
    def __init__(
        self,
        patient_data_list, # Changed from csv_file/dicom_root to list of dicts
        model_type="linear",
        transform=None,
        num_slices=10,
        num_samples=1,
        preprocessed_root=None
    ):
        """
        Args:
            patient_data_list (list): List of dictionaries, each containing patient info.
            model_type (str): Either 'linear' or 'time_to_event'.
            transform (callable): Torchvision transforms to apply on each slice.
            num_slices (int): Number of slices per sub-sample.
            num_samples (int): Number of sub-samples to extract from each patient's stack.
            preprocessed_root (str): Base directory for caching/loading preprocessed tensors.
        """
        self.transform = transform
        self.model_type = model_type
        self.num_slices = num_slices
        self.num_samples = num_samples
        self.preprocessed_root = preprocessed_root
        self.patient_data = patient_data_list # Directly use the provided list

        if DEBUG:
            print(f"[DEBUG] HCCDicomDataset initialized with {len(self.patient_data)} patients.")

    def _has_valid_images(self, dicom_dir, dataset_type): # Added dataset_type argument
        """
        Checks if the DICOM directory contains at least one valid image.
        Uses dataset_type to determine traversal logic.
        """
        if not os.path.exists(dicom_dir):
            return False

        if dataset_type == "TCGA": # Use passed dataset_type
            for top_folder in os.listdir(dicom_dir):
                top_folder_path = os.path.join(dicom_dir, top_folder)
                # Loosened check for flexibility
                if not os.path.isdir(top_folder_path): continue 
                if "MR" not in top_folder.upper(): 
                    continue
                for subfolder in os.listdir(top_folder_path):
                    subfolder_path = os.path.join(top_folder_path, subfolder)
                    if not os.path.isdir(subfolder_path): continue
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
        else:  # NYU or other types handled similarly
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
          Or returns None if the patient has no valid image data after loading.
        """
        patient = self.patient_data[idx]
        dicom_dir = patient['dicom_dir']
        patient_id = patient['patient_id']
        dataset_type = patient['dataset_type'] # Get type for this patient

        # Load the entire axial series (as a single 4D tensor: [total_slices, C, H, W]).
        full_stack = self.load_axial_series(dicom_dir, patient_id, dataset_type)

        # --- Check for empty stack --- 
        if full_stack.size(0) == 0:
             if DEBUG:
                 print(f"[DEBUG] Skipping patient {patient_id} ({dataset_type}) in __getitem__ due to empty full_stack.")
             return None # Signal to collate_fn to skip this sample

        if DEBUG:
            print(f"[DEBUG] __getitem__ for patient {patient_id} ({dataset_type}) - full stack shape: {full_stack.shape}")

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
            label = torch.tensor(patient['event'], dtype=torch.float32) # Assuming binary uses 'event'
            return sub_stacks, label
        elif self.model_type == "time_to_event":
            time_ = torch.tensor(patient['time'], dtype=torch.float32)
            event_ = torch.tensor(patient['event'], dtype=torch.float32)
            return sub_stacks, time_, event_

    def load_axial_series(self, dicom_dir, patient_id, dataset_type):
        """
        Loads (or retrieves from cache) all axial slices as a 4D tensor:
        [num_total_slices, 3, H, W].

        Uses the patient's dataset_type for path traversal and preprocessing path.
        
        If preprocessed_root is specified, tries to load from a .pt file to speed up.
        """
        # patient_id = os.path.basename(dicom_dir) # Already have patient_id

        # If we have preprocessed data, just load that
        if self.preprocessed_root:
            # Path includes dataset type: preprocessed_root/dataset_type/patient_id.pt
            preprocessed_dir = os.path.join(self.preprocessed_root, dataset_type.lower())
            preprocessed_path = os.path.join(preprocessed_dir, f"{patient_id}.pt")
            if os.path.exists(preprocessed_path):
                if DEBUG:
                    print(f"[DEBUG] Loading preprocessed tensor for patient {patient_id} ({dataset_type}) from {preprocessed_path}")
                try:
                    image_stack = torch.load(preprocessed_path)
                    if image_stack.nelement() == 0: # Check for empty tensor saved previously
                         if DEBUG: print(f"[DEBUG] Loaded empty tensor for {patient_id}. Will re-process.")
                    else:
                         return image_stack  # shape: [num_slices, C, H, W]
                except Exception as e:
                     print(f"[WARN] Error loading preprocessed file {preprocessed_path}, will re-process. Error: {e}")
                     # Fall through to re-process

        # --- File search based on dataset type --- 
        # Ensure directory exists before listing contents
        if not os.path.isdir(dicom_dir):
            print(f"[WARN] DICOM directory not found for patient {patient_id}: {dicom_dir}")
            image_stack = torch.zeros((0, 3, 224, 224))
            # Save empty tensor if preprocessing is enabled
            if self.preprocessed_root:
                preprocessed_dir = os.path.join(self.preprocessed_root, dataset_type.lower())
                os.makedirs(preprocessed_dir, exist_ok=True)
                preprocessed_path = os.path.join(preprocessed_dir, f"{patient_id}.pt")
                torch.save(image_stack, preprocessed_path)
                if DEBUG: print(f"[DEBUG] Saved empty preprocessed tensor for missing dir patient {patient_id} to {preprocessed_path}")
            return image_stack
            
        if dataset_type == "TCGA": # Use patient's dataset_type
            dcm_files = []
            # Step 1: Traverse only top-level series folders (e.g., MRI-AbdomenPelvis...)
            for top_folder in os.listdir(dicom_dir):
                top_folder_path = os.path.join(dicom_dir, top_folder)
                if not os.path.isdir(top_folder_path): continue
                # if "MR" not in top_folder.upper(): 
                #     continue
                for subfolder in os.listdir(top_folder_path):
                    subfolder_path = os.path.join(top_folder_path, subfolder)
                    if not os.path.isdir(subfolder_path): continue
                    # Step 3: Read DICOM files inside these subfolders
                    for fname in os.listdir(subfolder_path):
                        if not fname.endswith(".dcm"):
                            continue
                        file_path = os.path.join(subfolder_path, fname)
                        try:
                            # Read header first to potentially skip faster
                            dcm_hdr = pydicom.dcmread(file_path, stop_before_pixels=True) 
                            dcm_files.append(file_path)
                        except Exception as e:
                            # print(f"Minor error reading DICOM header {file_path}: {e}")
                            continue
            selected_files = dcm_files
        else:  # NYU or other types
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
                    # print(f"Minor error reading DICOM header {dcm_path}: {e}")
                    continue

            selected_files = []
            # Only include slices from axial series
            for orient, dcm_paths in series_dict.items():
                if self.is_axial(orient):
                    selected_files.extend(dcm_paths)

        if DEBUG:
            print(f"[DEBUG] load_axial_series {patient_id} ({dataset_type}): {len(selected_files)} axial slices found.")

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
            if DEBUG: print(f"[DEBUG] No valid axial slices found for patient {patient_id} ({dataset_type}). Returning empty tensor.")
            image_stack = torch.zeros((0, 3, 224, 224))

        # Save preprocessed if requested
        if self.preprocessed_root:
            preprocessed_dir = os.path.join(self.preprocessed_root, dataset_type.lower())
            os.makedirs(preprocessed_dir, exist_ok=True)
            preprocessed_path = os.path.join(preprocessed_dir, f"{patient_id}.pt")
            torch.save(image_stack, preprocessed_path)
            if DEBUG:
                print(f"[DEBUG] Saved preprocessed tensor ({image_stack.shape}) for patient {patient_id} ({dataset_type}) to {preprocessed_path}")

        return image_stack  # shape: [num_slices, C, H, W]

    def sample_sub_stack(self, full_stack):
        """
        Randomly selects `self.num_slices` slices from `full_stack`.
        If `full_stack` has fewer than `self.num_slices`, it will pad with zeros.
        Returns a 4D tensor: [num_slices, C, H, W].
        """
        total_slices = full_stack.size(0)
        # if DEBUG:
            # print(f"[DEBUG] sample_sub_stack - total slices: {total_slices}, needed: {self.num_slices}")

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
            start_index = random.randint(0, total_slices - self.num_slices) # Select contiguous block
            sub_stack = full_stack[start_index : start_index + self.num_slices]
            # selected_indices = torch.randperm(total_slices)[:self.num_slices]
            # selected_indices, _ = torch.sort(selected_indices)
            # sub_stack = full_stack[selected_indices]

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
        # Normalize to [0, 1]
        if img_max > img_min:
             img = (img - img_min) / (img_max - img_min)
        else:
             img = np.zeros_like(img) # Handle case of constant image

        if img.ndim == 2:
            img = np.repeat(img[..., np.newaxis], 3, axis=-1)
        elif img.ndim == 3 and img.shape[2] == 1: # Handle grayscale with channel dim
             img = np.repeat(img, 3, axis=-1)
        elif img.ndim == 3 and img.shape[2] != 3: # Unexpected channel count
             print(f"[WARN] Unexpected channel count {img.shape[2]}, taking first channel.")
             img = img[:,:,0:1] # Take the first channel
             img = np.repeat(img, 3, axis=-1) # Repeat it to make 3 channels

        tensor_img = torch.from_numpy(img).permute(2, 0, 1)
        # Resize tensor_img to (3, 224, 224) using torch.nn.functional.interpolate:
        if tensor_img.shape[1:] != (224, 224):
            tensor_img = tensor_img.unsqueeze(0)  # add batch dim for interpolate
            tensor_img = torch.nn.functional.interpolate(tensor_img, size=(224, 224), mode='bilinear', align_corners=False)
            tensor_img = tensor_img.squeeze(0)  # remove batch dim

        return tensor_img

class HCCDataModule:
    def __init__(
        self,
        train_csv_file, # Should be TCGA csv
        test_csv_file,  # Should be NYU csv
        train_dicom_root, # TCGA root
        test_dicom_root,  # NYU root
        model_type="linear",
        batch_size=2,
        num_slices=20,
        num_samples=1,
        num_workers=2,
        preprocessed_root=None,
        eval_batch_size=None,
        use_validation=True,
        cross_validation=False,
        cv_folds=10,
        leave_one_out=False,
        random_state=42
    ):
        # Renamed args for clarity when combining datasets
        if not isinstance(train_csv_file, str):
            raise ValueError("train_csv_file (TCGA) must be a string path to a CSV file")
        if test_csv_file is not None and not isinstance(test_csv_file, str):
            raise ValueError("test_csv_file (NYU) must be a string path to a CSV file or None")

        self.tcga_csv_file = train_csv_file
        self.nyu_csv_file = test_csv_file
        self.tcga_dicom_root = train_dicom_root
        self.nyu_dicom_root = test_dicom_root
        self.batch_size = batch_size
        self.model_type = model_type
        self.num_slices = num_slices
        self.num_samples = num_samples
        self.num_workers = num_workers
        self.preprocessed_root = preprocessed_root # Base directory
        self.eval_batch_size = eval_batch_size if eval_batch_size is not None else batch_size
        self.use_validation = use_validation
        self.cross_validation = cross_validation
        self.cv_folds = cv_folds
        self.leave_one_out = leave_one_out
        self.random_state = random_state

        self.transform = T.Compose([
            # Resize is now handled in dicom_to_tensor if needed
            # T.Resize((224, 224), antialias=True),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
        ])

    def _build_patient_list(self, csv_file, dicom_root, dataset_type):
        """Builds the list of patient data dictionaries from a CSV."""
        if not csv_file or not os.path.exists(csv_file):
             print(f"[WARN] CSV file not found or not provided: {csv_file}")
             return []
             
        df = pd.read_csv(csv_file)
        patient_list = []
        required_cols = ['Pre op MRI Accession number', 'time', 'event']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns in {csv_file}: {missing_cols}")

        for _, row in df.iterrows():
            patient_id = str(row['Pre op MRI Accession number'])
            pat_dicom_dir = os.path.join(dicom_root, patient_id)
            
            # Basic check if directory exists
            # More thorough check happens in HCCDicomDataset._has_valid_images
            # if not os.path.exists(pat_dicom_dir): 
            #      if DEBUG: print(f"[DEBUG] Skipping patient {patient_id} from {dataset_type}, directory not found: {pat_dicom_dir}")
            #      continue
                 
            data_entry = {
                'patient_id': patient_id,
                'dicom_dir': pat_dicom_dir, # Specific dicom dir for this patient
                'dataset_type': dataset_type, # TCGA or NYU
                'source': dataset_type # Use dataset_type as source identifier
            }
            if self.model_type == "linear":
                data_entry['event'] = row['event'] # Use event as label for binary
            elif self.model_type == "time_to_event":
                data_entry['time'] = row['time']
                data_entry['event'] = row['event']
            
            patient_list.append(data_entry)
        print(f"Found {len(patient_list)} initial patient entries in {dataset_type} ({csv_file})")
        return patient_list

    def _filter_patient_list(self, patient_list):
         """Filters a list of patient data based on image validity."""
         filtered_list = []
         print(f"Filtering {len(patient_list)} patients for valid images...")
         # Use a dummy dataset instance just for the filtering method
         temp_ds = HCCDicomDataset([], transform=self.transform) 
         for entry in tqdm(patient_list, desc="Filtering Patients"):
             if temp_ds._has_valid_images(entry['dicom_dir'], entry['dataset_type']):
                 filtered_list.append(entry)
             else:
                 if DEBUG:
                     print(f"[DEBUG] Filtering out patient {entry['patient_id']} ({entry['dataset_type']}) due to no valid images.")
         print(f"Retained {len(filtered_list)} patients after filtering.")
         return filtered_list

    def setup(self):
        # Build initial patient lists for TCGA and NYU
        tcga_patients = self._build_patient_list(self.tcga_csv_file, self.tcga_dicom_root, "TCGA")
        nyu_patients = self._build_patient_list(self.nyu_csv_file, self.nyu_dicom_root, "NYU")
        
        # Filter patient lists based on image validity
        tcga_patients_filtered = self._filter_patient_list(tcga_patients)
        nyu_patients_filtered = self._filter_patient_list(nyu_patients)

        # Combine filtered lists
        self.all_patients = tcga_patients_filtered + nyu_patients_filtered
        random.shuffle(self.all_patients) # Shuffle the combined list
        print(f"Total combined patients after filtering: {len(self.all_patients)}")

        if not self.all_patients:
             raise ValueError("No valid patients found in either dataset after filtering.")

        # Preprocess all patients if requested
        if self.preprocessed_root:
            print(f"Preprocessing all {len(self.all_patients)} combined patients...")
            # Create a temporary dataset with all patients
            temp_all_dataset = HCCDicomDataset(
                patient_data_list=self.all_patients,
                model_type=self.model_type,
                transform=self.transform, # Pass transform for resizing during preprocessing
                num_slices=self.num_slices, # Needed for dummy call
                num_samples=self.num_samples, # Needed for dummy call
                preprocessed_root=self.preprocessed_root
            )
            # Force preprocessing by accessing each patient via __getitem__
            # which calls load_axial_series -> dicom_to_tensor -> save
            for i in tqdm(range(len(temp_all_dataset)), desc="Preprocessing All Patients"):
                try:
                    _ = temp_all_dataset[i] # This triggers loading and saving
                except Exception as e:
                     print(f"[WARN] Error during preprocessing patient index {i}: {e}")
            print("Preprocessing complete.")

        if self.cross_validation:
            # Create cross-validation splits on the combined list of patient dicts
            patient_indices = list(range(len(self.all_patients)))
            patient_events = [p['event'] for p in self.all_patients]

            if self.leave_one_out:
                from sklearn.model_selection import LeaveOneOut
                self.cv_splitter = LeaveOneOut()
                self.cv_splits = list(self.cv_splitter.split(patient_indices))
            else:
                from sklearn.model_selection import StratifiedKFold
                skf = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
                # Check if there are enough samples for stratification
                if len(np.unique(patient_events)) < 2:
                     print("[WARN] Only one class present in the combined dataset. Using KFold instead of StratifiedKFold.")
                     from sklearn.model_selection import KFold
                     kf = KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
                     self.cv_splits = list(kf.split(patient_indices))
                elif any(np.bincount(patient_events) < self.cv_folds):
                     print(f"[WARN] The least populated class has {min(np.bincount(patient_events))} members, which is less than n_splits={self.cv_folds}. Using KFold instead.")
                     from sklearn.model_selection import KFold
                     kf = KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
                     self.cv_splits = list(kf.split(patient_indices))
                else:
                     self.cv_splits = list(skf.split(patient_indices, patient_events))
            
            print(f"Total patients for cross-validation: {len(self.all_patients)}")
            print(f"Number of folds: {len(self.cv_splits)}")
            
            self.current_fold = 0
            self._setup_fold(0)  # Setup first fold
            
        else: # Regular train/val/test split (on combined data)
            if self.use_validation:
                train_patients, test_patients = train_test_split(
                    self.all_patients,
                    test_size=0.2, # Or adjust as needed
                    random_state=self.random_state,
                    stratify=[p['event'] for p in self.all_patients]
                )
                train_patients, val_patients = train_test_split(
                    train_patients,
                    test_size=0.25, # 0.25 * 0.8 = 0.2
                    random_state=self.random_state,
                    stratify=[p['event'] for p in train_patients]
                )
            else:
                train_patients, test_patients = train_test_split(
                    self.all_patients,
                    test_size=0.2, # Or adjust as needed
                    random_state=self.random_state,
                    stratify=[p['event'] for p in self.all_patients]
                )
                val_patients = [] # No validation set

            print(f"Train: {len(train_patients)} patients, Positive cases: {sum(p['event'] for p in train_patients)}")
            print(f"Val: {len(val_patients)} patients, Positive cases: {sum(p['event'] for p in val_patients)}")
            print(f"Test: {len(test_patients)} patients, Positive cases: {sum(p['event'] for p in test_patients)}")

            self.train_dataset = HCCDicomDataset(train_patients, self.model_type, self.transform, self.num_slices, self.num_samples, self.preprocessed_root)
            self.val_dataset = HCCDicomDataset(val_patients, self.model_type, self.transform, self.num_slices, self.num_samples, self.preprocessed_root) if val_patients else None
            self.test_dataset = HCCDicomDataset(test_patients, self.model_type, self.transform, self.num_slices, self.num_samples, self.preprocessed_root)

    def _setup_fold(self, fold_idx):
        """Setup datasets for a specific cross-validation fold using the combined patient list."""
        train_indices, test_indices = self.cv_splits[fold_idx]
        
        # Get the patient data dictionaries for this fold
        fold_train_val_patients = [self.all_patients[i] for i in train_indices]
        fold_test_patients = [self.all_patients[i] for i in test_indices]

        # Split training data into train and validation sets
        if len(fold_train_val_patients) > 1 and self.use_validation:
             try:
                 fold_train_patients, fold_val_patients = train_test_split(
                     fold_train_val_patients,
                     test_size=0.2,
                     random_state=self.random_state,
                     stratify=[p['event'] for p in fold_train_val_patients]
                 )
             except ValueError as e:
                 # Handle cases where stratification isn't possible (e.g., only one class in split)
                 print(f"[WARN] Stratification failed for fold {fold_idx} validation split: {e}. Splitting without stratification.")
                 fold_train_patients, fold_val_patients = train_test_split(
                     fold_train_val_patients,
                     test_size=0.2,
                     random_state=self.random_state
                 )
        else:
             fold_train_patients = fold_train_val_patients
             fold_val_patients = [] # No validation split if only 1 sample or validation disabled

        # Create datasets for this fold using the patient lists
        self.train_dataset = HCCDicomDataset(
            patient_data_list=fold_train_patients,
            model_type=self.model_type,
            transform=self.transform,
            num_slices=self.num_slices,
            num_samples=self.num_samples,
            preprocessed_root=self.preprocessed_root
        )

        self.val_dataset = HCCDicomDataset(
            patient_data_list=fold_val_patients,
            model_type=self.model_type,
            transform=self.transform,
            num_slices=self.num_slices,
            num_samples=self.num_samples,
            preprocessed_root=self.preprocessed_root
        ) if fold_val_patients else None # Create only if val set exists

        self.test_dataset = HCCDicomDataset(
            patient_data_list=fold_test_patients,
            model_type=self.model_type,
            transform=self.transform,
            num_slices=self.num_slices,
            num_samples=self.num_samples,
            preprocessed_root=self.preprocessed_root
        )

        # Print actual patient counts after filtering
        train_events = sum(p['event'] for p in fold_train_patients)
        val_events = sum(p['event'] for p in fold_val_patients) if fold_val_patients else 0
        test_events = sum(p['event'] for p in fold_test_patients)
        print(f"[CV Fold {fold_idx+1}] Train: {len(self.train_dataset)} patients ({train_events} positive)")
        if self.val_dataset:
             print(f"[CV Fold {fold_idx+1}] Val:   {len(self.val_dataset)} patients ({val_events} positive)")
        print(f"[CV Fold {fold_idx+1}] Test:  {len(self.test_dataset)} patients ({test_events} positive)")

    def next_fold(self):
        """Move to the next cross-validation fold"""
        if not self.cross_validation:
            raise ValueError("Cross-validation is not enabled")
        
        self.current_fold += 1
        if self.current_fold >= len(self.cv_splits):
            return False
        
        self._setup_fold(self.current_fold)
        return True

    def get_current_fold(self):
        """Get the current fold index"""
        return self.current_fold

    def get_total_folds(self):
        """Get the total number of folds"""
        return len(self.cv_splits) if self.cross_validation else 1

    def train_dataloader(self):
        if not self.train_dataset:
            raise ValueError("Train dataset not setup.")
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self.skip_none_collate, # Use custom collate
            pin_memory=True
        )

    def val_dataloader(self):
        if not self.use_validation:
            return None 
        if not self.val_dataset:
            print("[WARN] Validation dataset is empty for this fold.")
            return None 
        return DataLoader(
            self.val_dataset,
            batch_size=self.eval_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.skip_none_collate, # Use custom collate
            drop_last=False,
            pin_memory=True
        )

    def test_dataloader(self):
        if not self.test_dataset:
            raise ValueError("Test dataset not setup.")
        return DataLoader(
            self.test_dataset,
            batch_size=self.eval_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.skip_none_collate, # Use custom collate
            drop_last=False,
            pin_memory=True
        )

    def skip_none_collate(self, batch):
        """A collate function that filters out None values returned by dataset.__getitem__."""
        # Filter out None samples
        original_len = len(batch)
        batch = [x for x in batch if x is not None]
        filtered_len = len(batch)
        skipped_count = original_len - filtered_len
        if skipped_count > 0 and DEBUG:
            print(f"[DEBUG] Collate: Skipped {skipped_count}/{original_len} samples due to loading errors.")
        
        # If all samples in the batch were None, return None or empty tensors
        if not batch:
            # Depending on what the training loop expects, 
            # returning None might be simplest, or return appropriately shaped empty tensors.
            # Let's try returning None and handle it in the training loop if necessary.
             print("[WARN] Collate: Entire batch was skipped.")
             return None # Or potentially (torch.empty(0,...), torch.empty(0,...))
            
        # Use the default collate function on the filtered batch
        try:
            return default_collate(batch)
        except Exception as e:
            print(f"[ERROR] Error in default_collate after filtering Nones: {e}")
            # You might want to investigate the structure of items in `batch` here
            # For now, returning None to avoid crashing the training loop
            return None

    # Keep train_collate_fn and collate_fn in case they are referenced elsewhere,
    # but point them to the new function or make them aliases.
    def train_collate_fn(self, batch):
        return self.skip_none_collate(batch)

    def collate_fn(self, batch):
        return self.skip_none_collate(batch)
