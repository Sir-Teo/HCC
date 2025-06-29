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
import torch.nn.functional as F

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

        # Check if this patient was marked as having no images
        has_images = patient.get('has_images', True)  # Default to True for backward compatibility
        
        if not has_images:
            # Patient has no images but valid clinical data - return zero tensor for images
            if DEBUG:
                print(f"[DEBUG] Patient {patient_id} has no images, using zero tensor placeholders.")
            
            # Create zero tensor with expected shape [num_samples, num_slices, C, H, W]
            if self.num_slices <= 0:
                sub_stacks = torch.zeros(1, 1, 3, 224, 224)  # [1, 1, C, H, W]
            else:
                sub_stacks = torch.zeros(self.num_samples, self.num_slices, 3, 224, 224)
        else:
            # Normal image loading for patients with images
            # Load the entire axial series (as a single 4D tensor: [total_slices, C, H, W]).
            full_stack = self.load_axial_series(dicom_dir, patient_id, dataset_type)

            # --- Check for empty stack --- 
            if full_stack.size(0) == 0:
                if DEBUG:
                    print(f"[DEBUG] Patient {patient_id} ({dataset_type}) has empty stack, using zero tensor placeholders.")
                # Instead of returning None (which gets filtered out), return zero tensors
                # This ensures ALL patients are included in the analysis
                if self.num_slices <= 0:
                    sub_stacks = torch.zeros(1, 1, 3, 224, 224)  # [1, 1, C, H, W]
                else:
                    sub_stacks = torch.zeros(self.num_samples, self.num_slices, 3, 224, 224)
            else:
                # Normal sampling for patients with valid images
                if self.num_slices <= 0:
                    # Use the entire axial series as one sub-stack
                    if self.transform is not None:
                        # Apply transform to each slice
                        transformed = [self.transform(s) for s in full_stack]
                        full_stack = torch.stack(transformed, dim=0)
                    # Pool across slice dimension to unify to one slice per patient
                    # full_stack: [total_slices, C, H, W] -> permute to [1, C, total_slices, H, W]
                    full_stack = full_stack.permute(1, 0, 2, 3).unsqueeze(0)
                    # Adaptive pooling to collapse slice dimension
                    pooled = F.adaptive_avg_pool3d(full_stack, (1, full_stack.size(3), full_stack.size(4)))
                    # pooled: [1, C, 1, H, W] -> get back to [1, C, H, W]
                    pooled = pooled.squeeze(2)
                    # Final sub_stacks: [num_samples=1, num_slices=1, C, H, W]
                    sub_stacks = pooled.unsqueeze(0)
                else:
                    # Generate num_samples 3D sub-stacks (each with num_slices)
                    stacks = []
                    for _ in range(self.num_samples):
                        sub = self.sample_sub_stack(full_stack)
                        if self.transform is not None:
                            sub = torch.stack([self.transform(s) for s in sub], dim=0)
                        stacks.append(sub)
                    # Shape: [num_samples, num_slices, C, H, W]
                    sub_stacks = torch.stack(stacks, dim=0)

        # Return according to model type
        if self.model_type == "linear":
            # Handle patients without clinical data
            event_val = patient.get('event', 0)  # Default to 0 if no event data
            if event_val is None:
                event_val = 0
            label = torch.tensor(event_val, dtype=torch.float32)
            return sub_stacks, label
        elif self.model_type == "time_to_event":
            # Handle patients without clinical data
            time_val = patient.get('time', 1.0)  # Default to 1.0 if no time data
            event_val = patient.get('event', 0)  # Default to 0 if no event data
            
            if time_val is None:
                time_val = 1.0
            if event_val is None:
                event_val = 0
                
            time_ = torch.tensor(time_val, dtype=torch.float32)
            event_ = torch.tensor(event_val, dtype=torch.float32)
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
            
        # NYU file structure - DICOM files directly in patient directory
        dcm_files = []
        for filename in os.listdir(dicom_dir):
            if filename.endswith('.dcm'):
                dcm_files.append(os.path.join(dicom_dir, filename))
        
        if not dcm_files:
            print(f"[WARN] No .dcm files found for patient {patient_id} in {dicom_dir}")
            image_stack = torch.zeros((0, 3, 224, 224))
            # Save empty tensor if preprocessing is enabled
            if self.preprocessed_root:
                preprocessed_dir = os.path.join(self.preprocessed_root, dataset_type.lower())
                os.makedirs(preprocessed_dir, exist_ok=True)
                preprocessed_path = os.path.join(preprocessed_dir, f"{patient_id}.pt")
                torch.save(image_stack, preprocessed_path)
                if DEBUG: print(f"[DEBUG] Saved empty preprocessed tensor for no DCM files patient {patient_id} to {preprocessed_path}")
            return image_stack

        image_stack = []
        for dcm_path in dcm_files:
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
        csv_file, # NYU csv
        dicom_root,  # NYU root
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
        self.csv_file = csv_file
        self.dicom_root = dicom_root
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
        
        # Ensure the normalization uses float32 - wrap the transform
        original_transform = self.transform
        def float32_transform(tensor):
            # Ensure input is float32
            if tensor.dtype != torch.float32:
                tensor = tensor.float()
            # Apply original transform
            result = original_transform(tensor)
            # Ensure output is float32
            if result.dtype != torch.float32:
                result = result.float()
            return result
        
        self.transform = float32_transform

    def _build_patient_list(self, csv_file, dicom_root, dataset_type):
        """Builds the list of patient data dictionaries from DICOM directories and CSV.
        Includes ALL patients from DICOM directories, even if missing from CSV."""
        
        # First, get all available DICOM directories
        if not dicom_root or not os.path.exists(dicom_root):
            print(f"[WARN] DICOM root not found: {dicom_root}")
            return []
            
        all_dicom_items = os.listdir(dicom_root)
        # Filter to only directories (exclude .txt, .csv files)
        dicom_directories = []
        for item in all_dicom_items:
            full_path = os.path.join(dicom_root, item)
            if os.path.isdir(full_path):
                dicom_directories.append(item)
        
        print(f"Found {len(dicom_directories)} DICOM patient directories")
        
        # Load CSV data if available
        csv_data = {}
        if csv_file and os.path.exists(csv_file):
            df = pd.read_csv(csv_file)
            required_cols = ['Deidentified ID', 'time', 'event']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                print(f"[WARN] Missing columns in CSV: {missing_cols}")
            else:
                # Build lookup for CSV data by accession number
                for _, row in df.iterrows():
                    acc_num = row.get('Pre op MRI Accession number')
                    if pd.notna(acc_num):
                        acc_num = str(acc_num)
                        if acc_num not in csv_data:  # Handle duplicates by keeping first occurrence
                            csv_data[acc_num] = {
                                'patient_id': row['Deidentified ID'],
                                'time': row['time'],
                                'event': row['event'],
                                'dataset_type': dataset_type
                            }
        
        # Build patient list from ALL DICOM directories
        patient_list = []
        csv_matched = 0
        dicom_only = 0
        
        for dicom_dir in dicom_directories:
            dicom_path = os.path.join(dicom_root, dicom_dir)
            
            if dicom_dir in csv_data:
                # Patient has both DICOM and CSV data
                patient_entry = csv_data[dicom_dir].copy()
                patient_entry.update({
                    'dicom_dir': dicom_path,
                    'has_images': True,
                    'has_clinical_data': True
                })
                csv_matched += 1
            else:
                # Patient has DICOM only (missing from CSV)
                patient_entry = {
                    'patient_id': f'DICOM_ONLY_{dicom_dir}',
                    'dicom_dir': dicom_path,
                    'time': None,  # No survival data
                    'event': None,  # No event data
                    'dataset_type': dataset_type,
                    'has_images': True,
                    'has_clinical_data': False
                }
                dicom_only += 1
            
            patient_list.append(patient_entry)
        
        print(f"Built patient list: {len(patient_list)} total patients")
        print(f"  - {csv_matched} patients with both DICOM and clinical data")
        print(f"  - {dicom_only} patients with DICOM only (missing from CSV)")
        
        return patient_list

    def _filter_patient_list(self, patient_list):
        """Include ALL patients without filtering - user wants every patient included.
        Just verify images exist but don't exclude patients during preprocessing."""
        
        print(f"Processing {len(patient_list)} patients for the dataset...")
        
        # Create a temporary dataset instance for image validation
        temp_dataset = HCCDicomDataset([], model_type=self.model_type, transform=self.transform)
        
        # Just verify image availability but include all patients
        patients_with_images = 0
        patients_without_images = 0
        
        for entry in tqdm(patient_list, desc="Verifying Patients"):
            # Check if patient has valid images using the dataset method
            has_valid_images = temp_dataset._has_valid_images(entry['dicom_dir'], entry['dataset_type'])
            entry['has_images'] = has_valid_images
            
            if has_valid_images:
                patients_with_images += 1
            else:
                patients_without_images += 1
                print(f"[INFO] Patient {entry['patient_id']} has no valid images but will be included")
        
        print(f"Dataset includes ALL {len(patient_list)} patients:")
        print(f"  - {patients_with_images} patients with valid images")
        print(f"  - {patients_without_images} patients without valid images (will use zero tensors)")
        
        # Return ALL patients - no filtering
        return patient_list

    def _should_include_clinical_only(self, patient_entry):
        """Determine if a patient should be included even without images based on clinical data quality."""
        # Include if patient has valid survival time and event data
        try:
            time_val = patient_entry.get('time', None)
            event_val = patient_entry.get('event', None)
            
            # Check if we have valid time and event data
            if time_val is not None and event_val is not None:
                time_float = float(time_val)
                event_int = int(event_val)
                
                # Include if time > 0 and event is 0 or 1
                if time_float > 0 and event_int in [0, 1]:
                    return True
            return False
        except (ValueError, TypeError):
            return False

    def setup(self):
        # Build patient list from NYU data only
        if not self.csv_file or not os.path.exists(self.csv_file):
            raise ValueError(f"NYU CSV file not found: {self.csv_file}")
            
        nyu_patients = self._build_patient_list(self.csv_file, self.dicom_root, "NYU")
        nyu_patients = self._filter_patient_list(nyu_patients)
        
        self.all_patients = nyu_patients
        print(f"Running with NYU data only.")
             
        random.shuffle(self.all_patients) # Shuffle the selected list
        print(f"Total patients: {len(self.all_patients)}")

        if not self.all_patients:
             raise ValueError(f"No valid patients found after filtering.")

        # Preprocess the selected patients if requested
        if self.preprocessed_root:
            print(f"Preprocessing {len(self.all_patients)} NYU patients...")
            temp_all_dataset = HCCDicomDataset(
                patient_data_list=self.all_patients,
                model_type=self.model_type,
                transform=self.transform, 
                num_slices=self.num_slices, 
                num_samples=self.num_samples, 
                preprocessed_root=self.preprocessed_root
            )
            for i in tqdm(range(len(temp_all_dataset)), desc="Preprocessing NYU"):
                try:
                    _ = temp_all_dataset[i] # Triggers loading and saving
                except Exception as e:
                     print(f"[WARN] Error during preprocessing patient index {i} (ID: {self.all_patients[i]['patient_id']}): {e}")
            print("Preprocessing complete.")

        # Setup splits based on whether cross_validation is enabled
        if self.cross_validation:
            # Use ALL patients for cross-validation (as requested by user)
            # First, ensure all patients have valid event/time data by updating self.all_patients directly
            patients_with_clinical = 0
            patients_dicom_only = 0
            
            for i, p in enumerate(self.all_patients):
                if p.get('event') is not None:
                    patients_with_clinical += 1
                else:
                    # DICOM-only patient - assign default values directly to self.all_patients
                    self.all_patients[i]['event'] = 0  # Default to negative event
                    self.all_patients[i]['time'] = 365.0  # Default survival time
                    patients_dicom_only += 1
            
            # Now all patients have valid event data, create CV setup
            cv_patients = self.all_patients
            cv_indices = list(range(len(self.all_patients)))
            patient_events = [p['event'] for p in cv_patients]
            
            print(f"Using ALL {len(cv_patients)} patients for cross-validation splits")
            print(f"  - {patients_with_clinical} patients with clinical data")
            print(f"  - {patients_dicom_only} DICOM-only patients (assigned event=0 for CV)")

            if self.leave_one_out:
                from sklearn.model_selection import LeaveOneOut
                self.cv_splitter = LeaveOneOut()
                # Create splits based on the filtered CV patients, then map back to original indices
                temp_splits = list(self.cv_splitter.split(range(len(cv_patients))))
                self.cv_splits = []
                for train_temp, test_temp in temp_splits:
                    train_orig = [cv_indices[i] for i in train_temp]
                    test_orig = [cv_indices[i] for i in test_temp]
                    self.cv_splits.append((train_orig, test_orig))
            else:
                from sklearn.model_selection import StratifiedKFold, KFold
                # Fall back to stratifying on event label
                if len(np.unique(patient_events)) < 2 or any(np.bincount(patient_events) < self.cv_folds):
                    print("[WARN] Stratification on events not possible; using KFold instead.")
                    kf = KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
                    temp_splits = list(kf.split(range(len(cv_patients))))
                    self.cv_splits = []
                    for train_temp, test_temp in temp_splits:
                        train_orig = [cv_indices[i] for i in train_temp]
                        test_orig = [cv_indices[i] for i in test_temp]
                        self.cv_splits.append((train_orig, test_orig))
                else:
                    skf = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
                    temp_splits = list(skf.split(range(len(cv_patients)), patient_events))
                    self.cv_splits = []
                    for train_temp, test_temp in temp_splits:
                        train_orig = [cv_indices[i] for i in train_temp]
                        test_orig = [cv_indices[i] for i in test_temp]
                        self.cv_splits.append((train_orig, test_orig))

            print(f"Created {len(self.cv_splits)} CV splits.")
            self.current_fold = -1 # Will be incremented in _setup_fold
            
            # Initialize datasets to None - they will be set up when next_fold() is called
            self.train_dataset = None
            self.val_dataset = None
            self.test_dataset = None
        else:
            # Standard train/val/test split - use ALL patients
            # First ensure all patients have valid event/time data
            for i, p in enumerate(self.all_patients):
                if p.get('event') is None:
                    # DICOM-only patient - assign default values
                    self.all_patients[i]['event'] = 0
                    self.all_patients[i]['time'] = 365.0
            
            cv_patients = self.all_patients
            cv_indices = list(range(len(self.all_patients)))
            
            print(f"Using ALL {len(cv_patients)} patients for train/val/test splits")
            
            from sklearn.model_selection import train_test_split
            train_temp, test_temp = train_test_split(
                list(range(len(cv_patients))),
                test_size=0.2,
                stratify=[p['event'] for p in cv_patients],
                random_state=self.random_state
            )
            
            # Map back to original indices
            train_indices = [cv_indices[i] for i in train_temp]
            test_indices = [cv_indices[i] for i in test_temp]
            
            if self.use_validation:
                train_temp, val_temp = train_test_split(
                    train_temp,
                    test_size=0.2,  # 20% of remaining for validation
                    stratify=[cv_patients[i]['event'] for i in train_temp],
                    random_state=self.random_state
                )
                val_indices = [cv_indices[i] for i in val_temp]
                train_indices = [cv_indices[i] for i in train_temp]
                self.val_indices = val_indices
            else:
                self.val_indices = []
            
            self.train_indices = train_indices
            self.test_indices = test_indices
            print(f"Standard split: train={len(self.train_indices)}, val={len(self.val_indices)}, test={len(self.test_indices)}")
            
            # Create datasets for standard split
            train_patients = [self.all_patients[i] for i in self.train_indices]
            val_patients = [self.all_patients[i] for i in self.val_indices] if self.val_indices else []
            test_patients = [self.all_patients[i] for i in self.test_indices]
            
            self.train_dataset = HCCDicomDataset(
                patient_data_list=train_patients,
                model_type=self.model_type,
                transform=self.transform,
                num_slices=self.num_slices,
                num_samples=self.num_samples,
                preprocessed_root=self.preprocessed_root
            )
            
            self.val_dataset = HCCDicomDataset(
                patient_data_list=val_patients,
                model_type=self.model_type,
                transform=self.transform,
                num_slices=self.num_slices,
                num_samples=self.num_samples,
                preprocessed_root=self.preprocessed_root
            ) if val_patients else None
            
            self.test_dataset = HCCDicomDataset(
                patient_data_list=test_patients,
                model_type=self.model_type,
                transform=self.transform,
                num_slices=self.num_slices,
                num_samples=self.num_samples,
                preprocessed_root=self.preprocessed_root
            )

    def _setup_fold(self, fold_idx):
        """Setup datasets for a specific cross-validation fold using the combined patient list."""
        train_indices, test_indices = self.cv_splits[fold_idx]
        
        # Special case for cross-dataset prediction (cv_folds=1)
        # We created a mock split with identical train and test indices
        # For this case, we need to ensure test_indices is not actually used
        # because the cross_predict dataset will be used for testing instead
        if self.cv_folds == 1 and train_indices == test_indices:
            print("[INFO] Using all dataset for training in single-fold mode (test set will be from another dataset)")
            mock_test_mode = True
        else:
            mock_test_mode = False
        
        # Get the patient data dictionaries for this fold
        fold_train_val_patients = [self.all_patients[i] for i in train_indices]
        # In mock test mode, create an empty test set as it will be replaced by cross_predict_data_module
        fold_test_patients = [] if mock_test_mode else [self.all_patients[i] for i in test_indices]

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
        if not hasattr(self, 'train_dataset') or self.train_dataset is None:
            if self.cross_validation:
                raise ValueError("Train dataset not setup. For cross-validation, call next_fold() first to set up the fold.")
            else:
                raise ValueError("Train dataset not setup. Call setup() first.")
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
        if not hasattr(self, 'val_dataset') or self.val_dataset is None:
            if self.cross_validation:
                print("[WARN] Validation dataset is empty for this fold.")
                return None
            else:
                raise ValueError("Validation dataset not setup. Call setup() first.")
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
        if not hasattr(self, 'test_dataset') or self.test_dataset is None:
            if self.cross_validation:
                raise ValueError("Test dataset not setup. For cross-validation, call next_fold() first to set up the fold.")
            else:
                raise ValueError("Test dataset not setup. Call setup() first.")
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

