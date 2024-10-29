import os
from torch.utils.data import Dataset
import torch
import pydicom
import nibabel as nib
import numpy as np
from PIL import Image
from torchvision import transforms as T

class UnlabeledMedicalImageDataset(Dataset):
    def __init__(self, root, extra=None, transforms=None, transform=None, target_transform=None):
        """
        Args:
            root (str): Directory with all the medical images.
            extra (str): Directory for storing extra files like metadata.
            transforms (callable, optional): Transform to be applied on the image after preprocessing.
            transform (callable, optional): Additional transform to be applied specifically to individual images.
            target_transform (callable, optional): Transform to be applied to the target, if available.
        """
        self.root = root
        self.extra = extra if extra else os.path.join(root, "extra_files")
        self.transform = transform
        self.target_transform = target_transform
        self.samples = []

        # Use the transforms passed as an argument or define default ones
        if transforms is not None:
            self.transforms = transforms
        else:
            self.transforms = T.Compose([
                T.ToTensor(),  # Convert PIL Image or ndarray to tensor
                T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])

        # Recursively collect all file paths with the desired extensions
        for dirpath, _, filenames in os.walk(self.root):
            for file in filenames:
                if file.lower().endswith(('.dcm', '.dicom', '.nii', '.nii.gz')):
                    full_path = os.path.join(dirpath, file)
                    self.samples.append(full_path)

        if len(self.samples) == 0:
            raise ValueError(f"No DICOM or NIfTI files found in {root}")

        print(f"Found {len(self.samples)} medical image files in {self.root}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_path = self.samples[idx]
        print(f"\nLoading file: {file_path}")

        # Load the image data based on file type
        if file_path.lower().endswith(('.dcm', '.dicom')):
            # Load DICOM file
            dicom_data = pydicom.dcmread(file_path)
            image = dicom_data.pixel_array
            target = self.extract_target(dicom_data)
            print(f"DICOM Image shape (raw): {image.shape}, Type: {image.dtype}")
        elif file_path.lower().endswith(('.nii', '.nii.gz')):
            # Load NIfTI file
            nii_data = nib.load(file_path)
            image = nii_data.get_fdata()
            print(f"NIfTI Image shape (raw): {image.shape}, Type: {image.dtype}")
            target = self.extract_target(nii_data)
        else:
            raise ValueError(f'Unsupported file type: {file_path}')

        # Preprocess the image
        image = self.preprocess_image(image).numpy()  # Convert torch.Tensor to ndarray
        print(f"Image shape after preprocessing: {image.shape}, Type: {image.dtype}")

        # Convert ndarray to PIL Image if transforms expect it
        if isinstance(image, np.ndarray):
            # Ensure image is in HxWxC format for PIL
            if image.ndim == 3 and image.shape[0] in [1, 3]:  # [C, H, W] to [H, W, C]
                image = np.transpose(image, (1, 2, 0))
                print(f"Transposed image to HxWxC format, new shape: {image.shape}")

            try:
                image = Image.fromarray((image * 255).astype(np.uint8))  # Rescale for uint8
                print("Converted image to PIL format for transforms")
            except TypeError as e:
                print(f"Error converting to PIL Image: {e}")
                raise ValueError(f"Incompatible image shape for PIL conversion: {image.shape}")

        # Apply individual image transform if any
        if self.transform:
            image = self.transform(image)
            print(f"Applied individual image transform, type after transform: {type(image)}")
            # If the transform returns a dict, extract 'image' and 'target'
            if isinstance(image, dict):
                target = image.get('target', target)
                image = image.get('image', image)
                print(f"Extracted image and target from dict, image type: {type(image)}")

        # Apply any additional transforms if provided
        if self.transforms:
            image = self.transforms(image)
            print(f"Applied additional transforms, type after transforms: {type(image)}")

        # Apply target transform if provided
        if self.target_transform:
            target = self.target_transform(target)
            print("Applied target transform")

        return image, target

    def preprocess_image(self, image):
        # Convert image to float32 for consistency and normalize it
        image = image.astype(np.float32)

        # Normalize the image to [0, 1] range
        min_val = np.min(image)
        max_val = np.max(image)
        if max_val > min_val:
            image = (image - min_val) / (max_val - min_val)
        else:
            image = np.zeros_like(image)

        # Handle image dimensions
        if image.ndim == 2:
            # 2D grayscale image, add channel dimension
            image = np.expand_dims(image, axis=0)  # [C, H, W]
            print(f"Added channel dimension, new shape: {image.shape}")
        elif image.ndim == 3:
            # Handle 3D data by selecting a middle slice
            slice_dim = np.argmin(image.shape)
            if slice_dim != 0:
                image = np.moveaxis(image, slice_dim, 0)
                print(f"Moved slice dimension to first axis, new shape: {image.shape}")
            middle_slice = image.shape[0] // 2
            image = image[middle_slice, :, :]
            image = np.expand_dims(image, axis=0)  # [C, H, W]
            print(f"Selected middle slice, new shape: {image.shape}")
        else:
            raise ValueError(f'Unsupported image dimensions: {image.shape}')

        # Convert to torch tensor
        image = torch.from_numpy(image)
        print(f"Image shape after conversion to tensor: {image.shape}, Type: {image.dtype}")

        # Broadcast grayscale to RGB if needed
        if image.shape[0] == 1:  # If single channel
            image = image.repeat(3, 1, 1)  # [3, H, W]
            print(f"Broadcasted grayscale to RGB, new shape: {image.shape}")

        return image

    def extract_target(self, data):
        """
        Placeholder method to generate or extract a target from image metadata.
        This can be replaced with user-defined logic.
        """
        if isinstance(data, pydicom.dataset.FileDataset):
            # Extract any metadata as target, e.g., patient ID or study description
            target = data.PatientID if 'PatientID' in data else "Unknown"
            print(f"Extracted DICOM target: {target}")
            return target
        elif isinstance(data, nib.Nifti1Image):
            # Optionally extract NIfTI metadata
            target = data.header.get('descrip', "No Description")
            print(f"Extracted NIfTI target: {target}")
            return target
        return None

    def _get_extra_full_path(self, extra_path: str) -> str:
        """
        Returns the full path to an extra file given its relative path.
        """
        return os.path.join(self.extra, extra_path)

    def _load_extra(self, extra_path: str) -> np.ndarray:
        """
        Loads extra dataset-related data from a .npy file.
        """
        extra_full_path = self._get_extra_full_path(extra_path)
        print(f"Loading extra file from {extra_full_path}")
        return np.load(extra_full_path, mmap_mode="r")

    def _save_extra(self, extra_array: np.ndarray, extra_path: str) -> None:
        """
        Saves extra dataset-related data to a .npy file.
        """
        extra_full_path = self._get_extra_full_path(extra_path)
        os.makedirs(self.extra, exist_ok=True)
        np.save(extra_full_path, extra_array)
        print(f"Saved extra data to {extra_full_path}")
