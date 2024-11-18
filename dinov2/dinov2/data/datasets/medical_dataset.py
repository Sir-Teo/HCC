import os
import torch
import pydicom
import nibabel as nib
import numpy as np
from PIL import Image
from torchvision import transforms as T
from typing import Any, Tuple
from .decoders import TargetDecoder, ImageDataDecoder
from io import BytesIO
from .extended import ExtendedVisionDataset

class UnlabeledMedicalImageDataset(ExtendedVisionDataset):
    def __init__(self, root, extra=None, transforms=None, transform=None, target_transform=None):
        super().__init__(root, transforms, transform, target_transform)
        self.root = root
        self.extra = extra if extra else os.path.join(root, "extra_files")

        # Gather samples from the root directory
        self.samples = []
        for dirpath, _, filenames in os.walk(self.root):
            for file in filenames:
                full_path = os.path.join(dirpath, file)
                if file.lower().endswith(('.dcm', '.dicom')):
                    # For DICOM files, append as is
                    self.samples.append((full_path, None))
                elif file.lower().endswith(('.nii', '.nii.gz')):
                    # For NIfTI files, append each slice as a separate sample
                    nii_data = nib.load(full_path)
                    shape = nii_data.shape  # Get the shape without loading the entire data
                    slicing_axis = 2  # Choose the axis along which to slice (usually the axial axis)
                    num_slices = shape[slicing_axis]
                    for slice_index in range(num_slices):
                        self.samples.append((full_path, slice_index))
                else:
                    continue

        if len(self.samples) == 0:
            raise ValueError(f"No DICOM or NIfTI files found in {root}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        try:
            # Get image data and decode it
            image_data = self.get_image_data(index)
            image = ImageDataDecoder(image_data).decode()
        except Exception as e:
            raise RuntimeError(f"Cannot read image for sample {index}") from e

        # Get target and decode it
        target = self.get_target(index)
        target = TargetDecoder(target).decode()

        # Apply transformations, if any
        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    def get_image_data(self, idx):
        file_path, slice_index = self.samples[idx]

        # Load the image and preprocess
        if file_path.lower().endswith(('.dcm', '.dicom')):
            dicom_data = pydicom.dcmread(file_path)

            # Fix the 'Bits Stored' warning if necessary
            if dicom_data.file_meta.TransferSyntaxUID.is_compressed:
                pixel_array = dicom_data.pixel_array
                actual_bits_stored = pixel_array.dtype.itemsize * 8
                if 'BitsStored' in dicom_data and dicom_data.BitsStored != actual_bits_stored:
                    dicom_data.BitsStored = actual_bits_stored

            image = dicom_data.pixel_array
        elif file_path.lower().endswith(('.nii', '.nii.gz')):
            nii_data = nib.load(file_path)
            # Read only the required slice
            slicing_axis = 2  # Same as in __init__
            # Build slicing object to get only the desired slice
            slice_obj = [slice(None)] * 3  # Assuming data is 3D
            slice_obj[slicing_axis] = slice(slice_index, slice_index + 1)
            image_slice = nii_data.dataobj[tuple(slice_obj)]
            image_slice = np.squeeze(image_slice)  # Remove the singleton dimension
            image = image_slice
        else:
            raise ValueError(f'Unsupported file type: {file_path}')

        # Preprocess image
        image = self.preprocess_image(image).numpy()
        if isinstance(image, np.ndarray):
            if image.ndim == 3 and image.shape[0] in [1, 3]:  # [C, H, W] to [H, W, C]
                image = np.transpose(image, (1, 2, 0))
            image = Image.fromarray((image * 255).astype(np.uint8))

        # Apply image-level transforms if any
        if self.transform:
            transformed_output = self.transform(image)
            if isinstance(transformed_output, dict):
                image = transformed_output.get('image', image)
            else:
                image = transformed_output

        # Convert the image to bytes
        with BytesIO() as output:
            image.save(output, format="PNG")  # Choose appropriate format (e.g., 'JPEG', 'PNG')
            image_bytes = output.getvalue()

        return image_bytes


    def get_target(self, idx):
        file_path, slice_index = self.samples[idx]

        if file_path.lower().endswith(('.dcm', '.dicom')):
            dicom_data = pydicom.dcmread(file_path)
            return self.extract_target(dicom_data)
        elif file_path.lower().endswith(('.nii', '.nii.gz')):
            nii_data = nib.load(file_path)
            return self.extract_target(nii_data)
        return None

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
            image = np.expand_dims(image, axis=0)  # [C, H, W]
        elif image.ndim == 3:
            # If the image is still 3D after slicing, select the middle slice
            slice_dim = np.argmin(image.shape)
            if slice_dim != 0:
                image = np.moveaxis(image, slice_dim, 0)
            middle_slice = image.shape[0] // 2
            image = image[middle_slice, :, :]
            image = np.expand_dims(image, axis=0)  # [C, H, W]
        else:
            raise ValueError(f'Unsupported image dimensions: {image.shape}')

        # Convert to torch tensor and broadcast grayscale to RGB if needed
        image = torch.from_numpy(image)
        if image.shape[0] == 1:
            image = image.repeat(3, 1, 1)  # [3, H, W]

        return image

    def extract_target(self, data):
        if isinstance(data, pydicom.dataset.FileDataset):
            target = data.PatientID if 'PatientID' in data else "Unknown"
            return target
        elif isinstance(data, nib.Nifti1Image):
            target = data.header.get('descrip', "No Description")
            return target
        return None
