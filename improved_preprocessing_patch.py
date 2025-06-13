import torch
import numpy as np
from improved_preprocessing_v2 import ImprovedPreprocessor

# Global preprocessor instance
IMPROVED_PREPROCESSOR = ImprovedPreprocessor()

def improved_preprocess_dicom_slice(pixel_array, series_description=''):
    """
    Improved preprocessing function for DICOM slices.
    """
    # Convert to float and normalize basic range
    if pixel_array.dtype != np.float32:
        pixel_array = pixel_array.astype(np.float32)
    
    # Apply improved preprocessing
    processed = IMPROVED_PREPROCESSOR.preprocess_slice(
        pixel_array, 
        sequence_name=series_description,
        apply_domain_adaptation=True
    )
    
    return processed

# Monkey patch the dataset class
def patch_dataset_preprocessing():
    """
    Patch the existing dataset to use improved preprocessing.
    """
    import sys
    sys.path.append('/gpfs/data/shenlab/wz1492/HCC')
    
    try:
        # Newer codebase uses HCCDicomDataset; older versions used HCCDataset.
        try:
            from data.dataset import HCCDicomDataset as _Dataset  # New name
            dataset_type = 'HCCDicomDataset'
        except ImportError:
            from data.dataset import HCCDataset as _Dataset       # Fallback to old name
            dataset_type = 'HCCDataset'

        # ------------------------------
        # 1) If the target dataset has dicom_to_tensor (new implementation)
        #    we patch that so every slice goes through the improved pipeline.
        # ------------------------------
        if hasattr(_Dataset, 'dicom_to_tensor'):
            if not hasattr(_Dataset, '_original_dicom_to_tensor'):
                _Dataset._original_dicom_to_tensor = _Dataset.dicom_to_tensor

            def improved_dicom_to_tensor(self, dcm, *args, **kwargs):
                """Convert a pydicom dataset to a 3-channel (3,224,224) tensor with improved preprocessing."""
                series_desc = getattr(dcm, 'SeriesDescription', '')
                # Apply advanced preprocessing on the raw pixel data
                processed = improved_preprocess_dicom_slice(dcm.pixel_array, series_desc)

                # Ensure numpy array, float32, range [0,1]
                img = processed.astype(np.float32)

                # Convert to 3-channel RGB-like array expected by ViT backbones
                if img.ndim == 2:
                    img = np.repeat(img[..., np.newaxis], 3, axis=-1)
                elif img.ndim == 3 and img.shape[2] == 1:
                    img = np.repeat(img, 3, axis=-1)
                elif img.ndim == 3 and img.shape[2] != 3:
                    # Unexpected number of channels → take first channel and replicate
                    img = np.repeat(img[:, :, 0:1], 3, axis=-1)

                tensor_img = torch.from_numpy(img).permute(2, 0, 1)  # (C,H,W)

                # Resize to (224,224) if needed
                if tensor_img.shape[1:] != (224, 224):
                    tensor_img = torch.nn.functional.interpolate(
                        tensor_img.unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False
                    ).squeeze(0)
                return tensor_img

            _Dataset.dicom_to_tensor = improved_dicom_to_tensor
            print(f"✅ Successfully patched {dataset_type}.dicom_to_tensor with improved preprocessing!")

        # ------------------------------
        # 2) Legacy fallback: if dataset exposes preprocess_slice we patch that.
        # ------------------------------
        elif hasattr(_Dataset, 'preprocess_slice'):
            if not hasattr(_Dataset, '_original_preprocess_slice'):
                _Dataset._original_preprocess_slice = _Dataset.preprocess_slice

            def improved_preprocess_slice_method(self, pixel_array, **kwargs):
                series_desc = getattr(self, 'current_series_description', '')
                return improved_preprocess_dicom_slice(pixel_array, series_desc)

            _Dataset.preprocess_slice = improved_preprocess_slice_method
            print(f"✅ Successfully patched {dataset_type}.preprocess_slice with improved preprocessing!")
        else:
            raise AttributeError(f"{dataset_type} has no dicom_to_tensor or preprocess_slice methods to patch.")

    except Exception as e:
        print(f"❌ Failed to patch dataset: {e}")
        print("Will apply preprocessing during visualization instead.")

if __name__ == "__main__":
    patch_dataset_preprocessing()
