# Patch `HCCDicomDataset.dicom_to_tensor` so all training/val/test pipelines use the
# improved 1–99% percentile normalisation + CLAHE + domain-adaptation.

from improved_preprocessing import ImprovedPreprocessor

_IMPROVED = ImprovedPreprocessor()

def _patched_dicom_to_tensor(self, dcm):
    """Replacement for `HCCDicomDataset.dicom_to_tensor` that applies the
    advanced normalisation before converting to a 3×224×224 tensor."""
    import numpy as np
    import torch

    raw = dcm.pixel_array.astype(np.float32)
    series_desc = getattr(dcm, 'SeriesDescription', '') or ''

    # Run improved preprocessing
    processed = _IMPROVED.preprocess_slice(raw, sequence_name=series_desc, apply_domain_adaptation=True)

    # Ensure [0,1]
    processed = np.clip(processed, 0, 1)

    if processed.ndim == 2:  # H×W → H×W×3
        processed = np.repeat(processed[..., np.newaxis], 3, axis=-1)
    elif processed.ndim == 3 and processed.shape[2] == 1:
        processed = np.repeat(processed, 3, axis=-1)

    tensor_img = torch.from_numpy(processed).permute(2, 0, 1)

    # Resize to 224×224 if needed
    if tensor_img.shape[1:] != (224, 224):
        tensor_img = torch.nn.functional.interpolate(tensor_img.unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False).squeeze(0)
    return tensor_img


def patch_dataset_preprocessing():
    import sys
    sys.path.append('/gpfs/data/shenlab/wz1492/HCC')
    try:
        from data.dataset import HCCDicomDataset

        if not hasattr(HCCDicomDataset, '_original_dicom_to_tensor'):
            HCCDicomDataset._original_dicom_to_tensor = HCCDicomDataset.dicom_to_tensor

        HCCDicomDataset.dicom_to_tensor = _patched_dicom_to_tensor
        print('✅ Patched HCCDicomDataset.dicom_to_tensor with improved preprocessing')
    except Exception as e:
        print(f'❌ Failed to patch dataset: {e}')
        print('Training will proceed with original preprocessing.')

if __name__ == "__main__":
    patch_dataset_preprocessing()
