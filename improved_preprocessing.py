#!/usr/bin/env python

import os
import numpy as np
import torch
import cv2
from pathlib import Path
import pydicom
from scipy import ndimage
from skimage import exposure, filters
import warnings
warnings.filterwarnings('ignore')

class ImprovedPreprocessor:
    """
    Improved preprocessing pipeline to handle domain shift and outlier patients.
    """
    
    def __init__(self):
        self.intensity_stats = {}
        
    def robust_intensity_normalization(self, image, method='percentile'):
        """
        Robust intensity normalization to handle different sequence types.
        """
        if method == 'percentile':
            # Use percentile-based normalization (more robust than min-max)
            p1, p99 = np.percentile(image, [1, 99])
            image_norm = np.clip((image - p1) / (p99 - p1), 0, 1)
        
        elif method == 'zscore_robust':
            # Robust z-score using median and MAD
            median = np.median(image)
            mad = np.median(np.abs(image - median))
            image_norm = (image - median) / (mad * 1.4826)  # 1.4826 makes MAD consistent with std
            # Clip to reasonable range and normalize to [0,1]
            image_norm = np.clip(image_norm, -3, 3)
            image_norm = (image_norm + 3) / 6
        
        elif method == 'histogram_matching':
            # Match histogram to a reference (will implement if needed)
            # For now, use adaptive histogram equalization
            image_norm = exposure.equalize_adapthist(image, clip_limit=0.02)
        
        return image_norm.astype(np.float32)
    
    def sequence_specific_preprocessing(self, image, sequence_name=''):
        """
        Apply sequence-specific preprocessing based on sequence type.
        """
        sequence_name = sequence_name.lower()
        
        if 'diffusion' in sequence_name or 'dwi' in sequence_name:
            # Diffusion sequences: enhance contrast, reduce noise
            image = filters.gaussian(image, sigma=0.5)  # Light denoising
            image = exposure.rescale_intensity(image, out_range=(0, 1))
            # Enhance contrast for diffusion
            image = exposure.adjust_gamma(image, gamma=0.8)
            
        elif 'haste' in sequence_name or 't2' in sequence_name:
            # T2/HASTE sequences: different intensity characteristics
            image = exposure.equalize_adapthist(image, clip_limit=0.03)
            
        elif 'vibe' in sequence_name or 't1' in sequence_name:
            # T1/VIBE sequences: standard processing
            image = self.robust_intensity_normalization(image, method='percentile')
            
        elif 'fiesta' in sequence_name or 'fsfiesta' in sequence_name:
            # Old GE sequences: need special handling
            # These often have different intensity ranges
            image = exposure.rescale_intensity(image, out_range=(0, 1))
            image = exposure.equalize_adapthist(image, clip_limit=0.02)
            
        else:
            # Default: robust percentile normalization
            image = self.robust_intensity_normalization(image, method='percentile')
            
        return image
    
    def enhance_image_quality(self, image):
        """
        Apply image quality enhancements.
        """
        # Remove noise while preserving edges
        image = filters.median(image, selem=np.ones((3, 3)))
        
        # Enhance contrast adaptively
        image = exposure.equalize_adapthist(image, clip_limit=0.02)
        
        # Slight sharpening
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        image_sharp = cv2.filter2D(image, -1, kernel * 0.1)
        image = 0.9 * image + 0.1 * image_sharp
        
        return np.clip(image, 0, 1)
    
    def domain_adaptation_normalization(self, image, target_mean=0.5, target_std=0.2):
        """
        Normalize images to have consistent statistics across domains.
        """
        # Compute current statistics
        current_mean = np.mean(image)
        current_std = np.std(image)
        
        if current_std > 0:
            # Standardize then rescale to target distribution
            image_norm = (image - current_mean) / current_std
            image_norm = image_norm * target_std + target_mean
            image_norm = np.clip(image_norm, 0, 1)
        else:
            image_norm = image
            
        return image_norm
    
    def preprocess_slice(self, image, sequence_name='', apply_domain_adaptation=True):
        """
        Complete preprocessing pipeline for a single slice.
        """
        # Step 1: Sequence-specific preprocessing
        image = self.sequence_specific_preprocessing(image, sequence_name)
        
        # Step 2: Enhance image quality
        image = self.enhance_image_quality(image)
        
        # Step 3: Domain adaptation normalization
        if apply_domain_adaptation:
            image = self.domain_adaptation_normalization(image)
        
        # Step 4: Final intensity normalization to [0, 1]
        image = np.clip(image, 0, 1)
        
        return image

def create_improved_preprocessing_script():
    """
    Create a script that patches the existing data loading to use improved preprocessing.
    """
    
    script_content = '''
import torch
import numpy as np
from improved_preprocessing import ImprovedPreprocessor

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
        from data.dataset import HCCDataset
        
        # Store original method
        if not hasattr(HCCDataset, '_original_preprocess_slice'):
            HCCDataset._original_preprocess_slice = HCCDataset.preprocess_slice if hasattr(HCCDataset, 'preprocess_slice') else None
        
        # Replace with improved version
        def improved_preprocess_slice_method(self, pixel_array, **kwargs):
            series_desc = getattr(self, 'current_series_description', '')
            return improved_preprocess_dicom_slice(pixel_array, series_desc)
        
        HCCDataset.preprocess_slice = improved_preprocess_slice_method
        print("✅ Successfully patched dataset preprocessing!")
        
    except Exception as e:
        print(f"❌ Failed to patch dataset: {e}")
        print("Will apply preprocessing during visualization instead.")

if __name__ == "__main__":
    patch_dataset_preprocessing()
'''
    
    with open('improved_preprocessing_patch.py', 'w') as f:
        f.write(script_content)
    
    print("Created improved preprocessing patch script!")

def test_preprocessing_on_outliers():
    """
    Test the improved preprocessing on outlier patient images.
    """
    print("=== TESTING IMPROVED PREPROCESSING ON OUTLIERS ===")
    
    outlier_paths = {
        'TCGA-DD-A1EH': '/gpfs/data/mankowskilab/HCC/data/TCGA/manifest-4lZjKqlp5793425118292424834/TCGA-LIHC',
        '16946681': '/gpfs/data/mankowskilab/HCC_Recurrence/dicom'
    }
    
    preprocessor = ImprovedPreprocessor()
    
    for patient_id, root_path in outlier_paths.items():
        print(f"\nTesting {patient_id}...")
        
        # Find a DICOM file
        if 'TCGA' in patient_id:
            dcm_files = list(Path(root_path).glob(f"**/{patient_id}/**/*.dcm"))
        else:
            dcm_files = list(Path(root_path).glob(f"**/{patient_id}/**/*.dcm"))
            if not dcm_files:
                dcm_files = list(Path(root_path).glob(f"**/*{patient_id}*/**/*.dcm"))
        
        if dcm_files:
            try:
                # Load and preprocess
                ds = pydicom.dcmread(dcm_files[0], force=True)
                original = ds.pixel_array.astype(np.float32)
                series_desc = getattr(ds, 'SeriesDescription', '')
                
                # Apply improved preprocessing
                improved = preprocessor.preprocess_slice(original, series_desc)
                
                print(f"  Original: shape={original.shape}, range=[{original.min():.1f}, {original.max():.1f}], mean={original.mean():.1f}")
                print(f"  Improved: shape={improved.shape}, range=[{improved.min():.3f}, {improved.max():.3f}], mean={improved.mean():.3f}")
                print(f"  Series: {series_desc}")
                
            except Exception as e:
                print(f"  ❌ Error processing {patient_id}: {e}")
        else:
            print(f"  ❌ No DICOM files found for {patient_id}")

if __name__ == "__main__":
    create_improved_preprocessing_script()
    test_preprocessing_on_outliers() 