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
            if p99 > p1:
                image_norm = np.clip((image - p1) / (p99 - p1), 0, 1)
            else:
                image_norm = np.clip(image / (image.max() + 1e-8), 0, 1)
        
        elif method == 'zscore_robust':
            # Robust z-score using median and MAD
            median = np.median(image)
            mad = np.median(np.abs(image - median))
            if mad > 0:
                image_norm = (image - median) / (mad * 1.4826)  # 1.4826 makes MAD consistent with std
                # Clip to reasonable range and normalize to [0,1]
                image_norm = np.clip(image_norm, -3, 3)
                image_norm = (image_norm + 3) / 6
            else:
                image_norm = np.clip(image / (image.max() + 1e-8), 0, 1)
        
        elif method == 'histogram_matching':
            # Use adaptive histogram equalization
            image_norm = exposure.equalize_adapthist(image, clip_limit=0.02)
        
        return image_norm.astype(np.float32)
    
    def sequence_specific_preprocessing(self, image, sequence_name=''):
        """
        Apply sequence-specific preprocessing based on sequence type.
        """
        sequence_name = sequence_name.lower()
        
        # Normalize to [0,1] first
        if image.max() > 1:
            image = image / (image.max() + 1e-8)
        
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
        # Remove noise while preserving edges using scipy
        image = ndimage.median_filter(image, size=3)
        
        # Enhance contrast adaptively
        image = exposure.equalize_adapthist(image, clip_limit=0.02)
        
        # Slight sharpening using OpenCV
        if len(image.shape) == 2:  # 2D image
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]], dtype=np.float32)
            image_sharp = cv2.filter2D(image.astype(np.float32), -1, kernel * 0.1)
            image = 0.9 * image + 0.1 * image_sharp
        
        return np.clip(image, 0, 1)
    
    def domain_adaptation_normalization(self, image, target_mean=0.5, target_std=0.2):
        """
        Normalize images to have consistent statistics across domains.
        """
        # Compute current statistics
        current_mean = np.mean(image)
        current_std = np.std(image)
        
        if current_std > 1e-8:
            # Standardize then rescale to target distribution
            image_norm = (image - current_mean) / current_std
            image_norm = image_norm * target_std + target_mean
            image_norm = np.clip(image_norm, 0, 1)
        else:
            image_norm = np.full_like(image, target_mean)
            
        return image_norm
    
    def preprocess_slice(self, image, sequence_name='', apply_domain_adaptation=True):
        """
        Complete preprocessing pipeline for a single slice.
        """
        # Ensure image is 2D
        if len(image.shape) > 2:
            image = image.squeeze()
        
        # Convert to float32
        image = image.astype(np.float32)
        
        # Handle edge cases
        if np.all(image == 0):
            return np.zeros_like(image)
        
        try:
            # Step 1: Sequence-specific preprocessing
            image = self.sequence_specific_preprocessing(image, sequence_name)
            
            # Step 2: Enhance image quality
            image = self.enhance_image_quality(image)
            
            # Step 3: Domain adaptation normalization
            if apply_domain_adaptation:
                image = self.domain_adaptation_normalization(image)
            
            # Step 4: Final intensity normalization to [0, 1]
            image = np.clip(image, 0, 1)
            
        except Exception as e:
            print(f"Warning: Preprocessing failed for sequence '{sequence_name}': {e}")
            # Fallback to simple normalization
            if image.max() > image.min():
                image = (image - image.min()) / (image.max() - image.min())
            else:
                image = np.zeros_like(image)
        
        return image

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
                
                print(f"  ✅ Successfully processed {patient_id}")
                print(f"     Original: shape={original.shape}, range=[{original.min():.1f}, {original.max():.1f}], mean={original.mean():.1f}")
                print(f"     Improved: shape={improved.shape}, range=[{improved.min():.3f}, {improved.max():.3f}], mean={improved.mean():.3f}")
                print(f"     Series: {series_desc}")
                
            except Exception as e:
                print(f"  ❌ Error processing {patient_id}: {e}")
        else:
            print(f"  ❌ No DICOM files found for {patient_id}")

def create_patched_visualize_script():
    """
    Create a modified version of visualize.py that uses improved preprocessing.
    """
    print("\n=== CREATING PATCHED VISUALIZATION SCRIPT ===")
    
    # Read the original visualize.py
    with open('visualize.py', 'r') as f:
        original_content = f.read()
    
    # Create the patched version
    patched_content = f'''#!/usr/bin/env python

# PATCHED VERSION WITH IMPROVED PREPROCESSING
from improved_preprocessing_v2 import ImprovedPreprocessor

# Global preprocessor
GLOBAL_PREPROCESSOR = ImprovedPreprocessor()

{original_content}

# Patch the extract_features function to use improved preprocessing
original_extract_features = extract_features

def extract_features_improved(data_loader, model, device, output_dir='debug_images'):
    """
    Extract features using improved preprocessing for outlier patients.
    """
    import torchvision.utils as vutils
    import torchvision.transforms as transforms

    model.eval()
    features = []
    image_save_dir = None
    if output_dir:
        image_save_dir = os.path.join(output_dir, "debug_images_improved")
        os.makedirs(image_save_dir, exist_ok=True)

    # Create transform for DINOv2 (224x224)
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(data_loader, desc="Extracting Features (Improved)")):
            images, _, patient_data = batch  # Get patient data to access series info
            images = images.to(device)
            batch_size, num_samples, num_slices, C, H, W = images.size()

            # Apply improved preprocessing to each slice
            improved_images = []
            for b in range(batch_size):
                patient_improved_samples = []
                for s in range(num_samples):
                    sample_improved_slices = []
                    for sl in range(num_slices):
                        # Get original slice
                        slice_tensor = images[b, s, sl]  # shape: (C, H, W)
                        
                        # Convert to numpy for preprocessing
                        if slice_tensor.shape[0] == 1:  # Grayscale
                            slice_np = slice_tensor.squeeze(0).cpu().numpy()
                        else:
                            # If RGB, convert to grayscale for medical images
                            slice_np = torch.mean(slice_tensor, dim=0).cpu().numpy()
                        
                        # Apply improved preprocessing
                        try:
                            # Get series description if available
                            series_desc = ''
                            if hasattr(patient_data, '__iter__') and len(patient_data) > b:
                                series_desc = getattr(patient_data[b], 'series_description', '')
                            
                            processed_slice = GLOBAL_PREPROCESSOR.preprocess_slice(slice_np, series_desc)
                            
                            # Convert back to tensor and apply DINOv2 transform
                            processed_slice = (processed_slice * 255).astype(np.uint8)
                            processed_tensor = transform(processed_slice).unsqueeze(0)  # Add batch dim
                            
                        except Exception as e:
                            print(f"Warning: Preprocessing failed for batch {{batch_idx}}, using original: {{e}}")
                            # Fallback to original processing
                            processed_tensor = slice_tensor.unsqueeze(0)
                        
                        sample_improved_slices.append(processed_tensor)
                    
                    # Stack slices
                    sample_tensor = torch.cat(sample_improved_slices, dim=0)  # (num_slices, C, H, W)
                    patient_improved_samples.append(sample_tensor)
                
                # Stack samples
                patient_tensor = torch.stack(patient_improved_samples, dim=0)  # (num_samples, num_slices, C, H, W)
                improved_images.append(patient_tensor)
            
            # Stack batch
            improved_batch = torch.stack(improved_images, dim=0).to(device)  # (batch_size, num_samples, num_slices, C, H, W)

            # Save sample images
            if image_save_dir and batch_idx < 5:
                sample_tensor = improved_batch[0, 0]  # First patient, first sample
                num_display = min(5, sample_tensor.shape[0])
                for i in range(num_display):
                    slice_tensor = sample_tensor[i]  # shape: (C, H, W)
                    
                    # Unnormalize for display
                    mean = torch.tensor([0.485, 0.456, 0.406], device=slice_tensor.device).view(3, 1, 1)
                    std = torch.tensor([0.229, 0.224, 0.225], device=slice_tensor.device).view(3, 1, 1)
                    
                    if slice_tensor.shape[0] == 1:  # Grayscale
                        img = slice_tensor.squeeze(0).cpu().numpy()
                        plt.imshow(img, cmap="gray")
                    else:
                        unnorm = slice_tensor * std + mean
                        img = unnorm.permute(1, 2, 0).cpu().numpy()
                        img = np.clip(img, 0, 1)
                        plt.imshow(img)

                    plt.axis("off")
                    plt.title(f"Improved Batch {{batch_idx}} - Slice {{i}}")
                    plt.savefig(os.path.join(image_save_dir, f"improved_batch{{batch_idx}}_slice{{i}}.png"))
                    plt.close()

            # Reshape for feature extraction
            batch_size, num_samples, num_slices, C, H, W = improved_batch.size()
            improved_batch = improved_batch.view(batch_size * num_samples * num_slices, C, H, W)
            feats = model.forward_features(improved_batch)
            feature_dim = feats.size(-1)
            feats = feats.view(batch_size, num_samples, num_slices, feature_dim)
            features.append(feats.cpu().numpy())

    features = np.concatenate(features, axis=0)
    return features

# Replace the extract_features function
extract_features = extract_features_improved
'''

    # Write the patched version
    with open('visualize_improved.py', 'w') as f:
        f.write(patched_content)
    
    print("✅ Created visualize_improved.py with enhanced preprocessing!")

if __name__ == "__main__":
    test_preprocessing_on_outliers()
    create_patched_visualize_script() 