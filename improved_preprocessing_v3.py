"""
Enhanced Preprocessing v3 - Advanced techniques for improved HCC data preprocessing
Focus on maximizing AUC and precision through better data quality and feature extraction
"""

import numpy as np
import torch
import torch.nn.functional as F
from skimage import filters, exposure, restoration, morphology, segmentation
from skimage.feature import local_binary_pattern
from scipy import ndimage
import cv2

class AdvancedHCCPreprocessor:
    """
    Advanced preprocessing pipeline for HCC images with focus on precision
    """
    
    def __init__(self, target_size=(224, 224), enhance_contrast=True, noise_reduction=True):
        self.target_size = target_size
        self.enhance_contrast = enhance_contrast
        self.noise_reduction = noise_reduction
        
    def adaptive_histogram_equalization(self, image):
        """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)"""
        # Convert to uint8 for CLAHE
        image_uint8 = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)
        
        # Apply CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(image_uint8)
        
        # Convert back to float
        return enhanced.astype(np.float32) / 255.0
    
    def advanced_noise_reduction(self, image):
        """Multi-scale noise reduction"""
        # Gaussian denoising
        denoised_gaussian = filters.gaussian(image, sigma=0.5, preserve_range=True)
        
        # Non-local means denoising
        try:
            denoised_nlm = restoration.denoise_nl_means(
                image, h=0.1, patch_size=7, patch_distance=11,
                fast_mode=True, preserve_range=True
            )
        except:
            denoised_nlm = denoised_gaussian
        
        # Bilateral filtering
        image_uint8 = (image * 255).astype(np.uint8)
        bilateral = cv2.bilateralFilter(image_uint8, 9, 75, 75)
        denoised_bilateral = bilateral.astype(np.float32) / 255.0
        
        # Weighted combination
        final_denoised = (
            0.4 * denoised_gaussian + 
            0.4 * denoised_nlm + 
            0.2 * denoised_bilateral
        )
        
        return final_denoised
    
    def liver_region_enhancement(self, image):
        """Enhance liver tissue contrast and visibility"""
        # Gamma correction for liver tissue
        gamma_corrected = exposure.adjust_gamma(image, gamma=0.8)
        
        # Sigmoid contrast adjustment
        sigmoid_enhanced = exposure.adjust_sigmoid(gamma_corrected, cutoff=0.5, gain=10)
        
        # Adaptive thresholding for tissue separation
        thresh = filters.threshold_otsu(image)
        
        # Create tissue mask
        tissue_mask = image > thresh * 0.3
        
        # Apply morphological operations to clean mask
        tissue_mask = morphology.binary_opening(tissue_mask, morphology.disk(3))
        tissue_mask = morphology.binary_closing(tissue_mask, morphology.disk(5))
        
        # Enhance tissue regions
        enhanced = np.where(tissue_mask, sigmoid_enhanced, image * 0.7)
        
        return enhanced
    
    def edge_enhancement(self, image):
        """Multi-scale edge enhancement for better feature detection"""
        # Sobel edge detection
        sobel_h = filters.sobel_h(image)
        sobel_v = filters.sobel_v(image)
        sobel_magnitude = np.sqrt(sobel_h**2 + sobel_v**2)
        
        # Laplacian edge detection
        laplacian = filters.laplace(image)
        
        # Scharr edge detection
        scharr_h = filters.scharr_h(image)
        scharr_v = filters.scharr_v(image)
        scharr_magnitude = np.sqrt(scharr_h**2 + scharr_v**2)
        
        # Combine edge information
        edge_combined = (sobel_magnitude + np.abs(laplacian) + scharr_magnitude) / 3
        
        # Normalize edge strength
        edge_normalized = (edge_combined - edge_combined.min()) / (edge_combined.max() - edge_combined.min() + 1e-8)
        
        # Apply edge enhancement
        alpha = 0.3  # Edge enhancement strength
        enhanced = image + alpha * edge_normalized
        
        return np.clip(enhanced, 0, 1)
    
    def texture_enhancement(self, image):
        """Enhance texture features using local binary patterns"""
        # LBP for texture enhancement
        radius = 2
        n_points = 8 * radius
        
        # Convert to uint8 for LBP
        image_uint8 = (image * 255).astype(np.uint8)
        
        # Compute LBP
        lbp = local_binary_pattern(image_uint8, n_points, radius, method='uniform')
        
        # Normalize LBP
        lbp_normalized = (lbp - lbp.min()) / (lbp.max() - lbp.min() + 1e-8)
        
        # Combine original image with texture information
        texture_weight = 0.2
        enhanced = (1 - texture_weight) * image + texture_weight * lbp_normalized
        
        return enhanced
    
    def multi_scale_processing(self, image):
        """Process image at multiple scales and combine"""
        scales = [0.5, 1.0, 1.5]
        processed_scales = []
        
        original_shape = image.shape
        
        for scale in scales:
            # Resize image
            new_shape = (int(original_shape[0] * scale), int(original_shape[1] * scale))
            if scale != 1.0:
                resized = cv2.resize(image, new_shape[::-1], interpolation=cv2.INTER_CUBIC)
            else:
                resized = image
            
            # Apply processing
            processed = self.liver_region_enhancement(resized)
            processed = self.edge_enhancement(processed)
            
            # Resize back to original
            if scale != 1.0:
                processed = cv2.resize(processed, original_shape[::-1], interpolation=cv2.INTER_CUBIC)
            
            processed_scales.append(processed)
        
        # Weighted combination of scales
        weights = [0.2, 0.6, 0.2]  # Favor original scale
        combined = sum(w * scale for w, scale in zip(weights, processed_scales))
        
        return combined
    
    def adaptive_windowing(self, image, window_center=50, window_width=400):
        """Apply adaptive windowing based on tissue characteristics"""
        # Compute image statistics
        mean_val = np.mean(image)
        std_val = np.std(image)
        
        # Adaptive window parameters
        adaptive_center = mean_val + 0.1 * std_val
        adaptive_width = 2 * std_val + window_width * 0.5
        
        # Apply windowing
        min_val = adaptive_center - adaptive_width / 2
        max_val = adaptive_center + adaptive_width / 2
        
        windowed = np.clip((image - min_val) / (max_val - min_val), 0, 1)
        
        return windowed
    
    def process_slice(self, image_slice):
        """
        Main preprocessing pipeline for a single slice
        """
        # Ensure input is float32
        if image_slice.dtype != np.float32:
            image_slice = image_slice.astype(np.float32)
        
        # Normalize to [0, 1]
        if image_slice.max() > 1.0:
            image_slice = image_slice / image_slice.max()
        
        # Step 1: Adaptive windowing
        processed = self.adaptive_windowing(image_slice).astype(np.float32)
        
        # Step 2: Noise reduction
        if self.noise_reduction:
            processed = self.advanced_noise_reduction(processed).astype(np.float32)
        
        # Step 3: Multi-scale processing
        processed = self.multi_scale_processing(processed).astype(np.float32)
        
        # Step 4: Contrast enhancement
        if self.enhance_contrast:
            processed = self.adaptive_histogram_equalization(processed).astype(np.float32)
        
        # Step 5: Texture enhancement
        processed = self.texture_enhancement(processed).astype(np.float32)
        
        # Final normalization
        processed = (processed - processed.mean()) / (processed.std() + 1e-8)
        processed = np.clip(processed, -3, 3).astype(np.float32)  # Ensure float32 and clip extreme values
        
        # Resize to target size
        if processed.shape != self.target_size:
            processed = cv2.resize(processed, self.target_size, interpolation=cv2.INTER_CUBIC).astype(np.float32)
        
        return processed

# Global preprocessor instance
_advanced_preprocessor = None

def get_advanced_preprocessor():
    """Get global preprocessor instance"""
    global _advanced_preprocessor
    if _advanced_preprocessor is None:
        _advanced_preprocessor = AdvancedHCCPreprocessor()
    return _advanced_preprocessor

def patch_dataset_preprocessing_v3():
    """
    Patch the dataset preprocessing to use advanced v3 preprocessing
    """
    try:
        from data.dataset import HCCDicomDataset
        
        # Store original preprocessing method
        if not hasattr(HCCDicomDataset, '_original_dicom_to_tensor_v3'):
            HCCDicomDataset._original_dicom_to_tensor_v3 = HCCDicomDataset.dicom_to_tensor
        
        def enhanced_dicom_to_tensor(self, dcm):
            """Enhanced DICOM to tensor with v3 preprocessing improvements"""
            # Get original pixel array
            img = dcm.pixel_array.astype(np.float32)
            img_min, img_max = img.min(), img.max()
            
            # Normalize to [0, 1]
            if img_max > img_min:
                img = (img - img_min) / (img_max - img_min)
            else:
                img = np.zeros_like(img, dtype=np.float32)
            
            # Apply advanced v3 preprocessing
            preprocessor = get_advanced_preprocessor()
            img = preprocessor.process_slice(img)
            
            # Ensure img is float32 after preprocessing
            if img.dtype != np.float32:
                img = img.astype(np.float32)
            
            # Ensure 3 channels for RGB
            if img.ndim == 2:
                img = np.repeat(img[..., np.newaxis], 3, axis=-1)
            elif img.ndim == 3 and img.shape[2] == 1:
                img = np.repeat(img, 3, axis=-1)
            elif img.ndim == 3 and img.shape[2] != 3:
                print(f"[WARN] Unexpected channel count {img.shape[2]}, taking first channel.")
                img = img[:,:,0:1]
                img = np.repeat(img, 3, axis=-1)
            
            # Ensure final array is float32 before tensor conversion
            img = img.astype(np.float32)
            
            # Convert to tensor and resize
            tensor_img = torch.from_numpy(img).permute(2, 0, 1)
            
            # Ensure tensor is float32
            if tensor_img.dtype != torch.float32:
                tensor_img = tensor_img.float()
            
            # Resize tensor_img to (3, 224, 224) using torch.nn.functional.interpolate
            if tensor_img.shape[1:] != (224, 224):
                tensor_img = tensor_img.unsqueeze(0)  # add batch dim for interpolate
                tensor_img = torch.nn.functional.interpolate(tensor_img, size=(224, 224), mode='bilinear', align_corners=False)
                tensor_img = tensor_img.squeeze(0)  # remove batch dim
            
            return tensor_img
        
        # Patch the method
        HCCDicomDataset.dicom_to_tensor = enhanced_dicom_to_tensor
        
        print("✅ Successfully patched HCCDicomDataset.dicom_to_tensor with advanced preprocessing v3!")
        
    except Exception as e:
        print(f"[WARN] Could not apply advanced preprocessing v3 patch: {e}")
        import traceback
        traceback.print_exc()

# Auto-apply patch when imported
if __name__ != "__main__":
    patch_dataset_preprocessing_v3() 