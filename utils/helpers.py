# utils/helpers.py
import numpy as np
from tqdm import tqdm
import torch

def extract_features(data_loader, model, device):
    """
    Extract features using the DINOv2 model.
    Patient-level features are computed by averaging per-slice features.
    """
    model.eval()
    features = []
    durations = []
    events = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Extracting Features"):
            images, t, e = batch
            images = images.to(device)
            # Unpack the 6 dimensions
            batch_size, num_samples, num_slices, C, H, W = images.size()
            # Combine batch, num_samples, and num_slices dimensions for feature extraction
            images = images.view(batch_size * num_samples * num_slices, C, H, W)
            
            feats = model.forward_features(images)
            feature_dim = feats.size(-1)
            # Reshape back: each sample now has `num_slices` feature vectors
            feats = feats.view(batch_size, num_samples, num_slices, feature_dim)
            # Average on the slice level
            feats = feats.mean(dim=2)  # Now shape is (batch_size, num_samples, feature_dim)
            
            # (Optional) If you want to further average across samples for a patient-level feature:
            # feats = feats.mean(dim=1)  # Now shape is (batch_size, feature_dim)

            features.append(feats.cpu().numpy())
            durations.append(t.cpu().numpy())
            events.append(e.cpu().numpy())

    features = np.concatenate(features, axis=0)
    durations = np.concatenate(durations, axis=0)
    events = np.concatenate(events, axis=0)
    return features, durations, events
