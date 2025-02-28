# utils/helpers.py
import numpy as np
from tqdm import tqdm
import torch

def extract_features(data_loader, model, device):
    """
    Extract features using the DINOv2 model.
    Each slice will have its own feature vector (no averaging on the slice level).
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
            # Reshape back: each sample now has num_slices feature vectors, one per slice
            feats = feats.view(batch_size, num_samples, num_slices, feature_dim)
            
            # No averaging on the slice level is performed here

            features.append(feats.cpu().numpy())
            durations.append(t.cpu().numpy())
            events.append(e.cpu().numpy())

    features = np.concatenate(features, axis=0)
    durations = np.concatenate(durations, axis=0)
    events = np.concatenate(events, axis=0)
    return features, durations, events



def validate_survival_data(durations, events):
    sort_idx = np.argsort(durations)
    sorted_durations = durations[sort_idx]
    sorted_events = events[sort_idx]
    for i in range(len(sorted_durations)):
        if sorted_events[i] == 1:
            current_time = sorted_durations[i]
            num_at_risk = np.sum(sorted_durations >= current_time)
            if num_at_risk == 0:
                raise ValueError(f"Event at {current_time} has no at-risk individuals.")
            elif num_at_risk == 1:
                print(f"Warning: Event at {current_time} has only 1 at-risk.")