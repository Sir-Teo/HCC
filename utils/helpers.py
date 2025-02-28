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

def upsample_training_data(x_train, durations, events):
    """
    Upsample the minority class so that both classes have equal representation.
    """
    idx_event = np.where(events == 1)[0]
    idx_no_event = np.where(events == 0)[0]

    if len(idx_event) == 0 or len(idx_no_event) == 0:
        print("Warning: One of the classes is empty. Skipping upsampling.")
        return x_train, durations, events

    if len(idx_event) < len(idx_no_event):
        minority_idx = idx_event
        majority_idx = idx_no_event
    else:
        minority_idx = idx_no_event
        majority_idx = idx_event

    n_to_sample = len(majority_idx) - len(minority_idx)
    sampled_minority_idx = np.random.choice(minority_idx, size=n_to_sample, replace=True)
    new_indices = np.concatenate([np.arange(len(events)), sampled_minority_idx])
    new_indices = np.random.permutation(new_indices)

    x_train_upsampled = x_train[new_indices]
    durations_upsampled = durations[new_indices]
    events_upsampled = events[new_indices]

    print(f"Upsampled training data from {len(events)} to {len(events_upsampled)} samples.")
    return x_train_upsampled, durations_upsampled, events_upsampled
