import argparse
import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, accuracy_score, average_precision_score
import datetime  # new import for timestamp
from sklearn.model_selection import KFold
# Custom module imports
from data.dataset import HCCDataModule
from models.dino import load_dinov2_model
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from imblearn.over_sampling import SMOTE, ADASYN
import json # Add json import
import copy
import random
import math
import shutil
import sys
# Import CustomMLP for optional MLP architecture
from models.mlp import CustomMLP

# Ensure improved preprocessing is applied
try:
    import improved_preprocessing_v3 as _imp_patch_v3
    print("[INFO] Using advanced preprocessing v3 for maximum precision")
except Exception as _e:
    try:
        import improved_preprocessing_patch as _imp_patch
        _imp_patch.patch_dataset_preprocessing()
        print("[INFO] Using improved preprocessing v2 (fallback)")
    except Exception as _e2:
        print(f"[WARN] Could not apply any improved preprocessing: {_e}, {_e2}")

sns.set(style="whitegrid")

# --- Ultra-Precision Architecture for Maximum AUC/Precision ---
class UltraPrecisionMLP(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.1):
        super().__init__()
        
        # Hierarchical feature extraction
        self.feature_extractor = nn.Sequential(
            nn.Linear(in_features, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        # Multi-head attention for feature importance
        self.attention_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(1024, 256),
                nn.ReLU(),
                nn.Linear(256, 1024),
                nn.Sigmoid()
            ) for _ in range(4)
        ])
        
        # Precision-focused classifier with heavy regularization
        self.precision_classifier = nn.Sequential(
            nn.Dropout(dropout * 1.5),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout * 0.8),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout * 0.3),
            nn.Linear(64, out_features)
        )
        
        # Feature gate for selective processing
        self.feature_gate = nn.Sequential(
            nn.Linear(in_features, 1024),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # Gate input features
        gate_weights = self.feature_gate(x)
        
        # Extract features
        features = self.feature_extractor(x)
        features = features * gate_weights
        
        # Multi-head attention
        attention_outputs = []
        for attention_head in self.attention_heads:
            att_weights = attention_head(features)
            attention_outputs.append(features * att_weights)
        
        # Combine attention heads
        combined_features = torch.stack(attention_outputs).mean(dim=0)
        
        # Final classification
        return self.precision_classifier(combined_features)

# --- Enhanced Precision-Weighted Ensemble ---
class PrecisionWeightedEnsemble(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.1):
        super().__init__()
        
        # Create diverse models with different architectures
        self.models = nn.ModuleList([
            UltraPrecisionMLP(in_features, out_features, dropout),
            PrecisionFocusedMLP(in_features, out_features, dropout),
            AdvancedMLP(in_features, out_features, dropout),
            EnhancedMLP(in_features, out_features, dropout),
        ])
        
        # Learned ensemble weights
        self.ensemble_weights = nn.Parameter(torch.ones(len(self.models)))
        
        # Meta-classifier for ensemble combination
        self.meta_classifier = nn.Sequential(
            nn.Linear(len(self.models), 16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # Get predictions from all models
        model_outputs = []
        for model in self.models:
            output = model(x)
            model_outputs.append(output)
        
        # Apply learned weights
        weighted_outputs = []
        for i, output in enumerate(model_outputs):
            weight = torch.softmax(self.ensemble_weights, dim=0)[i]
            weighted_outputs.append(output * weight)
        
        # Combine predictions
        ensemble_pred = torch.stack(weighted_outputs).sum(dim=0)
        
        # Meta-classifier refinement
        if len(model_outputs[0].shape) > 1 and model_outputs[0].shape[1] == 1:
            meta_input = torch.stack([torch.sigmoid(out).squeeze(-1) for out in model_outputs], dim=-1)
        else:
            meta_input = torch.stack([torch.sigmoid(out) for out in model_outputs], dim=-1)
        meta_weight = self.meta_classifier(meta_input)
        
        return ensemble_pred * meta_weight + ensemble_pred * (1 - meta_weight) * 0.5

# --- Advanced Precision-Recall Loss ---
class PrecisionRecallFocalLoss(nn.Module):
    def __init__(self, alpha=0.8, gamma=2.0, precision_weight=0.7, pos_weight=None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma 
        self.precision_weight = precision_weight
        self.pos_weight = pos_weight
        
    def forward(self, logits, targets):
        # Focal loss component
        bce_loss = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.pos_weight, reduction='none'
        )
        
        probas = torch.sigmoid(logits)
        p_t = probas * targets + (1 - probas) * (1 - targets)
        focal_term = (1 - p_t) ** self.gamma
        focal_loss = self.alpha * focal_term * bce_loss
        
        # Precision-recall balancing term
        predicted_pos = (probas > 0.5).float()
        true_pos = targets
        
        # Approximate precision and recall
        tp = (predicted_pos * true_pos).sum()
        fp = (predicted_pos * (1 - true_pos)).sum()
        fn = ((1 - predicted_pos) * true_pos).sum()
        
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        
        # Precision-recall loss (maximize F-beta with beta < 1 for precision emphasis)
        beta = 0.5  # Emphasize precision over recall
        f_beta = (1 + beta**2) * precision * recall / (beta**2 * precision + recall + 1e-8)
        pr_loss = 1 - f_beta
        
        return focal_loss.mean() + self.precision_weight * pr_loss

# --- Enhanced Model Architecture ---
class EnhancedMLP(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.1, use_batch_norm=True):
        super().__init__()
        self.use_batch_norm = use_batch_norm
        
        # More sophisticated architecture with skip connections
        self.input_layer = nn.Linear(in_features, 1024)
        self.bn1 = nn.BatchNorm1d(1024) if use_batch_norm else nn.Identity()
        
        self.hidden1 = nn.Linear(1024, 512)
        self.bn2 = nn.BatchNorm1d(512) if use_batch_norm else nn.Identity()
        
        self.hidden2 = nn.Linear(512, 256)
        self.bn3 = nn.BatchNorm1d(256) if use_batch_norm else nn.Identity()
        
        self.hidden3 = nn.Linear(256, 128)
        self.bn4 = nn.BatchNorm1d(128) if use_batch_norm else nn.Identity()
        
        # Skip connection layer
        self.skip_layer = nn.Linear(in_features, 128)
        
        self.output_layer = nn.Linear(128, out_features)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # Main path
        out = torch.relu(self.bn1(self.input_layer(x)))
        out = self.dropout(out)
        
        out = torch.relu(self.bn2(self.hidden1(out)))
        out = self.dropout(out)
        
        out = torch.relu(self.bn3(self.hidden2(out)))
        out = self.dropout(out)
        
        out = torch.relu(self.bn4(self.hidden3(out)))
        
        # Skip connection
        skip = torch.relu(self.skip_layer(x))
        
        # Combine main path and skip connection
        out = out + skip
        out = self.dropout(out)
        
        return self.output_layer(out)

# --- Weighted Ensemble Training ---
class WeightedEnsembleModel(nn.Module):
    def __init__(self, models, weights=None):
        super().__init__()
        self.models = nn.ModuleList(models)
        if weights is None:
            weights = [1.0 / len(models)] * len(models)
        self.register_buffer('weights', torch.tensor(weights))
        
    def forward(self, x):
        outputs = []
        for model in self.models:
            outputs.append(model(x))
        weighted_outputs = torch.stack(outputs) * self.weights.view(-1, 1, 1)
        return weighted_outputs.sum(dim=0)
    
    def predict_proba(self, x):
        with torch.no_grad():
            outputs = []
            for model in self.models:
                model.eval()
                logits = model(x)
                probs = torch.sigmoid(logits)
                outputs.append(probs)
            weighted_outputs = torch.stack(outputs) * self.weights.view(-1, 1, 1)
            return weighted_outputs.sum(dim=0)

# --- Cost-Sensitive Loss ---
class CostSensitiveLoss(nn.Module):
    def __init__(self, cost_matrix=None, pos_weight=None):
        super().__init__()
        # Default cost matrix: FN costs more than FP (precision-focused)
        if cost_matrix is None:
            cost_matrix = torch.tensor([[1.0, 5.0],  # [TN, FP]
                                       [10.0, 1.0]])  # [FN, TP]
        self.cost_matrix = cost_matrix
        self.pos_weight = pos_weight
        
    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        
        # Cost for each sample
        # Cost = targets * (FN_cost * (1-probs) + TP_cost * probs) + 
        #        (1-targets) * (TN_cost * (1-probs) + FP_cost * probs)
        
        fn_cost, tp_cost = self.cost_matrix[1, 0], self.cost_matrix[1, 1]
        tn_cost, fp_cost = self.cost_matrix[0, 0], self.cost_matrix[0, 1]
        
        positive_cost = targets * (fn_cost * (1 - probs) + tp_cost * probs)
        negative_cost = (1 - targets) * (tn_cost * (1 - probs) + fp_cost * probs)
        
        total_cost = positive_cost + negative_cost
        return total_cost.mean()

# --- Precision-Focused Architecture ---
class PrecisionFocusedMLP(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.1):
        super().__init__()
        
        # Multi-scale feature extraction
        self.scale1 = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.scale2 = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.BatchNorm1d(256), 
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU()
        )
        
        # Attention mechanism for precision focus
        self.attention = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, 1024),
            nn.Sigmoid()
        )
        
        # Final classification layers with heavy regularization
        self.classifier = nn.Sequential(
            nn.Dropout(dropout * 1.5),
            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(64, out_features)
        )
        
    def forward(self, x):
        # Multi-scale processing
        feat1 = self.scale1(x)
        feat2 = self.scale2(x)
        
        # Combine scales
        combined = torch.cat([feat1, feat2], dim=1)
        
        # Apply attention
        attention_weights = self.attention(combined)
        attended_features = combined * attention_weights
        
        # Final classification
        return self.classifier(attended_features)

# --- Advanced Regularized MLP ---
class AdvancedMLP(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.1):
        super().__init__()
        
        # Progressive dimension reduction with proper residual connections
        self.input_proj = nn.Linear(in_features, 1024)
        self.input_bn = nn.BatchNorm1d(1024)
        
        # Residual blocks with proper skip connections
        self.block1_main = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512)
        )
        self.block1_skip = nn.Linear(1024, 512)
        
        self.block2_main = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256)
        )
        self.block2_skip = nn.Linear(512, 256)
        
        self.block3_main = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128)
        )
        self.block3_skip = nn.Linear(256, 128)
        
        # Final layers
        self.final_layers = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(64, out_features)
        )
        
    def forward(self, x):
        # Input projection
        out = torch.relu(self.input_bn(self.input_proj(x)))
        
        # Residual block 1
        residual = self.block1_skip(out)
        out = torch.relu(self.block1_main(out) + residual)
        
        # Residual block 2
        residual = self.block2_skip(out)
        out = torch.relu(self.block2_main(out) + residual)
        
        # Residual block 3
        residual = self.block3_skip(out)
        out = torch.relu(self.block3_main(out) + residual)
        
        return self.final_layers(out)

# --- Label Smoothing Loss ---
class LabelSmoothingBCE(nn.Module):
    def __init__(self, smoothing=0.1, pos_weight=None):
        super().__init__()
        self.smoothing = smoothing
        self.pos_weight = pos_weight
        
    def forward(self, logits, targets):
        # Apply label smoothing
        targets_smooth = targets * (1 - self.smoothing) + 0.5 * self.smoothing
        
        # Use BCEWithLogitsLoss with smoothed targets
        if self.pos_weight is not None:
            loss_fn = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)
        else:
            loss_fn = nn.BCEWithLogitsLoss()
            
        return loss_fn(logits, targets_smooth)

# --- Feature Extraction --- 
# Moved from helpers and adapted for binary classification labels
def extract_features(data_loader, model, device):
    """
    Extract features using the DINOv2 model.
    Each slice will have its own feature vector (no averaging on the slice level).
    Returns features and event labels.
    """
    model.eval()
    features = []
    events = []
    patient_info = [] # To store patient dicts

    if data_loader is None:
        # Return empty arrays and list if dataloader is None (e.g., empty val set)
        return np.array([]).reshape(0, 1, 1, 768), np.array([]), [] # Assuming feature dim 768

    desc = "Extracting Features"
    # Check if we can get the dataset source for a more informative description
    try: 
         if hasattr(data_loader.dataset, 'patient_data') and len(data_loader.dataset.patient_data) > 0:
             # Attempt to get source from the first patient, default if fails
             source_example = data_loader.dataset.patient_data[0].get('source', 'NYU')
             desc = f"Extracting Features ({source_example} set)" 
         # Add fold info if available from datamodule
         if hasattr(data_loader, 'dataset') and hasattr(data_loader.dataset, '__fold_idx__'):
              desc += f" Fold {data_loader.dataset.__fold_idx__ + 1}"
    except Exception:
         pass # Keep default description if any error occurs

    with torch.no_grad():
        # Assuming dataloader yields (images, event_label) or (images, time, event_label)
        for batch in tqdm(data_loader, desc=desc):
            # Adapt based on actual dataloader output structure
            if len(batch) == 2:
                 images, e = batch
                 # Retrieve patient info if collate_fn passes it
                 # This part might need adjustment based on how patient info is handled in collate
                 # Assuming patient info is NOT directly in the batch here, retrieve later
            elif len(batch) == 3:
                 images, _, e = batch # Ignore time for binary classification
            else:
                 raise ValueError(f"Unexpected batch structure length: {len(batch)}")

            images = images.to(device)
            # Unpack the 5 dimensions (batch_size now refers to patients)
            batch_size, num_samples, num_slices, C, H, W = images.size()
            # Combine batch, num_samples, and num_slices dimensions for feature extraction
            images_reshaped = images.view(batch_size * num_samples * num_slices, C, H, W)
            
            feats = model.forward_features(images_reshaped)
            feature_dim = feats.size(-1)
            # Reshape back: each patient now has num_samples * num_slices feature vectors
            feats = feats.view(batch_size, num_samples * num_slices, feature_dim)
            
            # Average across the slice dimension (axis=1)
            # Result shape: (batch_size, feature_dim)
            # feats_avg_slice = feats.mean(dim=1) 
            # NO - keep slice dim for now, average later after scaling
            feats_reshaped = feats.view(batch_size, num_samples, num_slices, feature_dim)

            features.append(feats_reshaped.cpu().numpy()) # Keep [batch, samples, slices, feat]
            events.append(e.cpu().numpy())

    if not features: # Handle empty dataloader case
        print("[WARN] extract_features received empty data_loader or produced no features.")
        # Define shape based on expected feature dimension (e.g., 768 for DINOv2 base)
        feature_dim_expected = 768 
        return np.empty((0, 1, 1, feature_dim_expected), dtype=np.float32), np.empty((0,), dtype=np.float32), []

    features = np.concatenate(features, axis=0)
    events = np.concatenate(events, axis=0)
    
    # Retrieve corresponding patient data - IMPORTANT ASSUMPTION:
    # Assumes the order of patients in the dataloader matches the order in dataset.patient_data
    if hasattr(data_loader.dataset, 'patient_data'):
        patient_info = data_loader.dataset.patient_data[:len(features)] # Get info for the processed patients
    else:
        print("[WARN] Could not retrieve patient info from dataset.")
        # Create dummy patient info if needed elsewhere, though ideally it should be available
        patient_info = [{'patient_id': f'unknown_{i}', 'dataset_type': 'NYU'} for i in range(len(features))] 
        
    return features, events, patient_info # Return features, events, and patient metadata

# --- Upsampling (Simplified for binary event) --- 
def upsample_training_data(x_train, events):
    """
    Upsample the minority class (event=1 or event=0) for binary classification.
    Args:
        x_train (np.array): Feature array (e.g., [n_patients, n_slices, feat_dim])
        events (np.array): Binary event labels (0 or 1) [n_patients]
    Returns:
        Upsampled x_train, events
    """
    idx_event_1 = np.where(events == 1)[0]
    idx_event_0 = np.where(events == 0)[0]

    if len(idx_event_1) == 0 or len(idx_event_0) == 0:
        print("Warning: One of the classes is empty. Skipping upsampling.")
        return x_train, events

    n_minority = min(len(idx_event_1), len(idx_event_0))
    n_majority = max(len(idx_event_1), len(idx_event_0))

    if n_minority == n_majority:
         print("Classes are balanced. Skipping upsampling.")
         return x_train, events

    if len(idx_event_1) < len(idx_event_0):
        minority_idx = idx_event_1
        majority_idx = idx_event_0
    else:
        minority_idx = idx_event_0
        majority_idx = idx_event_1

    n_to_sample = n_majority - n_minority
    sampled_minority_idx = np.random.choice(minority_idx, size=n_to_sample, replace=True)
    
    # Combine original majority indices, original minority indices, and sampled minority indices
    new_indices = np.concatenate([majority_idx, minority_idx, sampled_minority_idx])
    # Shuffle the combined indices
    np.random.shuffle(new_indices)

    x_train_upsampled = x_train[new_indices]
    events_upsampled = events[new_indices]

    print(f"Upsampled training data from {len(events)} to {len(events_upsampled)} samples. Minority class originally had {n_minority} samples.")
    return x_train_upsampled, events_upsampled

def upsample_df(df, target_column='event'): # Keep this utility? Maybe not needed if upsampling features directly.
    df_majority = df[df[target_column] == 0]
    df_minority = df[df[target_column] == 1]
    if len(df_minority) == 0 or len(df_majority) == 0:
        return df
    df_minority_upsampled = df_minority.sample(len(df_majority), replace=True, random_state=1)
    return pd.concat([df_majority, df_minority_upsampled]).reset_index(drop=True)

def cross_validation_mode(args):
    """
    Perform cross-validation on the NYU dataset.
    """
    # Hyperparameters for training
    hyperparams = {
        'learning_rate': args.learning_rate,
    }

    # --- Data Module Setup --- 
    data_module = HCCDataModule(
        csv_file=args.nyu_csv_file,   # NYU csv only
        dicom_root=args.nyu_dicom_root,   # NYU root only
        model_type="linear", # Set explicitly for binary classification
        batch_size=args.batch_size,
        num_slices=args.num_slices,
        num_samples=args.num_samples_per_patient,
        num_workers=args.num_workers,
        preprocessed_root=args.preprocessed_root,
        cross_validation=True,
        cv_folds=args.cv_folds,
        leave_one_out=args.leave_one_out,
        random_state=42,
        use_validation=True # Use validation set within folds
    )
    # Setup does the combining, filtering, preprocessing, and splitting
    data_module.setup()

    # --- Model Setup --- 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    dino_model = load_dinov2_model(args.dinov2_weights)
    dino_model = dino_model.to(device)
    dino_model.eval()
    # Prevent DINOv2 from tracking gradients
    for param in dino_model.parameters():
        param.requires_grad = False

    # --- Results Storage --- 
    all_fold_results = [] # Store results dictionary for each test patient
    fold_test_accuracies = [] # Store accuracy per fold for averaging

    # --- Cross-Validation Loop ---
    total_folds = data_module.get_total_folds()
    print(f"[INFO] Starting cross-validation with {total_folds} folds…")

    for current_fold in range(total_folds):
        # Explicitly set up the desired fold (guarantees each fold is processed)
        data_module._setup_fold(current_fold)

        print(f"\n===== Processing Fold {current_fold}/{total_folds} =====")

        # Get dataloaders for current fold
        train_loader = data_module.train_dataloader()
        val_loader = data_module.val_dataloader() # Can be None
        test_loader = data_module.test_dataloader()

        # Add fold index to dataset for potential use in extract_features desc
        if hasattr(train_loader.dataset, '__fold_idx__'):
            train_loader.dataset.__fold_idx__ = current_fold
        if val_loader and hasattr(val_loader.dataset, '__fold_idx__'):
            val_loader.dataset.__fold_idx__ = current_fold
        if hasattr(test_loader.dataset, '__fold_idx__'):
            test_loader.dataset.__fold_idx__ = current_fold

        # --- Feature Extraction --- 
        # Note: extract_features now returns patient_info list as the third element
        x_train, y_train_events, train_patient_info = extract_features(train_loader, dino_model, device)
        x_val, y_val_events, _ = extract_features(val_loader, dino_model, device) # Will be empty if val_loader is None
        x_test, y_test_events, test_patient_info = extract_features(test_loader, dino_model, device)

        # Check for empty datasets post-extraction
        if x_train.size == 0:
            print(f"[WARN] Fold {current_fold}: No training features extracted. Skipping fold.")
            continue
        if x_test.size == 0:
            print(f"[WARN] Fold {current_fold}: No test features extracted. Skipping fold.")
            continue
        # Validation set can be empty, handle downstream
        has_validation_data = x_val.size > 0
        if not has_validation_data:
            print(f"[INFO] Fold {current_fold + 1}: No validation data available or extracted.")

        # --- Feature Preprocessing --- 
        # Average across samples dimension -> [patients, slices, features]
        x_train = x_train.mean(axis=1)
        if has_validation_data:
            x_val = x_val.mean(axis=1)
        x_test = x_test.mean(axis=1)

        # Remove zero-variance features (calculated only on training set)
        # Reshape for variance calculation: [patients * slices, features]
        n_train_p, n_train_s, n_train_f = x_train.shape
        x_train_flat_var = x_train.reshape(-1, n_train_f)
        variances = np.var(x_train_flat_var, axis=0)
        zero_var_indices = np.where(variances == 0)[0]
        if len(zero_var_indices) > 0:
            print(f"[INFO] Fold {current_fold + 1}: Removing {len(zero_var_indices)} features with zero variance based on training set.")
            non_zero_var_indices = np.where(variances != 0)[0]
            x_train = x_train[:, :, non_zero_var_indices]
            if has_validation_data:
                x_val = x_val[:, :, non_zero_var_indices]
            x_test = x_test[:, :, non_zero_var_indices]
            # Update feature dimension
            n_train_f = x_train.shape[2]
        else:
            print(f"[INFO] Fold {current_fold + 1}: No zero-variance features found in training set.")

        # Upsampling (optional, on training data)
        if args.upsampling and args.upsampling_method == 'random':
            print(f"[INFO] Fold {current_fold + 1}: Upsampling training data...")
            x_train, y_train_events = upsample_training_data(x_train, y_train_events)
            print(f"[INFO] Fold {current_fold + 1}: Training data shape after upsampling: {x_train.shape}")
        # Update train shape info after possible upsampling
        n_train_p, n_train_s, n_train_f = x_train.shape

        # Standardize features (fit on train, transform train/val/test)
        # Reshape needed for StandardScaler: [patients * slices, features]
        x_mapper = StandardScaler()
        x_train_reshaped = x_train.reshape(-1, n_train_f)
        x_train_scaled = x_mapper.fit_transform(x_train_reshaped).astype('float32')
        x_train_scaled = x_train_scaled.reshape(n_train_p, n_train_s, n_train_f)

        if has_validation_data:
            n_val_p, n_val_s, n_val_f = x_val.shape
            x_val_reshaped = x_val.reshape(-1, n_val_f)
            x_val_scaled = x_mapper.transform(x_val_reshaped).astype('float32')
            x_val_scaled = x_val_scaled.reshape(n_val_p, n_val_s, n_val_f)
        else:
            x_val_scaled = np.array([]) # Empty array if no val data

        n_test_p, n_test_s, n_test_f = x_test.shape
        x_test_reshaped = x_test.reshape(-1, n_test_f)
        x_test_scaled = x_mapper.transform(x_test_reshaped).astype('float32')
        x_test_scaled = x_test_scaled.reshape(n_test_p, n_test_s, n_test_f)

        # Collapse the slice dimension by adaptive average pooling -> [patients, features]
        slice_pool = nn.AdaptiveAvgPool1d(1)
        # Pool training set
        x_train_tensor = torch.from_numpy(x_train_scaled.transpose(0, 2, 1))  # [patients, features, slices]
        x_train_final = slice_pool(x_train_tensor).squeeze(-1).numpy()  # [patients, features]
        # Pool validation set
        if has_validation_data:
            x_val_tensor = torch.from_numpy(x_val_scaled.transpose(0, 2, 1))
            x_val_final = slice_pool(x_val_tensor).squeeze(-1).numpy()
        else:
            x_val_final = np.array([])
        # Pool test set
        x_test_tensor = torch.from_numpy(x_test_scaled.transpose(0, 2, 1))
        x_test_final = slice_pool(x_test_tensor).squeeze(-1).numpy()

        print(f"[INFO] Fold {current_fold + 1}: Final feature shapes: Train {x_train_final.shape}, Val {x_val_final.shape}, Test {x_test_final.shape}")
        # SMOTE upsampling (optional)
        if args.upsampling and args.upsampling_method == 'smote':
            print(f"[INFO] Fold {current_fold + 1}: Applying SMOTE upsampling for binary...")
            classes, counts = np.unique(y_train_events, return_counts=True)
            if len(classes) > 1:
                n_minority = counts.min()
                k = min(5, n_minority - 1)
                if k >= 1:
                    smote = SMOTE(random_state=42, k_neighbors=k)
                    x_train_final, y_train_events = smote.fit_resample(x_train_final, y_train_events)
                else:
                    print(f"[WARN] Fold {current_fold + 1}: minority class too small for SMOTE (n_minority={n_minority}). Skipping SMOTE.")
            else:
                print(f"[WARN] Fold {current_fold + 1}: single-class. Skipping SMOTE.")
            print(f"[INFO] After SMOTE: Train {x_train_final.shape}, events {int(sum(y_train_events))}")
        # ADASYN upsampling (optional)
        if args.upsampling and args.upsampling_method == 'adasyn':
            print(f"[INFO] Fold {current_fold + 1}: Applying ADASYN upsampling for binary...")
            classes, counts = np.unique(y_train_events, return_counts=True)
            if len(classes) > 1:
                adasyn = ADASYN(random_state=42)
                try:
                    x_train_final, y_train_events = adasyn.fit_resample(x_train_final, y_train_events)
                except ValueError as e:
                    print(f"[WARN] Fold {current_fold + 1}: ADASYN error {e}. Skipping ADASYN.")
            else:
                print(f"[WARN] Fold {current_fold + 1}: single-class. Skipping ADASYN.")
            print(f"[INFO] After ADASYN: Train {x_train_final.shape}, events {int(sum(y_train_events))}")
        # --- Model Training --- 
        in_features = x_train_final.shape[1]
        # Build requested architecture
        if args.model_arch == 'mlp':
            net = CustomMLP(in_features, 1, dropout=args.dropout)
        elif args.model_arch == 'enhanced_mlp':
            net = EnhancedMLP(in_features, 1, dropout=args.dropout, use_batch_norm=True)
        elif args.model_arch == 'ultra_precision':
            net = UltraPrecisionMLP(in_features, 1, dropout=args.dropout)
        elif args.model_arch == 'precision_weighted_ensemble':
            net = PrecisionWeightedEnsemble(in_features, 1, dropout=args.dropout)
        elif args.model_arch == 'ensemble':
            # Use the new precision-weighted ensemble as default ensemble
            net = PrecisionWeightedEnsemble(in_features, 1, dropout=args.dropout)
        elif args.model_arch == 'precision_focused':
            net = PrecisionFocusedMLP(in_features, 1, dropout=args.dropout)
        elif args.model_arch == 'advanced':
            net = AdvancedMLP(in_features, 1, dropout=args.dropout)
        else:
            net = nn.Linear(in_features, 1) # Default linear model
        net.to(device)

        optimizer = torch.optim.AdamW(net.parameters(), lr=hyperparams['learning_rate'], 
                                     weight_decay=5e-4, betas=(0.9, 0.999))
        
        # Learning rate scheduler for better convergence
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        
        # Use class-weighted BCE loss based on training class balance
        positives = int(y_train_events.sum())
        negatives = len(y_train_events) - positives
        if positives > 0:
            pos_weight = torch.tensor([negatives/positives], device=device)
        else:
            pos_weight = None

        # --- Loss Function ---
        # Class weight for positive samples (helps with imbalance)
        if args.focal_loss:
            # Define focal-loss wrapper around BCEWithLogits
            class FocalLoss(nn.Module):
                def __init__(self, gamma=2.0, pos_weight=None):
                    super().__init__()
                    self.gamma = gamma
                    self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction='none')

                def forward(self, logits, targets):
                    bce_loss = self.bce(logits, targets)
                    # p_t = sigmoid(logits) when target == 1, else 1 - sigmoid(logits)
                    probas = torch.sigmoid(logits.detach())
                    p_t = probas * targets + (1 - probas) * (1 - targets)
                    focal_term = (1 - p_t) ** self.gamma
                    loss = focal_term * bce_loss
                    return loss.mean()

            criterion = FocalLoss(gamma=args.focal_gamma, pos_weight=pos_weight)
        elif args.precision_recall_focal:
            # Use the new advanced precision-recall focal loss
            criterion = PrecisionRecallFocalLoss(alpha=0.8, gamma=args.focal_gamma, precision_weight=0.7, pos_weight=pos_weight)
        elif args.cost_sensitive:
            # Create cost matrix that heavily penalizes false negatives
            cost_matrix = torch.tensor([[1.0, 3.0],   # [TN, FP] 
                                       [8.0, 1.0]])   # [FN, TP]
            criterion = CostSensitiveLoss(cost_matrix=cost_matrix.to(device))
        elif args.label_smoothing > 0:
            criterion = LabelSmoothingBCE(smoothing=args.label_smoothing, pos_weight=pos_weight)
        else:
            if pos_weight is not None:
                criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            else:
                criterion = torch.nn.BCEWithLogitsLoss()

        # Prepare training, validation and test tensors
        train_tensor_x = torch.tensor(x_train_final, dtype=torch.float32).to(device)
        # Target shape should be [batch_size, 1] for BCEWithLogitsLoss
        train_tensor_y = torch.tensor(y_train_events, dtype=torch.float32).unsqueeze(1).to(device)

        val_tensor_x = torch.tensor(x_val_final, dtype=torch.float32).to(device) if has_validation_data else None
        val_tensor_y = torch.tensor(y_val_events, dtype=torch.float32).unsqueeze(1).to(device) if has_validation_data else None

        test_tensor_x = torch.tensor(x_test_final, dtype=torch.float32).to(device)
        test_tensor_y = torch.tensor(y_test_events, dtype=torch.float32).unsqueeze(1).to(device) # Ground truth for test

        # Training loop with progress display and early stopping
        best_val_loss = float('inf')
        best_model_state = None
        patience = args.early_stopping_patience # Use arg for patience
        epochs_no_improve = 0

        for epoch in range(args.epochs):
            # --- Training Phase --- 
            net.train()
            optimizer.zero_grad()
            outputs = net(train_tensor_x) # Raw logits, shape [batch, 1]
            loss = criterion(outputs, train_tensor_y)
            loss.backward()
            if args.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(net.parameters(), args.gradient_clip)
            optimizer.step()
            train_loss_epoch = loss.item()

            # --- Validation Phase --- 
            val_loss_epoch = None
            val_accuracy_epoch = None
            if has_validation_data and val_tensor_x is not None:
                net.eval()
                with torch.no_grad():
                    val_outputs = net(val_tensor_x) # Raw logits
                    val_loss = criterion(val_outputs, val_tensor_y)
                    val_loss_epoch = val_loss.item()

                    # Calculate validation accuracy
                    val_probs = torch.sigmoid(val_outputs) # Convert logits to probabilities
                    val_pred_labels = (val_probs >= 0.5).float() # Threshold probabilities
                    val_accuracy_epoch = accuracy_score(val_tensor_y.cpu().numpy(), val_pred_labels.cpu().numpy())
                net.train() # Set back to train mode

            # --- Logging --- 
            log_msg = f"[Fold {current_fold + 1}] Epoch {epoch+1}/{args.epochs} - Train Loss: {train_loss_epoch:.4f}"
            if val_loss_epoch is not None:
                log_msg += f" - Val Loss: {val_loss_epoch:.4f}"
            if val_accuracy_epoch is not None:
                log_msg += f" - Val Acc: {val_accuracy_epoch:.4f}"
            print(log_msg)

            # --- Learning Rate Scheduling ---
            if has_validation_data and val_loss_epoch is not None:
                scheduler.step(val_loss_epoch)
            
            # --- Early Stopping & Model Checkpointing --- 
            if args.early_stopping and has_validation_data:
                # Compute precision-recall based metric for early stopping
                if val_accuracy_epoch is not None:
                    with torch.no_grad():
                        val_logits = net(val_tensor_x)
                        val_probs = torch.sigmoid(val_logits).cpu().numpy().flatten()
                        val_true = val_tensor_y.cpu().numpy().flatten()
                        
                    # Calculate AUPRC as early stopping metric
                    if len(np.unique(val_true)) > 1:
                        val_auprc = average_precision_score(val_true, val_probs)
                        current_metric_for_stopping = -val_auprc  # Negative for minimization
                    else:
                        current_metric_for_stopping = val_loss_epoch
                else:
                    current_metric_for_stopping = val_loss_epoch
                    
                if current_metric_for_stopping < best_val_loss:
                    best_val_loss = current_metric_for_stopping
                    best_model_state = net.state_dict().copy() # Save best model
                    epochs_no_improve = 0
                    metric_name = "AUPRC" if val_accuracy_epoch is not None else "Loss"
                    metric_val = -current_metric_for_stopping if val_accuracy_epoch is not None else current_metric_for_stopping
                    print(f"[Fold {current_fold + 1}] New best validation {metric_name}: {metric_val:.4f}. Saving model.")
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= patience:
                        print(f"[Fold {current_fold + 1}] Early stopping triggered at epoch {epoch+1} after {patience} epochs with no improvement.")
                        break
            elif not args.early_stopping: 
                # If not early stopping, save the last model state
                best_model_state = net.state_dict().copy()

        # --- Post-Training: Load Best Model --- 
        if best_model_state is not None:
            print(f"[Fold {current_fold + 1}] Loading best model state for final evaluation.")
            net.load_state_dict(best_model_state)
        else:
            print(f"[WARN] Fold {current_fold + 1}: No best model state found (possibly due to no validation or error). Using last state.")
            # Ensure model is in eval mode even if loop finished early without saving state
            net.eval()  

        # Determine best threshold on validation set if available
        threshold = 0.5
        if has_validation_data and val_tensor_x is not None:
            net.eval()
            with torch.no_grad():
                val_logits = net(val_tensor_x)
                val_probs_full = torch.sigmoid(val_logits).cpu().numpy().flatten()
                val_true = y_val_events
            # Tune threshold based on selected metric; for AUROC/AUPRC we cannot optimise over threshold, so we use F1.
            metric_to_optimize = args.threshold_metric
            if metric_to_optimize == 'auprc':
                # AUPRC is independent of threshold; choose threshold that maximises F1 while reporting AUPRC separately
                best_score_pr = average_precision_score(val_true, val_probs_full) if len(np.unique(val_true)) > 1 else 0.0
                print(f"[Fold {current_fold + 1}] Validation AUPRC (scores): {best_score_pr:.4f}")
                metric_to_optimize = 'f1'

            # Initialize tracking variables for best threshold search
            best_score, best_thr = 0.0, 0.5
            
            # Also track precision-focused threshold
            best_precision_score, best_precision_thr = 0.0, 0.5

            for t in np.linspace(0.05, 0.95, 181):  # Finer grid for better precision
                preds = (val_probs_full >= t).astype(int)
                if metric_to_optimize == 'f1':
                    score = f1_score(val_true, preds, zero_division=0)
                else:
                    score = accuracy_score(val_true, preds)
                    
                # Track precision-focused metric
                precision = precision_score(val_true, preds, zero_division=0)
                recall = recall_score(val_true, preds, zero_division=0)
                # Precision-recall balance with precision emphasis
                pr_balance = 0.7 * precision + 0.3 * recall if (precision + recall) > 0 else 0
                
                if score > best_score:
                    best_score, best_thr = score, t
                    
                if pr_balance > best_precision_score:
                    best_precision_score, best_precision_thr = pr_balance, t
                    
            print(f"[Fold {current_fold + 1}] Best threshold‐tuned F1 {best_score:.4f} at threshold {best_thr:.2f}")
            print(f"[Fold {current_fold + 1}] Best precision-focused score {best_precision_score:.4f} at threshold {best_precision_thr:.2f}")
            
            # Use precision-focused threshold if it's significantly better
            if best_precision_score > 0.8 * best_score:
                threshold = best_precision_thr
                print(f"[Fold {current_fold + 1}] Using precision-focused threshold: {threshold:.2f}")
            else:
                threshold = best_thr

        # --- Final Evaluation on Test Set ---
        net.eval()
        with torch.no_grad():
            test_outputs = net(test_tensor_x) # Raw logits [n_test, 1]
            test_probs = torch.sigmoid(test_outputs).cpu().numpy().flatten()
            test_pred_labels = (test_probs >= threshold).astype(int) # use tuned threshold
            y_test_true = y_test_events # Ground truth [n_test,]

            # Calculate test accuracy for this fold
            test_accuracy = accuracy_score(y_test_true, test_pred_labels)
            fold_test_accuracies.append(test_accuracy)
            print(f"[Fold {current_fold + 1}] Final Test Accuracy: {test_accuracy:.4f}")

        # --- Store Fold Results --- 
        # Store predictions and metadata for each patient in the test set of this fold
        for i in range(len(test_patient_info)):
            patient_meta = test_patient_info[i]
            dataset_type = patient_meta.get('dataset_type', 'NYU')
                
            fold_results = {
                "patient_id": patient_meta['patient_id'],
                "fold": current_fold,
                "predicted_risk_score": test_probs[i],
                "predicted_label": test_pred_labels[i],
                "event_indicator": int(y_test_true[i]),
                "dataset_type": dataset_type
            }
            all_fold_results.append(fold_results)

    print("\n===== Finished all folds =====")

    # --- Aggregation and Final Reporting --- 
    if not all_fold_results:
        print("[ERROR] No results were collected from any fold. Exiting.")
        return

    # Convert collected results to DataFrame
    final_predictions_df = pd.DataFrame(all_fold_results)

    # Ensure DataFrame has the expected number of rows (total patients)
    total_patients_expected = len(data_module.all_patients)
    if len(final_predictions_df) != total_patients_expected:
        print(f"[WARN] Final predictions DataFrame has {len(final_predictions_df)} rows, but expected {total_patients_expected} (total unique patients). Check for patient dropouts or duplication.")
    else:
        print(f"Final predictions DataFrame contains results for all {len(final_predictions_df)} patients.")

    # --- Save Final Predictions --- 
    final_csv_path = os.path.join(args.output_dir, "final_cv_predictions_binary.csv")
    final_predictions_df.sort_values(by=["dataset_type", "patient_id"], inplace=True)
    final_predictions_df.to_csv(final_csv_path, index=False)
    print(f"Final aggregated predictions CSV saved to {final_csv_path}")

    # --- Calculate and Report Metrics --- 
    y_true_all = final_predictions_df["event_indicator"].values
    y_pred_scores_all = final_predictions_df["predicted_risk_score"].values
    y_pred_labels_all = final_predictions_df["predicted_label"].values

    def calculate_metrics(y_true, y_pred_scores, y_pred_labels, description):
        print(f"\n--- {description} Metrics --- ")
        if len(y_true) == 0:
            print("No samples found for this subset.")
            return

        try:
            accuracy = accuracy_score(y_true, y_pred_labels)
            precision = precision_score(y_true, y_pred_labels, zero_division=0)
            recall = recall_score(y_true, y_pred_labels, zero_division=0)
            f1 = f1_score(y_true, y_pred_labels, zero_division=0)
            # Check if both classes are present for ROC AUC
            if len(np.unique(y_true)) > 1:
                roc_auc = roc_auc_score(y_true, y_pred_scores)
                print(f"ROC AUC:   {roc_auc:.4f}")
            else:
                print("ROC AUC:   Not defined (only one class present)")
                roc_auc = np.nan

            print(f"Accuracy:  {accuracy:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall:    {recall:.4f}")
            print(f"F1 Score:  {f1:.4f}")
            print(f"Positive Class Samples: {int(sum(y_true))}/{len(y_true)}")
            return {'roc_auc': roc_auc, 'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}
        except Exception as e:
            print(f"Error calculating metrics for {description}: {e}")
            return None

    # Overall Metrics (on combined test sets from all folds)
    overall_metrics = calculate_metrics(y_true_all, y_pred_scores_all, y_pred_labels_all, "Overall Test Set (Aggregated)")

    # --- Per-Fold Accuracy Summary --- 
    fold_stats = {} # Initialize fold stats dict
    if fold_test_accuracies:
        mean_accuracy = np.mean(fold_test_accuracies)
        std_accuracy = np.std(fold_test_accuracies)
        min_accuracy = np.min(fold_test_accuracies)
        max_accuracy = np.max(fold_test_accuracies)
        print("\n--- Cross Validation Accuracy Statistics (Per-Fold Test Set Accuracies) ---")
        print(f"Mean: {mean_accuracy:.4f}")
        print(f"Standard Deviation: {std_accuracy:.4f}")
        print(f"Minimum: {min_accuracy:.4f}")
        print(f"Maximum: {max_accuracy:.4f}")
        fold_stats = {
            'mean_accuracy': mean_accuracy,
            'std_accuracy': std_accuracy,
            'min_accuracy': min_accuracy,
            'max_accuracy': max_accuracy,
            'fold_accuracies': fold_test_accuracies # Store individual fold accuracies
        }
    else:
        print("\n--- No per-fold accuracies recorded (likely due to errors or no completed folds) ---")

    # --- Save Run Summary ---
    try:
        summary_data = {
            "hyperparameters": vars(args),
            "metrics": {
                "overall": overall_metrics,
                "cv_fold_stats": fold_stats if fold_stats else None # Include CV stats if available
            }
        }
        summary_file_path = os.path.join(args.output_dir, "run_summary.json")
        # Convert numpy types to native Python types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.bool_):
                return bool(obj)
            elif pd.isna(obj): # Handle pandas NaNs specifically if they appear
                 return None
            return obj

        with open(summary_file_path, 'w') as f:
            json.dump(summary_data, f, indent=4, default=convert_numpy)
        print(f"Run summary saved to {summary_file_path}")
    except Exception as e:
        print(f"[ERROR] Failed to save run summary: {e}")

def main(args):
    # If hyper-parameter search is requested, training will be handled later.
    if args.hyper_search_iters <= 1:
        if args.cross_validation:
            cross_validation_mode(args)
        else:
            # Standard train/val/test split mode 
            print("Running in standard train/val/test mode (not cross-validation)")
            print("[WARN] Standard train/val/test mode not fully implemented yet.")

    print(f"Output directory: {args.output_dir}")

    # ------------------------------------------------------------
    # Optional internal hyper-parameter search (learning-rate only)
    # ------------------------------------------------------------

    def _run_single_trial(trial_idx: int, base_args):
        """Run a single training trial with a randomly sampled learning rate.
        Returns (auc, precision, trial_output_dir)."""
        trial_args = copy.deepcopy(base_args)

        # --------- Random hyper-parameter sampling ---------
        # Learning rate (log-uniform 1e-7 – 1e-3)
        lr_sample = 10 ** random.uniform(-7, -3)
        trial_args.learning_rate = lr_sample

        # Model architecture choice - bias towards successful architectures
        trial_args.model_arch = random.choices(
            ['linear', 'mlp', 'enhanced_mlp', 'ultra_precision', 'precision_weighted_ensemble', 'precision_focused', 'advanced'], 
            weights=[0.05, 0.1, 0.2, 0.3, 0.25, 0.1, 0.05]  # Heavily favor new precision architectures
        )[0]
        
        if trial_args.model_arch in ['mlp', 'enhanced_mlp', 'ultra_precision', 'precision_weighted_ensemble', 'precision_focused', 'advanced']:
            # Sample dropout - bias towards successful ranges
            trial_args.dropout = random.choices(
                [0.1, 0.2, 0.3, 0.4, 0.5], 
                weights=[0.2, 0.3, 0.3, 0.15, 0.05]
            )[0]

        # Batch size - bias towards smaller batches that performed well
        bs_choice = random.choices([8, 16, 32], weights=[0.5, 0.4, 0.1])[0]
        trial_args.batch_size = bs_choice

        # Num slices - bias towards successful ranges
        slices_choice = random.choices([16, 24, 32, 40, 48], weights=[0.1, 0.3, 0.3, 0.2, 0.1])[0]
        trial_args.num_slices = slices_choice

        # Upsampling method - strongly bias towards ADASYN
        upsampling_choices = ['smote', 'adasyn', 'random']
        up_method = random.choices(upsampling_choices, weights=[0.2, 0.7, 0.1])[0]
        trial_args.upsampling_method = up_method

        # Threshold metric for tuning - bias towards precision-recall balance
        trial_args.threshold_metric = random.choices(['auprc', 'f1'], weights=[0.6, 0.4])[0]

        # Loss function strategy - bias towards precision-focused losses
        loss_strategy = random.choices(['focal', 'precision_recall_focal', 'cost_sensitive', 'label_smooth', 'standard'], 
                                     weights=[0.3, 0.3, 0.2, 0.1, 0.1])[0]
        
        if loss_strategy == 'focal':
            trial_args.focal_loss = True
            trial_args.precision_recall_focal = False
            trial_args.cost_sensitive = False
            trial_args.label_smoothing = 0.0
            trial_args.focal_gamma = random.choices([1.5, 2.0, 2.5], weights=[0.3, 0.4, 0.3])[0]
        elif loss_strategy == 'precision_recall_focal':
            trial_args.focal_loss = False
            trial_args.precision_recall_focal = True
            trial_args.cost_sensitive = False
            trial_args.label_smoothing = 0.0
            trial_args.focal_gamma = random.choices([1.5, 2.0, 2.5], weights=[0.3, 0.4, 0.3])[0]
        elif loss_strategy == 'cost_sensitive':
            trial_args.focal_loss = False
            trial_args.precision_recall_focal = False
            trial_args.cost_sensitive = True
            trial_args.label_smoothing = 0.0
        elif loss_strategy == 'label_smooth':
            trial_args.focal_loss = False
            trial_args.precision_recall_focal = False
            trial_args.cost_sensitive = False
            trial_args.label_smoothing = random.choices([0.05, 0.1, 0.15], weights=[0.3, 0.5, 0.2])[0]
        else:  # standard
            trial_args.focal_loss = False
            trial_args.precision_recall_focal = False
            trial_args.cost_sensitive = False
            trial_args.label_smoothing = 0.0

        # ----------------------------------------------------

        # Re-define output directory to encode key params for traceability
        trial_dir_name = (
            f"trial_{trial_idx+1}_lr{lr_sample:.0e}_bs{bs_choice}_sl{slices_choice}_ups{up_method}_arch{trial_args.model_arch}_focal{int(trial_args.focal_loss)}_smooth{trial_args.label_smoothing:.2f}_thr{trial_args.threshold_metric}"
        )
        trial_args.output_dir = os.path.join(base_args.output_dir, trial_dir_name)
        os.makedirs(trial_args.output_dir, exist_ok=True)

        print(f"\n[SEARCH] Trial {trial_idx+1}/{base_args.hyper_search_iters}: LR={lr_sample:.2e}  -> {trial_args.output_dir}")

        # Run training/validation/testing for this set of hyper-params
        cross_validation_mode(trial_args)

        # Read back the metrics from the run_summary.json
        summary_path = os.path.join(trial_args.output_dir, "run_summary.json")
        if not os.path.isfile(summary_path):
            print(f"[SEARCH] Trial {trial_idx+1}: summary file not found at {summary_path}. Skipping candidate.")
            return -math.inf, -math.inf, trial_args.output_dir

        with open(summary_path, 'r') as fp:
            summ = json.load(fp)

        try:
            roc_auc = summ["metrics"]["overall"]["roc_auc"]
            precision_overall = summ["metrics"]["overall"]["precision"]
        except Exception as e:
            print(f"[SEARCH] Trial {trial_idx+1}: could not parse metrics ({e}). Skipping candidate.")
            roc_auc, precision_overall = -math.inf, -math.inf

        # Free GPU memory between trials
        torch.cuda.empty_cache()

        return roc_auc, precision_overall, trial_args.output_dir

    if args.hyper_search_iters > 1:
        print(f"\n========== Starting hyper-parameter search: {args.hyper_search_iters} trials ==========")

        base_output_dir = args.output_dir  # keep the user-specified directory as the parent
        best_auc, best_prec, best_dir = -math.inf, -math.inf, None

        for t in range(args.hyper_search_iters):
            auc, prec_o, out_dir = _run_single_trial(t, args)
            print(f"[SEARCH] Trial {t+1} finished: AUC={auc:.4f}, Prec_Ov={prec_o:.4f}")

            if auc > best_auc or (math.isclose(auc, best_auc) and prec_o > best_prec):
                best_auc, best_prec, best_dir = auc, prec_o, out_dir

        print("\n========== Hyper-parameter search completed ==========")
        if best_dir is not None:
            print(f"Best trial directory: {best_dir}")
            print(f"Best ROC-AUC: {best_auc:.4f}")
            print(f"Best Precision: {best_prec:.4f}")

            # Copy best run under a canonical name for convenience
            final_best_path = os.path.join(base_output_dir, "best_run")
            try:
                if os.path.exists(final_best_path):
                    shutil.rmtree(final_best_path)
                shutil.copytree(best_dir, final_best_path)
                print(f"Copied best run to {final_best_path}")
            except Exception as e:
                print(f"[WARN] Could not copy best run directory: {e}")

        else:
            print("[WARN] No successful trials completed – no best model to report.")

        # End the script after search – avoid running main() again.
        sys.exit(0)

    # --- No search requested, our work is already done earlier in this function ---
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train binary classification model with DINOv2 features using NYU dataset and cross-validation")
    
    # NYU-only arguments
    parser.add_argument("--nyu_dicom_root", type=str, default="/gpfs/data/mankowskilab/HCC_Recurrence/dicom",
                         help="Path to the NYU DICOM root directory.")
    parser.add_argument("--nyu_csv_file", type=str, default="/gpfs/data/shenlab/wz1492/HCC/spreadsheets/processed_patient_labels_nyu.csv",
                         help="Path to the NYU CSV metadata file.")
    
    parser.add_argument('--preprocessed_root', type=str, default='/gpfs/data/mankowskilab/HCC_Recurrence/preprocessed/', 
                         help='Base directory to store/load preprocessed image tensors.')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for data loaders')
    parser.add_argument('--num_slices', type=int, default=32,
                        help='Number of slices per patient sample')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of workers for data loaders')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Maximum number of training epochs per fold')
    parser.add_argument('--output_dir', type=str, default='checkpoints_binary_cv',
                        help='Base directory to save outputs and models')
    parser.add_argument('--learning_rate', type=float, default=1e-5,
                        help='Learning rate for the classification model')
    parser.add_argument('--gradient_clip', type=float, default=1.0,
                        help='Gradient clipping threshold. Set 0 to disable.')
    parser.add_argument('--num_samples_per_patient', type=int, default=1,
                        help='Number of times to sample slices from each patient series')
    parser.add_argument('--dinov2_weights', type=str, default=None,
                        help="Path to your local DINOv2 state dict file (.pth or .pt). If not provided, uses pretrained ImageNet DINO weights.")
    parser.add_argument('--upsampling', action='store_true',
                        help="If set, perform upsampling of the minority class in the training data for each fold")
    parser.add_argument('--upsampling_method', type=str, default='smote', choices=['random','smote','adasyn'], help="Upsampling method: 'random', 'smote', or 'adasyn'")
    parser.add_argument('--early_stopping', action='store_true',
                        help="If set, early stopping will be used based on validation loss within each fold")
    parser.add_argument('--early_stopping_patience', type=int, default=20, 
                        help="Number of epochs with no improvement to wait before stopping.")
    parser.add_argument('--cross_validation', action='store_true', default=True,
                        help="Enable cross validation mode")
    parser.add_argument('--cv_folds', type=int, default=7,
                        help="Number of cross validation folds")
    parser.add_argument('--leave_one_out', action='store_true',
                        help="Enable leave-one-out cross validation mode (overrides cv_folds)")
    parser.add_argument('--threshold_metric', type=str, choices=['auprc','f1'], default='auprc',
                        help="Metric to optimize threshold on validation: 'auprc' or 'f1'")
    
    # --- Hyper-parameter search ---
    parser.add_argument('--hyper_search_iters', type=int, default=1,
                        help='If >1, perform this many random hyper-parameter trials (learning-rate search) inside a single job and keep the best run based on ROC-AUC (ties broken by precision).')
    
    # New model architecture options
    parser.add_argument('--model_arch', type=str, default='enhanced_mlp', 
                        choices=['linear', 'mlp', 'enhanced_mlp', 'ultra_precision', 'precision_weighted_ensemble', 'precision_focused', 'advanced'],
                        help="Network architecture: 'linear', 'mlp', 'enhanced_mlp', 'ultra_precision', 'precision_weighted_ensemble', 'precision_focused', or 'advanced'.")
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate for MLP (if used).')
    # Optional focal-loss arguments
    parser.add_argument('--focal_loss', action='store_true',
                        help="Use focal loss instead of (or in addition to) BCEWithLogitsLoss.")
    parser.add_argument('--focal_gamma', type=float, default=2.0, help="Gamma parameter for focal loss.")
    
    # Precision-Recall Focal Loss option
    parser.add_argument('--precision_recall_focal', action='store_true',
                        help="Use advanced precision-recall focal loss for better precision optimization.")
    
    # Cost-sensitive learning option
    parser.add_argument('--cost_sensitive', action='store_true',
                        help="Use cost-sensitive loss that heavily penalizes false negatives.")
    
    # Label smoothing option
    parser.add_argument('--label_smoothing', type=float, default=0.1, help="Label smoothing factor for LabelSmoothingBCE loss.")
    
    args = parser.parse_args()

    # --- Output Directory Setup --- 
    # Create a unique subdirectory for each run using a timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"run_{timestamp}_lr{args.learning_rate}_bs{args.batch_size}_slices{args.num_slices}" 
    if args.upsampling: run_name += "_upsampled"
    if args.leave_one_out: run_name += "_loocv"
    else: run_name += f"_{args.cv_folds}fold"
        
    args.output_dir = os.path.join(args.output_dir, run_name)
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output directory: {args.output_dir}")
    
    main(args)