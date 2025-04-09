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
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, accuracy_score
import datetime  # new import for timestamp
from sklearn.model_selection import KFold
# Custom module imports
from data.dataset import HCCDataModule
from models.dino import load_dinov2_model
# Removed unused survival helpers and plots
# from utils.helpers import extract_features, validate_survival_data, upsample_training_data
# from callbacks.custom_callbacks import GradientClippingCallback, LossLogger, ParamCheckerCallback
# from utils.plotting import (
#                       plot_cv_metrics, 
#                       plot_survival_functions, plot_brier_score,
#                       plot_risk_score_distribution, plot_kaplan_meier, plot_calibration_plot,
#                       plot_multi_calibration, plot_cumulative_hazard, plot_survival_probability_distribution)
import numpy as np
from tqdm import tqdm
import torch

sns.set(style="whitegrid")

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
             source_example = data_loader.dataset.patient_data[0].get('source', 'Unknown')
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
        patient_info = [{'patient_id': f'unknown_{i}', 'dataset_type': 'Unknown'} for i in range(len(features))] 
        
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


# Removed train_and_evaluate function - logic merged into cross_validation_mode

def upsample_df(df, target_column='event'): # Keep this utility? Maybe not needed if upsampling features directly.
    df_majority = df[df[target_column] == 0]
    df_minority = df[df[target_column] == 1]
    if len(df_minority) == 0 or len(df_majority) == 0:
        return df
    df_minority_upsampled = df_minority.sample(len(df_majority), replace=True, random_state=1)
    return pd.concat([df_majority, df_minority_upsampled]).reset_index(drop=True)

def cross_validation_mode(args):
    """
    Perform cross-validation (combined, tcga, or nyu based on args.cv_mode).
    """
    # Hyperparameters for training
    hyperparams = {
        'learning_rate': args.learning_rate,
        # 'dropout': args.dropout, # Not used in Linear model
        # 'alpha': args.alpha, # Survival specific
        # 'gamma': args.gamma, # Survival specific
        # 'coxph_net': args.coxph_net # Survival specific
    }

    # --- Data Module Setup --- 
    data_module = HCCDataModule(
        train_csv_file=args.tcga_csv_file, # Pass TCGA csv
        test_csv_file=args.nyu_csv_file,   # Pass NYU csv
        train_dicom_root=args.tcga_dicom_root, # Pass TCGA root
        test_dicom_root=args.nyu_dicom_root,   # Pass NYU root
        model_type="linear", # Set explicitly for binary classification
        batch_size=args.batch_size,
        num_slices=args.num_slices,
        num_samples=args.num_samples_per_patient,
        num_workers=args.num_workers,
        preprocessed_root=args.preprocessed_root,
        cross_validation=True,
        cv_folds=args.cv_folds,
        cv_mode=args.cv_mode,
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
    # dino_model.register_forward_hook(lambda m, i, o: None) # Not needed if requires_grad=False

    # --- Results Storage --- 
    all_fold_results = [] # Store results dictionary for each test patient
    fold_test_accuracies = [] # Store accuracy per fold for averaging

    # --- Cross-Validation Loop --- 
    current_fold = -1 # Initialize fold counter
    while True:
        current_fold += 1
        print(f"\n===== Processing Fold {current_fold + 1}/{data_module.get_total_folds()} =====")

        # Get dataloaders for current fold
        # These now use the combined, pre-split data for the fold
    train_loader = data_module.train_dataloader()
        val_loader = data_module.val_dataloader() # Can be None
        test_loader = data_module.test_dataloader()

        # Add fold index to dataset for potential use in extract_features desc
        if hasattr(train_loader.dataset, '__fold_idx__'): train_loader.dataset.__fold_idx__ = current_fold
        if val_loader and hasattr(val_loader.dataset, '__fold_idx__'): val_loader.dataset.__fold_idx__ = current_fold
        if hasattr(test_loader.dataset, '__fold_idx__'): test_loader.dataset.__fold_idx__ = current_fold

        # --- Feature Extraction --- 
        # Note: extract_features now returns patient_info list as the third element
        x_train, y_train_events, _ = extract_features(train_loader, dino_model, device)
        x_val, y_val_events, _ = extract_features(val_loader, dino_model, device) # Will be empty if val_loader is None
        x_test, y_test_events, test_patient_info = extract_features(test_loader, dino_model, device)

        # Check for empty datasets post-extraction
        if x_train.size == 0:
             print(f"[WARN] Fold {current_fold + 1}: No training features extracted. Skipping fold.")
             if not data_module.next_fold(): break
             else: continue
        if x_test.size == 0:
             print(f"[WARN] Fold {current_fold + 1}: No test features extracted. Skipping fold.")
             if not data_module.next_fold(): break
             else: continue
        # Validation set can be empty, handle downstream
        has_validation_data = x_val.size > 0
        if not has_validation_data:
             print(f"[INFO] Fold {current_fold + 1}: No validation data available or extracted.")

        # --- Feature Preprocessing --- 
        # Average across samples dimension -> [patients, slices, features]
    x_train = x_train.mean(axis=1)
        if has_validation_data: x_val = x_val.mean(axis=1)
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
            if has_validation_data: x_val = x_val[:, :, non_zero_var_indices]
            x_test = x_test[:, :, non_zero_var_indices]
            # Update feature dimension
            n_train_f = x_train.shape[2]
        else:
             print(f"[INFO] Fold {current_fold + 1}: No zero-variance features found in training set.")

        # Upsampling (optional, on training data only)
    if args.upsampling:
            print(f"[INFO] Fold {current_fold + 1}: Performing upsampling on training data...")
            # Reshape train features before upsampling if needed, or adapt upsampling function
            # Assuming upsample works on [patients, ...] shape
            x_train_shape_before_upsample = x_train.shape
            x_train, y_train_events = upsample_training_data(x_train, y_train_events)
            print(f"[INFO] Fold {current_fold + 1}: Training data shape after upsampling: {x_train.shape}")
            # Update train shape info if needed
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

        # Collapse the slice dimension by averaging -> [patients, features]
        x_train_final = x_train_scaled.mean(axis=1)
        x_val_final = x_val_scaled.mean(axis=1) if has_validation_data else np.array([])
        x_test_final = x_test_scaled.mean(axis=1)
        
        print(f"[INFO] Fold {current_fold + 1}: Final feature shapes: Train {x_train_final.shape}, Val {x_val_final.shape}, Test {x_test_final.shape}")

        # --- Model Training --- 
        in_features = x_train_final.shape[1]
        net = nn.Linear(in_features, 1) # Single output for BCEWithLogitsLoss
    net.to(device)
    
    optimizer = torch.optim.Adam(net.parameters(), lr=hyperparams['learning_rate'], weight_decay=1e-4)
        # Use BCEWithLogitsLoss for binary classification with single output neuron
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

            # --- Early Stopping & Model Checkpointing --- 
            if args.early_stopping and has_validation_data:
                 current_loss_for_stopping = val_loss_epoch
                 if current_loss_for_stopping < best_val_loss:
                     best_val_loss = current_loss_for_stopping
                     best_model_state = net.state_dict().copy() # Save best model
                epochs_no_improve = 0
                     print(f"[Fold {current_fold + 1}] New best validation loss: {best_val_loss:.4f}. Saving model.")
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

        # --- Final Evaluation on Test Set --- 
    net.eval()
    with torch.no_grad():
            test_outputs = net(test_tensor_x) # Raw logits [n_test, 1]
            test_probs = torch.sigmoid(test_outputs).cpu().numpy().flatten() # Probabilities [n_test,]
            test_pred_labels = (test_probs >= 0.5).astype(int) # Predicted labels [n_test,]
            y_test_true = y_test_events # Ground truth [n_test,]
            
            # Calculate test accuracy for this fold
            test_accuracy = accuracy_score(y_test_true, test_pred_labels)
            fold_test_accuracies.append(test_accuracy)
            print(f"[Fold {current_fold + 1}] Final Test Accuracy: {test_accuracy:.4f}")

        # --- Store Fold Results --- 
        # Store predictions and metadata for each patient in the test set of this fold
        for i in range(len(test_patient_info)):
            patient_meta = test_patient_info[i]
            fold_results = {
                "patient_id": patient_meta['patient_id'],
                "fold": current_fold + 1,
                "predicted_risk_score": test_probs[i],
                "predicted_label": test_pred_labels[i],
                "event_indicator": int(y_test_true[i]),
                "dataset_type": patient_meta['dataset_type'] # Store TCGA or NYU
            }
            all_fold_results.append(fold_results)

        # --- Move to Next Fold --- 
        if not data_module.next_fold():
            print("\n===== Finished all folds =====")
            break

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
    final_csv_path = os.path.join(args.output_dir, "final_cv_predictions_combined.csv")
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

    # TCGA Metrics
    tcga_df = final_predictions_df[final_predictions_df["dataset_type"] == "TCGA"]
    tcga_metrics = calculate_metrics(tcga_df["event_indicator"].values, 
                                   tcga_df["predicted_risk_score"].values, 
                                   tcga_df["predicted_label"].values, 
                                   "TCGA Test Set (Aggregated)")

    # NYU Metrics
    nyu_df = final_predictions_df[final_predictions_df["dataset_type"] == "NYU"]
    nyu_metrics = calculate_metrics(nyu_df["event_indicator"].values, 
                                  nyu_df["predicted_risk_score"].values, 
                                  nyu_df["predicted_label"].values, 
                                  "NYU Test Set (Aggregated)")

    # --- Per-Fold Accuracy Summary --- 
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
    else:
         print("\n--- No per-fold accuracies recorded (likely due to errors or no completed folds) ---")

def cross_prediction_mode(args):
    """
    Train on one dataset (NYU or TCGA) and test on the other.
    Uses a validation split from the training set for early stopping.
    """
    hyperparams = {
        'learning_rate': args.learning_rate,
    }

    # Determine train/test datasets based on mode
    if args.run_mode == 'cp_nyu_tcga': # Train NYU, Test TCGA
        train_csv = args.nyu_csv_file
        train_root = args.nyu_dicom_root
        test_csv = args.tcga_csv_file
        test_root = args.tcga_dicom_root
        train_source_name = "NYU"
        test_source_name = "TCGA"
        print("--- Running Cross Prediction: Train on NYU, Test on TCGA ---")
    elif args.run_mode == 'cp_tcga_nyu': # Train TCGA, Test NYU
        train_csv = args.tcga_csv_file
        train_root = args.tcga_dicom_root
        test_csv = args.nyu_csv_file
        test_root = args.nyu_dicom_root
        train_source_name = "TCGA"
        test_source_name = "NYU"
        print("--- Running Cross Prediction: Train on TCGA, Test on NYU ---")
    else:
        raise ValueError(f"Invalid run_mode for cross_prediction: {args.run_mode}")

    # --- Data Module Setup (No CV) ---
    data_module = HCCDataModule(
        train_csv_file=train_csv,
        test_csv_file=test_csv,
        train_dicom_root=train_root,
        test_dicom_root=test_root,
        model_type="linear",
        batch_size=args.batch_size,
        num_slices=args.num_slices,
        num_samples=args.num_samples_per_patient,
        num_workers=args.num_workers,
        preprocessed_root=args.preprocessed_root,
        cross_validation=False, # IMPORTANT: Not doing CV splitting here
        use_validation=True    # IMPORTANT: Enable train/val split of the training data
        # cv_mode, cv_folds, leave_one_out, random_state are ignored when cross_validation=False
    )
    try:
        data_module.setup() 
    except ValueError as e:
        print(f"[ERROR] Failed to setup DataModule: {e}. Check dataset paths and integrity.")
        return

    # --- Model Setup --- 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    dino_model = load_dinov2_model(args.dinov2_weights)
    dino_model = dino_model.to(device)
    dino_model.eval()
    for param in dino_model.parameters(): param.requires_grad = False

    # --- Get Dataloaders ---
    try:
        train_loader = data_module.train_dataloader()
        val_loader = data_module.val_dataloader() 
        test_loader = data_module.test_dataloader()
    except ValueError as e:
         print(f"[ERROR] Failed to get dataloaders: {e}. Ensure datasets were setup correctly.")
         return

    if val_loader is None:
        print("[WARN] Validation loader is None. Early stopping will be disabled.")
        use_early_stopping_run = False
    else:
        use_early_stopping_run = args.early_stopping

    # --- Feature Extraction --- 
    print(f"Extracting features for {train_source_name} training set...")
    x_train, y_train_events, _ = extract_features(train_loader, dino_model, device)
    print(f"Extracting features for {train_source_name} validation set...")
    x_val, y_val_events, _ = extract_features(val_loader, dino_model, device) 
    print(f"Extracting features for {test_source_name} test set...")
    x_test, y_test_events, test_patient_info = extract_features(test_loader, dino_model, device)

    # --- Basic Data Checks ---
    if x_train.size == 0:
         print(f"[ERROR] No training data extracted for {train_source_name}. Cannot proceed.")
         return
    if x_test.size == 0:
         print(f"[ERROR] No test data extracted for {test_source_name}. Cannot proceed.")
         return
    has_validation_data = x_val.size > 0
    if not has_validation_data:
         print(f"[INFO] No validation data available or extracted for {train_source_name}.")
         use_early_stopping_run = False # Override if needed

    # --- Feature Preprocessing (as in CV mode) ---
    # Average across samples -> [patients, slices, features]
    x_train = x_train.mean(axis=1)
    if has_validation_data: x_val = x_val.mean(axis=1)
    x_test = x_test.mean(axis=1)

    # Remove zero-variance features (fit on train)
    n_train_p, n_train_s, n_train_f = x_train.shape
    x_train_flat_var = x_train.reshape(-1, n_train_f)
    variances = np.var(x_train_flat_var, axis=0)
    zero_var_indices = np.where(variances == 0)[0]
    if len(zero_var_indices) > 0:
        print(f"[INFO] Removing {len(zero_var_indices)} zero-variance features.")
        non_zero_var_indices = np.where(variances != 0)[0]
        x_train = x_train[:, :, non_zero_var_indices]
        if has_validation_data: x_val = x_val[:, :, non_zero_var_indices]
        x_test = x_test[:, :, non_zero_var_indices]
        n_train_f = x_train.shape[2]
    else:
         print(f"[INFO] No zero-variance features found.")

    # Upsampling (optional, on training data)
        if args.upsampling:
        print(f"[INFO] Upsampling {train_source_name} training data...")
        x_train, y_train_events = upsample_training_data(x_train, y_train_events)
        n_train_p, n_train_s, n_train_f = x_train.shape # Update shape

    # Standardize features (fit on train, transform all)
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
        x_val_scaled = np.array([]) 

    n_test_p, n_test_s, n_test_f = x_test.shape
    x_test_reshaped = x_test.reshape(-1, n_test_f)
    x_test_scaled = x_mapper.transform(x_test_reshaped).astype('float32')
    x_test_scaled = x_test_scaled.reshape(n_test_p, n_test_s, n_test_f)

    # Collapse slice dimension by averaging -> [patients, features]
    x_train_final = x_train_scaled.mean(axis=1)
    x_val_final = x_val_scaled.mean(axis=1) if has_validation_data else np.array([])
    x_test_final = x_test_scaled.mean(axis=1)
    
    print(f"[INFO] Final feature shapes: Train {x_train_final.shape}, Val {x_val_final.shape}, Test {x_test_final.shape}")

    # --- Binary Classification Model Training (similar to CV mode training loop) --- 
    in_features = x_train_final.shape[1]
    net = nn.Linear(in_features, 1) # Single output for BCEWithLogitsLoss
    net.to(device)
    
    optimizer = torch.optim.Adam(net.parameters(), lr=hyperparams['learning_rate'], weight_decay=1e-4)
    criterion = torch.nn.BCEWithLogitsLoss()

    # Prepare tensors
    train_tensor_x = torch.tensor(x_train_final, dtype=torch.float32).to(device)
    train_tensor_y = torch.tensor(y_train_events, dtype=torch.float32).unsqueeze(1).to(device)
    val_tensor_x = torch.tensor(x_val_final, dtype=torch.float32).to(device) if has_validation_data else None
    val_tensor_y = torch.tensor(y_val_events, dtype=torch.float32).unsqueeze(1).to(device) if has_validation_data else None
    test_tensor_x = torch.tensor(x_test_final, dtype=torch.float32).to(device)
    test_tensor_y = torch.tensor(y_test_events, dtype=torch.float32).unsqueeze(1).to(device)

    # Training loop
    best_val_loss = float('inf')
    best_model_state = None
    patience = args.early_stopping_patience
    epochs_no_improve = 0
    
    for epoch in range(args.epochs):
        net.train()
        optimizer.zero_grad()
        outputs = net(train_tensor_x) 
        loss = criterion(outputs, train_tensor_y)
        loss.backward()
        if args.gradient_clip > 0: torch.nn.utils.clip_grad_norm_(net.parameters(), args.gradient_clip)
        optimizer.step()
        train_loss_epoch = loss.item()

        val_loss_epoch, val_accuracy_epoch = None, None
        if has_validation_data and val_tensor_x is not None:
            net.eval()
            with torch.no_grad():
                val_outputs = net(val_tensor_x) 
                val_loss = criterion(val_outputs, val_tensor_y)
                val_loss_epoch = val_loss.item()
                val_probs = torch.sigmoid(val_outputs) 
                val_pred_labels = (val_probs >= 0.5).float() 
                val_accuracy_epoch = accuracy_score(val_tensor_y.cpu().numpy(), val_pred_labels.cpu().numpy())
            net.train() 

        log_msg = f"Epoch {epoch+1}/{args.epochs} - Train Loss: {train_loss_epoch:.4f}"
        if val_loss_epoch is not None: log_msg += f" - Val Loss: {val_loss_epoch:.4f}"
        if val_accuracy_epoch is not None: log_msg += f" - Val Acc: {val_accuracy_epoch:.4f}"
        print(log_msg)

        if use_early_stopping_run and has_validation_data:
             current_loss_for_stopping = val_loss_epoch
             if current_loss_for_stopping < best_val_loss:
                 best_val_loss = current_loss_for_stopping
                 best_model_state = net.state_dict().copy() 
                 epochs_no_improve = 0
                 print(f"New best validation loss: {best_val_loss:.4f}. Saving model.")
             else:
                 epochs_no_improve += 1
                 if epochs_no_improve >= patience:
                     print(f"Early stopping triggered at epoch {epoch+1}.")
                     break
        elif not use_early_stopping_run: 
             best_model_state = net.state_dict().copy()

    # --- Load Best Model --- 
    if best_model_state is not None:
        print("Loading best model state for final evaluation.")
        net.load_state_dict(best_model_state)
        else:
         print("[WARN] No best model state found. Using last state.")
         net.eval() 

    # --- Final Evaluation on Test Set ({test_source_name}) --- 
    net.eval()
    with torch.no_grad():
        test_outputs = net(test_tensor_x) 
        test_probs = torch.sigmoid(test_outputs).cpu().numpy().flatten()
        test_pred_labels = (test_probs >= 0.5).astype(int) 
        y_test_true = y_test_events
    
    print(f"\n--- Evaluating model on {test_source_name} test set --- ")
    def calculate_binary_metrics(y_true, y_pred_scores, y_pred_labels, description):
        # Reusing the function from cross_validation_mode
        print(f"\n--- {description} Metrics --- ")
        if len(y_true) == 0:
             print("No samples found.")
             return
        try:
            accuracy = accuracy_score(y_true, y_pred_labels)
            precision = precision_score(y_true, y_pred_labels, zero_division=0)
            recall = recall_score(y_true, y_pred_labels, zero_division=0)
            f1 = f1_score(y_true, y_pred_labels, zero_division=0)
            roc_auc = np.nan
            if len(np.unique(y_true)) > 1:
                 roc_auc = roc_auc_score(y_true, y_pred_scores)
                 print(f"ROC AUC:   {roc_auc:.4f}")
        else:
                 print("ROC AUC:   Not defined (only one class present)")
            print(f"Accuracy:  {accuracy:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall:    {recall:.4f}")
            print(f"F1 Score:  {f1:.4f}")
            print(f"Positive Class Samples: {int(sum(y_true))}/{len(y_true)}")
            return {'roc_auc': roc_auc, 'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}
        except Exception as e:
            print(f"Error calculating metrics for {description}: {e}")
            return None
            
    test_metrics = calculate_binary_metrics(y_test_true, test_probs, test_pred_labels, f"Test Set ({test_source_name})")
         
    # --- Save Predictions --- 
    predictions_data = []
    for i in range(len(test_patient_info)):
        patient_meta = test_patient_info[i]
        predictions_data.append({
            "patient_id": patient_meta['patient_id'],
            "predicted_risk_score": test_probs[i],
            "predicted_label": test_pred_labels[i],
            "event_indicator": int(y_test_true[i]),
            "dataset_type": patient_meta['dataset_type'] 
        })
    
    pred_df = pd.DataFrame(predictions_data)
    pred_csv_path = os.path.join(args.output_dir, f"cross_predict_results_{test_source_name}_test.csv")
    pred_df.sort_values(by="patient_id", inplace=True)
    pred_df.to_csv(pred_csv_path, index=False)
    print(f"Test set predictions saved to {pred_csv_path}")

def main(args):
    # Dispatch based on run mode
    if args.run_mode == 'cv':
        cross_validation_mode(args)
    elif args.run_mode in ['cp_nyu_tcga', 'cp_tcga_nyu']:
        cross_prediction_mode(args)
    else:
        print(f"[ERROR] Unknown run_mode '{args.run_mode}'. Exiting.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train binary classification model with DINOv2 features")
    
    # --- Arguments --- 
    # Data Paths
    parser.add_argument("--tcga_dicom_root", type=str, default="/gpfs/data/mankowskilab/HCC/data/TCGA/manifest-4lZjKqlp5793425118292424834/TCGA-LIHC", help="Path to TCGA DICOM root.")
    parser.add_argument("--nyu_dicom_root", type=str, default="/gpfs/data/mankowskilab/HCC_Recurrence/dicom", help="Path to NYU DICOM root.")
    parser.add_argument("--tcga_csv_file", type=str, default="/gpfs/data/shenlab/wz1492/HCC/spreadsheets/tcga.csv", help="Path to TCGA CSV.")
    parser.add_argument("--nyu_csv_file", type=str, default="/gpfs/data/shenlab/wz1492/HCC/spreadsheets/nyu_recurrence.csv", help="Path to NYU CSV.") 
    parser.add_argument('--preprocessed_root', type=str, default='/gpfs/data/mankowskilab/HCC_Recurrence/preprocessed/', help='Base directory for preprocessed tensors. Set to None or empty to disable.')

    # Run Mode
    parser.add_argument('--run_mode', type=str, default='cv', choices=['cv', 'cp_nyu_tcga', 'cp_tcga_nyu'],
                        help="Execution mode: 'cv' (cross-validation), 'cp_nyu_tcga' (train NYU, test TCGA), 'cp_tcga_nyu' (train TCGA, test NYU)")

    # Data Loading & Preprocessing
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--num_slices', type=int, default=32, help='Slices per sample')
    parser.add_argument('--num_workers', type=int, default=4, help='DataLoader workers')
    parser.add_argument('--num_samples_per_patient', type=int, default=1, help='Slice samples per patient series')
    parser.add_argument('--upsampling', action='store_true', help="Upsample minority class in training data")

    # Model & Training
    parser.add_argument('--epochs', type=int, default=100, help='Max epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--gradient_clip', type=float, default=1.0, help='Gradient clipping value (0 to disable)')
    parser.add_argument('--dinov2_weights', type=str, required=True, help="Path to DINOv2 weights.")
    parser.add_argument('--early_stopping', action='store_true', help="Use early stopping based on validation loss")
    parser.add_argument('--early_stopping_patience', type=int, default=10, help="Patience for early stopping.")

    # Cross-Validation Specific (used only if run_mode='cv')
    # parser.add_argument('--cross_validation', action='store_true', default=True, # Replaced by run_mode
    #                     help="Enable cross validation mode (combines TCGA/NYU)")
    parser.add_argument('--cv_folds', type=int, default=10, help="Number of CV folds (if run_mode='cv')")
    parser.add_argument('--cv_mode', type=str, default='combined', choices=['combined', 'tcga', 'nyu'], 
                        help="Dataset mode for cross-validation (if run_mode='cv')")
    parser.add_argument('--leave_one_out', action='store_true', help="Use LOOCV (if run_mode='cv', overrides cv_folds)")
    
    # Output
    parser.add_argument('--output_dir', type=str, default='checkpoints_binary', help='Base output directory') # Updated default
    
    args = parser.parse_args()

    # Handle empty string for preprocessed_root
    if isinstance(args.preprocessed_root, str) and not args.preprocessed_root.strip():
        args.preprocessed_root = None
        print("Preprocessing disabled (preprocessed_root is empty).")
    elif args.preprocessed_root:
         print(f"Using preprocessed root: {args.preprocessed_root}")
    else:
         print("Preprocessing disabled (preprocessed_root is None).")

    # Create unique output directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    # Base run name on the execution mode
    if args.run_mode == 'cv':
        run_name = f"run_{timestamp}_cv_mode-{args.cv_mode}"
        if args.leave_one_out: run_name += "_loocv"
        else: run_name += f"_{args.cv_folds}fold"
    elif args.run_mode == 'cp_nyu_tcga':
        run_name = f"run_{timestamp}_cp-nyu-on-tcga"
    elif args.run_mode == 'cp_tcga_nyu':
        run_name = f"run_{timestamp}_cp-tcga-on-nyu"
    else: # Fallback
        run_name = f"run_{timestamp}_{args.run_mode}"
        
    # Add common flags to run name
    if args.upsampling: run_name += "_upsampled"
    run_name += f"_lr{args.learning_rate}_bs{args.batch_size}"
        
    args.output_dir = os.path.join(args.output_dir, run_name)
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output directory: {args.output_dir}")
    
    main(args)