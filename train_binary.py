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
    Perform cross-validation or cross-prediction based on args.cv_mode.
    For CV modes (combined, tcga, nyu): Train/Test within folds.
    For Cross-Prediction modes: Train on one dataset, test on the other.
    Aggregate and report test metrics separately for TCGA and NYU sources if applicable.
    """
    hyperparams = {
        'learning_rate': args.learning_rate,
    }

    # --- Data Module Setup --- 
    data_module = HCCDataModule(
        train_csv_file=args.tcga_csv_file, 
        test_csv_file=args.nyu_csv_file,   
        train_dicom_root=args.tcga_dicom_root,
        test_dicom_root=args.nyu_dicom_root,   
        model_type="linear", 
        batch_size=args.batch_size,
        num_slices=args.num_slices,
        num_samples=args.num_samples_per_patient,
        num_workers=args.num_workers,
        preprocessed_root=args.preprocessed_root,
        # Pass cross_validation=True only if it's a CV splitting mode
        cross_validation=(args.cv_mode in ['combined', 'tcga', 'nyu']),
        cv_folds=args.cv_folds,
        cv_mode=args.cv_mode, 
        leave_one_out=args.leave_one_out,
        random_state=42,
        use_validation=True 
    )
    data_module.setup()

    # --- Model Setup --- 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    dino_model = load_dinov2_model(args.dinov2_weights)
    dino_model = dino_model.to(device)
    dino_model.eval()
    for param in dino_model.parameters(): param.requires_grad = False

    # --- Results Storage --- 
    all_test_results = [] # Use a consistent name
    fold_test_accuracies = [] # Still useful for k-fold CV modes

    # --- Training & Evaluation Logic --- 
    is_single_run = args.cv_mode in ['nyu-train_tcga-test', 'tcga-train_nyu-test']
    num_cycles = 1 if is_single_run else data_module.get_total_folds()
    
    for cycle_idx in range(num_cycles):
        current_fold = cycle_idx 
        if not is_single_run:
             print(f"\n===== Processing Fold {current_fold + 1}/{num_cycles} =====")
        else:
             print(f"\n===== Processing Cross-Prediction Run (Mode: {args.cv_mode}) =====")

        # Get dataloaders
        train_loader = data_module.train_dataloader()
        val_loader = data_module.val_dataloader()
        test_loader = data_module.test_dataloader()

        # Add fold index for logging
        if not is_single_run:
             if hasattr(train_loader.dataset, '__fold_idx__'): train_loader.dataset.__fold_idx__ = current_fold
             if val_loader and hasattr(val_loader.dataset, '__fold_idx__'): val_loader.dataset.__fold_idx__ = current_fold
             if hasattr(test_loader.dataset, '__fold_idx__'): test_loader.dataset.__fold_idx__ = current_fold

        # --- Feature Extraction --- 
        x_train, y_train_events, _ = extract_features(train_loader, dino_model, device)
        x_val, y_val_events, _ = extract_features(val_loader, dino_model, device) 
        x_test, y_test_events, test_patient_info = extract_features(test_loader, dino_model, device)

        # Handle empty datasets
        if x_train.size == 0:
             print(f"[WARN] Cycle {current_fold + 1}: No training features. Skipping cycle.")
             if not is_single_run and not data_module.next_fold(): break
             else: continue
        if x_test.size == 0:
             print(f"[WARN] Cycle {current_fold + 1}: No test features. Skipping cycle.")
             if not is_single_run and not data_module.next_fold(): break
             else: continue
        has_validation_data = x_val.size > 0
        use_early_stopping_cycle = args.early_stopping and has_validation_data
        # ... (Validation logic) ...

        # --- Feature Preprocessing --- 
        # ... (Remains the same: mean, zero-var, upsample, scale, average slices) ...

        # --- Model Training --- 
        in_features = x_train_final.shape[1]
        net = nn.Linear(in_features, 1) # Single output
        net.to(device)
        optimizer = torch.optim.Adam(net.parameters(), lr=hyperparams['learning_rate'], weight_decay=1e-4)
        criterion = torch.nn.BCEWithLogitsLoss()

        # Tensors
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
            # ... (Training phase remains the same) ...
            # ... (Validation phase remains the same) ...
            # ... (Logging remains the same, uses cycle_idx as fold number) ...
            # ... (Early stopping / Checkpointing remains the same) ...
        
        # Load Best Model
        # ... (Remains the same) ...

        # --- Final Evaluation on Test Set --- 
        net.eval()
        with torch.no_grad():
            test_outputs = net(test_tensor_x) 
            test_probs = torch.sigmoid(test_outputs).cpu().numpy().flatten()
            test_pred_labels = (test_probs >= 0.5).astype(int)
            y_test_true = y_test_events
            test_accuracy = accuracy_score(y_test_true, test_pred_labels)
            
            print(f"[Cycle {current_fold + 1}] Test Accuracy: {test_accuracy:.4f}")
            # Store accuracy only if it's a CV fold mode and not LOOCV
            if data_module.is_cv_split_mode and not args.leave_one_out:
                 fold_test_accuracies.append(test_accuracy)

        # --- Store Test Results --- 
        for i in range(len(test_patient_info)):
            patient_meta = test_patient_info[i]
            fold_results = {
                "patient_id": patient_meta['patient_id'],
                "fold": current_fold + 1,
                "predicted_risk_score": test_probs[i],
                "predicted_label": test_pred_labels[i],
                "event_indicator": int(y_test_true[i]),
                "dataset_type": patient_meta['dataset_type'] 
            }
            all_test_results.append(fold_results)

        # --- Move to Next Fold (Only for CV modes) --- 
        if data_module.is_cv_split_mode:
            if not data_module.next_fold():
                print("\n===== Finished all CV folds =====")
                break

    # --- Aggregation and Final Reporting --- 
    if not all_test_results:
         print("[ERROR] No test results collected. Exiting.")
         return
         
    final_predictions_df = pd.DataFrame(all_test_results)
    
    # Check row count consistency (for CV modes)
    if data_module.is_cv_split_mode:
        total_patients_expected = len(data_module.all_patients)
        if len(final_predictions_df) != total_patients_expected:
             print(f"[WARN] Final predictions DF has {len(final_predictions_df)} rows, expected {total_patients_expected} for CV mode '{args.cv_mode}'.")
        else:
             print(f"Final predictions DF contains results for all {len(final_predictions_df)} patients in CV mode '{args.cv_mode}'.")
    else:
         print(f"Final predictions DF contains results for {len(final_predictions_df)} patients in the test set.")
         
    # --- Save Final Predictions --- 
    mode_suffix = args.cv_mode.replace('-', '_')
    final_csv_path = os.path.join(args.output_dir, f"final_predictions_binary_{mode_suffix}.csv")
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

    # --- Per-Fold Accuracy Summary (Only for k-fold CV modes) --- 
    if data_module.is_cv_split_mode and not args.leave_one_out and fold_test_accuracies:
        mean_accuracy = np.mean(fold_test_accuracies)
        std_accuracy = np.std(fold_test_accuracies)
        min_accuracy = np.min(fold_test_accuracies)
        max_accuracy = np.max(fold_test_accuracies)
        print("\n--- Cross Validation Accuracy Statistics (Per-Fold Test Set Accuracies) ---")
        print(f"Mean: {mean_accuracy:.4f}")
        print(f"Standard Deviation: {std_accuracy:.4f}")
        print(f"Minimum: {min_accuracy:.4f}")
        print(f"Maximum: {max_accuracy:.4f}")
    elif data_module.is_cv_split_mode and not args.leave_one_out:
         print("\n--- No per-fold accuracies recorded for CV mode ---")

def main(args):
    if args.cross_validation:
        cross_validation_mode(args)
    else:
        # Standard train/val/test split mode (using the combined setup from DataModule)
        print("Running in standard train/val/test mode (not cross-validation)")
        # This part needs to be implemented if non-CV mode is desired
        # It would involve calling data_module.setup() with cross_validation=False
        # Then getting train/val/test loaders and running a single training loop
        # Similar logic to the inside of the CV loop but without fold iteration
        print("[WARN] Standard train/val/test mode not fully implemented yet.")
        # Example call structure (needs full implementation):
        # score = train_single_split(args, hyperparams={'learning_rate': args.learning_rate})
        # print(f"Final Test Accuracy: {score:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train binary classification model with DINOv2 features using combined datasets and cross-validation")
    
    # Changed arguments to reflect TCGA/NYU specifics
    parser.add_argument("--tcga_dicom_root", type=str, default="/gpfs/data/mankowskilab/HCC/data/TCGA/manifest-4lZjKqlp5793425118292424834/TCGA-LIHC",
                         help="Path to the TCGA DICOM root directory.")
    parser.add_argument("--nyu_dicom_root", type=str, default="/gpfs/data/mankowskilab/HCC_Recurrence/dicom",
                         help="Path to the NYU DICOM root directory.")
    parser.add_argument("--tcga_csv_file", type=str, default="/gpfs/data/shenlab/wz1492/HCC/spreadsheets/tcga.csv",
                         help="Path to the TCGA CSV metadata file.")
    parser.add_argument("--nyu_csv_file", type=str, default="/gpfs/data/shenlab/wz1492/HCC/spreadsheets/nyu_recurrence.csv", # Example path, adjust as needed
                         help="Path to the NYU CSV metadata file.")
    
    parser.add_argument('--preprocessed_root', type=str, default='/gpfs/data/mankowskilab/HCC_Recurrence/preprocessed/', 
                         help='Base directory to store/load preprocessed image tensors (will have tcga/ and nyu/ subfolders).')
    parser.add_argument('--batch_size', type=int, default=16, # Reduced default maybe
                        help='Batch size for data loaders')
    parser.add_argument('--num_slices', type=int, default=32, # Reduced default maybe
                        help='Number of slices per patient sample')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of workers for data loaders')
    parser.add_argument('--epochs', type=int, default=100, # Reduced default maybe
                        help='Maximum number of training epochs per fold')
    parser.add_argument('--output_dir', type=str, default='checkpoints_combined_cv',
                        help='Base directory to save outputs and models')
    parser.add_argument('--learning_rate', type=float, default=1e-5,
                        help='Learning rate for the classification model')
    parser.add_argument('--gradient_clip', type=float, default=1.0, # Adjusted default maybe
                        help='Gradient clipping threshold. Set 0 to disable.')
    parser.add_argument('--num_samples_per_patient', type=int, default=1,
                        help='Number of times to sample slices from each patient series')
    parser.add_argument('--dinov2_weights', type=str, required=True,
                        help="Path to your local DINOv2 state dict file (.pth or .pt).")
    parser.add_argument('--upsampling', action='store_true',
                        help="If set, perform upsampling of the minority class in the training data for each fold")
    parser.add_argument('--early_stopping', action='store_true',
                        help="If set, early stopping will be used based on validation loss within each fold")
    parser.add_argument('--early_stopping_patience', type=int, default=10, 
                        help="Number of epochs with no improvement to wait before stopping.")
    parser.add_argument('--cross_validation', action='store_true', default=True, # Defaulting to CV mode
                        help="Enable cross validation mode (combines TCGA/NYU)")
    parser.add_argument('--cv_folds', type=int, default=10,
                        help="Number of cross validation folds")
    parser.add_argument('--cv_mode', type=str, default='combined', choices=['combined', 'tcga', 'nyu', 'nyu-train_tcga-test', 'tcga-train_nyu-test'], 
                        help="Dataset mode: 'combined'(CV), 'tcga'(CV), 'nyu'(CV), 'nyu-train_tcga-test'(Predict), 'tcga-train_nyu-test'(Predict)")
    parser.add_argument('--leave_one_out', action='store_true',
                        help="Enable leave-one-out cross validation mode (overrides cv_folds)")
    
    args = parser.parse_args()

    # --- Output Directory Setup --- 
    # Create a unique subdirectory for each run using a timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    # Add more info to run name if desired (e.g., learning rate, batch size)
    # Include cv_mode in run name
    run_name = f"run_{timestamp}_mode{args.cv_mode}_lr{args.learning_rate}_bs{args.batch_size}_slices{args.num_slices}" 
    if args.upsampling: run_name += "_upsampled"
    if args.leave_one_out: run_name += "_loocv"
    else: run_name += f"_{args.cv_folds}fold"
        
    args.output_dir = os.path.join(args.output_dir, run_name)
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output directory: {args.output_dir}")
    
    main(args)