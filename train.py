import argparse
import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, train_test_split
import torchtuples as tt
from pycox.evaluation import EvalSurv
from lifelines import KaplanMeierFitter
from lifelines.utils import concordance_index  # for overall C-index computation
import datetime  # new import for timestamp
from tqdm import tqdm # Add progress bar

# Custom module imports
from data.dataset import HCCDataModule # Use the updated DataModule
from models.dino import load_dinov2_model
from models.mlp import CustomMLP, CenteredModel, CoxPHWithL1 # Keep survival models
# Removed old helpers, will use adapted versions
# from utils.helpers import extract_features, validate_survival_data, upsample_training_data 
# from callbacks.custom_callbacks import GradientClippingCallback, LossLogger, ParamCheckerCallback # Callbacks handled differently or integrated
from utils.plotting import (plot_cv_metrics, plot_survival_functions, plot_brier_score,
                      plot_risk_score_distribution, plot_kaplan_meier, plot_calibration_plot,
                      plot_multi_calibration, plot_cumulative_hazard, plot_survival_probability_distribution) # Keep plotting utils
                      
sns.set(style="whitegrid")

# --- Adapted Feature Extraction (from train_binary, handles survival labels) ---
def extract_features(data_loader, model, device):
    """
    Extract features using the DINOv2 model for survival data.
    Each slice will have its own feature vector initially.
    Returns features, durations, events, and patient_info list.
    """
    model.eval()
    features = []
    durations = []
    events = []
    patient_info = [] # To store patient dicts

    if data_loader is None:
        # Return empty arrays and list if dataloader is None (e.g., empty val set)
        # Shape assumes DINOv2 base feature dimension 768
        return np.array([]).reshape(0, 1, 1, 768), np.array([]), np.array([]), [] 

    desc = "Extracting Features"
    # Attempt to make description more informative
    try: 
        if hasattr(data_loader.dataset, 'patient_data') and len(data_loader.dataset.patient_data) > 0:
            source_example = data_loader.dataset.patient_data[0].get('source', 'Unknown')
            desc = f"Extracting Features ({source_example} set)" 
        if hasattr(data_loader, 'dataset') and hasattr(data_loader.dataset, '__fold_idx__'):
             desc += f" Fold {data_loader.dataset.__fold_idx__ + 1}"
    except Exception: pass

    with torch.no_grad():
        # Dataloader yields (images, time, event_label) for time_to_event model
        for batch_data in tqdm(data_loader, desc=desc):
             # Handle potential None batches from collate_fn
             if batch_data is None:
                  print("[WARN] extract_features: Skipping None batch from DataLoader.")
                  continue
                  
             if len(batch_data) == 3:
                  images, t, e = batch_data
             else:
                  # Handle cases where the dataloader might yield different structures
                  # or if a batch was completely skipped by collate_fn and returned None previously
                  print(f"[WARN] Unexpected batch structure length: {len(batch_data)}. Skipping batch.")
                  continue

             images = images.to(device)
             batch_size, num_samples, num_slices, C, H, W = images.size()
             images_reshaped = images.view(batch_size * num_samples * num_slices, C, H, W)
             
             feats = model.forward_features(images_reshaped)
             feature_dim = feats.size(-1)
             # Reshape back: [batch, num_samples, num_slices, feature_dim]
             feats_reshaped = feats.view(batch_size, num_samples, num_slices, feature_dim)

             features.append(feats_reshaped.cpu().numpy()) 
             durations.append(t.cpu().numpy())
             events.append(e.cpu().numpy())

    if not features: 
        print("[WARN] extract_features produced no features. Returning empty arrays.")
        feature_dim_expected = 768 
        return np.empty((0, 1, 1, feature_dim_expected), dtype=np.float32), np.empty((0,), dtype=np.float32), np.empty((0,), dtype=np.float32), []

    features = np.concatenate(features, axis=0)
    durations = np.concatenate(durations, axis=0)
    events = np.concatenate(events, axis=0)
    
    # Retrieve corresponding patient data (assumes dataloader order matches dataset.patient_data)
    if hasattr(data_loader.dataset, 'patient_data'):
        # Ensure we only take info for patients that were actually processed
        num_patients_processed = features.shape[0]
        patient_info = data_loader.dataset.patient_data[:num_patients_processed] 
        if len(patient_info) != num_patients_processed:
             print(f"[WARN] Mismatch between processed features ({num_patients_processed}) and retrieved patient info ({len(patient_info)}). Check dataloader/dataset logic.")
             # Fallback or error handling might be needed here
             patient_info = [{'patient_id': f'processed_{i}', 'dataset_type': 'Unknown'} for i in range(num_patients_processed)]
    else:
        print("[WARN] Could not retrieve patient info from dataset in extract_features.")
        num_patients_processed = features.shape[0]
        patient_info = [{'patient_id': f'unknown_{i}', 'dataset_type': 'Unknown'} for i in range(num_patients_processed)]
        
    return features, durations, events, patient_info

# --- Upsampling for Survival Data ---
def upsample_training_data(x_train, durations, events):
    """
    Upsample the minority event class for survival data.
    Keeps features, durations, and events aligned.
    """
    idx_event_1 = np.where(events == 1)[0]
    idx_event_0 = np.where(events == 0)[0]

    if len(idx_event_1) == 0 or len(idx_event_0) == 0:
        print("Warning: One class (event/censored) is empty. Skipping upsampling.")
        return x_train, durations, events

    n_minority = min(len(idx_event_1), len(idx_event_0))
    n_majority = max(len(idx_event_1), len(idx_event_0))

    if n_minority == n_majority:
         print("Event/censored classes are balanced. Skipping upsampling.")
         return x_train, durations, events

    if len(idx_event_1) < len(idx_event_0):
        minority_idx = idx_event_1
        majority_idx = idx_event_0
        print(f"Upsampling event=1 class (Minority size: {n_minority})")
    else:
        minority_idx = idx_event_0
        majority_idx = idx_event_1
        print(f"Upsampling event=0 class (Minority size: {n_minority})")

    n_to_sample = n_majority - n_minority
    sampled_minority_idx = np.random.choice(minority_idx, size=n_to_sample, replace=True)
    
    new_indices = np.concatenate([majority_idx, minority_idx, sampled_minority_idx])
    np.random.shuffle(new_indices)

    x_train_upsampled = x_train[new_indices]
    durations_upsampled = durations[new_indices]
    events_upsampled = events[new_indices]

    print(f"Upsampled training data from {len(events)} to {len(events_upsampled)} samples.")
    return x_train_upsampled, durations_upsampled, events_upsampled
    
# --- Data Validation --- 
def validate_survival_data(durations, events):
     # Basic checks
     if np.any(durations <= 0):
          print("[WARN] Found non-positive durations.")
     if not np.all((events == 0) | (events == 1)):
          print("[WARN] Found event indicators other than 0 or 1.")
     # Add more checks if needed (e.g., from original train.py)

# Removed old train_and_evaluate function

def cross_validation_mode(args):
    """
    Perform cross-validation on the combined TCGA and NYU dataset for survival analysis.
    Train on combined data within each fold.
    Aggregate and report test metrics (C-index) separately for TCGA and NYU sources.
    """
    hyperparams = {
        'learning_rate': args.learning_rate,
        'dropout': args.dropout,
        'alpha': args.alpha, # Regularization for CoxPHWithL1
        'gamma': args.gamma, # Regularization for CoxPHWithL1
        'coxph_net': args.coxph_net # 'mlp' or 'linear'
    }

    # --- Data Module Setup --- 
    data_module = HCCDataModule(
        train_csv_file=args.tcga_csv_file, # TCGA csv
        test_csv_file=args.nyu_csv_file,   # NYU csv
        train_dicom_root=args.tcga_dicom_root, # TCGA root
        test_dicom_root=args.nyu_dicom_root,   # NYU root
        model_type="time_to_event", # Set for survival
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
    data_module.setup() # Combines, filters, preprocesses, splits

    # --- Model Setup --- 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    dino_model = load_dinov2_model(args.dinov2_weights)
    dino_model = dino_model.to(device)
    dino_model.eval()
    for param in dino_model.parameters(): param.requires_grad = False

    # --- Results Storage --- 
    all_fold_results = [] # Store results dict for each test patient
    fold_test_cindices = [] # Store C-index per fold

    # --- Cross-Validation Loop --- 
    current_fold = -1 
    while True:
        current_fold += 1
        print(f"\n===== Processing Fold {current_fold + 1}/{data_module.get_total_folds()} =====")

        # Get dataloaders for current fold 
        train_loader = data_module.train_dataloader()
        val_loader = data_module.val_dataloader() # Can be None
        test_loader = data_module.test_dataloader()

        # Add fold index to dataset for description
        if hasattr(train_loader.dataset, '__fold_idx__'): train_loader.dataset.__fold_idx__ = current_fold
        if val_loader and hasattr(val_loader.dataset, '__fold_idx__'): val_loader.dataset.__fold_idx__ = current_fold
        if hasattr(test_loader.dataset, '__fold_idx__'): test_loader.dataset.__fold_idx__ = current_fold

        # --- Feature Extraction --- 
        x_train, y_train_durations, y_train_events, _ = extract_features(train_loader, dino_model, device)
        x_val, y_val_durations, y_val_events, _ = extract_features(val_loader, dino_model, device) 
        x_test, y_test_durations, y_test_events, test_patient_info = extract_features(test_loader, dino_model, device)

        # Handle empty datasets after extraction
        if x_train.size == 0 or y_train_durations.size == 0:
             print(f"[WARN] Fold {current_fold + 1}: No training data extracted. Skipping fold.")
             if not data_module.next_fold(): break
             else: continue
        if x_test.size == 0 or y_test_durations.size == 0:
             print(f"[WARN] Fold {current_fold + 1}: No test data extracted. Skipping fold.")
             if not data_module.next_fold(): break
             else: continue
        has_validation_data = x_val.size > 0 and y_val_durations.size > 0
        if not has_validation_data:
             print(f"[INFO] Fold {current_fold + 1}: No validation data available or extracted.")
             # Disable early stopping if no validation data
             use_early_stopping_fold = False
        else:
             use_early_stopping_fold = args.early_stopping
             validate_survival_data(y_val_durations, y_val_events) # Validate val data

        # Validate train data
        validate_survival_data(y_train_durations, y_train_events)

        # --- Feature Preprocessing --- 
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
            print(f"[INFO] Fold {current_fold + 1}: Removing {len(zero_var_indices)} zero-variance features.")
            non_zero_var_indices = np.where(variances != 0)[0]
            x_train = x_train[:, :, non_zero_var_indices]
            if has_validation_data: x_val = x_val[:, :, non_zero_var_indices]
            x_test = x_test[:, :, non_zero_var_indices]
            n_train_f = x_train.shape[2]
        else:
             print(f"[INFO] Fold {current_fold + 1}: No zero-variance features found.")

        # Upsampling (optional, on training data)
        if args.upsampling:
            print(f"[INFO] Fold {current_fold + 1}: Upsampling training data...")
            x_train, y_train_durations, y_train_events = upsample_training_data(x_train, y_train_durations, y_train_events)
            n_train_p, n_train_s, n_train_f = x_train.shape # Update shape after upsampling

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
        
        print(f"[INFO] Fold {current_fold + 1}: Final feature shapes: Train {x_train_final.shape}, Val {x_val_final.shape}, Test {x_test_final.shape}")

        # --- Survival Model Training --- (Corrected Indentation Starts Here)
        in_features = x_train_final.shape[1]
        out_features = 1 # Cox model has 1 output (log hazard ratio)
    
        # Build the network based on hyperparameters 
        if hyperparams['coxph_net'] == 'mlp':
            net = CustomMLP(in_features, out_features, dropout=hyperparams['dropout'])
        elif hyperparams['coxph_net'] == 'linear':
            from torch import nn
            net = nn.Linear(in_features, out_features, bias=False) # Usually no bias in Cox
        else:
            raise ValueError("Unknown coxph_net option. Choose 'mlp' or 'linear'.")
        
        if args.center_risk:
            net = CenteredModel(net) # Wrap if centering is enabled
        
        # Instantiate the CoxPH model with L1/L2 regularization 
        # Use Adam optimizer by default from pycox
        model = CoxPHWithL1(net, tt.optim.Adam, alpha=hyperparams['alpha'], gamma=hyperparams['gamma'])
        model.optimizer.set_lr(hyperparams['learning_rate'])
        # Note: pycox Adam usually doesn't use weight_decay directly
            
        # Prepare validation data tuple for pycox fit method 
        val_data_pycox = None
        if has_validation_data:
            val_data_pycox = (x_val_final, (y_val_durations, y_val_events))
            if len(y_val_durations) < args.batch_size:
                print(f"[WARN] Fold {current_fold + 1}: Validation set size ({len(y_val_durations)}) is smaller than batch size ({args.batch_size}). Validation loss might be noisy.")

        # Define callbacks for pycox fit method 
        callbacks = []
        if use_early_stopping_fold:
            callbacks.append(tt.callbacks.EarlyStopping(patience=args.early_stopping_patience))
        
        print(f"[INFO] Fold {current_fold + 1}: Training the CoxPH model...")
        # Fit the model 
        log = model.fit(
            x_train_final,
            (y_train_durations, y_train_events),
            batch_size=args.batch_size,
            epochs=args.epochs,
            callbacks=callbacks,
            verbose=True, # Prints epoch loss
            val_data=val_data_pycox,
            val_batch_size=args.batch_size # Use same batch size for validation
        )

        # --- Post-Training: Load Best Model (if early stopping used) --- 
        if use_early_stopping_fold and hasattr(model, 'load_model_weights'):
            try:
                print(f"[INFO] Fold {current_fold + 1}: Loading best model weights based on validation loss.")
                model.load_model_weights() # pycox EarlyStopping saves best weights internally
            except Exception as e:
                print(f"[WARN] Fold {current_fold + 1}: Could not load best model weights: {e}. Using last state.")
        else:
             print(f"[INFO] Fold {current_fold + 1}: Using model state from the last epoch.")

        # --- Final Evaluation on Test Set --- 
        # Compute baseline hazards AFTER loading best model (if applicable)
        try:
            model.compute_baseline_hazards() # Needed for survival predictions
        except Exception as e:
            print(f"[WARN] Fold {current_fold + 1}: Error computing baseline hazards: {e}. Survival predictions might fail.")

        # Predict risk scores (negative log partial hazard)
        # Ensure model is in eval mode (usually handled by pycox predict)
        test_risk_scores = -model.predict(x_test_final).reshape(-1) # Higher score = higher risk
        y_test_true_durations = y_test_durations
        y_test_true_events = y_test_events
        
        # Calculate test C-index for this fold ONLY IF NOT in LOOCV mode
        if not args.leave_one_out:
            try:
                fold_cindex = concordance_index(y_test_true_durations, test_risk_scores, event_observed=y_test_true_events)
                fold_test_cindices.append(fold_cindex)
                print(f"[Fold {current_fold + 1}] Final Test Concordance Index: {fold_cindex:.4f}")
            except ZeroDivisionError:
                 print(f"[WARN] Fold {current_fold + 1}: Could not calculate Concordance Index (likely too few comparable pairs in test set).")
                 fold_test_cindices.append(np.nan) # Record NaN if calculation fails
        else:
            # In LOOCV mode, we don't calculate C-index per fold
            print(f"[Fold {current_fold + 1}] LOOCV Fold Complete. Storing prediction.")
            # We still need to append something or handle the list later
            # Let's append NaN so the list structure is maintained, but filter later
            fold_test_cindices.append(np.nan) 

        # --- Store Fold Results --- 
        for i in range(len(test_patient_info)):
            patient_meta = test_patient_info[i]
            fold_results = {
                "patient_id": patient_meta['patient_id'],
                "fold": current_fold + 1,
                "predicted_risk_score": test_risk_scores[i],
                "duration": y_test_true_durations[i],
                "event_indicator": int(y_test_true_events[i]),
                "dataset_type": patient_meta['dataset_type'] # TCGA or NYU
            }
            all_fold_results.append(fold_results)

        # --- Move to Next Fold --- 
        if not data_module.next_fold():
            print("\n===== Finished all folds =====")
            break

    # --- Aggregation and Final Reporting --- 
    if not all_fold_results:
         print("[ERROR] No results collected. Exiting.")
         return
         
    final_predictions_df = pd.DataFrame(all_fold_results)
    
    total_patients_expected = len(data_module.all_patients)
    if len(final_predictions_df) != total_patients_expected:
         print(f"[WARN] Final predictions DF has {len(final_predictions_df)} rows, expected {total_patients_expected}. Check patient processing.")
    else:
         print(f"Final predictions DF contains results for all {len(final_predictions_df)} patients.")
         
    # --- Save Final Predictions --- 
    final_csv_path = os.path.join(args.output_dir, "final_cv_predictions_survival.csv")
    final_predictions_df.sort_values(by=["dataset_type", "patient_id"], inplace=True)
    final_predictions_df.to_csv(final_csv_path, index=False)
    print(f"Final aggregated survival predictions CSV saved to {final_csv_path}")

    # --- Calculate and Report Metrics (C-Index) --- 
    y_durations_all = final_predictions_df["duration"].values
    y_events_all = final_predictions_df["event_indicator"].values
    y_pred_scores_all = final_predictions_df["predicted_risk_score"].values

    def calculate_cindex(durations, scores, events, description):
        print(f"\n--- {description} Metrics --- ")
        if len(durations) == 0:
             print("No samples found for this subset.")
             return np.nan
        try:
            c_index = concordance_index(durations, scores, event_observed=events)
            print(f"Concordance Index: {c_index:.4f}")
            print(f"Number of samples: {len(durations)}")
            print(f"Number of events:  {int(sum(events))}")
            return c_index
        except Exception as e:
            print(f"Error calculating C-index for {description}: {e}")
            return np.nan

    # Overall C-Index
    overall_cindex = calculate_cindex(y_durations_all, y_pred_scores_all, y_events_all, "Overall Test Set (Aggregated)")

    # TCGA C-Index
    tcga_df = final_predictions_df[final_predictions_df["dataset_type"] == "TCGA"]
    tcga_cindex = calculate_cindex(tcga_df["duration"].values, 
                                   tcga_df["predicted_risk_score"].values, 
                                   tcga_df["event_indicator"].values, 
                                   "TCGA Test Set (Aggregated)")

    # NYU C-Index
    nyu_df = final_predictions_df[final_predictions_df["dataset_type"] == "NYU"]
    nyu_cindex = calculate_cindex(nyu_df["duration"].values, 
                                  nyu_df["predicted_risk_score"].values, 
                                  nyu_df["event_indicator"].values, 
                                  "NYU Test Set (Aggregated)")

    # --- Per-Fold C-Index Summary (Only if not LOOCV) --- 
    valid_fold_cindices = [c for c in fold_test_cindices if not np.isnan(c)]
    if not args.leave_one_out and valid_fold_cindices:
        mean_cindex = np.mean(valid_fold_cindices)
        std_cindex = np.std(valid_fold_cindices)
        min_cindex = np.min(valid_fold_cindices)
        max_cindex = np.max(valid_fold_cindices)
        print("\n--- Cross Validation C-Index Statistics (Per-Fold Test Set C-Indices) ---")
        print(f"Mean: {mean_cindex:.4f}")
        print(f"Standard Deviation: {std_cindex:.4f}")
        print(f"Minimum: {min_cindex:.4f}")
        print(f"Maximum: {max_cindex:.4f}")
        print(f"Number of valid folds included: {len(valid_fold_cindices)}/{len(fold_test_cindices)}")
    elif not args.leave_one_out:
         print("\n--- No valid per-fold C-indices recorded ---")
    # No per-fold summary needed for LOOCV
         
    # --- Optional: Plotting on Aggregated Results ---
    # Add calls to plotting functions here if desired, passing the aggregated df
    # e.g., plot_cv_metrics(y_pred_scores_all, y_durations_all, y_events_all, args.output_dir)
    # Or plot separately for TCGA/NYU using tcga_df, nyu_df
    print("\nPlotting is currently disabled in this refactored script. Re-enable if needed.")

def main(args):
    if args.cross_validation:
        cross_validation_mode(args)
    else:
        # Standard train/val/test split mode (using combined data)
        print("Running in standard train/val/test mode (not cross-validation)")
        print("[WARN] Standard train/val/test mode not fully implemented yet for survival.")
        # Needs implementation: Setup DataModule with cross_validation=False, 
        # get loaders, run single training/evaluation pass similar to inside CV loop.

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CoxPH survival model with DINOv2 features using combined datasets and cross-validation")
    
    # Updated arguments for combined dataset handling
    parser.add_argument("--tcga_dicom_root", type=str, default="/gpfs/data/mankowskilab/HCC/data/TCGA/manifest-4lZjKqlp5793425118292424834/TCGA-LIHC", help="Path to the TCGA DICOM root directory.")
    parser.add_argument("--nyu_dicom_root", type=str, default="/gpfs/data/mankowskilab/HCC_Recurrence/dicom", help="Path to the NYU DICOM root directory.")
    parser.add_argument("--tcga_csv_file", type=str, default="/gpfs/data/shenlab/wz1492/HCC/spreadsheets/tcga.csv", help="Path to the TCGA CSV metadata file.")
    parser.add_argument("--nyu_csv_file", type=str, default="/gpfs/data/shenlab/wz1492/HCC/spreadsheets/nyu_recurrence.csv", help="Path to the NYU CSV metadata file.") # Adjust default NYU path
    
    parser.add_argument('--preprocessed_root', type=str, default='/gpfs/data/mankowskilab/HCC_Recurrence/preprocessed/', help='Base directory to store/load preprocessed tensors (will have tcga/ and nyu/ subfolders). Set to None or empty string to disable.')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--num_slices', type=int, default=32, help='Number of slices per patient sample')
    parser.add_argument('--num_workers', type=int, default=4, help='DataLoader workers')
    parser.add_argument('--epochs', type=int, default=100, help='Max epochs per fold')
    parser.add_argument('--output_dir', type=str, default='checkpoints_survival_combined_cv', help='Base output directory')
    parser.add_argument('--learning_rate', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--gradient_clip', type=float, default=1.0, help='Gradient clipping (NOTE: Manual loop needed for this with pycox fit)')
    parser.add_argument('--center_risk', action='store_true', help='Center risk scores (for CenteredModel wrapper)')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate for MLP')
    parser.add_argument('--num_samples_per_patient', type=int, default=1, help='Number of slice samples per patient series')
    parser.add_argument('--coxph_net', type=str, default='mlp', choices=['mlp', 'linear'], help='Network type for CoxPH')
    parser.add_argument('--dinov2_weights', type=str, required=True, help="Path to DINOv2 weights (.pth or .pt).")
    parser.add_argument('--alpha', type=float, default=0.5, help="L1/L2 regularization weight (alpha for CoxPHWithL1)")
    parser.add_argument('--gamma', type=float, default=0.5, help="L1 vs L2 balance (gamma for CoxPHWithL1, 0=L2, 1=L1)")
    parser.add_argument('--upsampling', action='store_true', help="Upsample minority event class in training data per fold")
    parser.add_argument('--early_stopping', action='store_true', help="Use early stopping based on validation loss")
    parser.add_argument('--early_stopping_patience', type=int, default=10, help="Patience epochs for early stopping.")
    parser.add_argument('--cross_validation', action='store_true', default=True, help="Enable combined cross validation mode")
    parser.add_argument('--cv_folds', type=int, default=10, help="Number of CV folds")
    parser.add_argument('--cv_mode', type=str, default='combined', choices=['combined', 'tcga', 'nyu'], 
                        help="Dataset mode for cross-validation: 'combined', 'tcga' (uses tcga_csv_file), 'nyu' (uses nyu_csv_file)")
    parser.add_argument('--leave_one_out', action='store_true', help="Use LOOCV (overrides cv_folds)")
    
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
    # Include cv_mode in the run name
    run_name = f"run_{timestamp}_mode{args.cv_mode}_lr{args.learning_rate}_bs{args.batch_size}_net{args.coxph_net}"
    if args.upsampling: run_name += "_upsampled"
    if args.leave_one_out: run_name += "_loocv"
    else: run_name += f"_{args.cv_folds}fold"
        
    args.output_dir = os.path.join(args.output_dir, run_name)
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output directory: {args.output_dir}")
    
    main(args)
