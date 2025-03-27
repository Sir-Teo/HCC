import argparse
import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
# Import necessary metrics
from sklearn.metrics import (
    roc_auc_score, roc_curve, 
    precision_recall_curve, average_precision_score, # For PR Curve and PR-AUC
    precision_score, recall_score, f1_score # Precision, Recall, F1
)
from sklearn.model_selection import StratifiedKFold, LeaveOneOut
import datetime
from tqdm import tqdm
import warnings
import copy # For saving best model state in early stopping

# Custom module imports
# Ensure these paths are correct relative to your script location
from data.dataset import HCCDataModule
from models.dino import load_dinov2_model

# Suppress UndefinedMetricWarning (often occurs with F1/Precision/Recall in single-class/no-prediction scenarios)
warnings.filterwarnings("ignore", category=RuntimeWarning) 
warnings.filterwarnings("ignore", category=UserWarning, module='sklearn')


def extract_features(data_loader, model, device):
    """ Extracts features using the DINOv2 model. """
    model.eval()
    features = []
    events = []
    patient_ids = [] 

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Extracting Features"):
            if len(batch) == 3:
                images, e, pids = batch
                patient_ids.extend(pids)
            else:
                images, e = batch
            
            images = images.to(device)
            b, s, n, c, h, w = images.shape # Use descriptive names
            images = images.view(b * s * n, c, h, w)
            
            feats = model.forward_features(images) 
            feat_dim = feats.shape[-1]
            
            # Reshape assuming output is (B*S*N, Dim) -> (B, S, N, Dim)
            feats = feats.view(b, s, n, feat_dim)

            features.append(feats.cpu().numpy())
            events.append(e.cpu().numpy())

    features = np.concatenate(features, axis=0)
    events = np.concatenate(events, axis=0)
    
    if patient_ids:
       return features, events, patient_ids
    else:
       return features, events

#############################################
# Plotting functions (ROC, PR, Loss)       #
#############################################

def plot_roc_curve(y_true, y_scores, output_dir, fold_id=""):
    """Plots and saves the ROC curve."""
    if len(np.unique(y_true)) < 2:
        print(f"Skipping ROC plot for {fold_id}: Only one class present.")
        return
    try:
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = roc_auc_score(y_true, y_scores)
    except ValueError as e:
        print(f"Could not generate ROC curve for {fold_id}: {e}")
        return

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], linestyle='--', color='grey', label='Chance')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve {fold_id}')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, f'roc_curve_{fold_id}.png'))
    plt.close()

def plot_pr_curve(y_true, y_scores, output_dir, fold_id=""):
    """Plots and saves the Precision-Recall curve."""
    if len(np.unique(y_true)) < 2:
        print(f"Skipping PR plot for {fold_id}: Only one class present.")
        return
    try:
        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        pr_auc = average_precision_score(y_true, y_scores)
    except ValueError as e:
        print(f"Could not generate PR curve for {fold_id}: {e}")
        return
        
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label=f'PR curve (AP = {pr_auc:.4f})')
    # Calculate baseline (proportion of positives)
    baseline = np.sum(y_true) / len(y_true) if len(y_true) > 0 else 0
    plt.axhline(baseline, linestyle='--', color='grey', label=f'Baseline ({baseline:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve {fold_id}')
    plt.legend(loc="lower left") # Often better for PR curves
    plt.grid(True)
    # Consider setting limits if needed, e.g., plt.ylim([0.0, 1.05])
    plt.savefig(os.path.join(output_dir, f'pr_curve_{fold_id}.png'))
    plt.close()


def plot_loss_curves(train_losses, val_losses, output_dir, fold_id=""):
    """Plots and saves the training and validation loss curves."""
    plt.figure(figsize=(8, 6))
    plt.plot(train_losses, label="Training Loss")
    if val_losses: # Only plot validation loss if available (for early stopping)
        plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Loss Curve {fold_id}")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, f"loss_curve_{fold_id}.png"))
    plt.close()

#############################################
#  Upsampling function for classification   #
#############################################

def upsample_training_data_classifier(x, y):
    """ Upsamples minority class in feature arrays (x, y). """
    unique_classes, counts = np.unique(y, return_counts=True)
    if len(unique_classes) < 2: return x, y 
    minority_class = unique_classes[np.argmin(counts)]
    majority_class = unique_classes[np.argmax(counts)]
    idx_minority = np.where(y == minority_class)[0]
    idx_majority = np.where(y == majority_class)[0]
    n_majority, n_minority = len(idx_majority), len(idx_minority)
    if n_minority == 0 or n_majority == 0 or n_minority == n_majority: return x, y
    sampled_indices = np.random.choice(idx_minority, size=n_majority - n_minority, replace=True)
    x_upsampled = np.concatenate([x, x[sampled_indices]], axis=0)
    y_upsampled = np.concatenate([y, y[sampled_indices]], axis=0)
    shuffle_perm = np.random.permutation(len(y_upsampled))
    x_upsampled, y_upsampled = x_upsampled[shuffle_perm], y_upsampled[shuffle_perm]
    return x_upsampled, y_upsampled

#############################################
#       Binary Classifier Model             #
#############################################

class BinaryClassifier(torch.nn.Module):
    """Simple Linear Binary Classifier."""
    def __init__(self, in_features):
        super(BinaryClassifier, self).__init__()
        self.linear = torch.nn.Linear(in_features, 1)
    def forward(self, x):
        return self.linear(x)

#############################################
#       Metrics Calculation Helper          #
#############################################

def calculate_metrics(y_true, y_prob, threshold=0.5, prefix=""):
    """Calculates ROC-AUC, PR-AUC, Precision, Recall, F1-Score."""
    metrics = {
        f'{prefix}ROC_AUC': np.nan, 
        f'{prefix}PR_AUC': np.nan, 
        f'{prefix}Precision': np.nan, 
        f'{prefix}Recall': np.nan, 
        f'{prefix}F1': np.nan
    }
    
    if y_true is None or y_prob is None or len(y_true) == 0 or len(y_prob) == 0:
        print(f"Warning: Empty data provided for metric calculation ({prefix}).")
        return metrics
        
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)

    # --- AUC Metrics ---
    # Check if both classes are present for AUC calculations
    if len(np.unique(y_true)) >= 2:
        try:
            metrics[f'{prefix}ROC_AUC'] = roc_auc_score(y_true, y_prob)
        except ValueError as e: print(f"Warning: ROC AUC Error ({prefix}): {e}")
        try:
            # PR-AUC (Average Precision)
            metrics[f'{prefix}PR_AUC'] = average_precision_score(y_true, y_prob) 
        except ValueError as e: print(f"Warning: PR AUC Error ({prefix}): {e}")
    else:
         # print(f"Warning: Only one class present ({prefix}). ROC AUC / PR AUC are not defined.")
         pass # Avoid repetitive warnings

    # --- Threshold-based Metrics (Precision, Recall, F1) ---
    try:
        y_pred = (y_prob >= threshold).astype(int)
        # Use zero_division=0 to return 0 instead of raising error
        metrics[f'{prefix}Precision'] = precision_score(y_true, y_pred, zero_division=0)
        metrics[f'{prefix}Recall'] = recall_score(y_true, y_pred, zero_division=0)
        metrics[f'{prefix}F1'] = f1_score(y_true, y_pred, zero_division=0) 
    except Exception as e:
        print(f"Warning: Could not compute P/R/F1 ({prefix}). Reason: {e}")
        
    return metrics

#############################################
#    Training and Evaluation Function       #
#############################################

def train_and_evaluate(args, train_csv, val_csv, hyperparams, fold_id="", 
                       final_eval=True, return_predictions=False):
    """
    Trains a linear binary classifier with early stopping, calculates requested metrics.
    Returns metrics dict OR (probabilities, true_labels, sample_indices).
    """
    torch.manual_seed(42)
    np.random.seed(42)
    
    # --- Data Setup ---
    data_module = HCCDataModule(
        train_csv_file=train_csv, train_dicom_root=args.train_dicom_root,
        test_csv_file=val_csv, test_dicom_root=args.test_dicom_root,
        model_type="linear", batch_size=args.batch_size, num_slices=args.num_slices,
        num_workers=args.num_workers, preprocessed_root=args.preprocessed_root,
        num_samples=args.num_samples_per_patient
    )
    data_module.setup()
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()  # For early stopping loss calc

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # --- Feature Extraction ---
    dino_model = load_dinov2_model(args.dinov2_weights).to(device).eval()
    print(f"[Fold {fold_id}] Extracting training features...")
    extracted_train = extract_features(train_loader, dino_model, device)
    x_train, y_train = extracted_train[0], extracted_train[1]
    
    print(f"[Fold {fold_id}] Extracting validation features...")
    extracted_val = extract_features(val_loader, dino_model, device)
    # Check if sample indices were returned from the loader
    if len(extracted_val) == 3:
        x_val, y_val, val_indices = extracted_val
    else:
        x_val, y_val = extracted_val
        val_indices = None

    # --- Fallback: if no indices from the loader and val_csv is a DataFrame, use its index ---
    if val_indices is None and isinstance(val_csv, pd.DataFrame):
        num_val_samples = x_val.shape[0]
        val_indices = np.array(val_csv.index)[:num_val_samples]
    
    # Average features & remove zero variance
    x_train = x_train.mean(axis=(1, 2)) 
    x_val = x_val.mean(axis=(1, 2)) 
    variances = np.var(x_train, axis=0)
    non_zero_var_mask = variances != 0
    if np.any(~non_zero_var_mask):
        num_zero = np.sum(~non_zero_var_mask)
        print(f"[Fold {fold_id}] Warning: Removing {num_zero} features with zero variance...")
        x_train = x_train[:, non_zero_var_mask]
        x_val = x_val[:, non_zero_var_mask]
        if x_train.shape[1] == 0: 
            raise ValueError(f"[Fold {fold_id}] All features removed.")
    
    # --- Preprocessing (Upsampling, Scaling) ---
    if args.upsampling:
        print(f"[Fold {fold_id}] Upsampling training features...")
        x_train, y_train = upsample_training_data_classifier(x_train, y_train)
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train).astype("float32")
    x_val = scaler.transform(x_val).astype("float32")
    y_train = np.array(y_train).astype("float32")
    y_val = np.array(y_val).astype("float32")
    
    # --- Model & Training Setup ---
    in_features = x_train.shape[1]
    model = BinaryClassifier(in_features).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=hyperparams['learning_rate'])
    criterion = torch.nn.BCEWithLogitsLoss() 
    
    train_dataset = torch.utils.data.TensorDataset(torch.tensor(x_train), torch.tensor(y_train))
    val_dataset = torch.utils.data.TensorDataset(torch.tensor(x_val), torch.tensor(y_val))
    train_loader_cls = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader_cls = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    # --- Early Stopping Initialization ---
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_state = None
    train_losses = []
    val_losses = []
    
    print(f"[Fold {fold_id}] Starting training (max {args.epochs} epochs, Early Stopping patience={args.early_stopping_patience})...")
    # --- Training Loop ---
    for epoch in range(args.epochs):
        model.train()
        epoch_train_losses = []
        for batch_x, batch_y in train_loader_cls:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device).unsqueeze(1)
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            if args.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
            optimizer.step()
            epoch_train_losses.append(loss.item())
        avg_train_loss = np.mean(epoch_train_losses)
        train_losses.append(avg_train_loss)

        model.eval()
        epoch_val_losses = []
        with torch.no_grad():
            for batch_x_val, batch_y_val in val_loader_cls:
                batch_x_val, batch_y_val = batch_x_val.to(device), batch_y_val.to(device).unsqueeze(1)
                val_logits = model(batch_x_val)
                val_loss = criterion(val_logits, batch_y_val)
                epoch_val_losses.append(val_loss.item())
        avg_val_loss = np.mean(epoch_val_losses)
        val_losses.append(avg_val_loss)

        log_msg = f"[Fold {fold_id}] Epoch {epoch+1}/{args.epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}"
        if avg_val_loss < best_val_loss - args.early_stopping_delta:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            best_model_state = copy.deepcopy(model.state_dict())
            log_msg += " (New best)"
        else:
            epochs_no_improve += 1
            log_msg += f" (No improve {epochs_no_improve}/{args.early_stopping_patience})"
        if (epoch + 1) % 10 == 0 or epoch == args.epochs - 1 or epochs_no_improve == 0:
             print(log_msg)
        if epochs_no_improve >= args.early_stopping_patience:
            print(f"[Fold {fold_id}] Early stopping triggered at epoch {epoch+1}. Best Val Loss: {best_val_loss:.4f}")
            break
            
    if best_model_state:
        print(f"[Fold {fold_id}] Loading best model state from epoch {epoch + 1 - epochs_no_improve}.")
        model.load_state_dict(best_model_state)
    else:
        print(f"[Fold {fold_id}] Warning: No best model state saved. Using final model state.")

    print(f"[Fold {fold_id}] Final evaluation on validation set...")
    model.eval()
    all_probs = []
    all_labels = []
    with torch.no_grad():
        for batch_x, batch_y in val_loader_cls:
            batch_x = batch_x.to(device)
            logits = model(batch_x)
            probs = torch.sigmoid(logits)
            all_probs.extend(probs.cpu().numpy().flatten().tolist())
            all_labels.extend(batch_y.numpy().flatten().tolist())

    fold_metrics = calculate_metrics(all_labels, all_probs, prefix="")
    print(f"[Fold {fold_id}] Final Validation Metrics: " + ", ".join([f"{k}: {v:.4f}" for k, v in fold_metrics.items()]))

    if final_eval:
        os.makedirs(args.output_dir, exist_ok=True)
        plot_roc_curve(all_labels, all_probs, args.output_dir, fold_id=fold_id)
        plot_pr_curve(all_labels, all_probs, args.output_dir, fold_id=fold_id)
        plot_loss_curves(train_losses, val_losses, args.output_dir, fold_id=fold_id)

    if return_predictions:
        # Now we are sure val_indices is not empty if val_csv was provided
        return np.array(all_probs), np.array(all_labels), np.array(val_indices)
    else:
        return fold_metrics



#############################################
#         Cross-Validation Mode             #
#############################################

def cross_validation_mode(args):
    # --- Data Preparation (Option B) ---
    df_train_full = pd.read_csv(args.train_csv_file)
    df_train_full["dicom_root"] = args.train_dicom_root
    n_train_original = len(df_train_full)
    original_test_indices_in_all = None 
    n_test_original = 0

    if args.test_csv_file:
        try:
            df_test_full = pd.read_csv(args.test_csv_file)
            df_test_full["dicom_root"] = args.test_dicom_root
            n_test_original = len(df_test_full)
            common_cols = list(set(df_train_full.columns) & set(df_test_full.columns))
            if 'event' not in common_cols: raise ValueError("'event' column missing.")
            df_all = pd.concat([df_train_full[common_cols].copy(), df_test_full[common_cols].copy()])
            original_test_indices_in_all = np.arange(n_train_original, n_train_original + n_test_original)
            print(f"[CV Mode Option B] Combined train ({n_train_original}) and test ({n_test_original}) datasets. Total: {len(df_all)}")
        except Exception as e:
            print(f"Warning: Error combining CSVs ({e}). CV on training data only.")
            df_all = df_train_full.copy()
            n_test_original = 0
    else:
        df_all = df_train_full.copy()
        print(f"No test CSV provided. CV on training data only (Size: {len(df_all)}).")
    
    # --- CV Setup ---
    all_predicted_probs, all_labels, all_fold_test_indices = [], [], []
    hyperparams = {"learning_rate": args.learning_rate}
    
    if args.leave_one_out:
        cv_method, n_splits, cv_type = LeaveOneOut(), len(df_all), "LOOCV"
        splits = list(cv_method.split(df_all)) 
    else:
        cv_method = StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=42)
        n_splits, cv_type = args.cv_folds, f"{args.cv_folds}-Fold_CV"
        try: 
            splits = list(cv_method.split(df_all, df_all["event"]))
        except Exception as e: 
            print(f"StratifiedKFold Error: {e}. Aborting."); 
            return

    print(f"\nStarting {cv_type} using dataset (size={len(df_all)})...")

    # --- CV Loop ---
    for fold, (train_idx, test_idx) in enumerate(splits):
        df_new_train = df_all.iloc[train_idx]
        df_new_test = df_all.iloc[test_idx]

        fold_id_str = f"{cv_type}_fold_{fold+1}_of_{n_splits}"
        print(f"\n--- {fold_id_str} ---")
        print(f"Train: {len(df_new_train)} (Pos: {df_new_train['event'].sum()}), Test: {len(df_new_test)} (Pos: {df_new_test['event'].sum()})")
        
        # Capture predictions, labels, and sample indices from the evaluated fold
        preds, labels, indices = train_and_evaluate(
            args, train_csv=df_new_train, val_csv=df_new_test,
            hyperparams=hyperparams, fold_id=fold_id_str,
            final_eval=False, return_predictions=True
        )
        all_predicted_probs.extend(preds.tolist())
        all_labels.extend(labels.tolist())
        all_fold_test_indices.extend(indices.tolist()) 

    # --- Aggregated Evaluation ---
    print("\n--- Cross-Validation Aggregated Results ---")
    if not all_labels: 
        print("Error: No results aggregated."); 
        return
        
    all_labels = np.array(all_labels)
    all_predicted_probs = np.array(all_predicted_probs)
    all_fold_test_indices = np.array(all_fold_test_indices)

    print("\nOverall Aggregated Metrics (on all test folds):")
    overall_metrics = calculate_metrics(all_labels, all_predicted_probs, prefix="Overall_")
    for name, value in overall_metrics.items(): 
        print(f"  {name}: {value:.4f}")
    
    # Plot aggregated curves
    os.makedirs(args.output_dir, exist_ok=True)
    plot_roc_curve(all_labels, all_predicted_probs, args.output_dir, fold_id="aggregated_CV")
    plot_pr_curve(all_labels, all_predicted_probs, args.output_dir, fold_id="aggregated_CV")

    # --- Dataset-Specific Reporting ---
    if n_test_original > 0 and original_test_indices_in_all is not None:
        print("\nSubset Metrics (based on original dataset source):")
        mask_original_test = np.isin(all_fold_test_indices, original_test_indices_in_all)
        mask_original_train = ~mask_original_test

        print(f"\n  Metrics on subset from original TRAIN data ({np.sum(mask_original_train)} samples):")
        train_subset_metrics = calculate_metrics(all_labels[mask_original_train], all_predicted_probs[mask_original_train], prefix="TrainSubset_")
        for name, value in train_subset_metrics.items(): 
            print(f"    {name}: {value:.4f}")

        print(f"\n  Metrics on subset from original TEST data ({np.sum(mask_original_test)} samples):")
        test_subset_metrics = calculate_metrics(all_labels[mask_original_test], all_predicted_probs[mask_original_test], prefix="TestSubset_")
        for name, value in test_subset_metrics.items(): 
            print(f"    {name}: {value:.4f}")
    elif n_test_original == 0:
         print("\nNote: Only original training data used for CV.")

    print("--- End Cross-Validation ---")


#############################################
#                Main Function              #
#############################################

def main(args):
    # Add early stopping defaults if not provided
    if args.early_stopping_patience <= 0:
        print("Early stopping disabled (patience <= 0).")
        args.early_stopping_patience = args.epochs # Effectively disable it
        
    if args.cross_validation or args.leave_one_out:
        if args.leave_one_out: args.cross_validation = True; print("LOOCV enabled.")
        elif args.cv_folds < 2: print("Warning: cv_folds < 2, setting to 5."); args.cv_folds = 5
        cross_validation_mode(args)
    else:
        if not args.test_csv_file: raise ValueError("test_csv_file is required for non-CV runs.")
        print("Running standard train/test evaluation...")
        metrics = train_and_evaluate(
            args, args.train_csv_file, args.test_csv_file,
            hyperparams={"learning_rate": args.learning_rate},
            fold_id="TrainTestRun", final_eval=True, return_predictions=False
        )
        print("\n--- Final Evaluation Results on Test Set ---")
        for name, value in metrics.items(): print(f"  {name}: {value:.4f}")
        print("--- End Final Evaluation ---")

#############################################
#          Argument Parsing & Run           #
#############################################

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train/Evaluate linear binary classifier with DINOv2 features.")
    # Paths
    parser.add_argument("--train_dicom_root", type=str, required=True, help="Base DICOM directory for training.")
    parser.add_argument("--test_dicom_root", type=str, required=True, help="Base DICOM directory for testing.")
    parser.add_argument("--train_csv_file", type=str, required=True, help="Path to the training CSV.")
    parser.add_argument("--test_csv_file", type=str, default=None, help="Path to the test CSV.")
    parser.add_argument('--preprocessed_root', type=str, default=None, help='Optional: Preprocessed tensors directory.')
    parser.add_argument('--dinov2_weights', type=str, required=True, help="Path to DINOv2 weights.")
    parser.add_argument('--output_dir', type=str, default='checkpoints_classifier', help='Base directory for outputs.')

    # Data Loading & Feature Extraction
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size.')
    parser.add_argument('--num_slices', type=int, default=64, help='Slices per patient.')
    parser.add_argument('--num_workers', type=int, default=4, help='Dataloader workers.')
    parser.add_argument('--num_samples_per_patient', type=int, default=1, help='Samples per patient.')
    
    # Training Params
    parser.add_argument('--epochs', type=int, default=100, help='Max training epochs.') # Increased default for early stopping
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Classifier learning rate.')
    parser.add_argument('--gradient_clip', type=float, default=0.1, help='Gradient clipping threshold (<=0 disables).')
    parser.add_argument('--upsampling', action='store_true', help="Enable upsampling on training features.")
    
    # --- Early Stopping Arguments ---
    parser.add_argument('--early_stopping_patience', type=int, default=10, 
                        help='Epochs to wait for validation loss improvement before stopping. <=0 to disable.')
    parser.add_argument('--early_stopping_delta', type=float, default=1e-5, # Small delta to detect meaningful improvement
                        help='Minimum change in validation loss to qualify as improvement.')

    # Cross-Validation Params
    parser.add_argument('--cross_validation', action='store_true', help="Enable K-Fold CV (Option B).")
    parser.add_argument('--cv_folds', type=int, default=5, help="K-Fold CV folds.")
    parser.add_argument('--leave_one_out', action='store_true', help="Enable LOOCV.")
    
    args = parser.parse_args()
    
    # Setup output directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name_parts = [f"run_{timestamp}"]
    if args.cross_validation: run_name_parts.append(f"{args.cv_folds}fold" if not args.leave_one_out else "LOOCV")
    run_name_parts.append(f"lr{args.learning_rate}")
    if args.upsampling: run_name_parts.append("upsampled")
    if args.early_stopping_patience > 0: run_name_parts.append(f"ES{args.early_stopping_patience}")
    run_name = "_".join(run_name_parts)
    args.output_dir = os.path.join(args.output_dir, run_name)
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("="*50)
    print("Starting Run Configuration:")
    for k, v in vars(args).items(): print(f"  {k}: {v}")
    print(f"  Output Directory: {args.output_dir}")
    # Use current date/time based on system
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"  Current Time: {current_time}") # Add current time info
    print("="*50)
       
    main(args)

    print("\n" + "="*50)
    print(f"Run finished. Outputs saved to: {args.output_dir}")
    print("="*50)