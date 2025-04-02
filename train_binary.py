import argparse
import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, train_test_split
import datetime  # new import for timestamp

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

# Custom module imports
from data.dataset import HCCDataModule
from models.dino import load_dinov2_model
# The following model imports for survival models are no longer used:
# from models.mlp import CustomMLP, CenteredModel, CoxPHWithL1
from utils.helpers import extract_features, validate_survival_data, upsample_training_data
from callbacks.custom_callbacks import GradientClippingCallback, LossLogger, ParamCheckerCallback
from utils.plotting import (plot_cv_metrics,  # if you wish to adapt plotting for classification, modify accordingly.
                      plot_survival_functions, plot_brier_score,
                      plot_risk_score_distribution, plot_kaplan_meier, plot_calibration_plot,
                      plot_multi_calibration, plot_cumulative_hazard, plot_survival_probability_distribution)

sns.set(style="whitegrid")

def train_and_evaluate(args, train_csv, val_csv, hyperparams, fold_id="", final_eval=True, return_predictions=False):
    """
    Train the model on data from train_csv and evaluate on val_csv.
    In this version, if args.cross_validation is True, the evaluation is done on the test CSV
    (using the test_dataloader) instead of the validation split from the training CSV.
    """
    # Set seed for reproducibility
    torch.manual_seed(42)
    
    # Instantiate the data module.
    # Here we pass the CSVs (or DataFrames) directly to the data module.
    data_module = HCCDataModule(
        train_csv_file=train_csv,
        test_csv_file=val_csv,  # when in CV mode, this is the fold test DataFrame/CSV
        train_dicom_root=args.train_dicom_root,
        test_dicom_root=args.test_dicom_root,
        model_type="time_to_event",
        batch_size=args.batch_size,
        num_slices=args.num_slices,
        num_samples=args.num_samples_per_patient,
        num_workers=args.num_workers,
        preprocessed_root=args.preprocessed_root
    )
    data_module.setup()
    
    # Load the DINOv2 feature extractor
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dino_model = load_dinov2_model(args.dinov2_weights)
    dino_model = dino_model.to(device)
    dino_model.eval()
    dino_model.register_forward_hook(lambda m, i, o: None)
    
    # Use the training dataloader for feature extraction on training data.
    train_loader = data_module.train_dataloader()
    # For evaluation, if in CV mode, use the test dataloader (which is built from the fold test CSV)
    # otherwise, use the val_dataloader.
    if args.cross_validation:
        eval_loader = data_module.test_dataloader()
    else:
        eval_loader = data_module.val_dataloader()
    
    # Feature extraction for training and evaluation data
    x_train, y_train_durations, y_train_events = extract_features(train_loader, dino_model, device)
    x_eval, y_eval_durations, y_eval_events = extract_features(eval_loader, dino_model, device)
    
    # Average across samples (for each patient) if needed
    x_train = x_train.mean(axis=1)
    x_eval = x_eval.mean(axis=1)
    
    # Remove zero-variance features
    variances = np.var(x_train, axis=0)
    if np.any(variances == 0):
        zero_var_features = np.where(variances == 0)[0]
        print(f"[Fold {fold_id}] Warning: Features with zero variance: {zero_var_features}")
        x_train = x_train[:, variances != 0]
        x_eval = x_eval[:, variances != 0]
    
    if args.upsampling:
        print(f"[Fold {fold_id}] Performing upsampling on training data...")
        x_train, y_train_durations, y_train_events = upsample_training_data(
            x_train, y_train_durations, y_train_events
        )
    
    # Standardize features (fit only on train)
    x_mapper = StandardScaler()
    n_train, n_slices, feat_dim = x_train.shape
    x_train_reshaped = x_train.reshape(-1, feat_dim)
    x_train_scaled = x_mapper.fit_transform(x_train_reshaped).astype('float32')
    x_train_scaled = x_train_scaled.reshape(n_train, n_slices, feat_dim)
    
    n_eval, n_slices_eval, feat_dim_eval = x_eval.shape
    x_eval_reshaped = x_eval.reshape(-1, feat_dim_eval)
    x_eval_scaled = x_mapper.transform(x_eval_reshaped).astype('float32')
    x_eval_scaled = x_eval_scaled.reshape(n_eval, n_slices_eval, feat_dim_eval)
    
    # Collapse the slice dimension by averaging
    x_train_std = x_train_scaled.mean(axis=1)
    x_eval_std = x_eval_scaled.mean(axis=1)
    
    # Validate survival data (if applicable)
    validate_survival_data(y_train_durations, y_train_events)
    
    in_features = x_train_std.shape[1]
    
    # For classification, replace the survival model with a single linear layer.
    from torch import nn
    net = nn.Linear(in_features, 1)
    net.to(device)
    
    # Define optimizer and loss function
    optimizer = torch.optim.Adam(net.parameters(), lr=hyperparams['learning_rate'], weight_decay=1e-4)
    criterion = torch.nn.BCEWithLogitsLoss()
    
    # Prepare training and evaluation tensors
    train_tensor_x = torch.tensor(x_train_std, dtype=torch.float32).to(device)
    train_tensor_y = torch.tensor(y_train_events.astype(np.float32), dtype=torch.float32).to(device)
    eval_tensor_x = torch.tensor(x_eval_std, dtype=torch.float32).to(device)
    eval_tensor_y = torch.tensor(y_eval_events.astype(np.float32), dtype=torch.float32).to(device)
    
    train_dataset = torch.utils.data.TensorDataset(train_tensor_x, train_tensor_y)
    eval_dataset = torch.utils.data.TensorDataset(eval_tensor_x, eval_tensor_y)
    
    train_loader_cls = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    eval_loader_cls = torch.utils.data.DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Training loop with early stopping if enabled
    best_val_loss = float('inf')
    patience = 10
    epochs_no_improve = 0
    for epoch in range(args.epochs):
        net.train()
        train_losses = []
        for batch_x, batch_y in train_loader_cls:
            optimizer.zero_grad()
            outputs = net(batch_x).squeeze()  # shape: (batch_size)
            loss = criterion(outputs, batch_y)
            loss.backward()
            if args.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(net.parameters(), args.gradient_clip)
            optimizer.step()
            train_losses.append(loss.item())
        avg_train_loss = np.mean(train_losses)
        
        # Validation/Evaluation
        net.eval()
        eval_losses = []
        all_eval_preds = []
        with torch.no_grad():
            for batch_x, batch_y in eval_loader_cls:
                outputs = net(batch_x).squeeze()
                loss = criterion(outputs, batch_y)
                eval_losses.append(loss.item())
                all_eval_preds.append(outputs)
        avg_eval_loss = np.mean(eval_losses)
        all_eval_preds = torch.cat(all_eval_preds)
        eval_probs = torch.sigmoid(all_eval_preds)
        eval_pred_labels = (eval_probs >= 0.5).float()
        eval_accuracy = (eval_pred_labels.cpu().numpy() == eval_tensor_y.cpu().numpy()).mean()
        
        print(f"[Fold {fold_id}] Epoch {epoch+1}/{args.epochs} - Train Loss: {avg_train_loss:.4f} - Eval Loss: {avg_eval_loss:.4f} - Eval Accuracy: {eval_accuracy:.4f}")
        
        if args.early_stopping:
            if avg_eval_loss < best_val_loss:
                best_val_loss = avg_eval_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f"[Fold {fold_id}] Early stopping triggered at epoch {epoch+1}")
                    break
                    
    # After training, evaluate on the full evaluation set
    net.eval()
    with torch.no_grad():
        outputs = net(eval_tensor_x).squeeze()
        eval_probs = torch.sigmoid(outputs).cpu().numpy()
        eval_pred_labels = (eval_probs >= 0.5).astype(int)
        eval_accuracy = (eval_pred_labels == y_eval_events).mean()
    
    if return_predictions:
        # Return predicted probabilities along with durations and events.
        return eval_probs, y_eval_durations, y_eval_events
    else:
        print(f"[Fold {fold_id}] Accuracy: {eval_accuracy:.4f}")
        if final_eval and not return_predictions:
            os.makedirs(args.output_dir, exist_ok=True)
            with open(os.path.join(args.output_dir, f"fold_{fold_id}_accuracy.txt"), "w") as f:
                f.write(f"Validation Accuracy: {eval_accuracy:.4f}\n")
        return eval_accuracy



def upsample_df(df, target_column='event'):
    df_majority = df[df[target_column] == 0]
    df_minority = df[df[target_column] == 1]
    if len(df_minority) == 0 or len(df_majority) == 0:
        return df
    df_minority_upsampled = df_minority.sample(len(df_majority), replace=True, random_state=1)
    return pd.concat([df_majority, df_minority_upsampled]).reset_index(drop=True)

def cross_validation_mode(args):
    """
    For each fold, we create training and test DataFrames,
    add a source column to the test set, and pass them directly to the data module.
    We also record the fold id for each prediction.
    """
    # Load the full training CSV and add a source column for later tracking.
    df_train_full = pd.read_csv(args.train_csv_file)
    df_train_full['dicom_root'] = args.train_dicom_root
    df_train_full['source'] = 'train'

    df_test_full = None
    if args.test_csv_file:
        df_test_full = pd.read_csv(args.test_csv_file)
        df_test_full['dicom_root'] = args.test_dicom_root
        df_test_full['source'] = 'test'
        df_all = pd.concat([df_train_full, df_test_full], ignore_index=True)
        print(f"[CV Mode] Combined train ({len(df_train_full)}) and test ({len(df_test_full)}) datasets. Total: {len(df_all)}")
    else:
        df_all = df_train_full.copy().reset_index(drop=True)
        print(f"No test CSV provided. CV on training data only (Size: {len(df_all)}).")

    # Hyperparameters for training
    hyperparams = {
        'learning_rate': args.learning_rate,
        'dropout': args.dropout,
        'alpha': args.alpha,
        'gamma': args.gamma,
        'coxph_net': args.coxph_net
    }

    # Prepare arrays to collect predictions and sources.
    all_predicted_risk_scores = []
    all_event_times = []
    all_event_indicators = []
    all_sources = []  # will store 'train' or 'test' from the fold test CSV
    all_fold_ids = []  # record the fold id for each prediction
    fold_accuracies = []

    from sklearn.model_selection import KFold
    kf = KFold(n_splits=args.cv_folds, shuffle=True, random_state=1)
    for fold, (train_idx, test_idx) in enumerate(kf.split(df_all)):
        df_fold_train = df_all.iloc[train_idx].reset_index(drop=True)
        df_fold_test = df_all.iloc[test_idx].reset_index(drop=True)
        print(f"[CV Fold {fold}] Train: {len(df_fold_train)} patients, Positive events: {df_fold_train['event'].sum()}")
        print(f"[CV Fold {fold}] Test: {len(df_fold_test)} patients, Positive events: {df_fold_test['event'].sum()}")

        if args.upsampling:
            print(f"[CV Fold {fold}] Performing upsampling on training data...")
            df_fold_train = upsample_df(df_fold_train, target_column='event')

        # Ensure the fold test set has a source column.
        if 'source' not in df_fold_test.columns:
            df_fold_test['source'] = 'test'

        # Instead of writing temporary CSV files, we pass the DataFrames directly.
        # (Alternatively, you can save them as CSVs and pass the file paths.)
        fold_train_csv = df_fold_train  # DataFrame
        fold_test_csv = df_fold_test    # DataFrame

        # Call train_and_evaluate (it will use the test_dataloader because args.cross_validation is True)
        risk_scores, durations, events = train_and_evaluate(
            args,
            train_csv=fold_train_csv,
            val_csv=fold_test_csv,
            hyperparams=hyperparams,
            fold_id=f"fold_{fold}",
            final_eval=False,
            return_predictions=True
        )
        fold_accuracy = np.mean((risk_scores >= 0.5).astype(int) == events)
        print(f"[CV Fold {fold}] Accuracy: {fold_accuracy:.4f}")
        fold_accuracies.append(fold_accuracy)

        # Extend overall arrays.
        all_predicted_risk_scores.extend(risk_scores.tolist())
        all_event_times.extend(durations.tolist())
        all_event_indicators.extend(events.tolist())
        # Use the source column from df_fold_test.
        fold_sources = df_fold_test['source'].tolist()[:len(risk_scores)]
        all_sources.extend(fold_sources)
        all_fold_ids.extend([fold] * len(risk_scores))

    # Save final CSV with aggregated predictions from all folds.
    final_predictions = pd.DataFrame({
         "fold": all_fold_ids,
         "predicted_risk_score": all_predicted_risk_scores,
         "event_time": all_event_times,
         "event_indicator": all_event_indicators,
         "source": all_sources
    })
    final_csv_path = os.path.join(args.output_dir, "final_cv_predictions.csv")
    final_predictions.to_csv(final_csv_path, index=False)
    print(f"Final predictions CSV saved to {final_csv_path}")

    # Compute overall accuracy from aggregated predictions.
    all_predicted_risk_scores = np.array(all_predicted_risk_scores)
    all_event_indicators = np.array(all_event_indicators)
    overall_accuracy = np.mean((all_predicted_risk_scores >= 0.5).astype(int) == all_event_indicators)
    print(f"\nOverall Accuracy from aggregated CV predictions: {overall_accuracy:.4f}")

    # Print summary stats for per-fold accuracies.
    mean_accuracy = np.mean(fold_accuracies)
    std_accuracy = np.std(fold_accuracies)
    min_accuracy = np.min(fold_accuracies)
    max_accuracy = np.max(fold_accuracies)
    print("\nCross Validation Accuracy Statistics (combined):")
    print(f"Mean: {mean_accuracy:.4f}")
    print(f"Standard Deviation: {std_accuracy:.4f}")
    print(f"Minimum: {min_accuracy:.4f}")
    print(f"Maximum: {max_accuracy:.4f}")

    # Optionally, report separate accuracies if the original test CSV was provided.
    if df_test_full is not None:
        all_sources = np.array(all_sources)
        test_indices = np.where(all_sources == 'test')[0]
        train_indices = np.where(all_sources == 'train')[0]

        if len(train_indices) > 0:
            train_risk_scores = all_predicted_risk_scores[train_indices]
            train_event_indicators = all_event_indicators[train_indices]
            train_accuracy = np.mean((train_risk_scores >= 0.5).astype(int) == train_event_indicators)
            print(f"\nTraining CSV Only Accuracy: {train_accuracy:.4f}")
        else:
            print("\nNo training CSV samples found in the CV folds.")

        if len(test_indices) > 0:
            test_risk_scores = all_predicted_risk_scores[test_indices]
            test_event_indicators = all_event_indicators[test_indices]
            test_accuracy = np.mean((test_risk_scores >= 0.5).astype(int) == test_event_indicators)
            print(f"Test CSV Only Accuracy: {test_accuracy:.4f}")
        else:
            print("No test CSV samples found in the CV folds.")

def main(args):
    if args.cross_validation:
        cross_validation_mode(args)
    else:
        score = train_and_evaluate(
            args,
            args.train_csv_file,
            args.test_csv_file,
            hyperparams={
                'learning_rate': args.learning_rate,
                'dropout': args.dropout,
                'alpha': args.alpha,
                'gamma': args.gamma,
                'coxph_net': args.coxph_net
            },
            fold_id="full",
            final_eval=True,
            return_predictions=False
        )
        print(f"Final Accuracy: {score:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train binary classification model with DINOv2 features")
    parser.add_argument("--train_dicom_root", type=str, default="/gpfs/data/mankowskilab/HCC/data/TCGA/manifest-4lZjKqlp5793425118292424834/TCGA-LIHC",
                         help="Path to the training DICOM directory.")
    parser.add_argument("--test_dicom_root", type=str, default="/gpfs/data/mankowskilab/HCC_Recurrence/dicom",
                         help="Path to the testing DICOM directory.")
    parser.add_argument("--train_csv_file", type=str, default="/gpfs/data/shenlab/wz1492/HCC/spreadsheets/tcga.csv",
                         help="Path to the training CSV file.")
    parser.add_argument("--test_csv_file", type=str, default="/gpfs/data/shenlab/wz1492/HCC/spreadsheets/processed_patient_labels_nyu.csv",
                         help="Path to the testing CSV file. If not provided, a train-test split will be performed on the training dataset.")
    parser.add_argument('--preprocessed_root', type=str, default=None, 
                         help='Directory to store/load preprocessed image tensors')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for data loaders')
    parser.add_argument('--num_slices', type=int, default=64,
                        help='Number of slices per patient')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of workers for data loaders')
    parser.add_argument('--epochs', type=int, default=500,
                        help='Number of training epochs')
    parser.add_argument('--output_dir', type=str, default='checkpoints',
                        help='Base directory to save outputs and models')
    parser.add_argument('--learning_rate', type=float, default=1e-5,
                        help='Learning rate for the classification model')
    parser.add_argument('--gradient_clip', type=float, default=0.1,
                        help='Gradient clipping threshold. Set 0 to disable.')
    parser.add_argument('--center_risk', action='store_true',
                        help='(Unused in classification) If set, center risk scores for numerical stability')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='(Unused in linear classification) Dropout rate if using an MLP')
    parser.add_argument('--num_samples_per_patient', type=int, default=1,
                        help='Number of times to sample from each patient per epoch')
    parser.add_argument('--coxph_net', type=str, default='mlp', choices=['mlp', 'linear'],
                        help='(Unused in classification) Type of network for pycox survival regression.')
    parser.add_argument('--dinov2_weights', type=str, required=True,
                        help="Path to your local DINOv2 state dict file (.pth or .pt).")
    parser.add_argument('--alpha', type=float, default=0.5,
                        help="(Unused in classification) Weight for the regularization term relative to the Cox loss")
    parser.add_argument('--gamma', type=float, default=0.5,
                        help="(Unused in classification) Relative weight between L1 and L2 in the regularizer")
    parser.add_argument('--upsampling', action='store_true',
                        help="If set, perform upsampling of the minority class in the training data")
    parser.add_argument('--early_stopping', action='store_true',
                        help="If set, early stopping will be used during training")
    parser.add_argument('--cross_validation', action='store_true',
                        help="Enable cross validation mode")
    parser.add_argument('--cv_folds', type=int, default=2,
                        help="Number of cross validation folds")
    parser.add_argument('--leave_one_out', action='store_true',
                        help="Enable leave-one-out cross validation mode (combines CSVs and uses LOOCV)")
    
    args = parser.parse_args()

    # Create a unique subdirectory for each run using a timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    args.output_dir = os.path.join(args.output_dir, f"run_{timestamp}")
    os.makedirs(args.output_dir, exist_ok=True)
    
    main(args)