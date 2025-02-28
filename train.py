import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, ParameterGrid
import torch
from pytorch_lightning import seed_everything
import torchtuples as tt
from pycox.evaluation import EvalSurv
from data.dataset import HCCDataModule
from models.dino import load_dinov2_model
from models.mlp import CustomMLP, CenteredModel, CoxPHWithL1
from utils.helpers import extract_features, validate_survival_data, upsample_training_data
from callbacks.custom_callbacks import GradientClippingCallback, LossLogger, ParamCheckerCallback
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from utils.plotting import (plot_training_log, plot_survival_functions, plot_brier_score, 
                            plot_risk_score_distribution, plot_kaplan_meier, 
                            plot_calibration_plot, plot_multi_calibration)

sns.set(style="whitegrid")


def train_and_evaluate(args, train_csv, val_csv, hyperparams, fold_id="", final_eval=False):
    """
    Train the model on data from train_csv and evaluate on val_csv.
    hyperparams is a dict containing parameters such as learning_rate, dropout, alpha, gamma, and coxph_net.
    If final_eval is True, additional evaluation (e.g., survival plots) may be produced.
    Returns the chosen evaluation metric (here, the concordance index on the validation fold).
    """
    # Set seed for reproducibility
    seed_everything(42, workers=True)

    # Instantiate the data module with the given CSV files
    data_module = HCCDataModule(
        train_csv_file=train_csv,
        test_csv_file=val_csv,  # for evaluation, we treat the validation fold as "test"
        train_dicom_root=args.train_dicom_root,
        test_dicom_root=args.test_dicom_root,
        model_type="time_to_event",
        batch_size=args.batch_size,
        num_slices=args.num_slices,
        num_workers=args.num_workers,
        preprocessed_root=args.preprocessed_root,
        num_samples=args.num_samples_per_patient
    )
    data_module.setup()

    # Load the DINOv2 feature extractor
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dino_model = load_dinov2_model(args.dinov2_weights)
    dino_model = dino_model.to(device)
    dino_model.eval()
    dino_model.register_forward_hook(lambda m, i, o: None)

    # Get dataloaders for training and validation
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()

    # Feature extraction for train and validation data
    x_train, y_train_durations, y_train_events = extract_features(train_loader, dino_model, device)
    x_val, y_val_durations, y_val_events = extract_features(val_loader, dino_model, device)

    # Average across slices
    x_train = x_train.mean(axis=1)
    x_val = x_val.mean(axis=1)

    # Check for NaNs and remove zero-variance features
    variances = np.var(x_train, axis=0)
    if np.any(variances == 0):
        zero_var_features = np.where(variances == 0)[0]
        print(f"[Fold {fold_id}] Warning: Features with zero variance: {zero_var_features}")
        x_train = x_train[:, variances != 0]
        x_val = x_val[:, variances != 0]

    if args.upsampling:
        print(f"[Fold {fold_id}] Performing upsampling on training data...")
        x_train, y_train_durations, y_train_events = upsample_training_data(
            x_train, y_train_durations, y_train_events
        )

    # Standardize features (fitting only on train)
    x_mapper = StandardScaler()
    n_train, n_slices, feat_dim = x_train.shape
    x_train_reshaped = x_train.reshape(-1, feat_dim)
    x_train_scaled = x_mapper.fit_transform(x_train_reshaped).astype('float32')
    x_train_scaled = x_train_scaled.reshape(n_train, n_slices, feat_dim)

    n_val, n_slices_val, feat_dim_val = x_val.shape
    x_val_reshaped = x_val.reshape(-1, feat_dim_val)
    x_val_scaled = x_mapper.transform(x_val_reshaped).astype('float32')
    x_val_scaled = x_val_scaled.reshape(n_val, n_slices_val, feat_dim_val)

    # For the Cox model, collapse the slice dimension by averaging
    x_train_std = x_train_scaled.mean(axis=1)
    x_val_std = x_val_scaled.mean(axis=1)

    # Validate survival data
    validate_survival_data(y_train_durations, y_train_events)

    in_features = x_train_std.shape[1]
    out_features = 1

    # Build the network based on hyperparameters
    if hyperparams['coxph_net'] == 'mlp':
        net = CustomMLP(in_features, out_features, dropout=hyperparams['dropout'])
    elif hyperparams['coxph_net'] == 'linear':
        from torch import nn
        net = nn.Linear(in_features, out_features, bias=False)
    else:
        raise ValueError("Unknown coxph_net option. Choose 'mlp' or 'linear'.")

    if args.center_risk:
        net = CenteredModel(net)
    net.register_forward_hook(lambda m, i, o: None)

    # Instantiate the CoxPH model with L1/L2 regularization
    model = CoxPHWithL1(net, tt.optim.Adam, alpha=hyperparams['alpha'], gamma=hyperparams['gamma'])
    model.optimizer.set_lr(hyperparams['learning_rate'])
    model.optimizer.param_groups[0]['weight_decay'] = 1e-4
    model.x_train_std = x_train_std

    # Define callbacks
    callbacks = [LossLogger(), ParamCheckerCallback()]
    if args.early_stopping:
        callbacks.append(tt.callbacks.EarlyStopping())
    if args.gradient_clip > 0:
        callbacks.insert(1, GradientClippingCallback(args.gradient_clip))

    # Train the model
    print(f"[Fold {fold_id}] Training the CoxPH model...")
    log = model.fit(
        x_train_std,
        (y_train_durations, y_train_events),
        args.batch_size,
        args.epochs,
        callbacks,
        verbose=True,
        val_data=(x_val_std, (y_val_durations, y_val_events)),
        val_batch_size=args.batch_size
    )

    # (Optional) For final evaluation, you might plot survival curves etc.
    model.compute_baseline_hazards()
    surv = model.predict_surv_df(x_val_std)
    ev = EvalSurv(surv, y_val_durations, y_val_events, censor_surv='km')
    concordance = ev.concordance_td()
    print(f"[Fold {fold_id}] Concordance Index: {concordance:.4f}")

    # Optionally, if doing final evaluation (outer loop), you can add more plots and statistics here.
    if final_eval:
        # e.g., plot survival functions, calibration curves, risk stratification, etc.
        plot_training_log(log, args.output_dir + f"/training_log_fold{fold_id}")
        # (Add any additional plotting/evaluation as needed)
        pass

    return concordance


def nested_cross_validation(args):
    """
    Perform nested cross validation.
    The outer loop estimates the generalization performance while the inner loop selects hyperparameters.
    """
    # Read the entire training CSV into a DataFrame
    full_df = pd.read_csv(args.train_csv_file)

    outer_cv = KFold(n_splits=args.outer_splits, shuffle=True, random_state=42)
    outer_scores = []

    # Define a hyperparameter grid for the inner CV search.
    hyperparameter_grid = {
        'learning_rate': [1e-8, 1e-7],
        'dropout': [0.2, 0.5],
        'alpha': [0.5, 1.0],
        'gamma': [0.5, 1.0],
        'coxph_net': ['mlp']  # you could also try 'linear'
    }

    fold = 0
    for outer_train_idx, outer_test_idx in outer_cv.split(full_df):
        fold += 1
        print(f"\n=== Outer Fold {fold}/{args.outer_splits} ===")
        outer_train_df = full_df.iloc[outer_train_idx].reset_index(drop=True)
        outer_test_df = full_df.iloc[outer_test_idx].reset_index(drop=True)

        # Inner loop for hyperparameter tuning on the outer training set
        inner_cv = KFold(n_splits=args.inner_splits, shuffle=True, random_state=42)
        best_inner_score = -np.inf
        best_hyperparams = None

        for hyperparams in ParameterGrid(hyperparameter_grid):
            inner_scores = []
            # For each hyperparameter configuration, evaluate with inner CV
            for inner_train_idx, inner_val_idx in inner_cv.split(outer_train_df):
                inner_train_df = outer_train_df.iloc[inner_train_idx].reset_index(drop=True)
                inner_val_df = outer_train_df.iloc[inner_val_idx].reset_index(drop=True)
                # For the inner loop, we assume that we can write these dataframes to temporary CSVs.
                inner_train_csv = f"temp_inner_train_fold{fold}.csv"
                inner_val_csv = f"temp_inner_val_fold{fold}.csv"
                inner_train_df.to_csv(inner_train_csv, index=False)
                inner_val_df.to_csv(inner_val_csv, index=False)
                score = train_and_evaluate(args, inner_train_csv, inner_val_csv, hyperparams, fold_id=f"{fold}-inner")
                inner_scores.append(score)
            avg_inner_score = np.mean(inner_scores)
            print(f"[Fold {fold}] Hyperparams {hyperparams} achieved inner avg concordance: {avg_inner_score:.4f}")
            if avg_inner_score > best_inner_score:
                best_inner_score = avg_inner_score
                best_hyperparams = hyperparams

        print(f"[Fold {fold}] Selected hyperparameters: {best_hyperparams} with score {best_inner_score:.4f}")

        # Now train on the entire outer training set using the best hyperparameters and evaluate on the outer test set.
        outer_train_csv = f"temp_outer_train_fold{fold}.csv"
        outer_test_csv = f"temp_outer_test_fold{fold}.csv"
        outer_train_df.to_csv(outer_train_csv, index=False)
        outer_test_df.to_csv(outer_test_csv, index=False)
        outer_score = train_and_evaluate(args, outer_train_csv, outer_test_csv, best_hyperparams, fold_id=f"{fold}-outer", final_eval=True)
        outer_scores.append(outer_score)
        print(f"[Fold {fold}] Outer fold concordance: {outer_score:.4f}")

    avg_outer = np.mean(outer_scores)
    print(f"\n=== Nested CV Average Concordance Index: {avg_outer:.4f} ===")


def main(args):
    if args.nested_cv:
        nested_cross_validation(args)
    else:
        # Original training (non-nested) using provided CSVs.
        # This is essentially the code you already have.
        # For brevity, we call train_and_evaluate once on the full training and test CSVs.
        score = train_and_evaluate(args, args.train_csv_file, args.test_csv_file, 
                                   hyperparams={
                                       'learning_rate': args.learning_rate,
                                       'dropout': args.dropout,
                                       'alpha': args.alpha,
                                       'gamma': args.gamma,
                                       'coxph_net': args.coxph_net
                                   }, fold_id="full", final_eval=True)
        print(f"Final Concordance Index: {score:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train CoxPH model with DINOv2 features and nested cross validation"
    )
    parser.add_argument("--train_dicom_root", type=str, default="/gpfs/data/mankowskilab/HCC/data/TCGA/manifest-4lZjKqlp5793425118292424834/TCGA-LIHC",
                        help="Path to the training DICOM directory.")
    parser.add_argument("--test_dicom_root", type=str, default="/gpfs/data/mankowskilab/HCC_Recurrence/dicom",
                        help="Path to the testing DICOM directory.")
    parser.add_argument("--train_csv_file", type=str, default="/gpfs/data/shenlab/wz1492/HCC/spreadsheets/tcga.csv",
                        help="Path to the training CSV file.")
    parser.add_argument("--test_csv_file", type=str, default="/gpfs/data/shenlab/wz1492/HCC/spreadsheets/processed_patient_labels_nyu.csv",
                        help="Path to the testing CSV file.")
    parser.add_argument('--preprocessed_root', type=str, default=None, 
                        help='Directory to store/load preprocessed image tensors')
    parser.add_argument('--batch_size', type=int, default=32, 
                        help='Batch size for data loaders')
    parser.add_argument('--num_slices', type=int, default=2, 
                        help='Number of slices per patient')
    parser.add_argument('--num_workers', type=int, default=4, 
                        help='Number of workers for data loaders')
    parser.add_argument('--epochs', type=int, default=500, 
                        help='Number of training epochs')
    parser.add_argument('--output_dir', type=str, default='checkpoints', 
                        help='Directory to save outputs and models')
    parser.add_argument('--learning_rate', type=float, default=1e-8, 
                        help='Learning rate for the CoxPH model')
    parser.add_argument('--gradient_clip', type=float, default=0.05, 
                        help='Gradient clipping threshold. Set 0 to disable.')
    parser.add_argument('--center_risk', action='store_true', 
                        help='If set, center risk scores for numerical stability')
    parser.add_argument('--dropout', type=float, default=0.2, 
                        help='Dropout rate for the MLP if used')
    parser.add_argument('--num_samples_per_patient', type=int, default=1,
                        help='Number of times to sample from each patient per epoch')
    parser.add_argument('--coxph_net', type=str, default='mlp', choices=['mlp', 'linear'],
                        help='Type of network for pycox survival regression.')
    parser.add_argument('--dinov2_weights', type=str, required=True,
                        help="Path to your local DINOv2 state dict file (.pth or .pt).")
    parser.add_argument('--alpha', type=float, default=0.5,
                        help="Weight for the regularization term relative to the Cox loss")
    parser.add_argument('--gamma', type=float, default=0.5,
                        help="Relative weight between L1 and L2 in the regularizer")
    parser.add_argument('--upsampling', action='store_true',
                        help="If set, perform upsampling of the minority class in the training data")
    parser.add_argument('--early_stopping', action='store_true',
                        help="If set, early stopping will be used during training")
    # Arguments for nested CV
    parser.add_argument('--nested_cv', action='store_true', 
                        help="If set, perform nested cross validation instead of a single train/test split")
    parser.add_argument('--outer_splits', type=int, default=3,
                        help="Number of outer folds for nested CV")
    parser.add_argument('--inner_splits', type=int, default=3,
                        help="Number of inner folds for hyperparameter tuning")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    main(args)
