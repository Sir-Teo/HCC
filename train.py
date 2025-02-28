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
# Updated import with additional plotting functions
from utils.plotting import (plot_training_log, plot_survival_functions, plot_brier_score, 
                            plot_risk_score_distribution, plot_kaplan_meier, 
                            plot_calibration_plot, plot_multi_calibration, 
                            plot_cumulative_hazard, plot_survival_probability_distribution)

sns.set(style="whitegrid")


def bootstrap_evaluation(model, x_val_std, y_val_durations, y_val_events, num_bootstrap=1000):
    """
    Perform bootstrap evaluation to estimate the variability of the Concordance Index.
    
    Parameters:
        model: Trained CoxPH model.
        x_val_std (np.ndarray): Standardized validation features.
        y_val_durations (np.ndarray): Survival durations for validation data.
        y_val_events (np.ndarray): Event indicators for validation data.
        num_bootstrap (int): Number of bootstrap samples.
    
    Returns:
        np.ndarray: Array of bootstrapped concordance index values.
    """
    bootstrapped_concordances = []
    n_samples = len(y_val_durations)
    for i in range(num_bootstrap):
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        x_boot = x_val_std[indices]
        durations_boot = y_val_durations[indices]
        events_boot = y_val_events[indices]
        surv_boot = model.predict_surv_df(x_boot)
        ev_boot = EvalSurv(surv_boot, durations_boot, events_boot, censor_surv='km')
        boot_c_index = ev_boot.concordance_td()
        bootstrapped_concordances.append(boot_c_index)
    return np.array(bootstrapped_concordances)


def train_and_evaluate(args, train_csv, val_csv, hyperparams, fold_id="", final_eval=False):
    """
    Train the model on data from train_csv and evaluate on val_csv.
    hyperparams is a dict containing parameters such as learning_rate, dropout, alpha, gamma, and coxph_net.
    If final_eval is True, additional evaluation plots are produced.
    Returns the evaluation metric (here, the concordance index).
    """
    # Set seed for reproducibility
    seed_everything(42, workers=True)

    # Instantiate the data module with the given CSV files
    data_module = HCCDataModule(
        train_csv_file=train_csv,
        test_csv_file=val_csv,  # treat the validation fold as "test"
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

    # Standardize features (fit only on train)
    x_mapper = StandardScaler()
    n_train, n_slices, feat_dim = x_train.shape
    x_train_reshaped = x_train.reshape(-1, feat_dim)
    x_train_scaled = x_mapper.fit_transform(x_train_reshaped).astype('float32')
    x_train_scaled = x_train_scaled.reshape(n_train, n_slices, feat_dim)

    n_val, n_slices_val, feat_dim_val = x_val.shape
    x_val_reshaped = x_val.reshape(-1, feat_dim_val)
    x_val_scaled = x_mapper.transform(x_val_reshaped).astype('float32')
    x_val_scaled = x_val_scaled.reshape(n_val, n_slices_val, feat_dim_val)

    # Collapse the slice dimension by averaging (for the Cox model)
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

    # Compute baseline hazards and predict survival functions
    model.compute_baseline_hazards()
    surv = model.predict_surv_df(x_val_std)
    ev = EvalSurv(surv, y_val_durations, y_val_events, censor_surv='km')
    concordance = ev.concordance_td()
    print(f"[Fold {fold_id}] Concordance Index: {concordance:.4f}")

    # Perform bootstrap evaluation if requested
    if args.bootstrap:
        print(f"[Fold {fold_id}] Performing bootstrap evaluation...")
        boot_results = bootstrap_evaluation(model, x_val_std, y_val_durations, y_val_events, num_bootstrap=1000)
        mean_boot = np.mean(boot_results)
        ci_lower = np.percentile(boot_results, 2.5)
        ci_upper = np.percentile(boot_results, 97.5)
        print(f"[Fold {fold_id}] Bootstrap Concordance Index: Mean = {mean_boot:.4f}, 95% CI = [{ci_lower:.4f}, {ci_upper:.4f}]")

    # If final evaluation is enabled, produce additional plots
    if final_eval:
        # Create a subdirectory for this fold's outputs
        fold_output_dir = os.path.join(args.output_dir, f"fold_{fold_id}")
        os.makedirs(fold_output_dir, exist_ok=True)

        # Plot training log
        plot_training_log(log, fold_output_dir)

        # Plot survival functions with recurrence markers
        plot_survival_functions(surv, y_val_durations, y_val_events, fold_output_dir, num_samples=15)

        # Compute and plot Brier score over a time grid
        time_grid = np.linspace(y_val_durations.min(), y_val_durations.max(), 100)
        brier_score = ev.brier_score(time_grid)
        plot_brier_score(time_grid, brier_score, fold_output_dir)

        # Compute risk scores, stratify, and plot risk score distribution
        risk_scores = model.predict(x_val_std).reshape(-1)
        median_risk = np.median(risk_scores)
        plot_risk_score_distribution(risk_scores, median_risk, fold_output_dir)

        # Stratify patients into low- and high-risk groups and plot Kaplan-Meier curves
        low_risk_idx = risk_scores <= median_risk
        high_risk_idx = risk_scores > median_risk
        kmf_low = KaplanMeierFitter()
        kmf_high = KaplanMeierFitter()
        plot_kaplan_meier(kmf_low, kmf_high, y_val_durations, y_val_events, low_risk_idx, high_risk_idx, fold_output_dir)

        # Plot calibration curve at a fixed time point (e.g., 24 time units)
        fixed_time = 24
        if fixed_time not in surv.index:
            nearest_idx = np.abs(surv.index - fixed_time).argmin()
            fixed_time = surv.index[nearest_idx]
            print(f"[Fold {fold_id}] Fixed time not found; using nearest time {fixed_time} instead.")
        predicted_surv_probs = surv.loc[fixed_time].values
        plot_calibration_plot(surv, fixed_time, predicted_surv_probs, y_val_durations, y_val_events, fold_output_dir)

        # Optional: Plot multi-time calibration curves at selected time points
        time_points = [12, 24, 36]
        plot_multi_calibration(surv, y_val_durations, y_val_events, time_points, fold_output_dir)

        # New: Plot cumulative hazard functions (H(t) = -log(S(t)))
        plot_cumulative_hazard(surv, fold_output_dir)

        # New: Plot distribution of predicted survival probabilities at the fixed time
        plot_survival_probability_distribution(surv, fold_output_dir, time_point=fixed_time)

    return concordance


def main(args):
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
        final_eval=True
    )
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
    parser.add_argument('--bootstrap', action='store_true',
                        help="If set, perform bootstrap evaluation for the Concordance Index")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    main(args)
