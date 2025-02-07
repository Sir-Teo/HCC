import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import torch
from pytorch_lightning import seed_everything

import torchtuples as tt
from pycox.evaluation import EvalSurv
from data.dataset import HCCDataModule
from models.dino import load_dinov2_model
from models.mlp import CustomMLP, CenteredModel, CoxPHWithL1
from utils.helpers import extract_features
from callbacks.custom_callbacks import GradientClippingCallback, LossLogger, ParamCheckerCallback

# New imports for additional visualization and statistics
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test

# Use a seaborn style for improved aesthetics
sns.set(style="whitegrid")


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


def main(args):
    seed_everything(42, workers=True)

    data_module = HCCDataModule(
        csv_file=args.csv_file,
        dicom_root=args.dicom_root,
        model_type="time_to_event",
        batch_size=args.batch_size,
        num_slices=args.num_slices,
        num_workers=args.num_workers,
        preprocessed_root=args.preprocessed_root,
        num_samples=args.num_samples_per_patient
    )
    data_module.setup()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dino_model = load_dinov2_model(args.dinov2_weights)
    dino_model = dino_model.to(device)
    dino_model.eval()
    dino_model.register_forward_hook(lambda m, i, o: None)  # Optionally register a hook

    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()
    test_loader = data_module.test_dataloader()

    print("Extracting train features...")
    x_train, y_train_durations, y_train_events = extract_features(train_loader, dino_model, device)
    print("Extracting validation features...")
    x_val, y_val_durations, y_val_events = extract_features(val_loader, dino_model, device)
    print("Extracting test features...")
    x_test, y_test_durations, y_test_events = extract_features(test_loader, dino_model, device)

    x_train = x_train.mean(axis=1)  # New shape: (n_samples, n_features)
    x_val = x_val.mean(axis=1)
    x_test = x_test.mean(axis=1)


    print("Checking for NaNs in features...")
    print("x_train contains NaNs:", np.isnan(x_train).any())
    print("x_val contains NaNs:", np.isnan(x_val).any())
    print("x_test contains NaNs:", np.isnan(x_test).any())

    variances = np.var(x_train, axis=0)
    if np.any(variances == 0):
        zero_var_features = np.where(variances == 0)[0]
        print(f"Warning: Features with zero variance: {zero_var_features}")
        x_train = x_train[:, variances != 0]
        x_val = x_val[:, variances != 0]
        x_test = x_test[:, variances != 0]

    if args.upsampling:
        print("Performing upsampling on training data...")
        x_train, y_train_durations, y_train_events = upsample_training_data(
            x_train, y_train_durations, y_train_events
        )

    y_train = (y_train_durations, y_train_events)
    y_val = (y_val_durations, y_val_events)
    y_test = (y_test_durations, y_test_events)

    x_mapper = StandardScaler()
    # --- Process training data ---
    n_train, n_slices, feat_dim = x_train.shape
    # Reshape to 2D: (n_patients * num_slices, feature_dim)
    x_train_reshaped = x_train.reshape(-1, feat_dim)
    # Fit the scaler and transform
    x_train_scaled = x_mapper.fit_transform(x_train_reshaped).astype('float32')
    # Reshape back to original 3D shape
    x_train_scaled = x_train_scaled.reshape(n_train, n_slices, feat_dim)

    # --- Process validation data ---
    n_val, n_slices_val, feat_dim_val = x_val.shape
    x_val_reshaped = x_val.reshape(-1, feat_dim_val)
    x_val_scaled = x_mapper.transform(x_val_reshaped).astype('float32')
    x_val_scaled = x_val_scaled.reshape(n_val, n_slices_val, feat_dim_val)

    # --- Process test data ---
    n_test, n_slices_test, feat_dim_test = x_test.shape
    x_test_reshaped = x_test.reshape(-1, feat_dim_test)
    x_test_scaled = x_mapper.transform(x_test_reshaped).astype('float32')
    x_test_scaled = x_test_scaled.reshape(n_test, n_slices_test, feat_dim_test)

    # --- Flatten slice dimension if the Cox model requires one feature vector per patient ---
    # (Concatenate the slice and feature dimensions into a single vector.)
    x_train_std = x_train_scaled.reshape(n_train, -1)
    x_val_std = x_val_scaled.reshape(n_val, -1)
    x_test_std = x_test_scaled.reshape(n_test, -1)
    print("Feature value ranges (train):")
    print(f"  Min: {x_train_std.min()}, Max: {x_train_std.max()}")
    print(f"  Mean abs: {np.mean(np.abs(x_train_std))}, Std: {np.std(x_train_std)}")

    validate_survival_data(y_train_durations, y_train_events)

    in_features = x_train_std.shape[1]
    print("Input feature dimension:", in_features)
    out_features = 1

    if args.coxph_net == 'mlp':
        net = CustomMLP(in_features, out_features, dropout=args.dropout)
    elif args.coxph_net == 'linear':
        from torch import nn
        net = nn.Linear(in_features, out_features, bias=False)
    else:
        raise ValueError("Unknown coxph_net option. Choose 'mlp' or 'linear'.")

    if args.center_risk:
        net = CenteredModel(net)

    net.register_forward_hook(lambda m, i, o: None)

    model = CoxPHWithL1(net, tt.optim.Adam, alpha=args.alpha, gamma=args.gamma)
    model.optimizer.set_lr(args.learning_rate)
    model.optimizer.param_groups[0]['weight_decay'] = 1e-4

    model.x_train_std = x_train_std
    callbacks = [LossLogger(), ParamCheckerCallback()]
    if args.early_stopping:
        callbacks.append(tt.callbacks.EarlyStopping())
    if args.gradient_clip > 0:
        callbacks.insert(1, GradientClippingCallback(args.gradient_clip))

    verbose = True
    batch_size = args.batch_size

    print("Training the CoxPH model...")
    log = model.fit(
        x_train_std,
        (y_train_durations, y_train_events),
        batch_size,
        args.epochs,
        callbacks,
        verbose,
        val_data=(x_val_std, (y_val_durations, y_val_events)),
        val_batch_size=batch_size
    )

    # Enhanced Training Log Plot
    plt.figure(figsize=(10, 6))
    log.plot()
    plt.title("Training Loss Over Epochs", fontsize=14)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "training_log.png"))
    plt.close()

    partial_ll = model.partial_log_likelihood(x_val_std, (y_val_durations, y_val_events)).mean()
    print(f"Partial Log-Likelihood on Validation Set: {partial_ll}")

    model.compute_baseline_hazards()
    surv = model.predict_surv_df(x_test_std)

    # Improved Survival Functions Plot with Recurrence Highlighted
    plt.figure(figsize=(10, 6))
    ax = plt.gca()
    # Plot the survival curves for the first 5 test samples.
    for i in range(5):
        # Plot the survival function for sample i.
        ax.plot(surv.index, surv.iloc[:, i], lw=2, label=f"Sample {i}")
        
        # If an event (recurrence) was observed for sample i, add a marker.
        if y_test_events[i] == 1:
            # Get the observed recurrence time.
            event_time = y_test_durations[i]
            # Find the nearest time point in the survival function's time grid.
            nearest_idx = np.abs(surv.index - event_time).argmin()
            nearest_time = surv.index[nearest_idx]
            survival_prob_at_event = surv.iloc[nearest_idx, i]
            # Plot a red marker at the recurrence.
            ax.scatter(nearest_time, survival_prob_at_event, color='red', s=50, zorder=5,
                    label='Recurrence' if i == 0 else "")  # Label only once for the legend

    ax.set_ylabel("Survival Probability", fontsize=12)
    ax.set_xlabel("Time", fontsize=12)
    ax.set_title("Survival Functions for Test Samples with Recurrence Highlighted", fontsize=14)
    ax.legend(title="Sample Index", fontsize=10)
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "survival_functions.png"))
    plt.close()


    ev = EvalSurv(surv, y_test_durations, y_test_events, censor_surv='km')
    concordance = ev.concordance_td()
    print(f"Concordance Index: {concordance}")

    time_grid = np.linspace(y_test_durations.min(), y_test_durations.max(), 100)
    brier_score = ev.brier_score(time_grid)
    plt.figure(figsize=(10, 6))
    brier_score.plot(lw=2)
    plt.title("Brier Score Over Time", fontsize=14)
    plt.xlabel("Time", fontsize=12)
    plt.ylabel("Brier Score", fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "brier_score.png"))
    plt.close()

    integrated_brier = ev.integrated_brier_score(time_grid)
    integrated_nbll = ev.integrated_nbll(time_grid)
    print(f"Integrated Brier Score: {integrated_brier}")
    print(f"Integrated NBLL: {integrated_nbll}")

    # ---- Additional Visualization & Interpretation ----
    # Risk Stratification based on the median predicted risk
    risk_scores = model.predict(x_test_std).reshape(-1)
    median_risk = np.median(risk_scores)
    low_risk_idx = risk_scores <= median_risk
    high_risk_idx = risk_scores > median_risk

    # Enhanced Risk Score Distribution Plot with Histogram and Inset Boxplot
    plt.figure(figsize=(10, 6))
    sns.histplot(risk_scores, bins=30, color='skyblue', edgecolor='k')
    plt.axvline(median_risk, color='red', linestyle='--', label=f'Median Risk ({median_risk:.2f})')
    plt.xlabel("Predicted Risk Score", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.title("Distribution of Predicted Risk Scores", fontsize=14)
    plt.legend()
    plt.grid(True)
    # Inset boxplot
    inset_ax = plt.axes([0.65, 0.65, 0.2, 0.2])
    sns.boxplot(x=risk_scores, ax=inset_ax, color='lightgreen')
    inset_ax.set_xlabel("")
    inset_ax.set_yticks([])
    plt.savefig(os.path.join(args.output_dir, "risk_score_distribution.png"))
    plt.close()

    # Kaplan-Meier curves for low- and high-risk groups
    kmf_low = KaplanMeierFitter()
    kmf_high = KaplanMeierFitter()

    plt.figure(figsize=(10, 6))
    kmf_low.fit(y_test_durations[low_risk_idx], event_observed=y_test_events[low_risk_idx], label='Low Risk')
    ax = kmf_low.plot(ci_show=True, color='green', lw=2)
    kmf_high.fit(y_test_durations[high_risk_idx], event_observed=y_test_events[high_risk_idx], label='High Risk')
    kmf_high.plot(ci_show=True, ax=ax, color='red', lw=2)
    plt.title("Kaplan-Meier Survival Curves by Risk Group", fontsize=14)
    plt.xlabel("Time", fontsize=12)
    plt.ylabel("Survival Probability", fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "km_risk_stratification.png"))
    plt.close()

    # Log-Rank Test between risk groups
    lr_results = logrank_test(
        y_test_durations[low_risk_idx],
        y_test_durations[high_risk_idx],
        event_observed_A=y_test_events[low_risk_idx],
        event_observed_B=y_test_events[high_risk_idx]
    )
    print(f"Log-Rank Test p-value: {lr_results.p_value:.4f}")

    # Median survival times for each group
    median_low = kmf_low.median_survival_time_
    median_high = kmf_high.median_survival_time_
    print(f"Median Survival Time (Low Risk): {median_low}")
    print(f"Median Survival Time (High Risk): {median_high}")

    # 2. Calibration Plot at a Fixed Time Point (e.g., 24 months)
    fixed_time = 24  # desired time point (e.g., months)
    if fixed_time not in surv.index:
        nearest_idx = np.abs(surv.index - fixed_time).argmin()
        nearest_time = surv.index[nearest_idx]
        print(f"Fixed time {fixed_time} not found in survival index, using nearest time {nearest_time} instead.")
        fixed_time = nearest_time

    predicted_surv_probs = surv.loc[fixed_time].values

    # Group patients by deciles of predicted survival probability.
    decile_bins = np.percentile(predicted_surv_probs, np.arange(0, 110, 10))
    bin_indices = np.digitize(predicted_surv_probs, decile_bins, right=True)

    observed_probs = []
    predicted_avg = []
    for i in range(1, 11):
        idx = bin_indices == i
        if np.sum(idx) > 0:
            kmf = KaplanMeierFitter()
            kmf.fit(y_test_durations[idx], event_observed=y_test_events[idx])
            observed_probs.append(kmf.predict(fixed_time))
            predicted_avg.append(np.mean(predicted_surv_probs[idx]))

    plt.figure(figsize=(10, 6))
    plt.plot(predicted_avg, observed_probs, 'o-', lw=2, label='Calibration Curve')
    plt.plot([0, 1], [0, 1], '--', color='gray', label='Ideal Calibration')
    plt.xlabel(f"Predicted Survival Probability at {fixed_time} months", fontsize=12)
    plt.ylabel("Observed Survival Probability", fontsize=12)
    plt.title(f"Calibration Plot at {fixed_time} Months", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "calibration_plot.png"))
    plt.close()

    # Additional statistics reporting
    overall_kmf = KaplanMeierFitter()
    overall_kmf.fit(y_test_durations, event_observed=y_test_events)
    overall_surv = overall_kmf.predict(fixed_time)

    print("\nAdditional Statistics:")
    print(f"  Mean Predicted Risk Score: {np.mean(risk_scores):.4f}")
    print(f"  Std of Predicted Risk Scores: {np.std(risk_scores):.4f}")
    print(f"  Median Predicted Survival Probability at {fixed_time} months: {np.median(predicted_surv_probs):.4f}")
    print(f"  Observed Survival Probability at {fixed_time} months (overall): {overall_surv:.4f}")

    # Optionally, compute and display the average predicted survival probability along with its spread.
    print(f"  Mean Predicted Survival Probability at {fixed_time} months: {np.mean(predicted_surv_probs):.4f}")
    print(f"  Std of Predicted Survival Probabilities at {fixed_time} months: {np.std(predicted_surv_probs):.4f}")

    # You can add further statistical tests or visuals here as needed.


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train CoxPH model with DINOv2 features (local weights) and custom MLP with L1/L2 regularization"
    )

    parser.add_argument("--dicom_root", type=str, default="/gpfs/data/mankowskilab/HCC_Recurrence/dicom",
                        help="Path to the root DICOM directory.")
    parser.add_argument("--csv_file", type=str, default="processed_patient_labels.csv",
                        help="Path to processed CSV with columns [Pre op MRI Accession number, recurrence post tx, time, event].")
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
    parser.add_argument('--coxph_method', type=str, default='pycox', choices=['pycox', 'traditional'],
                        help='Regression method: "pycox" or "traditional" (lifelines) for CoxPH.')
    parser.add_argument('--dinov2_weights', type=str, required=True,
                        help="Path to your local DINOv2 state dict file (.pth or .pt).")
    parser.add_argument('--alpha', type=float, default=0.5,
                        help="Weight for the regularization term relative to the Cox loss")
    parser.add_argument('--gamma', type=float, default=0.5,
                        help="Relative weight between L1 and L2 in the regularizer")
    parser.add_argument('--upsampling', action='store_true',
                        help="If set, perform upsampling of the minority class in the training data")
    parser.add_argument('--early_stopping', action='store_true',)

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    main(args)
