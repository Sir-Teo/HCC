# main.py
import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import torch
from pytorch_lightning import seed_everything

import torchtuples as tt
from pycox.evaluation import EvalSurv

# Import our modules
from data.dataset import HCCDataModule
from models.dino import load_dinov2_model
from models.mlp import CustomMLP, CenteredModel, CoxPHWithL1
from utils.helpers import extract_features
from callbacks.custom_callbacks import GradientClippingCallback, LossLogger, ParamCheckerCallback

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
        num_samples_per_patient=args.num_samples_per_patient
    )
    data_module.setup()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dino_model = load_dinov2_model(args.dinov2_weights)
    dino_model = dino_model.to(device)
    dino_model.eval()
    dino_model.register_forward_hook(lambda m, i, o: None)  # Optionally register nan_hook

    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()
    test_loader = data_module.test_dataloader()

    print("Extracting train features...")
    x_train, y_train_durations, y_train_events = extract_features(train_loader, dino_model, device)
    print("Extracting validation features...")
    x_val, y_val_durations, y_val_events = extract_features(val_loader, dino_model, device)
    print("Extracting test features...")
    x_test, y_test_durations, y_test_events = extract_features(test_loader, dino_model, device)

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

    y_train = (y_train_durations, y_train_events)
    y_val = (y_val_durations, y_val_events)
    y_test = (y_test_durations, y_test_events)

    x_mapper = StandardScaler()
    x_train_std = x_mapper.fit_transform(x_train).astype('float32')
    x_val_std = x_mapper.transform(x_val).astype('float32')
    x_test_std = x_mapper.transform(x_test).astype('float32')

    print("Feature value ranges (train):")
    print(f"  Min: {x_train_std.min()}, Max: {x_train_std.max()}")
    print(f"  Mean abs: {np.mean(np.abs(x_train_std))}, Std: {np.std(x_train_std)}")

    validate_survival_data(y_train_durations, y_train_events)

    # For traditional CoxPH via lifelines (if desired), you could insert that branch here.

    in_features = x_train_std.shape[1]
    print("Input feature dimension:", in_features)
    out_features = 1

    if args.coxph_net == 'mlp':
        num_nodes = [2048, 2048]
        net = CustomMLP(
            in_features,
            num_nodes,
            out_features,
            dropout=args.dropout,
            l1_lambda=args.l1_lambda
        )
    elif args.coxph_net == 'linear':
        from torch import nn
        net = nn.Linear(in_features, out_features, bias=False)
    else:
        raise ValueError("Unknown coxph_net option. Choose 'mlp' or 'linear'.")

    if args.center_risk:
        net = CenteredModel(net)

    net.register_forward_hook(lambda m, i, o: None)  # Optionally register nan_hook

    model = CoxPHWithL1(net, tt.optim.Adam)
    model.optimizer.set_lr(args.learning_rate)
    model.optimizer.param_groups[0]['weight_decay'] = 1e-4

    model.x_train_std = x_train_std

    callbacks = [tt.callbacks.EarlyStopping(), LossLogger(), ParamCheckerCallback()]
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

    plt.figure()
    log.plot()
    plt.title("Training Log (Loss)")
    plt.savefig(os.path.join(args.output_dir, "training_log.png"))
    plt.close()

    partial_ll = model.partial_log_likelihood(x_val_std, (y_val_durations, y_val_events)).mean()
    print(f"Partial Log-Likelihood on Validation Set: {partial_ll}")

    model.compute_baseline_hazards()
    surv = model.predict_surv_df(x_test_std)
    plt.figure()
    surv.iloc[:, :5].plot()
    plt.ylabel("S(t | x)")
    plt.xlabel("Time")
    plt.title("Survival Functions for Test Samples")
    plt.savefig(os.path.join(args.output_dir, "survival_functions.png"))
    plt.close()

    ev = EvalSurv(surv, y_test_durations, y_test_events, censor_surv='km')
    concordance = ev.concordance_td()
    print(f"Concordance Index: {concordance}")

    time_grid = np.linspace(y_test_durations.min(), y_test_durations.max(), 100)
    brier_score = ev.brier_score(time_grid)
    plt.figure()
    brier_score.plot()
    plt.title("Brier Score Over Time")
    plt.xlabel("Time")
    plt.ylabel("Brier Score")
    plt.savefig(os.path.join(args.output_dir, "brier_score.png"))
    plt.close()

    print(f"Integrated Brier Score: {ev.integrated_brier_score(time_grid)}")
    print(f"Integrated NBLL: {ev.integrated_nbll(time_grid)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train CoxPH model with DINOv2 features (local weights) and custom MLP with L1 regularization"
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
    parser.add_argument('--epochs', type=int, default=20, 
                        help='Number of training epochs')
    parser.add_argument('--output_dir', type=str, default='checkpoints', 
                        help='Directory to save outputs and models')
    parser.add_argument('--learning_rate', type=float, default=1e-6, 
                        help='Learning rate for the CoxPH model')
    parser.add_argument('--gradient_clip', type=float, default=0, 
                        help='Gradient clipping threshold. Set 0 to disable.')
    parser.add_argument('--center_risk', action='store_true', 
                        help='If set, center risk scores for numerical stability')
    parser.add_argument('--dropout', type=float, default=0.0, 
                        help='Dropout rate for the MLP if used')
    parser.add_argument('--num_samples_per_patient', type=int, default=1,
                        help='Number of times to sample from each patient per epoch')
    parser.add_argument('--coxph_net', type=str, default='mlp', choices=['mlp', 'linear'],
                        help='Type of network for pycox survival regression.')
    parser.add_argument('--coxph_method', type=str, default='pycox', choices=['pycox', 'traditional'],
                        help='Regression method: "pycox" or "traditional" (lifelines) for CoxPH.')
    parser.add_argument('--dinov2_weights', type=str, required=True,
                        help="Path to your local DINOv2 state dict file (.pth or .pt).")
    parser.add_argument('--l1_lambda', type=float, default=1e-5,
                        help="L1 regularization strength for the custom MLP.")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    main(args)
