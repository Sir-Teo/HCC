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

# Custom module imports
from data.dataset import HCCDataModule
from models.dino import load_dinov2_model
from models.mlp import CustomMLP, CenteredModel, CoxPHWithL1
from utils.helpers import extract_features, validate_survival_data, upsample_training_data
from callbacks.custom_callbacks import GradientClippingCallback, LossLogger, ParamCheckerCallback
from utils.plotting import (plot_cv_metrics, plot_survival_functions, plot_brier_score,
                      plot_risk_score_distribution, plot_kaplan_meier, plot_calibration_plot,
                      plot_multi_calibration, plot_cumulative_hazard, plot_survival_probability_distribution)
                      
sns.set(style="whitegrid")


def train_and_evaluate(args, train_csv, val_csv, hyperparams, fold_id="", final_eval=True, return_predictions=False):
    """
    Train the model on data from train_csv and evaluate on val_csv.
    
    If return_predictions is False (default), computes the concordance index on the validation set 
    (and optionally produces additional evaluation plots). 
    If return_predictions is True, returns the predicted risk scores for the validation set along with 
    the corresponding event times and event indicators.
    """
    # Set seed for reproducibility
    torch.manual_seed(42)
    
    # Instantiate the data module with CSV files or DataFrames
    data_module = HCCDataModule(
        train_csv_file=train_csv,
        test_csv_file=val_csv,  # treat validation as test
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
    
    # Remove zero-variance features
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
    
    # Compute baseline hazards
    model.compute_baseline_hazards()
    
    if return_predictions:
        # Predict risk scores for the validation set and return along with ground-truth
        risk_scores = -model.predict(x_val_std).reshape(-1)
        return risk_scores, y_val_durations, y_val_events
    else:
        # Compute survival predictions and evaluate concordance
        surv = model.predict_surv_df(x_val_std)
        ev = EvalSurv(surv, y_val_durations, y_val_events, censor_surv='km')
        concordance = ev.concordance_td()
        print(f"[Fold {fold_id}] Concordance Index: {concordance:.4f}")
    
        if final_eval and not return_predictions:
            os.makedirs(args.output_dir, exist_ok=True)
    
            plot_survival_functions(surv, y_val_durations, y_val_events, args.output_dir, num_samples=15)
            
            time_grid = np.linspace(y_val_durations.min(), y_val_durations.max(), 100)
            brier_score = ev.brier_score(time_grid)
            plot_brier_score(time_grid, brier_score, args.output_dir)
            
            risk_scores = model.predict(x_val_std).reshape(-1)
            median_risk = np.median(risk_scores)
            plot_risk_score_distribution(risk_scores, median_risk, args.output_dir)
            
            low_risk_idx = risk_scores <= median_risk
            high_risk_idx = risk_scores > median_risk
            kmf_low = KaplanMeierFitter()
            kmf_high = KaplanMeierFitter()
            plot_kaplan_meier(kmf_low, kmf_high, y_val_durations, y_val_events,
                              low_risk_idx, high_risk_idx, args.output_dir)
            
            fixed_time = 24
            if fixed_time not in surv.index:
                nearest_idx = np.abs(surv.index - fixed_time).argmin()
                fixed_time = surv.index[nearest_idx]
                print(f"[Fold {fold_id}] Fixed time not found; using nearest time {fixed_time} instead.")
            predicted_surv_probs = surv.loc[fixed_time].values
            plot_calibration_plot(surv, fixed_time, predicted_surv_probs, y_val_durations, y_val_events, args.output_dir)
            
            time_points = [12, 24, 36]
            plot_multi_calibration(surv, y_val_durations, y_val_events, time_points, args.output_dir)
            
            plot_cumulative_hazard(surv, args.output_dir)
            plot_survival_probability_distribution(surv, args.output_dir, time_point=fixed_time)
    
        return concordance


def upsample_df(df, target_column='event'):
    df_majority = df[df[target_column] == 0]
    df_minority = df[df[target_column] == 1]
    if len(df_minority) == 0 or len(df_majority) == 0:
        return df
    df_minority_upsampled = df_minority.sample(len(df_majority), replace=True, random_state=1)
    return pd.concat([df_majority, df_minority_upsampled]).reset_index(drop=True)


def cross_validation_mode(args):
    # Read training CSV and add a source column
    df_train_full = pd.read_csv(args.train_csv_file)
    df_train_full['dicom_root'] = args.train_dicom_root
    df_train_full['source'] = 'train'  # mark train samples

    if args.test_csv_file:
        df_test_full = pd.read_csv(args.test_csv_file)
        df_test_full['dicom_root'] = args.test_dicom_root
        df_test_full['source'] = 'test'  # mark test samples
    else:
        df_test_full = None
        print("No test CSV provided. Performing cross validation on the training dataset only.")

    all_predicted_risk_scores = []
    all_event_times = []
    all_event_indicators = []
    all_sources = []  # New list to store source info for each prediction
    fold_cindices = []  # List to store per-fold concordance indices

    hyperparams = {
        'learning_rate': args.learning_rate,
        'dropout': args.dropout,
        'alpha': args.alpha,
        'gamma': args.gamma,
        'coxph_net': args.coxph_net
    }
    
    # Determine combined dataset based on whether test CSV is provided
    if args.leave_one_out:
        from sklearn.model_selection import LeaveOneOut
        df_all = pd.concat([df_train_full, df_test_full]).reset_index(drop=True) if df_test_full is not None else df_train_full
        print(f"[LOO CV] Combined dataset has {len(df_all)} patients.")
        loo = LeaveOneOut()
        splits = list(loo.split(df_all))
        for fold, (train_idx, test_idx) in enumerate(splits):
            df_new_train = df_all.iloc[train_idx].reset_index(drop=True)
            df_new_test = df_all.iloc[test_idx].reset_index(drop=True)
            print(f"[LOO Fold {fold}] Train patients: {len(df_new_train)}, Positive events: {(df_new_train['event'] == 1).sum()}")
            print(f"[LOO Fold {fold}] Test patient: {len(df_new_test)}, Positive events: {(df_new_test['event'] == 1).sum()}")
            if args.upsampling:
                print(f"[LOO Fold {fold}] Performing upsampling on training data...")
                df_new_train = upsample_df(df_new_train, target_column='event')
            
            risk_scores, durations, events = train_and_evaluate(
                args,
                train_csv=df_new_train,
                val_csv=df_new_test,
                hyperparams=hyperparams,
                fold_id=f"LOO_fold_{fold}",
                final_eval=False,
                return_predictions=True
            )
            # Extend the aggregated lists; also record the source for these predictions
            all_predicted_risk_scores.extend(risk_scores.tolist())
            all_event_times.extend(durations.tolist())
            all_event_indicators.extend(events.tolist())
            all_sources.extend(df_new_test['source'].tolist())
    else:
        # Non-LOOCV cross validation mode with combined data
        df_all = pd.concat([df_train_full, df_test_full]).reset_index(drop=True) if df_test_full is not None else df_train_full
        print(f"[CV] Combined dataset has {len(df_all)} patients.")
        from sklearn.model_selection import StratifiedKFold
        skf = StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=1)
        for fold, (train_idx, test_idx) in enumerate(skf.split(df_all, df_all['event'])):
            df_new_train = df_all.iloc[train_idx].reset_index(drop=True)
            df_new_test = df_all.iloc[test_idx].reset_index(drop=True)
            print(f"[CV Fold {fold}] Train: {len(df_new_train)} patients, Positive events: {df_new_train['event'].sum()}")
            print(f"[CV Fold {fold}] Test: {len(df_new_test)} patients, Positive events: {df_new_test['event'].sum()}")
            if args.upsampling:
                print(f"[CV Fold {fold}] Performing upsampling on training data...")
                df_new_train = upsample_df(df_new_train, target_column='event')
            
            risk_scores, durations, events = train_and_evaluate(
                args,
                train_csv=df_new_train,
                val_csv=df_new_test,
                hyperparams=hyperparams,
                fold_id=f"fold_{fold}_test",
                final_eval=False,
                return_predictions=True
            )
            # Compute and report the fold's concordance index
            fold_cindex = concordance_index(durations, risk_scores, event_observed=events)
            print(f"[CV Fold {fold}] Concordance Index: {fold_cindex:.4f}")
            fold_cindices.append(fold_cindex)
            
            all_predicted_risk_scores.extend(risk_scores.tolist())
            all_event_times.extend(durations.tolist())
            all_event_indicators.extend(events.tolist())
            all_sources.extend(df_new_test['source'].tolist())
    
    # Compute overall concordance index on aggregated predictions (combined across sources)
    overall_cindex = concordance_index(all_event_times, np.array(all_predicted_risk_scores), event_observed=np.array(all_event_indicators))
    print(f"Overall Concordance Index from aggregated CV predictions: {overall_cindex:.4f}")
    
    # Report summary statistics for per-fold concordance indices (combined)
    mean_cindex = np.mean(fold_cindices)
    std_cindex = np.std(fold_cindices)
    min_cindex = np.min(fold_cindices)
    max_cindex = np.max(fold_cindices)
    
    print("\nCross Validation Concordance Index Statistics (combined):")
    print(f"Mean: {mean_cindex:.4f}")
    print(f"Standard Deviation: {std_cindex:.4f}")
    print(f"Minimum: {min_cindex:.4f}")
    print(f"Maximum: {max_cindex:.4f}")

    # Now, if both CSVs are provided, compute and report stats separately for train and test sources.
    if df_test_full is not None:
        all_sources = np.array(all_sources)
        all_predicted_risk_scores = np.array(all_predicted_risk_scores)
        all_event_times = np.array(all_event_times)
        all_event_indicators = np.array(all_event_indicators)
        
        # Identify indices for training and test samples
        train_indices = np.where(all_sources == 'train')[0]
        test_indices = np.where(all_sources == 'test')[0]
        
        if len(train_indices) > 0:
            train_risk_scores = all_predicted_risk_scores[train_indices]
            train_event_times = all_event_times[train_indices]
            train_event_indicators = all_event_indicators[train_indices]
            train_cindex = concordance_index(train_event_times, train_risk_scores, event_observed=train_event_indicators)
            print(f"\nTraining CSV Only Concordance Index: {train_cindex:.4f}")
        else:
            print("\nNo training CSV samples found in the CV folds.")
            
        if len(test_indices) > 0:
            test_risk_scores = all_predicted_risk_scores[test_indices]
            test_event_times = all_event_times[test_indices]
            test_event_indicators = all_event_indicators[test_indices]
            test_cindex = concordance_index(test_event_times, test_risk_scores, event_observed=test_event_indicators)
            print(f"Test CSV Only Concordance Index: {test_cindex:.4f}")
        else:
            print("No test CSV samples found in the CV folds.")

    # Optionally, plot CV metrics using your existing plotting functions
    plot_cv_metrics(np.array(all_predicted_risk_scores), np.array(all_event_times), np.array(all_event_indicators), args.output_dir)


def main(args):
    if args.cross_validation:
        cross_validation_mode(args)
    else:
        # For non-cross-validation mode, test CSV must be provided.
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
        print(f"Final Concordance Index: {score:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CoxPH model with DINOv2 features")
    parser.add_argument("--train_dicom_root", type=str, default="/gpfs/data/mankowskilab/HCC/data/TCGA/manifest-4lZjKqlp5793425118292424834/TCGA-LIHC",
                         help="Path to the training DICOM directory.")
    parser.add_argument("--test_dicom_root", type=str, default="/gpfs/data/mankowskilab/HCC_Recurrence/dicom",
                         help="Path to the testing DICOM directory.")
    parser.add_argument("--train_csv_file", type=str, default="/gpfs/data/shenlab/wz1492/HCC/spreadsheets/tcga.csv",
                         help="Path to the training CSV file.")
    parser.add_argument("--test_csv_file", type=str, default="",
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
                        help='Learning rate for the CoxPH model')
    parser.add_argument('--gradient_clip', type=float, default=0.1,
                        help='Gradient clipping threshold. Set 0 to disable.')
    parser.add_argument('--center_risk', action='store_true',
                        help='If set, center risk scores for numerical stability')
    parser.add_argument('--dropout', type=float, default=0.1,
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
    parser.add_argument('--cross_validation', action='store_true',
                        help="Enable cross validation mode")
    parser.add_argument('--cv_folds', type=int, default=10,
                        help="Number of cross validation folds")
    parser.add_argument('--leave_one_out', action='store_true',
                        help="Enable leave-one-out cross validation mode (combines CSVs and uses LOOCV)")
    
    args = parser.parse_args()

    # Create a unique subdirectory for each run using a timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    args.output_dir = os.path.join(args.output_dir, f"run_{timestamp}")
    os.makedirs(args.output_dir, exist_ok=True)
    
    main(args)
