import argparse
import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold, LeaveOneOut
import datetime
from tqdm import tqdm

# Custom module imports
from data.dataset import HCCDataModule
from models.dino import load_dinov2_model
# (The following import is from our previous survival code; if not used, it can be removed)
from utils.plotting import plot_cv_metrics  

def extract_features(data_loader, model, device):
    """
    Extract features using the DINOv2 model.
    Each slice will have its own feature vector (no averaging on the slice level).
    """
    model.eval()
    features = []
    events = []  # previously 't' and 'e', now only 'e' is returned

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Extracting Features"):
            images, e = batch  # Updated: only two values unpacked
            images = images.to(device)
            # Unpack the 6 dimensions
            batch_size, num_samples, num_slices, C, H, W = images.size()
            # Combine batch, num_samples, and num_slices dimensions for feature extraction
            images = images.view(batch_size * num_samples * num_slices, C, H, W)
            
            feats = model.forward_features(images)
            feature_dim = feats.size(-1)
            # Reshape back: each sample now has num_slices feature vectors, one per slice
            feats = feats.view(batch_size, num_samples, num_slices, feature_dim)

            features.append(feats.cpu().numpy())
            events.append(e.cpu().numpy())

    features = np.concatenate(features, axis=0)
    events = np.concatenate(events, axis=0)
    return features, events

#############################################
# Plotting functions for ROC curve & loss  #
#############################################

def plot_roc_curve(y_true, y_scores, output_dir, fold_id=""):
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    auc_value = roc_auc_score(y_true, y_scores)
    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {auc_value:.4f})')
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve {fold_id}')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(output_dir, f'roc_curve_{fold_id}.png'))
    plt.close()

def plot_training_loss(losses, output_dir, fold_id=""):
    plt.figure()
    plt.plot(losses, label="Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Training Loss Curve {fold_id}")
    plt.legend()
    plt.savefig(os.path.join(output_dir, f"training_loss_{fold_id}.png"))
    plt.close()

#############################################
#  Upsampling function for classification   #
#############################################

def upsample_training_data_classifier(x, y):
    """
    Upsamples the minority class so that both classes have equal number of samples.
    x: numpy array of shape (n_samples, feature_dim)
    y: numpy array of shape (n_samples,)
    """
    idx_0 = np.where(y == 0)[0]
    idx_1 = np.where(y == 1)[0]
    if len(idx_0) == 0 or len(idx_1) == 0:
        return x, y
    # Determine the minority class indices.
    if len(idx_0) < len(idx_1):
        minority = idx_0
    else:
        minority = idx_1
    n_to_sample = abs(len(idx_0) - len(idx_1))
    sampled_indices = np.random.choice(minority, size=n_to_sample, replace=True)
    x_upsampled = np.concatenate([x, x[sampled_indices]], axis=0)
    y_upsampled = np.concatenate([y, y[sampled_indices]], axis=0)
    return x_upsampled, y_upsampled

#############################################
#       Binary Classifier Model             #
#############################################

class BinaryClassifier(torch.nn.Module):
    def __init__(self, in_features):
        super(BinaryClassifier, self).__init__()
        self.linear = torch.nn.Linear(in_features, 1)
        
    def forward(self, x):
        # Output logits; we will use BCEWithLogitsLoss
        logits = self.linear(x)
        return logits

def upsample_df(df, target_column='event'):
    df_majority = df[df[target_column] == 0]
    df_minority = df[df[target_column] == 1]
    if len(df_minority) == 0 or len(df_majority) == 0:
        return df
    df_minority_upsampled = df_minority.sample(len(df_majority), replace=True, random_state=1)
    return pd.concat([df_majority, df_minority_upsampled]).reset_index(drop=True)

#############################################
#    Training and Evaluation Function       #
#############################################

def train_and_evaluate(args, train_csv, val_csv, hyperparams, fold_id="", 
                       final_eval=True, return_predictions=False):
    """
    Trains a simple linear binary classifier using DINOv2 features to predict recurrence.
    Computes metrics like AUC and plots the ROC curve and training loss.
    """
    # For reproducibility
    torch.manual_seed(42)
    
    data_module = HCCDataModule(
        train_csv_file=train_csv,
        test_csv_file=val_csv,
        train_dicom_root=args.train_dicom_root,
        test_dicom_root=args.test_dicom_root,
        model_type="linear",
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
    # Register a dummy forward hook (if needed)
    dino_model.register_forward_hook(lambda m, i, o: None)
    
    # Obtain DataLoaders for training and validation
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()
    
    # Extract DINOv2 features.
    # For binary classification, we expect extract_features to return two outputs: (features, labels)
    x_train, y_train = extract_features(train_loader, dino_model, device)
    x_val, y_val = extract_features(val_loader, dino_model, device)
    
    # Average across slices (shape becomes: (n_samples, feature_dim))
    x_train = x_train.mean(axis=(1, 2))
    x_val = x_val.mean(axis=(1, 2))

    # Remove any zero-variance features to avoid issues during scaling.
    variances = np.var(x_train, axis=0)
    if np.any(variances == 0):
        zero_var_features = np.where(variances == 0)[0]
        print(f"[Fold {fold_id}] Warning: Features with zero variance: {zero_var_features}")
        x_train = x_train[:, variances != 0]
        x_val = x_val[:, variances != 0]
    
    # If upsampling is requested, balance the classes in the training data.
    if args.upsampling:
        print(f"[Fold {fold_id}] Performing upsampling on training data...")
        x_train, y_train = upsample_training_data_classifier(x_train, y_train)
    
    # Standardize features (fit only on training data)
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train).astype("float32")
    x_val = scaler.transform(x_val).astype("float32")
    
    # Ensure labels are numpy arrays of type float32.
    y_train = np.array(y_train).astype("float32")
    y_val = np.array(y_val).astype("float32")
    
    in_features = x_train.shape[1]
    
    # Build the binary classifier model.
    model = BinaryClassifier(in_features)
    model = model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=hyperparams['learning_rate'])
    criterion = torch.nn.BCEWithLogitsLoss()
    
    # Create PyTorch datasets and dataloaders.
    train_dataset = torch.utils.data.TensorDataset(torch.tensor(x_train), torch.tensor(y_train))
    val_dataset = torch.utils.data.TensorDataset(torch.tensor(x_val), torch.tensor(y_val))
    
    train_loader_cls = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader_cls = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    num_epochs = args.epochs
    training_losses = []
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        epoch_losses = []
        for batch_x, batch_y in train_loader_cls:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device).unsqueeze(1)  # shape: (batch_size, 1)
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            if args.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
            optimizer.step()
            epoch_losses.append(loss.item())
        avg_loss = np.mean(epoch_losses)
        training_losses.append(avg_loss)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"[Fold {fold_id}] Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
    
    # Evaluation on the validation set.
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
    
    auc_score = roc_auc_score(all_labels, all_probs)
    print(f"[Fold {fold_id}] Validation AUC: {auc_score:.4f}")
    
    if final_eval:
        os.makedirs(args.output_dir, exist_ok=True)
        plot_roc_curve(all_labels, all_probs, args.output_dir, fold_id=fold_id)
        plot_training_loss(training_losses, args.output_dir, fold_id=fold_id)
    
    if return_predictions:
        return np.array(all_probs), np.array(all_labels)
    else:
        return auc_score

#############################################
#         Cross-Validation Mode             #
#############################################

def cross_validation_mode(args):
    # Read the training CSV (and test CSV if provided)
    df_train_full = pd.read_csv(args.train_csv_file)
    df_train_full["dicom_root"] = args.train_dicom_root

    if args.test_csv_file:
        df_test_full = pd.read_csv(args.test_csv_file)
        df_test_full["dicom_root"] = args.test_dicom_root
    else:
        df_test_full = None
        print("No test CSV provided. Performing cross validation on the training dataset only.")
    
    all_predicted_probs = []
    all_labels = []
    
    hyperparams = {"learning_rate": args.learning_rate}
    
    if args.leave_one_out:
        # For LOOCV, combine data (if test CSV exists, combine it with train CSV)
        if df_test_full is not None:
            df_all = pd.concat([df_train_full, df_test_full]).reset_index(drop=True)
        else:
            df_all = df_train_full
        print(f"[LOO CV] Combined dataset has {len(df_all)} patients.")
        loo = LeaveOneOut()
        splits = list(loo.split(df_all))
        for fold, (train_idx, test_idx) in enumerate(splits):
            df_new_train = df_all.iloc[train_idx].reset_index(drop=True)
            df_new_test = df_all.iloc[test_idx].reset_index(drop=True)
            print(f"[LOO Fold {fold}] Train patients: {len(df_new_train)}, Positive events: {df_new_train['event'].sum()}")
            print(f"[LOO Fold {fold}] Test patient: {len(df_new_test)}, Positive events: {df_new_test['event'].sum()}")
            if args.upsampling:
                print(f"[LOO Fold {fold}] Performing upsampling on training data...")
                # Here we use an external helper upsample_df for dataframes.
                df_new_train = upsample_df(df_new_train, target_column="event")
            
            preds, labels = train_and_evaluate(
                args,
                train_csv=df_new_train,
                val_csv=df_new_test,
                hyperparams=hyperparams,
                fold_id=f"LOO_fold_{fold}",
                final_eval=False,
                return_predictions=True
            )
            all_predicted_probs.extend(preds.tolist())
            all_labels.extend(labels.tolist())
    else:
        if df_test_full is not None:
            skf_train = StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=1)
            skf_test = StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=1)
            train_splits = list(skf_train.split(df_train_full, df_train_full["event"]))
            test_splits = list(skf_test.split(df_test_full, df_test_full["event"]))
            for fold in range(args.cv_folds):
                train_train_idx, _ = train_splits[fold]
                test_train_idx, test_test_idx = test_splits[fold]
                
                df_train_train = df_train_full.iloc[train_train_idx].reset_index(drop=True)
                df_test_fold = df_test_full.iloc[test_test_idx].reset_index(drop=True)
                
                print(f"[CV Fold {fold}] df_train_train: {len(df_train_train)} patients, Positive events: {df_train_train['event'].sum()}")
                print(f"[CV Fold {fold}] df_test_fold: {len(df_test_fold)} patients, Positive events: {df_test_fold['event'].sum()}")
                
                if args.upsampling:
                    print(f"[CV Fold {fold}] Performing upsampling on training subset if needed...")
                    df_train_train = upsample_df(df_train_train, target_column="event")
                
                preds, labels = train_and_evaluate(
                    args,
                    train_csv=df_train_train,
                    val_csv=df_test_fold,
                    hyperparams=hyperparams,
                    fold_id=f"fold_{fold}_test",
                    final_eval=False,
                    return_predictions=True
                )
                all_predicted_probs.extend(preds.tolist())
                all_labels.extend(labels.tolist())
        else:
            skf = StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=1)
            for fold, (train_idx, test_idx) in enumerate(skf.split(df_train_full, df_train_full["event"])):
                df_new_train = df_train_full.iloc[train_idx].reset_index(drop=True)
                df_new_test = df_train_full.iloc[test_idx].reset_index(drop=True)
                print(f"[CV Fold {fold}] Train: {len(df_new_train)} patients, Positive events: {df_new_train['event'].sum()}")
                print(f"[CV Fold {fold}] Test: {len(df_new_test)} patients, Positive events: {df_new_test['event'].sum()}")
                
                preds, labels = train_and_evaluate(
                    args,
                    train_csv=df_new_train,
                    val_csv=df_new_test,
                    hyperparams=hyperparams,
                    fold_id=f"fold_{fold}_test",
                    final_eval=False,
                    return_predictions=True
                )
                all_predicted_probs.extend(preds.tolist())
                all_labels.extend(labels.tolist())
    
    overall_auc = roc_auc_score(all_labels, np.array(all_predicted_probs))
    print(f"Overall AUC from CV: {overall_auc:.4f}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    plot_roc_curve(all_labels, all_predicted_probs, args.output_dir, fold_id="aggregated")

#############################################
#                Main Function              #
#############################################

def main(args):
    if args.cross_validation:
        cross_validation_mode(args)
    else:
        auc_score = train_and_evaluate(
            args,
            args.train_csv_file,
            args.test_csv_file,
            hyperparams={"learning_rate": args.learning_rate},
            fold_id="full",
            final_eval=True,
            return_predictions=False
        )
        print(f"Final Validation AUC: {auc_score:.4f}")

#############################################
#          Argument Parsing & Run           #
#############################################

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a simple linear binary classifier with DINOv2 features to predict recurrence"
    )
    parser.add_argument("--train_dicom_root", type=str, 
                        default="/gpfs/data/mankowskilab/HCC/data/TCGA/manifest-4lZjKqlp5793425118292424834/TCGA-LIHC",
                        help="Path to the test DICOM directory.")
    parser.add_argument("--test_dicom_root", type=str, 
                        default="/gpfs/data/mankowskilab/HCC_Recurrence/dicom",
                        help="Path to the training DICOM directory.")
    parser.add_argument("--test_csv_file", type=str, default="/gpfs/data/shenlab/wz1492/HCC/spreadsheets/processed_patient_labels_nyu.csv",
                        help="Path to the test CSV file for evaluation.")
    parser.add_argument("--train_csv_file", type=str, 
                        default="/gpfs/data/shenlab/wz1492/HCC/spreadsheets/tcga.csv",
                        help="Path to the train CSV file.")
    parser.add_argument("--preprocessed_root", type=str, default=None, 
                        help="Directory to store/load preprocessed image tensors")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for data loaders")
    parser.add_argument("--num_slices", type=int, default=64,
                        help="Number of slices per patient")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of workers for data loaders")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of training epochs")
    parser.add_argument("--output_dir", type=str, default="checkpoints",
                        help="Base directory to save outputs and models")
    parser.add_argument("--learning_rate", type=float, default=1e-3,
                        help="Learning rate for the binary classifier")
    parser.add_argument("--gradient_clip", type=float, default=0.05,
                        help="Gradient clipping threshold. Set 0 to disable.")
    parser.add_argument("--upsampling", action="store_true",
                        help="If set, perform upsampling of the minority class in the training data")
    parser.add_argument("--cross_validation", action="store_true",
                        help="Enable cross validation mode")
    parser.add_argument("--cv_folds", type=int, default=5,
                        help="Number of cross validation folds")
    parser.add_argument("--leave_one_out", action="store_true",
                        help="Enable leave-one-out cross validation mode (combines CSVs and uses LOOCV)")
    parser.add_argument("--num_samples_per_patient", type=int, default=1,
                        help="Number of times to sample from each patient per epoch")
    parser.add_argument("--dinov2_weights", type=str, required=True,
                        help="Path to your local DINOv2 state dict file (.pth or .pt).")
    
    args = parser.parse_args()
    
    # Create a unique subdirectory for each run using a timestamp.
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    args.output_dir = os.path.join(args.output_dir, f"run_{timestamp}")
    os.makedirs(args.output_dir, exist_ok=True)
    
    main(args)
