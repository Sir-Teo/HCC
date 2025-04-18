# HCC: Hepatocellular Carcinoma Recurrence Prediction

**A modular pipeline for time-to-event survival analysis and binary classification using DINOv2 features**

---

## Table of Contents

- [Project Overview](#project-overview)
- [Repository Structure](#repository-structure)
- [Datasets](#datasets)
- [Installation](#installation)
- [Quick Start](#quick-start)
  - [Survival Analysis Training](#survival-analysis-training)
  - [Binary Classification Training](#binary-classification-training)
  - [Embedding Visualization](#embedding-visualization)
- [Command-Line Interfaces](#command-line-interfaces)
- [Notebooks](#notebooks)

---

## Project Overview

This repository implements a pipeline for predicting hepatocellular carcinoma recurrence using features extracted from frozen DINOv2 vision transformer representations. We support:

1. **Survival Analysis (Time-to-Event)**: Train a Cox Proportional Hazards (CoxPH) model (MLP or linear) on patient-level features.
2. **Binary Classification**: Predict recurrence as a binary event using a simple classifier on top of DINOv2 features.
3. **Embedding Visualization**: Project high-dimensional features to 2D (PCA, t-SNE, UMAP) and inspect clusters/outliers.

Key features:
- Modular data handling via `data.dataset.HCCDataModule` and `HCCDicomDataset`.
- Flexible training scripts: `train.py` (survival), `train_binary.py` (classification), `visualize.py`.
- Built-in cross-validation, upsampling support, and comprehensive plotting utilities.
- Seamless integration of DINOv2 for feature extraction.

---

## Repository Structure

```
HCC/
├── data/                   # Data loading and preprocessing modules
│   └── dataset.py          # HCCDataModule, DICOM dataset loaders
├── models/                 # Model definitions
│   ├── dino.py             # Utilities to load and wrap DINOv2
│   └── mlp.py              # Custom MLP and CoxPH-with-L1 implementations
├── utils/                  # Plotting and helper utilities
├── train.py                # CLI for survival analysis training
├── train_binary.py         # CLI for binary classification training
├── visualize.py            # CLI for embedding visualization
├── dinov2/                 # DINOv2 source and training code (SSL)
├── notebooks/              # Notebooks for EDA and examples
└── readme.md               # This file
```

---

## Datasets

- **NYU Internal HCC**: Retrospective series of HCC patient CT volumes with recurrence labels.
- **TCGA-LIHC**: Public liver cancer CT dataset with survival follow-up.
- **External SSL Sources** (in `dinov2/`): LLD-MMRI, AMOS, LiverHCCSeg, CHAOS for self-supervised pretraining.

> **Note**: Ensure DICOM volumes are organized according to the provided CSV metadata.

---

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/your_org/HCC.git
   cd HCC
   ```

2. **Create and activate a Python environment**
   ```bash
   conda create -n hcc_env python=3.10 -y
   conda activate hcc_env
   ```

3. **Install DINOv2 dependencies** (for feature extraction and SSL):
   ```bash
   cd dinov2
   pip install -r requirements.txt
   pip install -r requirements-extras.txt
   pip install -e .
   cd ..
   ```

4. **Install HCC project dependencies**:
   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn torchtuples pycox lifelines tqdm imbalanced-learn umap-learn
   ```

---

## Quick Start

### Survival Analysis Training

Train a time-to-event CoxPH model with cross-validation and optional upsampling:
```bash
python train.py \
  --tcga_dicom_root /path/to/TCGA/DICOM \
  --nyu_dicom_root /path/to/NYU/DICOM \
  --tcga_csv_file /path/to/tcga.csv \
  --nyu_csv_file /path/to/nyu.csv \
  --dinov2_weights /path/to/dinov2_weights.pth \
  --preprocessed_root /path/to/preprocessed \
  --output_dir checkpoints_survival_cv \
  --batch_size 16 \
  --num_slices 0 \
  --epochs 100 \
  --cv_folds 10 \
  --coxph_net mlp \
  --learning_rate 1e-5 \
  --upsampling
```

### Binary Classification Training

Train a binary classifier to predict recurrence:
```bash
python train_binary.py \
  --tcga_dicom_root /path/to/TCGA/DICOM \
  --nyu_dicom_root /path/to/NYU/DICOM \
  --tcga_csv_file /path/to/tcga.csv \
  --nyu_csv_file /path/to/nyu.csv \
  --dinov2_weights /path/to/dinov2_weights.pth \
  --preprocessed_root /path/to/preprocessed \
  --output_dir checkpoints_binary_cv \
  --batch_size 16 \
  --num_slices 0 \
  --epochs 100 \
  --cv_folds 10 \
  --learning_rate 1e-5 \
  --upsampling
```

### Embedding Visualization

Project DINOv2 features into 2D to inspect patient clusters and outliers:
```bash
python visualize.py \
  --train_dicom_root /path/to/TCGA/DICOM \
  --test_dicom_root /path/to/NYU/DICOM \
  --train_csv_file /path/to/tcga.csv \
  --test_csv_file /path/to/nyu.csv \
  --dinov2_weights /path/to/dinov2_weights.pth \
  --output_dir visualization_outputs \
  --batch_size 32 \
  --num_slices 0 \
  --num_samples_per_patient 1
```

---

## Command-Line Interfaces

Each script provides `--help` for full argument details:

- **Survival**: `python train.py --help`
- **Classification**: `python train_binary.py --help`
- **Visualization**: `python visualize.py --help`

---

## Notebooks

Explore examples and EDA in the `notebooks/` directory:
- `EDA.ipynb`, `survival.ipynb`, `semantic_segmentation.ipynb`, etc.

---

