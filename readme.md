# HCC

This is the repository for the Hepatocellular Carcinoma Recurrence Prediction project. This repository implements a survival analysis pipeline that leverages features extracted from DINOv2 and a Cox Proportional Hazards (CoxPH) model using either a custom multi-layer perceptron (MLP) or a linear layer. The code is organized into modular components for data handling, feature extraction, model building, and training, as well as dinov2 SSL code


## Dataset

patient 13850412 might not have axial

## Training

```
python main.py \
    --dicom_root /path/to/dicom \
    --csv_file processed_patient_labels.csv \
    --dinov2_weights /path/to/dinov2_weights.pth \
    --output_dir checkpoints \
    --batch_size 32 \
    --num_slices 2 \
    --epochs 20
```

---

## SSL 

- [**LLD-MMRI Dataset**](https://github.com/LMMMEng/LLD-MMRI-Dataset): An open-access dataset for liver lesion diagnosis on multi-phase MRI, comprising 498 cases with 7 different lesion types. (~220,000 images)
  
- [**AMOS Dataset**](https://arxiv.org/abs/2206.08023): A large-scale abdominal multi-organ benchmark for versatile medical image segmentation, including 500 CT and 100 MRI scans with annotations for 15 abdominal organs.
  
- [**LiverHCCSeg Dataset**](https://www.sciencedirect.com/science/article/pii/S2352340923007473) 17 HCC Cases

- [**CHAOS Dataset**](https://chaos.grand-challenge.org/): The Combined (CT-MR) Healthy Abdominal Organ Segmentation dataset, featuring 40 CT and 120 MRI volumes with annotations for liver, kidneys, and spleen.

With all combined, there are 330,000 images (MRI) in total.

### Classification Benchmark

- [**Duke Liver Dataset**](https://scholars.duke.edu/publication/1589665): A publicly available liver MRI dataset with liver segmentation masks and series labels, consisting of 2,146 abdominal MRI series from 105 patients, including 310 image series with corresponding manually segmented liver masks.

Dinov2 official evaluation code trains **52 linear classifiers** on top of frozen DINOv2 features to evaluate the model's representation quality. These classifiers are trained with different configurations, combining:

1. **Number of Blocks Used**: Last 1 or last 4 blocks.
2. **Average Pooling**: With and without pooling.
3. **Learning Rates**: 13 different values.

After training, each classifier's performance is evaluated, and the best one is selected based on accuracy.

### Our Task Dataset

- NYU Internal HCC Dataset

### How to Run SSL

```
python dinov2/run/train/train.py \
--config-file dinov2/configs/train/vitl14.yaml \
--wandb-project-name "Experiment-003-Teo" \
--output-dir "./models/Experiment_003" \
--ngpus 2 \
--timeout 10079 \
--partition a100_long,radiology \
train.dataset_path=UnlabeledMedicalImageDataset:root=/gpfs/data/mankowskilab/HCC/data/images
```

### How to Run Classification Benchmark

#### For Untrained Model

```
sbatch gpu_train_untrained.sbatch
```

#### For Pretrained Model

```
sbatch gpu_train_pretrained.sbatch
```

#### For our own SSL Model

```
sbatch gpu_train.sbatch
```

## Time to Event Model Training

