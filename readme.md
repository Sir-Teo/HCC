# HCC

This is the repository for the Hepatocellular Carcinoma Reoccurance Prediction project.

## How to Run

```
python dinov2/run/train/train.py     --nodes 1     --config-file dinov2/configs/train/vitl16_short.yaml     train.dataset_path=UnlabeledMedicalImageDataset:root=/gpfs/data/mankowskilab/HCC/data/images     output_dir=/gpfs/data/shenlab/wz1492/HCC/dinov2/experiments2
```

## Dataset

### SSL

- LLD dataset
- Amos
- LiverSegHCC
- CHAOS

### Classification Benchmark

- Duke Liver Dataset

### Our Task Dataset

- NYU Internal HCC Dataset