from dinov2.data.datasets import ImageNet,UnlabeledMedicalImageDataset, CustomMRIClassificationDataset

for split in ImageNet.Split:
    dataset = CustomMRIClassificationDataset(split=split, root="/gpfs/data/mankowskilab/HCC/data/Series_Classification", extra="/gpfs/data/mankowskilab/HCC/data/Series_Classification")
    dataset.dump_extra()