from dinov2.data.datasets import ImageNet,UnlabeledMedicalImageDataset

for split in ImageNet.Split:
    dataset = ImageNet(split=split, root="/gpfs/data/mankowskilab/HCC/data/images", extra="/gpfs/data/mankowskilab/HCC/data/images")
    dataset.dump_extra()