import os
import pydicom
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from mmseg.models import build_segmentor
from mmcv import Config
from mmseg.datasets import build_dataset
from mmseg.datasets.builder import DATASETS
from mmseg.datasets.custom import CustomDataset
from mmseg.apis import train_segmentor, inference_segmentor, single_gpu_test
import matplotlib.pyplot as plt
import mmcv
from mmcv.runner import load_checkpoint
from mmseg.datasets import build_dataloader
from mmseg.datasets.pipelines import Compose
from mmseg.datasets.builder import PIPELINES
from mmseg.datasets.pipelines import LoadAnnotations

import torch.nn as nn
from collections import OrderedDict


class DinoV2Backbone(nn.Module):
    def __init__(self, dino_model, patch_size=14):
        super().__init__()
        self.dino_model = dino_model
        self.patch_size = patch_size

    def forward(self, x):
        # Forward through DINOv2 to get features
        features = self.dino_model.forward_features(x)

        
        patch_tokens = features['x_norm_patchtokens']  # shape [N, 256, 768]

        N, P, C = patch_tokens.shape
        # Check that P = 256
        H = W = int(P**0.5)  # H = W = 16

        # Rearrange from [N, P, C] to [N, C, H, W]
        patch_tokens = patch_tokens.permute(0, 2, 1).reshape(N, C, H, W)

        
        return [patch_tokens]



# --------------------------------------
# 1. Dataset Loader for DICOM Images (If needed)
# --------------------------------------
class DICOMLiverDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform=None, target_transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]
        
        dicom_image = pydicom.dcmread(image_path)
        image = dicom_image.pixel_array.copy()  # Ensure writable
        dicom_mask = pydicom.dcmread(mask_path)
        mask = dicom_mask.pixel_array.copy()  # Ensure writable
        
        # Normalize image
        image = (image - np.min(image)) / (np.max(image) - np.min(image)) * 255.0
        image = np.stack([image] * 3, axis=-1)  # Convert to RGB

        image = Image.fromarray(image.astype(np.uint8))
        mask = Image.fromarray(mask.astype(np.uint8))

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            mask = self.target_transform(mask)

        return image, mask


# --------------------------------------
# 2. Data Preparation
# --------------------------------------
def prepare_data(root_dir, test_size=0.2):
    image_paths = []
    mask_paths = []

    for root, dirs, files in os.walk(root_dir):
        if "images" in dirs and "masks" in dirs:
            images_dir = os.path.join(root, "images")
            masks_dir = os.path.join(root, "masks")
            
            for file in sorted(os.listdir(images_dir)):
                if file.endswith(".dicom"):
                    image_paths.append(os.path.join(images_dir, file))
                    mask_paths.append(os.path.join(masks_dir, file))

    if not image_paths or not mask_paths:
        raise ValueError(f"No images or masks found in {root_dir}. Check the directory structure and file extensions.")

    train_image_paths, test_image_paths, train_mask_paths, test_mask_paths = train_test_split(
        image_paths, mask_paths, test_size=test_size, random_state=42
    )

    return train_image_paths, train_mask_paths, test_image_paths, test_mask_paths

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

target_transform = transforms.Compose([
    transforms.Resize((224, 224), interpolation=Image.NEAREST),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.long())  # Ensure correct type for the loss function
])


root_dir = "/gpfs/scratch/wz1492/Segmentation"  
train_image_paths, train_mask_paths, test_image_paths, test_mask_paths = prepare_data(root_dir)

# --------------------------------------
# 3. Load Pretrained DINOv2 Backbone
# --------------------------------------
BACKBONE_SIZE = "base"
backbone_archs = {
    "small": "vits14",
    "base": "vitb14",
    "large": "vitl14",
    "giant": "vitg14",
}
backbone_arch = backbone_archs[BACKBONE_SIZE]
backbone_model = torch.hub.load("facebookresearch/dinov2", f"dinov2_{backbone_arch}")
backbone_model.eval().cuda()

# --------------------------------------
# 4. Create Segmentation Model Config
# --------------------------------------
HEAD_NUM_CLASSES = 2

cfg = Config({
    'model': {
        'type': 'EncoderDecoder',
        'pretrained': None,
        'backbone': {
            'type': 'VisionTransformer',
            'img_size': (224, 224),
            'embed_dims': 768,
            'num_layers': 12,
            'num_heads': 12,
            'patch_size': 14,
            'in_channels': 3,
            'drop_rate': 0.0,
            'attn_drop_rate': 0.0,
            'drop_path_rate': 0.1,
            'with_cls_token': True,
            'output_cls_token': False,
            'norm_cfg': {'type': 'LN', 'eps': 1e-6},
            'act_cfg': {'type': 'GELU'},
            'final_norm': False,
            'out_indices': (11,),
        },
        'decode_head': {
            'type': 'FCNHead',
            'in_channels': 768,
            'in_index': 0,
            'channels': 512,
            'num_convs': 2,
            'concat_input': False,
            'dropout_ratio': 0.1,
            'num_classes': HEAD_NUM_CLASSES,
            'norm_cfg': {'type': 'BN', 'requires_grad': True},
            'align_corners': False,
            'loss_decode': {
                'type': 'CrossEntropyLoss',
                'use_sigmoid': False,
                'loss_weight': 1.0
            },
        },
        'train_cfg': {},
        'test_cfg': {'mode': 'whole'}
    },
    'dataset_type': 'DICOMLiverSegmentationDataset',
    'data_root': root_dir,
    'img_norm_cfg': {
        'mean': [123.675, 116.28, 103.53],
        'std': [58.395, 57.12, 57.375],
        'to_rgb': True
    },
    'seed': 42,
    'gpu_ids': range(1),
})

# --------------------------------------
# Custom Pipeline Steps for DICOM
# --------------------------------------
@PIPELINES.register_module()
class LoadDicomImageFromFile:
    def __call__(self, results):
        # Load the image
        dicom = pydicom.dcmread(results['img_info']['filename'])
        image = dicom.pixel_array.copy()
        # Normalize
        image = (image - image.min()) / (image.max() - image.min()) * 255.0
        image = image.astype(np.uint8)

        # If grayscale, convert to 3-channel
        if image.ndim == 2:
            image = np.stack([image, image, image], axis=-1)

        results['filename'] = results['img_info']['filename']
        results['ori_filename'] = results['img_info']['filename']
        results['img'] = image
        results['img_shape'] = image.shape[:2]     # (H, W)
        results['ori_shape'] = image.shape[:2]     # (H, W)
        results['pad_shape'] = image.shape[:2]     # (H, W)

        results['scale_factor'] = 1.0
        results['img_norm_cfg'] = dict(
            mean=np.array([123.675, 116.28, 103.53], dtype=np.float32),
            std=np.array([58.395, 57.12, 57.375], dtype=np.float32),
            to_rgb=True
        )
        return results
    
@PIPELINES.register_module()
class LoadDicomAnnotations:
    def __call__(self, results):
        seg_map_filename = results['ann_info']['seg_map']
        dicom = pydicom.dcmread(seg_map_filename)
        
        # Convert to numpy array and ensure proper scaling
        mask = dicom.pixel_array
        
        # Debug print statements
        print(f"Mask unique values before processing: {np.unique(mask)}")
        
        # Ensure binary mask (0 and 1 values)
        mask = (mask > 0).astype(np.int64)
        
        print(f"Mask unique values after processing: {np.unique(mask)}")
        print(f"Mask shape: {mask.shape}")
        
        # Add channel dimension if needed
        if mask.ndim == 2:
            mask = mask[..., None]
        
        results['gt_semantic_seg'] = mask
        
        if 'seg_fields' not in results:
            results['seg_fields'] = []
        results['seg_fields'].append('gt_semantic_seg')
        
        return results




# --------------------------------------
# 5. Update the Data Pipeline to Use DICOM Loaders
# --------------------------------------
# Set up training pipeline
cfg.train_pipeline = [
    dict(type='LoadDicomImageFromFile'),
    dict(type='LoadDicomAnnotations'),
    dict(type='Resize', img_scale=(224, 224), keep_ratio=False),
    dict(type='RandomFlip', prob=0.5),
    dict(type='Normalize', **cfg.img_norm_cfg),
    dict(type='Pad', size=(224, 224), pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
# Set up test pipeline
cfg.test_pipeline = [
    dict(type='LoadDicomImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(224, 224),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=False),
            dict(type='RandomFlip'),
            dict(type='Normalize', **cfg.img_norm_cfg),
            dict(type='Pad', size=(224, 224), pad_val=0, seg_pad_val=255),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]




# --------------------------------------
# 6. Implement the Custom Dataset
# --------------------------------------
@DATASETS.register_module()
class DICOMLiverSegmentationDataset(CustomDataset):
    CLASSES = ('background', 'liver')
    PALETTE = [[0, 0, 0], [255, 0, 0]]

    def __init__(self, img_paths, mask_paths, pipeline, **kwargs):
        self.img_paths = img_paths
        self.mask_paths = mask_paths
        self.img_infos = self.load_annotations()
        
        super().__init__(
            img_dir='',
            ann_dir='',
            img_suffix='',
            seg_map_suffix='',
            reduce_zero_label=False,
            pipeline=pipeline,
            classes=self.CLASSES,
            palette=self.PALETTE,
            **kwargs
        )

    def load_annotations(self, img_dir=None, img_suffix=None, ann_dir=None, seg_map_suffix=None, split=None):
        img_infos = []
        for img_path, ann_path in zip(self.img_paths, self.mask_paths):
            if not os.path.exists(img_path):
                raise FileNotFoundError(f"Image file not found: {img_path}")
            if not os.path.exists(ann_path):
                raise FileNotFoundError(f"Annotation file not found: {ann_path}")
            
            img_info = dict(
                filename=img_path,
                ann=dict(seg_map=ann_path)
            )
            img_infos.append(img_info)
        return img_infos

    def get_ann_info(self, idx):
        return self.img_infos[idx]['ann']

    def pre_pipeline(self, results):
        results['seg_fields'] = []
        results['img_prefix'] = None
        results['seg_prefix'] = None

    def get_gt_seg_map_by_idx(self, index):
        """Get ground truth segmentation map by index."""
        ann_info = self.get_ann_info(index)
        seg_map = pydicom.dcmread(ann_info['seg_map']).pixel_array
        return seg_map

    def pre_eval(self, preds, indices):
        """Pre-evaluation for metrics calculation with shape correction."""
        if not isinstance(indices, list):
            indices = [indices]
        if not isinstance(preds, list):
            preds = [preds]
            
        pre_eval_results = []
        
        for pred, index in zip(preds, indices):
            # Get ground truth segmentation map
            seg_map = self.get_gt_seg_map_by_idx(index)
            
            # Ensure prediction and ground truth have same shape
            if pred.shape != seg_map.shape:
                # First resize prediction to match ground truth dimensions
                pred = mmcv.imresize(
                    pred.astype(np.uint8),
                    seg_map.shape[::-1],  # Note: imresize expects (width, height)
                    interpolation='nearest'
                )
                
                # Ensure the arrays have the same orientation
                if pred.shape != seg_map.shape:
                    pred = pred.T
            
            # Calculate metrics for each class
            class_results = []
            for class_id in range(len(self.CLASSES)):
                pred_mask = (pred == class_id)
                gt_mask = (seg_map == class_id)
                
                # Ensure masks have the same shape
                assert pred_mask.shape == gt_mask.shape, \
                    f"Shape mismatch: pred_mask {pred_mask.shape} vs gt_mask {gt_mask.shape}"
                
                intersect = np.logical_and(pred_mask, gt_mask).sum()
                union = np.logical_or(pred_mask, gt_mask).sum()
                pred_label = pred_mask.sum()
                label = gt_mask.sum()
                
                class_results.append([intersect, union, pred_label, label])
                
            pre_eval_results.append(np.array(class_results))
            
        return pre_eval_results

    def evaluate(self, results, metric='mIoU', logger=None, **kwargs):
        """Evaluate the dataset."""
        if isinstance(metric, str):
            metric = [metric]
        allowed_metrics = ['mIoU', 'mDice']
        if not set(metric).issubset(set(allowed_metrics)):
            raise KeyError(f'metric {metric} is not supported')

        eval_results = {}
        
        # Get pre-eval results for all predictions
        pre_eval_results = self.pre_eval(results, list(range(len(self))))
        
        # Calculate metrics
        ret_metrics = pre_eval_to_metrics(
            pre_eval_results=pre_eval_results,
            metrics=metric,
            num_classes=len(self.CLASSES))

        # Format metrics for return
        formatted_metrics = {}
        for key, value in ret_metrics.items():
            if isinstance(value, np.ndarray):
                formatted_metrics[key] = value.tolist()
            else:
                formatted_metrics[key] = float(value)
        
        eval_results.update(formatted_metrics)
        
        return eval_results

def pre_eval_to_metrics(pre_eval_results, metrics, num_classes):
    """Convert pre-eval results to metrics."""
    # Extract the actual pre-evaluation results
    pre_eval_results = pre_eval_results[0]  # Get the first element since it's the sum of all results
    
    # Ensure we have the correct shape
    assert pre_eval_results.shape[0] == num_classes, \
        f"pre_eval_results should have shape (num_classes, 4), but got {pre_eval_results.shape}"
    
    total_area_intersect = pre_eval_results[:, 0]
    total_area_union = pre_eval_results[:, 1]
    total_area_pred_label = pre_eval_results[:, 2]
    total_area_label = pre_eval_results[:, 3]
    
    all_acc = total_area_intersect.sum() / total_area_label.sum()
    ret_metrics = OrderedDict({'aAcc': all_acc})
    
    for metric in metrics:
        if metric == 'mIoU':
            iou = total_area_intersect / np.maximum(total_area_union, 1)
            acc = total_area_intersect / np.maximum(total_area_label, 1)
            ret_metrics['IoU'] = iou
            ret_metrics['Acc'] = acc
            ret_metrics['mIoU'] = np.nanmean(iou)
            ret_metrics['mAcc'] = np.nanmean(acc)
        elif metric == 'mDice':
            dice = 2 * total_area_intersect / np.maximum(
                total_area_pred_label + total_area_label, 1)
            ret_metrics['Dice'] = dice
            ret_metrics['mDice'] = np.nanmean(dice)
    
    # Add per-class metrics
    for metric in metrics:
        if metric == 'mIoU':
            ret_metrics['IoU_per_class'] = ret_metrics.pop('IoU')
            ret_metrics['Acc_per_class'] = ret_metrics.pop('Acc')
        elif metric == 'mDice':
            ret_metrics['Dice_per_class'] = ret_metrics.pop('Dice')
    
    return ret_metrics
# --------------------------------------
# 7. Build the Dataset
# --------------------------------------
train_dataset = DICOMLiverSegmentationDataset(
    img_paths=train_image_paths,
    mask_paths=train_mask_paths,
    pipeline=cfg.train_pipeline
)

val_dataset = DICOMLiverSegmentationDataset(
    img_paths=test_image_paths,
    mask_paths=test_mask_paths,
    pipeline=cfg.test_pipeline
)

cfg.data = dict(
    samples_per_gpu=4,
    workers_per_gpu=2,
    train=dict(
        type='DICOMLiverSegmentationDataset',
        img_paths=train_image_paths,
        mask_paths=train_mask_paths,
        pipeline=cfg.train_pipeline,
    ),
    val=dict(
        type='DICOMLiverSegmentationDataset',
        img_paths=test_image_paths,
        mask_paths=test_mask_paths,
        pipeline=cfg.test_pipeline,
    ),
    test=dict(
        type='DICOMLiverSegmentationDataset',
        img_paths=test_image_paths,
        mask_paths=test_mask_paths,
        pipeline=cfg.test_pipeline,
    )
)
cfg.model['decode_head']['loss_decode'] = dict(
    type='CrossEntropyLoss',
    use_sigmoid=False,
    loss_weight=1.0,
    class_weight=[0.5, 2.0]  # Give more weight to liver class
)
# --------------------------------------
# 8. Build the Model
# --------------------------------------
model = build_segmentor(cfg.model)
model.init_weights()

# Attach the DINOv2 backbone
model.backbone = DinoV2Backbone(backbone_model, patch_size=14)

model.cuda()

# --------------------------------------
# 9. Update the Optimizer and Learning Policy
# --------------------------------------
cfg.log_level = 'INFO'
cfg.optimizer = {
    'type': 'AdamW',
    'lr': 0.00006,
    'weight_decay': 0.01,
    'betas': (0.9, 0.999),
}
cfg.optimizer_config = {'type': 'OptimizerHook', 'grad_clip': None}
cfg.lr_config = {
    'policy': 'poly',
    'power': 1.0,
    'min_lr': 1e-6,
    'by_epoch': False
}
cfg.runner = {
    'type': 'IterBasedRunner',
    'max_iters': 5000
}
cfg.checkpoint_config = {
    'by_epoch': False,
    'interval': 500
}
cfg.evaluation = {
    'interval': 500,
    'metric': 'mIoU',
    'pre_eval': True
}
cfg.log_config = {
    'interval': 50,
    'hooks': [
        {'type': 'TextLoggerHook'},
    ]
}
cfg.workflow = [('train', 1)]
cfg.device = 'cuda' if torch.cuda.is_available() else 'cpu'
cfg.work_dir = './work_dirs/dicom_liver_segmentation'
cfg.resume_from = None
cfg.load_from = None

# --------------------------------------
# 10. Training the Model
# --------------------------------------
datasets = [train_dataset]
train_segmentor(model, datasets, cfg, distributed=False, validate=True)

# --------------------------------------
# 11. Inference and Visualization
# --------------------------------------
model.eval()
test_image_path = test_image_paths[0]
dicom_test = pydicom.dcmread(test_image_path)
image = dicom_test.pixel_array.copy()
image = (image - np.min(image)) / (np.max(image) - np.min(image)) * 255.0
image = np.stack([image] * 3, axis=-1)
image_pil = Image.fromarray(image.astype(np.uint8))
image_tensor = transform(image_pil).unsqueeze(0).cuda()

with torch.no_grad():
    img_metas = [{
        'img_shape': (224, 224),
        'ori_shape': (224, 224),
        'pad_shape': (224, 224),
        'filename': test_image_path,
        'scale_factor': 1.0,
        'flip': False
    }]
    result = model.encode_decode(image_tensor, img_metas=img_metas)
    result = result.squeeze().cpu().numpy().argmax(axis=0)

gt_mask = pydicom.dcmread(test_mask_paths[0]).pixel_array
gt_mask_resized = Image.fromarray(gt_mask.astype(np.uint8)).resize((224, 224), resample=Image.NEAREST)
gt_mask_np = np.array(gt_mask_resized)

plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.imshow(image_pil.resize((224, 224)))
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(gt_mask_np, cmap='gray')
plt.title('Ground Truth Mask')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(result, cmap='gray')
plt.title('Predicted Segmentation')
plt.axis('off')

output_figure_path = 'result_prediction.png'
plt.savefig(output_figure_path)
print(f"Saved inference results to {output_figure_path}")

# --------------------------------------
# 12. Evaluate Metrics on the Test/Validation Set
# --------------------------------------
def process_data_container(data):
    """Process DataContainer objects in the data dict."""
    processed_data = {}
    for key, value in data.items():
        if isinstance(value, mmcv.parallel.DataContainer):
            if value.stack:
                processed_data[key] = value.data
            else:
                processed_data[key] = [item for item in value.data]
        else:
            processed_data[key] = value
    return processed_data

val_loader = build_dataloader(
    val_dataset,
    samples_per_gpu=1,
    workers_per_gpu=2,
    dist=False,
    shuffle=False
)
def evaluate_model(model, val_dataset, cfg):
    """Evaluate model on validation dataset with proper data handling."""
    val_loader = build_dataloader(
        val_dataset,
        samples_per_gpu=1,
        workers_per_gpu=2,
        dist=False,
        shuffle=False
    )
    
    model.eval()
    outputs = []
    
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            # Handle both DataContainer and direct tensor formats
            if isinstance(data['img'], (list, torch.Tensor)):
                img = data['img'][0].cuda() if isinstance(data['img'], list) else data['img'].cuda()
            else:
                img = data['img'].data[0].cuda()
                
            # Handle image metadata
            if isinstance(data['img_metas'], (list, dict)):
                img_metas = data['img_metas'][0] if isinstance(data['img_metas'], list) else data['img_metas']
            else:
                img_metas = data['img_metas'].data[0]
            
            # Forward pass
            try:
                result = model.encode_decode(img.unsqueeze(0) if img.dim() == 3 else img, 
                                          img_metas=[img_metas] if isinstance(img_metas, dict) else img_metas)
                
                # Convert the output to segmentation map
                if isinstance(result, torch.Tensor):
                    result = result.squeeze(0).cpu().numpy()
                    result = np.argmax(result, axis=0)  # Convert to class indices
                
                outputs.append(result)
                
            except Exception as e:
                print(f"Error processing batch {i}: {str(e)}")
                continue
    
    # Evaluate results
    try:
        eval_metrics = val_dataset.evaluate(outputs, metric=['mIoU', 'mDice'])
        
        # Print evaluation results
        print("\nEvaluation Metrics on Validation Set:")
        for metric_name, metric_value in eval_metrics.items():
            if isinstance(metric_value, (float, int)):
                print(f"{metric_name}: {metric_value:.4f}")
            elif isinstance(metric_value, list):
                print(f"{metric_name}:")
                for i, val in enumerate(metric_value):
                    print(f"  Class {i}: {val:.4f}")
            else:
                print(f"{metric_name}: {metric_value}")
                
        return eval_metrics
        
    except Exception as e:
        print(f"Error during evaluation: {str(e)}")
        return None

def run_evaluation(model, val_dataset, cfg):
    """Run the complete evaluation pipeline with proper error handling."""
    try:
        print("Starting model evaluation...")
        eval_metrics = evaluate_model(model, val_dataset, cfg)
        
        if eval_metrics is not None:
            # Save metrics to a file
            metrics_output_path = os.path.join(cfg.work_dir, 'evaluation_metrics.txt')
            os.makedirs(cfg.work_dir, exist_ok=True)
            
            with open(metrics_output_path, 'w') as f:
                f.write("Evaluation Metrics on Validation Set:\n")
                for metric_name, metric_value in eval_metrics.items():
                    if isinstance(metric_value, (float, int)):
                        f.write(f"{metric_name}: {metric_value:.4f}\n")
                    elif isinstance(metric_value, list):
                        f.write(f"{metric_name}:\n")
                        for i, val in enumerate(metric_value):
                            f.write(f"  Class {i}: {val:.4f}\n")
                    else:
                        f.write(f"{metric_name}: {metric_value}\n")
            
            print(f"\nMetrics saved to {metrics_output_path}")
            return eval_metrics
        else:
            print("Evaluation failed to produce metrics.")
            return None
            
    except Exception as e:
        print(f"Error during evaluation pipeline: {str(e)}")
        return None
eval_metrics = evaluate_model(model, val_dataset, cfg)
    
if eval_metrics is not None:
    # Save metrics to a file
    metrics_output_path = os.path.join(cfg.work_dir, 'evaluation_metrics.txt')
    with open(metrics_output_path, 'w') as f:
        for metric_name, metric_value in eval_metrics.items():
            f.write(f"{metric_name}: {metric_value}\n")
    print(f"\nMetrics saved to {metrics_output_path}")