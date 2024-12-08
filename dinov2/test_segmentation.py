import os
import pydicom
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import torch.nn.functional as F

# ---------------------------------------------
# Improved ASPP and Decoder for DeepLabV3+
# ---------------------------------------------
class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, atrous_rates=(6,12,18)):
        super(ASPP, self).__init__()
        modules = []
        # 1x1 convolution
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ))
        # Atrous convolutions
        for rate in atrous_rates:
            modules.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=rate, dilation=rate, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ))
        # Global pooling
        modules.append(nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(len(modules)*out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        res = []
        for conv in self.convs[:-1]:
            res.append(conv(x))
        # Global pooling branch
        global_pool = self.convs[-1](x)
        global_pool = F.interpolate(global_pool, size=x.shape[2:], mode='bilinear', align_corners=False)
        res.append(global_pool)
        
        x = torch.cat(res, dim=1)
        x = self.project(x)
        return x

class DeepLabV3PlusHead(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.aspp = ASPP(in_channels, 256, atrous_rates=(6,12,18))
        self.decoder = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, 1)
        )

    def forward(self, x):
        x = self.aspp(x)
        # Use scale_factor=14 to go from 16x16 to 224x224
        x = F.interpolate(x, scale_factor=14, mode='bilinear', align_corners=True)
        x = self.decoder(x)
        return x

# ---------------------------------------------
# DINOv2 Backbone
# ---------------------------------------------
# ---------------------------------------------
# DINOv2 Backbone with Local Weight Loading
# ---------------------------------------------
class DinoV2Backbone(nn.Module):
    def __init__(self, weights_path, backbone_size="base"):
        super().__init__()
        # Map backbone size to model configuration
        backbone_configs = {
            "small": {"embed_dim": 384, "depth": 12, "num_heads": 6, "patch_size": 14},
            "base": {"embed_dim": 768, "depth": 12, "num_heads": 12, "patch_size": 14},
            "large": {"embed_dim": 1024, "depth": 24, "num_heads": 16, "patch_size": 14},
            "giant": {"embed_dim": 1536, "depth": 40, "num_heads": 24, "patch_size": 14}
        }
        
        if backbone_size not in backbone_configs:
            raise ValueError(f"Unsupported backbone size: {backbone_size}")
            
        config = backbone_configs[backbone_size]
        
        # Initialize vision transformer with correct configuration
        self.dino_model = torch.hub.load(
            "facebookresearch/dinov2", 
            f"dinov2_vit{backbone_size[0]}{config['patch_size']}", 
            pretrained=False
        )
        
        # Load local weights
        try:
            state_dict = torch.load(weights_path, map_location='cpu')
            # Handle both cases where state_dict is wrapped in 'state_dict' key or not
            if 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']
            # Remove 'module.' prefix if present (from DataParallel/DistributedDataParallel)
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            missing_keys, unexpected_keys = self.dino_model.load_state_dict(state_dict, strict=False)
            print(f"Loaded weights from {weights_path}")
            if missing_keys:
                print(f"Missing keys: {missing_keys}")
            if unexpected_keys:
                print(f"Unexpected keys: {unexpected_keys}")
        except Exception as e:
            raise Exception(f"Error loading weights from {weights_path}: {str(e)}")
        
        self.dino_model.eval()
        
        # Freeze DINOv2 parameters
        for param in self.dino_model.parameters():
            param.requires_grad = False
            
        # Uncomment below lines to partially unfreeze last few blocks for adaptation
        # for blk in self.dino_model.blocks[-2:]:  # Unfreeze last two blocks
        #     for param in blk.parameters():
        #         param.requires_grad = True
        # self.dino_model.norm.weight.requires_grad = True
        # self.dino_model.norm.bias.requires_grad = True

    def forward(self, x):
        x = x.to(next(self.parameters()).device)
        x = self.dino_model.prepare_tokens_with_masks(x)
        for blk in self.dino_model.blocks:
            x = blk(x)
        x = self.dino_model.norm(x)
        patch_tokens = x[:, 1:, :]  # Remove CLS token
        N, P, C = patch_tokens.shape
        H = W = int((P)**0.5)
        return patch_tokens.permute(0, 2, 1).reshape(N, C, H, W)
# ---------------------------------------------
# Full Model
# ---------------------------------------------
class LiverSegmentationModel(nn.Module):
    def __init__(self, weights_path, num_classes=2, backbone_size="base"):
        super().__init__()
        self.backbone = DinoV2Backbone(weights_path=weights_path, backbone_size=backbone_size)
        # DINOv2 base has output channel size of 768
        self.segmentation_head = DeepLabV3PlusHead(768, num_classes)
        
    def forward(self, x):
        features = self.backbone(x)
        return self.segmentation_head(features)

# ---------------------------------------------
# Dataset and Data Preparation
# ---------------------------------------------
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
        image = dicom_image.pixel_array.copy()
        
        dicom_mask = pydicom.dcmread(mask_path)
        mask = dicom_mask.pixel_array.copy()
        
        # Normalize image
        image = (image - np.min(image)) / (np.max(image) - np.min(image)) * 255.0
        image = np.stack([image] * 3, axis=-1)  # Convert single channel to RGB
        
        # Binarize mask
        mask = (mask > 0).astype(np.float32)
        
        # Convert to PIL
        image = Image.fromarray(image.astype(np.uint8))
        mask = Image.fromarray((mask * 255).astype(np.uint8))
        
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            mask = self.target_transform(mask)
            mask = (mask > 0.5).float()
        
        return image, mask

def prepare_data(root_dir, test_size=0.2):
    image_paths = []
    mask_paths = []
    
    for root, dirs, files in os.walk(root_dir):
        if "images" in dirs and "masks" in dirs:
            images_dir = os.path.join(root, "images")
            masks_dir = os.path.join(root, "masks")
            
            for file in sorted(os.listdir(images_dir)):
                if file.endswith(".dicom"):
                    image_path = os.path.join(images_dir, file)
                    mask_path = os.path.join(masks_dir, file)
                    
                    if os.path.exists(image_path) and os.path.exists(mask_path):
                        image_paths.append(image_path)
                        mask_paths.append(mask_path)
                    else:
                        print(f"Warning: Missing pair for {file}")
    
    if not image_paths:
        raise ValueError(f"No valid DICOM file pairs found in {root_dir}")
    
    print(f"Found {len(image_paths)} image-mask pairs")
    train_images, val_images, train_masks, val_masks = train_test_split(
        image_paths, mask_paths, test_size=test_size, random_state=42
    )
    
    print(f"Training samples: {len(train_images)}")
    print(f"Validation samples: {len(val_images)}")
    
    return train_images, train_masks, val_images, val_masks

def calculate_metrics(pred, target):
    print(f"Prediction shape: {pred.shape}, unique values: {torch.unique(pred)}")
    print(f"Target shape: {target.shape}, unique values: {torch.unique(target)}")
    
    pred = pred.bool()
    target = target.bool()
    
    intersection = (pred & target).float().sum()
    union = (pred | target).float().sum()
    pred_sum = pred.float().sum()
    target_sum = target.float().sum()
    
    print(f"Intersection: {intersection.item()}")
    print(f"Union: {union.item()}")
    print(f"Pred sum: {pred_sum.item()}")
    print(f"Target sum: {target_sum.item()}")
    
    dice = (2.0 * intersection) / (pred_sum + target_sum + 1e-6)
    iou = intersection / (union + 1e-6)
    
    dice_val = dice.item() if not torch.isnan(dice) else 0.0
    iou_val = iou.item() if not torch.isnan(iou) else 0.0
    
    print(f"Dice: {dice_val}, IoU: {iou_val}")
    return dice_val, iou_val

@torch.no_grad()
def evaluate_model(model, dataloader, device, save_dir='validation_results'):
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    
    total_dice = 0
    total_iou = 0
    num_samples = 0
    all_metrics = []
    
    for batch_idx, (images, masks) in enumerate(dataloader):
        print(f"\nProcessing batch {batch_idx}")
        
        images = images.to(device)
        masks = masks.to(device)
        
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)
        predictions = (probs[:, 1] > 0.5).float()
        
        print(f"Batch predictions shape: {predictions.shape}")
        print(f"Batch masks shape: {masks.shape}")
        
        # Save visualizations for first batch
        if batch_idx == 0:
            for i in range(min(4, len(images))):
                fig, axes = plt.subplots(1, 5, figsize=(25, 5))
                
                img = images[i].cpu().permute(1, 2, 0)
                img = (img - img.min()) / (img.max() - img.min())
                axes[0].imshow(img)
                axes[0].set_title('Original Image')
                
                axes[1].imshow(masks[i].cpu().squeeze(), cmap='gray')
                axes[1].set_title(f'Ground Truth\nUnique: {torch.unique(masks[i]).cpu().numpy()}')
                
                axes[2].imshow(probs[i, 1].cpu(), cmap='jet')
                axes[2].set_title('Probability Map')
                
                axes[3].imshow(predictions[i].cpu(), cmap='gray')
                axes[3].set_title(f'Prediction\nUnique: {torch.unique(predictions[i]).cpu().numpy()}')
                
                gt_mask = masks[i].cpu().squeeze().numpy()
                pred_mask = predictions[i].cpu().squeeze().numpy()
                
                overlay = np.zeros((*gt_mask.shape, 3))
                overlay[gt_mask == 1] = [0, 1, 0]  # Green for GT
                overlay[pred_mask == 1] = [1, 0, 0]  # Red for pred
                overlay[(gt_mask == 1) & (pred_mask == 1)] = [1, 1, 0]  # Yellow for overlap

                alpha = 0.5
                blended = (1 - alpha) * img.numpy() + alpha * overlay
                blended = np.clip(blended, 0, 1)
                
                axes[4].imshow(blended)
                axes[4].set_title('Overlay\nG: GT, R: Pred, Y: Overlap')
                
                for ax in axes:
                    ax.axis('off')
                
                plt.savefig(os.path.join(save_dir, f'sample_{i}_visualization.png'))
                plt.close()
        
        # Calculate metrics
        print(f"\nCalculating metrics for {len(images)} samples in batch")
        for i in range(images.shape[0]):
            try:
                print(f"\nSample {i}:")
                dice, iou = calculate_metrics(predictions[i], masks[i].squeeze())
                
                all_metrics.append({
                    'batch': batch_idx,
                    'sample': i,
                    'dice': dice,
                    'iou': iou
                })
                
                total_dice += dice
                total_iou += iou
                num_samples += 1
            except Exception as e:
                print(f"Error processing sample {i}: {str(e)}")
                continue
    
    if num_samples > 0:
        avg_dice = total_dice / num_samples
        avg_iou = total_iou / num_samples
    else:
        avg_dice = 0
        avg_iou = 0
    
    with open(os.path.join(save_dir, 'validation_metrics.txt'), 'w') as f:
        f.write(f"Number of processed samples: {num_samples}\n")
        f.write(f"Average Dice: {avg_dice:.4f}\n")
        f.write(f"Average IoU: {avg_iou:.4f}\n\n")
        
        for metric in all_metrics:
            f.write(f"Batch {metric['batch']}, Sample {metric['sample']}:\n")
            f.write(f"  Dice: {metric['dice']:.4f}\n")
            f.write(f"  IoU: {metric['iou']:.4f}\n\n")
    
    return {'dice': avg_dice, 'iou': avg_iou}

# ---------------------------------------------
# Loss Functions
# ---------------------------------------------
def dice_loss(pred, target, smooth=1e-6):
    # pred, target are probabilities or binary [N, H, W]
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return 1.0 - dice

def combined_loss(outputs, targets):
    # outputs: [N, C, H, W], targets: [N, 1, H, W]
    # Use CE + Dice
    ce = F.cross_entropy(outputs, targets.long().squeeze(1))
    # Convert outputs to probabilities for dice calculation of the foreground
    probs = torch.softmax(outputs, dim=1)[:, 1]
    # Dice loss on the foreground channel only
    d = dice_loss(probs, targets.squeeze(1))
    return ce + d

# ---------------------------------------------
# Main function for evaluation (adjust for training)
# ---------------------------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    root_dir = "/gpfs/scratch/wz1492/Segmentation"
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    target_transform = transforms.Compose([
        transforms.Resize((224, 224), interpolation=Image.NEAREST),
        transforms.ToTensor()
    ])
    
    # Prepare data
    print("\nPreparing data...")
    try:
        train_image_paths, train_mask_paths, val_image_paths, val_mask_paths = prepare_data(root_dir)
        
        print("\nChecking data paths:")
        print(f"Number of validation images: {len(val_image_paths)}")
        print(f"Number of validation masks: {len(val_mask_paths)}")
        
        for i in range(min(3, len(val_image_paths))):
            print(f"\nSample {i}:")
            print(f"Image: {val_image_paths[i]}")
            print(f"Mask: {val_mask_paths[i]}")
            print(f"Files exist: {os.path.exists(val_image_paths[i])} (image), {os.path.exists(val_mask_paths[i])} (mask)")
        
    except Exception as e:
        print(f"Error during data preparation: {str(e)}")
        raise
    
    val_dataset = DICOMLiverDataset(
        val_image_paths, 
        val_mask_paths,
        transform=transform,
        target_transform=target_transform
    )
    
    print(f"\nValidation dataset size: {len(val_dataset)}")
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=2,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    print("\nCreating model...")
    # In main()
    weights_path = "/gpfs/data/shenlab/wz1492/HCC/dinov2/experiments_large_dataset/eval/training_124999/teacher_checkpoint.pth"
    model = LiverSegmentationModel(
        weights_path=weights_path,
        num_classes=2, 
        backbone_size="large"
    )
    model = model.to(device)
    
    # Here you would load trained weights if you have them
    # model.load_state_dict(torch.load("your_trained_model.pth", map_location=device))
    
    print("\nStarting validation...")
    metrics = evaluate_model(model, val_loader, device)
    
    print(f"\nFinal Results:")
    print(f"Dice Coefficient: {metrics['dice']:.4f}")
    print(f"IoU Score: {metrics['iou']:.4f}")

if __name__ == "__main__":
    main()
