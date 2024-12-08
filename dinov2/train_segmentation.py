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
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import time
from tqdm import tqdm

# Keep the original ASPP, DeepLabV3PlusHead, and DinoV2Backbone classes the same
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
            if 'teacher' in state_dict:
                state_dict = state_dict['teacher']

            # Remove 'module.' and 'backbone.' prefixes if present (from DataParallel/DistributedDataParallel)
            state_dict = {k.replace('module.', '').replace('backbone.', ''): v for k, v in state_dict.items()}
            
            # Identify keys to exclude
            exclude_keys = ['pos_embed', 'patch_embed.proj.weight', 'patch_embed.proj.bias']
            
            # Create a new state_dict excluding the mismatched keys
            filtered_state_dict = {k: v for k, v in state_dict.items() if not any(k.startswith(ex_key) for ex_key in exclude_keys)}
            
            # Load the filtered state_dict
            missing_keys, unexpected_keys = self.dino_model.load_state_dict(filtered_state_dict, strict=False)
            print(f"Loaded weights from {weights_path}")
            if missing_keys:
                print(f"Missing keys: {missing_keys}")
            if unexpected_keys:
                print(f"Unexpected keys: {unexpected_keys}")
            
            # Count loaded parameters that match the model's state_dict
            model_state_dict = self.dino_model.state_dict()
            loaded_params = 0
            for k, v in filtered_state_dict.items():
                if k in model_state_dict and v.shape == model_state_dict[k].shape:
                    loaded_params += v.numel()
            
            total_params = sum(p.numel() for p in model_state_dict.values())
            print(f"Loaded {loaded_params} / {total_params} parameters ({(loaded_params / total_params) * 100:.2f}%)")
            
        except Exception as e:
            raise Exception(f"Error loading weights from {weights_path}: {str(e)}")
        
        self.dino_model.eval()
        
        # Freeze DINOv2 parameters
        for param in self.dino_model.parameters():
            param.requires_grad = False
            
        # Optionally, re-initialize excluded layers if necessary
        # self._initialize_excluded_layers()
        
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
        
        # Map backbone sizes to their corresponding output channel dimensions
        backbone_embed_dims = {
            "small": 384,
            "base": 768,
            "large": 1024,
            "giant": 1536
        }
        
        if backbone_size not in backbone_embed_dims:
            raise ValueError(f"Unsupported backbone size: {backbone_size}")
        
        in_channels = backbone_embed_dims[backbone_size]
        self.segmentation_head = DeepLabV3PlusHead(in_channels, num_classes)
        
    def forward(self, x):
        features = self.backbone(x)
        return self.segmentation_head(features)


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

def prepare_data_splits(root_dir, train_size=0.7, val_size=0.15, test_size=0.15):
    """
    Split data into train, validation, and test sets
    """
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
    
    if not image_paths:
        raise ValueError(f"No valid DICOM file pairs found in {root_dir}")
    
    # First split: separate test set
    train_val_images, test_images, train_val_masks, test_masks = train_test_split(
        image_paths, mask_paths, test_size=test_size, random_state=42
    )
    
    # Second split: separate train and validation from remaining data
    train_images, val_images, train_masks, val_masks = train_test_split(
        train_val_images, train_val_masks, 
        test_size=val_size/(train_size + val_size), 
        random_state=42
    )
    
    return {
        'train': (train_images, train_masks),
        'val': (val_images, val_masks),
        'test': (test_images, test_masks)
    }

class DiceBCELoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceBCELoss, self).__init__()
        self.smooth = smooth
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, pred, target):
        pred_sigmoid = torch.sigmoid(pred)
        
        # Flatten predictions and targets
        pred_flat = pred_sigmoid.view(-1)
        targets_flat = target.view(-1)
        
        # Calculate Dice Loss
        intersection = (pred_flat * targets_flat).sum()
        union = pred_flat.sum() + targets_flat.sum()
        dice_loss = 1 - ((2. * intersection + self.smooth) / (union + self.smooth))
        
        # Calculate BCE Loss
        bce_loss = self.bce(pred, target)
        
        # Combine losses
        return 0.5 * dice_loss + 0.5 * bce_loss

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    progress_bar = tqdm(dataloader, desc='Training')
    
    for images, masks in progress_bar:
        images = images.to(device)
        masks = masks.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        
        # Adjust output for binary segmentation
        outputs = outputs[:, 1:2, :, :]  # Take only the foreground channel
        loss = criterion(outputs, masks)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        progress_bar.set_postfix({'loss': loss.item()})
    
    return total_loss / len(dataloader)

def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    dice_scores = []
    
    with torch.no_grad():
        for images, masks in dataloader:
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            outputs = outputs[:, 1:2, :, :]  # Take only the foreground channel
            
            # Calculate loss
            loss = criterion(outputs, masks)
            total_loss += loss.item()
            
            # Calculate Dice score
            pred = (torch.sigmoid(outputs) > 0.5).float()
            dice = (2.0 * (pred * masks).sum()) / (pred.sum() + masks.sum() + 1e-6)
            dice_scores.append(dice.item())
    
    return total_loss / len(dataloader), np.mean(dice_scores)

def save_prediction_visualization(image, mask, pred, prob, save_path, alpha=0.5):
    """
    Save a visualization of the prediction including:
    - Original Image
    - Ground Truth Mask Overlay
    - Prediction Mask Overlay
    - Probability Heat Map

    Args:
        image (PIL.Image or numpy.ndarray): The original image.
        mask (torch.Tensor or numpy.ndarray): The ground truth mask.
        pred (torch.Tensor or numpy.ndarray): The predicted mask.
        prob (torch.Tensor or numpy.ndarray): The predicted probabilities.
        save_path (str): Path to save the visualization.
        alpha (float): Transparency factor for overlays.
    """
    # Convert tensors to numpy arrays if necessary
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy().transpose(1, 2, 0)
        # Denormalize
        image = (image * np.array([0.229, 0.224, 0.225])) + np.array([0.485, 0.456, 0.406])
        image = np.clip(image, 0, 1)
    elif isinstance(image, Image.Image):
        image = np.array(image) / 255.0

    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy().squeeze()
    if isinstance(pred, torch.Tensor):
        pred = pred.cpu().numpy().squeeze()
    if isinstance(prob, torch.Tensor):
        prob = prob.cpu().numpy().squeeze()

    # Create figure
    fig, axs = plt.subplots(2, 2, figsize=(12, 12))

    # Original Image
    axs[0, 0].imshow(image)
    axs[0, 0].set_title('Original Image')
    axs[0, 0].axis('off')

    # Ground Truth Overlay
    axs[0, 1].imshow(image)
    axs[0, 1].imshow(mask, cmap='jet', alpha=alpha)
    axs[0, 1].set_title('Ground Truth Overlay')
    axs[0, 1].axis('off')

    # Prediction Overlay
    axs[1, 0].imshow(image)
    axs[1, 0].imshow(pred, cmap='jet', alpha=alpha)
    axs[1, 0].set_title('Prediction Overlay')
    axs[1, 0].axis('off')

    # Probability Heat Map
    axs[1, 1].imshow(prob, cmap='jet')
    axs[1, 1].set_title('Probability Heat Map')
    axs[1, 1].axis('off')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)

def main():
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root_dir = "/gpfs/scratch/wz1492/Segmentation"
    save_dir = "seg_models"
    os.makedirs(save_dir, exist_ok=True)
    
    # Directory to save prediction visualizations
    preds_save_dir = os.path.join(save_dir, "predictions")
    os.makedirs(preds_save_dir, exist_ok=True)
    
    # Data transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    target_transform = transforms.Compose([
        transforms.Resize((224, 224), interpolation=Image.NEAREST),
        transforms.ToTensor()
    ])
    
    # Prepare datasets
    data_splits = prepare_data_splits(root_dir)
    datasets = {}
    dataloaders = {}
    
    for split in ['train', 'val', 'test']:
        datasets[split] = DICOMLiverDataset(
            *data_splits[split],
            transform=transform,
            target_transform=target_transform
        )
        
        dataloaders[split] = DataLoader(
            datasets[split],
            batch_size=8 if split == 'train' else 4,
            shuffle=(split == 'train'),
            num_workers=4,
            pin_memory=True
        )
    
    # Initialize model
    dino_weights_path = "/gpfs/data/shenlab/wz1492/HCC/dinov2/experiments_large_dataset/eval/training_124999/teacher_checkpoint.pth"
    model = LiverSegmentationModel(
        weights_path=dino_weights_path,
        num_classes=2,
        backbone_size="large"
    ).to(device)
    
    # Training setup
    criterion = DiceBCELoss()
    optimizer = Adam(model.parameters(), lr=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)
    
    # Training loop
    num_epochs = 1
    best_val_dice = 0
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        # Train
        train_loss = train_epoch(model, dataloaders['train'], criterion, optimizer, device)
        
        # Validate
        val_loss, val_dice = validate(model, dataloaders['val'], criterion, device)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Dice: {val_dice:.4f}")
        
        # Save best model
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_dice': val_dice,
            }, os.path.join(save_dir, 'best_model.pth'))
    
    # Final evaluation on test set
    model.load_state_dict(torch.load(os.path.join(save_dir, 'best_model.pth'))['model_state_dict'])
    test_loss, test_dice = validate(model, dataloaders['test'], criterion, device)
    print(f"\nFinal Test Results:")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Dice Score: {test_dice:.4f}")
    
    # Save Prediction Visualizations
    model.eval()
    num_visualizations = 10  # Number of samples to visualize
    saved = 0
    with torch.no_grad():
        for images, masks in dataloaders['test']:
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            outputs = outputs[:, 1:2, :, :]  # Take only the foreground channel
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()
            
            for i in range(images.size(0)):
                if saved >= num_visualizations:
                    break
                image = images[i].cpu()
                mask = masks[i].cpu()
                pred = preds[i].cpu()
                prob = probs[i].cpu()
                
                # Reverse normalization for visualization
                inv_normalize = transforms.Normalize(
                    mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
                    std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
                )
                image_denorm = inv_normalize(image)
                image_denorm = torch.clamp(image_denorm, 0, 1)
                
                # Convert tensors to PIL Images
                image_pil = transforms.ToPILImage()(image_denorm)
                mask_pil = transforms.ToPILImage()(mask)
                pred_pil = transforms.ToPILImage()(pred)
                prob_pil = transforms.ToPILImage()(prob)
                
                # Define save path
                save_path = os.path.join(preds_save_dir, f"prediction_{saved+1}.png")
                
                # Save visualization
                save_prediction_visualization(
                    image_pil, 
                    mask_pil, 
                    pred_pil, 
                    prob_pil, 
                    save_path
                )
                
                saved += 1
            if saved >= num_visualizations:
                break
    
    print(f"\nSaved {saved} prediction visualizations in '{preds_save_dir}'")

if __name__ == "__main__":
    main()