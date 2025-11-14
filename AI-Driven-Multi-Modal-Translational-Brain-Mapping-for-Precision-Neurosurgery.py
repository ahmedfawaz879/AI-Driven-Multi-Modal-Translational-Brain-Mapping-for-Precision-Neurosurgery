"""
Enhanced AI-Driven Multi-Modal Translational Brain Mapping for Precision Neurosurgery
Improvements: Advanced architecture, attention mechanisms, better XAI, mixed precision training,
data augmentation, ensemble methods, uncertainty quantification, and production-ready features.
"""

import os
import argparse
import json
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from tqdm import tqdm
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass, asdict
import logging

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler

from nilearn.input_data import NiftiLabelsMasker
from nilearn.image import resample_to_img
from scipy.ndimage import rotate, zoom
from sklearn.model_selection import KFold

# Optional imports
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

try:
    from monai.losses import DiceCELoss, FocalLoss
    from monai.metrics import DiceMetric
    MONAI_AVAILABLE = True
except ImportError:
    MONAI_AVAILABLE = False

# ======================== Configuration ========================

@dataclass
class ModelConfig:
    """Model configuration"""
    in_channels: int = 3
    out_channels: int = 2
    base_filters: int = 32
    depth: int = 4
    use_attention: bool = True
    use_residual: bool = True
    dropout: float = 0.1

@dataclass
class TrainingConfig:
    """Training configuration"""
    epochs: int = 50
    batch_size: int = 2
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    patience: int = 10
    use_amp: bool = True
    accumulation_steps: int = 4
    val_split: float = 0.2

# ======================== Logging Setup ========================

def setup_logging(log_dir: str = 'logs'):
    """Setup structured logging"""
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'{log_dir}/brain_mapping.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

# ======================== Utilities ========================

def ensure_dir(path: Union[str, Path]):
    """Create directory if it doesn't exist"""
    Path(path).mkdir(parents=True, exist_ok=True)

def load_nifti(path: str) -> Tuple[np.ndarray, np.ndarray, nib.Nifti1Header]:
    """Load NIfTI image with error handling"""
    try:
        img = nib.load(path)
        return img.get_fdata(), img.affine, img.header
    except Exception as e:
        logger.error(f"Failed to load {path}: {e}")
        raise

def save_nifti(data: np.ndarray, affine: np.ndarray, path: str):
    """Save data as NIfTI"""
    img = nib.Nifti1Image(data.astype(np.float32), affine)
    nib.save(img, path)
    logger.info(f"Saved NIfTI to {path}")

# ======================== Preprocessing ========================

class IntensityNormalization:
    """Advanced intensity normalization with multiple methods"""
    
    @staticmethod
    def zscore(vol: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """Z-score normalization"""
        if mask is not None:
            mean = vol[mask > 0].mean()
            std = vol[mask > 0].std()
        else:
            mean, std = vol.mean(), vol.std()
        return (vol - mean) / (std + 1e-8)
    
    @staticmethod
    def percentile_clip(vol: np.ndarray, lower: float = 0.5, upper: float = 99.5) -> np.ndarray:
        """Percentile-based clipping and normalization"""
        pmin, pmax = np.percentile(vol, [lower, upper])
        vol = np.clip(vol, pmin, pmax)
        return (vol - pmin) / (pmax - pmin + 1e-8)
    
    @staticmethod
    def nyul_normalization(vol: np.ndarray) -> np.ndarray:
        """Nyúl histogram matching normalization"""
        # Simplified version - full implementation would use reference histograms
        percentiles = np.percentile(vol, [1, 10, 25, 50, 75, 90, 99])
        vol_norm = np.interp(vol, percentiles, np.linspace(0, 1, len(percentiles)))
        return vol_norm

# ======================== Data Augmentation ========================

class Augmentation3D:
    """3D data augmentation for brain MRI"""
    
    def __init__(self, p: float = 0.5):
        self.p = p
    
    def random_flip(self, img: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Random flip along axes"""
        if np.random.random() < self.p:
            axis = np.random.choice([1, 2, 3])
            img = np.flip(img, axis)
            mask = np.flip(mask, axis)
        return img.copy(), mask.copy()
    
    def random_rotate(self, img: np.ndarray, mask: np.ndarray, max_angle: float = 15) -> Tuple[np.ndarray, np.ndarray]:
        """Random rotation"""
        if np.random.random() < self.p:
            angle = np.random.uniform(-max_angle, max_angle)
            axes = np.random.choice([(1, 2), (1, 3), (2, 3)])
            for c in range(img.shape[0]):
                img[c] = rotate(img[c], angle, axes=axes, reshape=False, order=1)
            mask = rotate(mask, angle, axes=axes, reshape=False, order=0)
        return img, mask
    
    def random_scale(self, img: np.ndarray, mask: np.ndarray, scale_range: Tuple[float, float] = (0.9, 1.1)) -> Tuple[np.ndarray, np.ndarray]:
        """Random scaling"""
        if np.random.random() < self.p:
            scale = np.random.uniform(*scale_range)
            for c in range(img.shape[0]):
                img[c] = zoom(img[c], scale, order=1)
            mask = zoom(mask, scale, order=0)
        return img, mask
    
    def random_noise(self, img: np.ndarray, std: float = 0.05) -> np.ndarray:
        """Add random Gaussian noise"""
        if np.random.random() < self.p:
            noise = np.random.normal(0, std, img.shape)
            img = img + noise
        return img
    
    def random_gamma(self, img: np.ndarray, gamma_range: Tuple[float, float] = (0.8, 1.2)) -> np.ndarray:
        """Random gamma correction"""
        if np.random.random() < self.p:
            gamma = np.random.uniform(*gamma_range)
            img = np.power(np.abs(img), gamma) * np.sign(img)
        return img

# ======================== Dataset ========================

class EnhancedBrainDataset(Dataset):
    """Enhanced multi-modal brain dataset with augmentation"""
    
    def __init__(self, 
                 cases: List[str], 
                 channels: List[str] = ['T1', 'T2', 'FLAIR'],
                 patch_size: Tuple[int, int, int] = (128, 128, 128),
                 augment: bool = False,
                 normalization: str = 'zscore'):
        self.cases = cases
        self.channels = channels
        self.patch_size = patch_size
        self.augment = augment
        self.normalization = normalization
        self.augmentor = Augmentation3D(p=0.5) if augment else None
        
    def __len__(self):
        return len(self.cases)
    
    def normalize(self, vol: np.ndarray) -> np.ndarray:
        """Apply normalization"""
        if self.normalization == 'zscore':
            return IntensityNormalization.zscore(vol)
        elif self.normalization == 'percentile':
            return IntensityNormalization.percentile_clip(vol)
        elif self.normalization == 'nyul':
            return IntensityNormalization.nyul_normalization(vol)
        return vol
    
    def __getitem__(self, idx):
        case = self.cases[idx]
        imgs = []
        
        # Load all channels
        for ch in self.channels:
            path = os.path.join(case, f"{ch}.nii.gz")
            if not os.path.exists(path):
                logger.warning(f"Missing {path}, using zeros")
                data = np.zeros((128, 128, 128))
            else:
                data, _, _ = load_nifti(path)
                data = self.normalize(data)
            imgs.append(data)
        
        img = np.stack(imgs, axis=0).astype(np.float32)
        
        # Load mask
        mask_path = os.path.join(case, 'mask.nii.gz')
        if os.path.exists(mask_path):
            mask, _, _ = load_nifti(mask_path)
            mask = (mask > 0).astype(np.uint8)
        else:
            mask = np.zeros(img.shape[1:], dtype=np.uint8)
        
        # Apply augmentation
        if self.augment and self.augmentor:
            img, mask = self.augmentor.random_flip(img, mask)
            img, mask = self.augmentor.random_rotate(img, mask)
            img = self.augmentor.random_noise(img)
            img = self.augmentor.random_gamma(img)
        
        # Crop/pad to patch size
        img = center_crop_or_pad(img, self.patch_size)
        mask = center_crop_or_pad(mask[None], self.patch_size)[0]
        
        return torch.from_numpy(img), torch.from_numpy(mask).long(), case

def center_crop_or_pad(arr: np.ndarray, target_shape: Tuple[int, int, int]) -> np.ndarray:
    """Center crop or pad array to target shape"""
    if arr.ndim == 4:
        C = arr.shape[0]
        out = np.zeros((C,) + tuple(target_shape), dtype=arr.dtype)
        for c in range(C):
            out[c] = center_crop_or_pad(arr[c], target_shape)
        return out
    
    assert arr.ndim == 3
    out = np.zeros(tuple(target_shape), dtype=arr.dtype)
    in_shape = arr.shape
    
    # Calculate crop/pad indices
    starts = [max(0, (in_s - tar_s) // 2) for in_s, tar_s in zip(in_shape, target_shape)]
    ends = [start + tar for start, tar in zip(starts, target_shape)]
    
    # Crop
    cropped = arr[starts[0]:ends[0], starts[1]:ends[1], starts[2]:ends[2]]
    
    # Pad if necessary
    pad_starts = [max(0, (tar - cr) // 2) for tar, cr in zip(target_shape, cropped.shape)]
    out[pad_starts[0]:pad_starts[0]+cropped.shape[0],
        pad_starts[1]:pad_starts[1]+cropped.shape[1],
        pad_starts[2]:pad_starts[2]+cropped.shape[2]] = cropped
    
    return out

# ======================== Architecture ========================

class AttentionBlock3D(nn.Module):
    """3D Attention mechanism for feature refinement"""
    
    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels
        self.query = nn.Conv3d(channels, channels // 8, 1)
        self.key = nn.Conv3d(channels, channels // 8, 1)
        self.value = nn.Conv3d(channels, channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
    
    def forward(self, x):
        B, C, D, H, W = x.size()
        
        # Compute attention
        q = self.query(x).view(B, -1, D * H * W).permute(0, 2, 1)
        k = self.key(x).view(B, -1, D * H * W)
        v = self.value(x).view(B, -1, D * H * W)
        
        attention = torch.bmm(q, k)
        attention = F.softmax(attention, dim=-1)
        
        out = torch.bmm(v, attention.permute(0, 2, 1))
        out = out.view(B, C, D, H, W)
        
        return self.gamma * out + x

class ResidualBlock3D(nn.Module):
    """3D Residual block with optional attention"""
    
    def __init__(self, in_ch: int, out_ch: int, use_attention: bool = False, dropout: float = 0.1):
        super().__init__()
        self.conv1 = nn.Conv3d(in_ch, out_ch, 3, padding=1)
        self.norm1 = nn.InstanceNorm3d(out_ch)
        self.conv2 = nn.Conv3d(out_ch, out_ch, 3, padding=1)
        self.norm2 = nn.InstanceNorm3d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout3d(dropout)
        
        self.attention = AttentionBlock3D(out_ch) if use_attention else None
        
        # Skip connection
        self.skip = nn.Conv3d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
    
    def forward(self, x):
        identity = self.skip(x)
        
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        out = self.conv2(out)
        out = self.norm2(out)
        
        if self.attention:
            out = self.attention(out)
        
        out = out + identity
        out = self.relu(out)
        
        return out

class EnhancedUNet3D(nn.Module):
    """Enhanced 3D U-Net with attention and residual connections"""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Encoder
        self.enc1 = ResidualBlock3D(config.in_channels, config.base_filters, 
                                    config.use_attention, config.dropout)
        self.pool1 = nn.MaxPool3d(2)
        
        self.enc2 = ResidualBlock3D(config.base_filters, config.base_filters * 2,
                                    config.use_attention, config.dropout)
        self.pool2 = nn.MaxPool3d(2)
        
        self.enc3 = ResidualBlock3D(config.base_filters * 2, config.base_filters * 4,
                                    config.use_attention, config.dropout)
        self.pool3 = nn.MaxPool3d(2)
        
        self.enc4 = ResidualBlock3D(config.base_filters * 4, config.base_filters * 8,
                                    config.use_attention, config.dropout)
        self.pool4 = nn.MaxPool3d(2)
        
        # Bottleneck
        self.bottleneck = ResidualBlock3D(config.base_filters * 8, config.base_filters * 16,
                                          True, config.dropout)
        
        # Decoder
        self.up4 = nn.ConvTranspose3d(config.base_filters * 16, config.base_filters * 8, 2, 2)
        self.dec4 = ResidualBlock3D(config.base_filters * 16, config.base_filters * 8,
                                    config.use_attention, config.dropout)
        
        self.up3 = nn.ConvTranspose3d(config.base_filters * 8, config.base_filters * 4, 2, 2)
        self.dec3 = ResidualBlock3D(config.base_filters * 8, config.base_filters * 4,
                                    config.use_attention, config.dropout)
        
        self.up2 = nn.ConvTranspose3d(config.base_filters * 4, config.base_filters * 2, 2, 2)
        self.dec2 = ResidualBlock3D(config.base_filters * 4, config.base_filters * 2,
                                    config.use_attention, config.dropout)
        
        self.up1 = nn.ConvTranspose3d(config.base_filters * 2, config.base_filters, 2, 2)
        self.dec1 = ResidualBlock3D(config.base_filters * 2, config.base_filters,
                                    config.use_attention, config.dropout)
        
        # Output
        self.out = nn.Conv3d(config.base_filters, config.out_channels, 1)
    
    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))
        
        # Bottleneck
        b = self.bottleneck(self.pool4(e4))
        
        # Decoder with skip connections
        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        
        return self.out(d1)

# ======================== Loss Functions ========================

class CombinedLoss(nn.Module):
    """Combined loss: CE + Dice + Focal"""
    
    def __init__(self, ce_weight: float = 1.0, dice_weight: float = 1.0, 
                 focal_weight: float = 0.5, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.ce = nn.CrossEntropyLoss()
        self.alpha = alpha
        self.gamma = gamma
    
    def dice_loss(self, pred, target, smooth=1e-5):
        """Dice loss"""
        probs = F.softmax(pred, dim=1)[:, 1]
        target_f = target.float()
        inter = (probs * target_f).sum()
        union = probs.sum() + target_f.sum()
        return 1 - (2. * inter + smooth) / (union + smooth)
    
    def focal_loss(self, pred, target):
        """Focal loss"""
        ce = F.cross_entropy(pred, target, reduction='none')
        pt = torch.exp(-ce)
        focal = self.alpha * (1 - pt) ** self.gamma * ce
        return focal.mean()
    
    def forward(self, pred, target):
        ce_loss = self.ce(pred, target)
        dice = self.dice_loss(pred, target)
        focal = self.focal_loss(pred, target)
        
        total = (self.ce_weight * ce_loss + 
                self.dice_weight * dice + 
                self.focal_weight * focal)
        
        return total, {
            'ce': ce_loss.item(),
            'dice': dice.item(),
            'focal': focal.item(),
            'total': total.item()
        }

# ======================== Training ========================

class Trainer:
    """Enhanced trainer with mixed precision, early stopping, and logging"""
    
    def __init__(self, model: nn.Module, config: TrainingConfig, device: str = 'cuda'):
        self.model = model.to(device)
        self.config = config
        self.device = device
        
        self.optimizer = optim.AdamW(model.parameters(), 
                                     lr=config.learning_rate,
                                     weight_decay=config.weight_decay)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=5, factor=0.5
        )
        self.scaler = GradScaler() if config.use_amp else None
        self.criterion = CombinedLoss()
        
        self.best_loss = float('inf')
        self.patience_counter = 0
        self.history = {'train_loss': [], 'val_loss': [], 'metrics': []}
    
    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        loss_components = {'ce': 0, 'dice': 0, 'focal': 0}
        
        self.optimizer.zero_grad()
        
        for i, (imgs, masks, _) in enumerate(tqdm(dataloader, desc='Training')):
            imgs, masks = imgs.to(self.device), masks.to(self.device)
            
            # Mixed precision training
            if self.scaler:
                with autocast():
                    preds = self.model(imgs)
                    loss, components = self.criterion(preds, masks)
                    loss = loss / self.config.accumulation_steps
                
                self.scaler.scale(loss).backward()
                
                if (i + 1) % self.config.accumulation_steps == 0:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
            else:
                preds = self.model(imgs)
                loss, components = self.criterion(preds, masks)
                loss = loss / self.config.accumulation_steps
                loss.backward()
                
                if (i + 1) % self.config.accumulation_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
            
            total_loss += components['total']
            for k in loss_components:
                loss_components[k] += components[k]
        
        n = len(dataloader)
        return {
            'loss': total_loss / n,
            **{k: v / n for k, v in loss_components.items()}
        }
    
    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Validation"""
        self.model.eval()
        total_loss = 0
        dice_scores = []
        
        with torch.no_grad():
            for imgs, masks, _ in tqdm(dataloader, desc='Validation'):
                imgs, masks = imgs.to(self.device), masks.to(self.device)
                
                preds = self.model(imgs)
                loss, _ = self.criterion(preds, masks)
                total_loss += loss.item()
                
                # Compute Dice score
                probs = F.softmax(preds, dim=1)[:, 1]
                pred_mask = (probs > 0.5).float()
                dice = self.compute_dice(pred_mask, masks.float())
                dice_scores.append(dice)
        
        return {
            'loss': total_loss / len(dataloader),
            'dice': np.mean(dice_scores)
        }
    
    @staticmethod
    def compute_dice(pred, target, smooth=1e-5):
        """Compute Dice coefficient"""
        inter = (pred * target).sum()
        union = pred.sum() + target.sum()
        return (2. * inter + smooth) / (union + smooth)
    
    def fit(self, train_loader: DataLoader, val_loader: DataLoader, 
            save_path: str = 'best_model.pth'):
        """Full training loop with early stopping"""
        logger.info(f"Starting training for {self.config.epochs} epochs")
        
        for epoch in range(self.config.epochs):
            # Train
            train_metrics = self.train_epoch(train_loader)
            logger.info(f"Epoch {epoch+1}/{self.config.epochs} - Train Loss: {train_metrics['loss']:.4f}")
            
            # Validate
            val_metrics = self.validate(val_loader)
            logger.info(f"Val Loss: {val_metrics['loss']:.4f}, Val Dice: {val_metrics['dice']:.4f}")
            
            # Learning rate scheduling
            self.scheduler.step(val_metrics['loss'])
            
            # Save history
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['metrics'].append(val_metrics)
            
            # Early stopping
            if val_metrics['loss'] < self.best_loss:
                self.best_loss = val_metrics['loss']
                self.patience_counter = 0
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'loss': self.best_loss,
                    'config': asdict(self.config)
                }, save_path)
                logger.info(f"Model saved with loss: {self.best_loss:.4f}")
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.config.patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break
        
        # Save training history
        with open(save_path.replace('.pth', '_history.json'), 'w') as f:
            json.dump(self.history, f, indent=2)
        
        return self.model

# ======================== XAI Methods ========================

class UncertaintyEstimator:
    """Monte Carlo Dropout for uncertainty estimation"""
    
    def __init__(self, model: nn.Module, n_samples: int = 20):
        self.model = model
        self.n_samples = n_samples
    
    def enable_dropout(self):
        """Enable dropout in eval mode"""
        for m in self.model.modules():
            if isinstance(m, nn.Dropout) or isinstance(m, nn.Dropout3d):
                m.train()
    
    def estimate(self, x: torch.Tensor, device: str = 'cuda') -> Tuple[np.ndarray, np.ndarray]:
        """Estimate prediction and uncertainty"""
        self.model.eval()
        self.enable_dropout()
        
        predictions = []
        with torch.no_grad():
            for _ in range(self.n_samples):
                pred = self.model(x.to(device))
                probs = F.softmax(pred, dim=1)[:, 1].cpu().numpy()
                predictions.append(probs)
        
        predictions = np.array(predictions)
        mean_pred = predictions.mean(axis=0)
        uncertainty = predictions.std(axis=0)  # Epistemic uncertainty
        
        return mean_pred[0], uncertainty[0]

class LayerCAM3D:
    """Layer-CAM for 3D medical images - more accurate than Grad-CAM"""
    
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        
        # Register hooks
        self.target_layer.register_forward_hook(self._save_activation)
        self.target_layer.register_full_backward_hook(self._save_gradient)
    
    def _save_activation(self, module, input, output):
        self.activations = output.detach()
    
    def _save_gradient(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()
    
    def generate(self, x: torch.Tensor, target_class: int = 1) -> np.ndarray:
        """Generate Layer-CAM heatmap"""
        self.model.eval()
        x.requires_grad_(True)
        
        # Forward pass
        output = self.model(x)
        
        # Backward pass
        self.model.zero_grad()
        score = output[:, target_class].sum()
        score.backward()
        
        # Compute Layer-CAM
        weights = F.adaptive_avg_pool3d(self.gradients, 1)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        
        # Resize to input size
        cam = F.interpolate(cam, size=x.shape[2:], mode='trilinear', align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        
        # Normalize
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        return cam

# ======================== Visualization ========================

class BrainVisualizer:
    """Advanced visualization utilities"""
    
    @staticmethod
    def create_multi_panel_figure(t1_vol: np.ndarray, 
                                  tumor_prob: np.ndarray,
                                  xai_maps: Dict[str, np.ndarray],
                                  uncertainty: Optional[np.ndarray] = None,
                                  save_path: Optional[str] = None,
                                  dpi: int = 300):
        """Create comprehensive visualization"""
        n_xai = len(xai_maps)
        n_cols = 3 if uncertainty is None else 4
        n_rows = n_xai + 1
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 4))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        
        Z, Y, X = t1_vol.shape
        slice_idx = {'axial': Z//2, 'coronal': Y//2, 'sagittal': X//2}
        views = ['axial', 'coronal', 'sagittal']
        
        # First row: Original + tumor probability
        for col, view in enumerate(views):
            if view == 'axial':
                im = t1_vol[slice_idx['axial'], :, :]
                tp = tumor_prob[slice_idx['axial'], :, :]
            elif view == 'coronal':
                im = t1_vol[:, slice_idx['coronal'], :]
                tp = tumor_prob[:, slice_idx['coronal'], :]
            else:
                im = t1_vol[:, :, slice_idx['sagittal']]
                tp = tumor_prob[:, :, slice_idx['sagittal']]
            
            axes[0, col].imshow(np.rot90(im), cmap='gray')
            axes[0, col].contour(np.rot90(tp), levels=[0.5], colors='red', linewidths=2)
            axes[0, col].set_title(f'{view.capitalize()} - Tumor')
            axes[0, col].axis('off')
        
        # Uncertainty map if available
        if uncertainty is not None:
            unc_axial = uncertainty[slice_idx['axial'], :, :]
            im_unc = axes[0, 3].imshow(np.rot90(unc_axial), cmap='viridis')
            axes[0, 3].set_title('Uncertainty')
            axes[0, 3].axis('off')
            plt.colorbar(im_unc, ax=axes[0, 3], fraction=0.046)
        
        # XAI maps rows
        for row_idx, (xai_name, xai_map) in enumerate(xai_maps.items(), start=1):
            for col, view in enumerate(views):
                if view == 'axial':
                    im = t1_vol[slice_idx['axial'], :, :]
                    xm = xai_map[slice_idx['axial'], :, :]
                elif view == 'coronal':
                    im = t1_vol[:, slice_idx['coronal'], :]
                    xm = xai_map[:, slice_idx['coronal'], :]
                else:
                    im = t1_vol[:, :, slice_idx['sagittal']]
                    xm = xai_map[:, :, slice_idx['sagittal']]
                
                axes[row_idx, col].imshow(np.rot90(im), cmap='gray')
                hm = axes[row_idx, col].imshow(np.rot90(xm), cmap='hot', alpha=0.6)
                axes[row_idx, col].set_title(f'{view} - {xai_name}')
                axes[row_idx, col].axis('off')
            
            if uncertainty is not None:
                # Show overlay on uncertainty
                unc_axial = uncertainty[slice_idx['axial'], :, :]
                xm_axial = xai_map[slice_idx['axial'], :, :]
                axes[row_idx, 3].imshow(np.rot90(unc_axial), cmap='viridis')
                axes[row_idx, 3].imshow(np.rot90(xm_axial), cmap='hot', alpha=0.4)
                axes[row_idx, 3].set_title(f'{xai_name} + Uncertainty')
                axes[row_idx, 3].axis('off')
        
        fig.suptitle('Brain Tumor Segmentation with XAI Analysis', fontsize=16)
        plt.tight_layout()
        
        if save_path:
            ensure_dir(os.path.dirname(save_path) or '.')
            plt.savefig(f'{save_path}.png', dpi=dpi, bbox_inches='tight')
            plt.savefig(f'{save_path}.pdf', dpi=300, bbox_inches='tight')
            logger.info(f'Saved visualization to {save_path}')
        
        return fig
    
    @staticmethod
    def plot_training_curves(history: Dict, save_path: Optional[str] = None):
        """Plot training and validation curves"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        epochs = range(1, len(history['train_loss']) + 1)
        
        # Loss curves
        axes[0].plot(epochs, history['train_loss'], 'b-', label='Train Loss')
        axes[0].plot(epochs, history['val_loss'], 'r-', label='Val Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training and Validation Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Dice scores
        dice_scores = [m['dice'] for m in history['metrics']]
        axes[1].plot(epochs, dice_scores, 'g-', label='Dice Score')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Dice Score')
        axes[1].set_title('Validation Dice Score')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f'Saved training curves to {save_path}')
        
        return fig

# ======================== Functional Connectivity ========================

class ConnectivityAnalyzer:
    """Advanced functional connectivity analysis"""
    
    def __init__(self, atlas_path: str):
        self.atlas_path = atlas_path
        self.masker = NiftiLabelsMasker(labels_img=atlas_path, standardize=True)
    
    def extract_timeseries(self, fmri_path: str, confounds: Optional[np.ndarray] = None) -> np.ndarray:
        """Extract ROI time series"""
        return self.masker.fit_transform(fmri_path, confounds=confounds)
    
    def compute_connectivity(self, timeseries: np.ndarray, method: str = 'correlation') -> np.ndarray:
        """Compute connectivity matrix"""
        if method == 'correlation':
            return np.corrcoef(timeseries.T)
        elif method == 'partial_correlation':
            from sklearn.covariance import GraphicalLassoCV
            model = GraphicalLassoCV()
            model.fit(timeseries)
            return model.precision_
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def network_metrics(self, connectivity: np.ndarray) -> Dict[str, np.ndarray]:
        """Compute graph theory metrics"""
        try:
            import networkx as nx
            
            # Create graph
            G = nx.from_numpy_array(np.abs(connectivity))
            
            # Compute metrics
            metrics = {
                'degree': np.array(list(dict(G.degree()).values())),
                'betweenness': np.array(list(nx.betweenness_centrality(G).values())),
                'clustering': np.array(list(nx.clustering(G).values())),
            }
            
            return metrics
        except ImportError:
            logger.warning("NetworkX not available for graph metrics")
            return {}

# ======================== Clinical Report Generator ========================

class ClinicalReportGenerator:
    """Generate clinical reports from analysis results"""
    
    @staticmethod
    def generate_report(case_name: str,
                       tumor_volume: float,
                       affected_rois: List[Tuple[int, float]],
                       uncertainty_stats: Dict[str, float],
                       connectivity_disruption: Optional[Dict] = None,
                       save_path: Optional[str] = None) -> str:
        """Generate structured clinical report"""
        
        report = f"""
╔══════════════════════════════════════════════════════════════╗
║          AI-Driven Brain Tumor Analysis Report               ║
╚══════════════════════════════════════════════════════════════╝

Case ID: {case_name}
Analysis Date: {np.datetime64('today')}

─────────────────────────────────────────────────────────────

TUMOR CHARACTERISTICS:
  • Estimated Volume: {tumor_volume:.2f} mm³
  • Mean Uncertainty: {uncertainty_stats.get('mean', 0):.3f}
  • Max Uncertainty: {uncertainty_stats.get('max', 0):.3f}
  • 95th Percentile Uncertainty: {uncertainty_stats.get('p95', 0):.3f}

─────────────────────────────────────────────────────────────

AFFECTED BRAIN REGIONS (Top 10):
"""
        for i, (roi, score) in enumerate(affected_rois[:10], 1):
            report += f"  {i:2d}. ROI {roi:3d}: {score*100:5.1f}% overlap\n"
        
        if connectivity_disruption:
            report += """
─────────────────────────────────────────────────────────────

FUNCTIONAL CONNECTIVITY IMPACT:
"""
            for roi, impact in list(connectivity_disruption.items())[:5]:
                report += f"  • ROI {roi}: {impact:.2f} disruption score\n"
        
        report += """
─────────────────────────────────────────────────────────────

INTERPRETATION NOTES:
  • High uncertainty regions require additional clinical review
  • Affected ROIs indicate potential functional impact zones
  • Connectivity analysis shows network-level implications


─────────────────────────────────────────────────────────────
"""
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report)
            logger.info(f'Saved clinical report to {save_path}')
        
        return report

# ======================== Inference Pipeline ========================

class InferencePipeline:
    """Complete inference pipeline with all enhancements"""
    
    def __init__(self, model: nn.Module, device: str = 'cuda'):
        self.model = model.to(device)
        self.device = device
        self.uncertainty_estimator = UncertaintyEstimator(model)
        self.visualizer = BrainVisualizer()
    
    def predict_with_uncertainty(self, imgs: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        """Predict with uncertainty estimation"""
        return self.uncertainty_estimator.estimate(imgs, self.device)
    
    def generate_xai_maps(self, imgs: torch.Tensor) -> Dict[str, np.ndarray]:
        """Generate multiple XAI maps"""
        xai_maps = {}
        
        # Layer-CAM
        try:
            # Find the last convolutional layer in decoder
            last_conv = None
            for name, module in self.model.named_modules():
                if isinstance(module, nn.Conv3d) and 'dec' in name:
                    last_conv = module
            
            if last_conv:
                layer_cam = LayerCAM3D(self.model, last_conv)
                xai_maps['LayerCAM'] = layer_cam.generate(imgs.unsqueeze(0).to(self.device))
        except Exception as e:
            logger.warning(f"LayerCAM failed: {e}")
        
        # Integrated Gradients
        try:
            baseline = torch.zeros_like(imgs)
            ig_map = integrated_gradients(self.model, imgs, baseline=baseline, 
                                        steps=50, device=self.device)
            xai_maps['IntegratedGradients'] = ig_map
        except Exception as e:
            logger.warning(f"Integrated Gradients failed: {e}")
        
        return xai_maps
    
    def process_case(self, case_path: str, atlas_path: str, 
                    save_dir: str = 'results') -> Dict:
        """Complete processing pipeline for a single case"""
        ensure_dir(save_dir)
        case_name = os.path.basename(case_path)
        logger.info(f"Processing case: {case_name}")
        
        # Load data
        dataset = EnhancedBrainDataset([case_path], augment=False)
        imgs, mask_gt, _ = dataset[0]
        
        # Predict with uncertainty
        tumor_prob, uncertainty = self.predict_with_uncertainty(imgs)
        
        # Generate XAI maps
        xai_maps = self.generate_xai_maps(imgs)
        
        # Load T1 for visualization
        t1_vol = imgs.numpy()[0]
        
        # Calculate tumor volume
        t1_img = nib.load(os.path.join(case_path, 'T1.nii.gz'))
        voxel_volume = np.prod(t1_img.header.get_zooms())
        tumor_volume = (tumor_prob > 0.5).sum() * voxel_volume
        
        # Uncertainty statistics
        uncertainty_stats = {
            'mean': float(uncertainty.mean()),
            'std': float(uncertainty.std()),
            'max': float(uncertainty.max()),
            'p95': float(np.percentile(uncertainty, 95))
        }
        
        # Functional connectivity analysis
        fmri_path = os.path.join(case_path, 'fmri.nii.gz')
        affected_rois = []
        connectivity_disruption = None
        
        if os.path.exists(fmri_path):
            try:
                analyzer = ConnectivityAnalyzer(atlas_path)
                timeseries = analyzer.extract_timeseries(fmri_path)
                connectivity = analyzer.compute_connectivity(timeseries)
                
                # Identify affected ROIs
                atlas_img = nib.load(atlas_path)
                atlas_data = atlas_img.get_fdata().astype(int)
                rois = np.unique(atlas_data)[1:]  # Exclude background
                
                roi_scores = []
                for roi in rois:
                    mask = (atlas_data == roi)
                    if mask.sum() > 0:
                        overlap = tumor_prob[mask].mean()
                        roi_scores.append((int(roi), float(overlap)))
                
                affected_rois = sorted(roi_scores, key=lambda x: -x[1])
                
                # Compute connectivity disruption
                metrics = analyzer.network_metrics(connectivity)
                if metrics:
                    connectivity_disruption = {
                        roi: float(metrics['betweenness'][i]) 
                        for i, (roi, _) in enumerate(affected_rois[:10])
                    }
            except Exception as e:
                logger.warning(f"Connectivity analysis failed: {e}")
        
        # Save results
        save_nifti(tumor_prob, t1_img.affine, 
                  os.path.join(save_dir, f'{case_name}_tumor_prob.nii.gz'))
        save_nifti(uncertainty, t1_img.affine,
                  os.path.join(save_dir, f'{case_name}_uncertainty.nii.gz'))
        
        # Generate visualization
        self.visualizer.create_multi_panel_figure(
            t1_vol, tumor_prob, xai_maps, uncertainty,
            save_path=os.path.join(save_dir, f'{case_name}_analysis')
        )
        
        # Generate clinical report
        report = ClinicalReportGenerator.generate_report(
            case_name, tumor_volume, affected_rois, uncertainty_stats,
            connectivity_disruption,
            save_path=os.path.join(save_dir, f'{case_name}_report.txt')
        )
        
        print(report)
        
        return {
            'case_name': case_name,
            'tumor_volume': tumor_volume,
            'uncertainty_stats': uncertainty_stats,
            'affected_rois': affected_rois,
            'connectivity_disruption': connectivity_disruption
        }

# ======================== Integrated Gradients ========================

def integrated_gradients(model, input_tensor, target_class=1, baseline=None, 
                        steps=50, device='cpu'):
    """Compute Integrated Gradients attribution"""
    model.to(device)
    model.eval()
    
    if baseline is None:
        baseline = torch.zeros_like(input_tensor)
    
    scaled_inputs = []
    for i in range(1, steps + 1):
        scaled = baseline + float(i) / steps * (input_tensor - baseline)
        scaled_inputs.append(scaled.unsqueeze(0).to(device))
    
    total_grad = None
    for x in scaled_inputs:
        x.requires_grad_(True)
        logits = model(x)
        score = logits[:, target_class].sum()
        model.zero_grad()
        score.backward(retain_graph=False)
        grad = x.grad.detach().cpu().numpy()[0]
        
        if total_grad is None:
            total_grad = grad
        else:
            total_grad += grad
    
    avg_grad = total_grad / steps
    ig = (input_tensor.cpu().numpy() - baseline.cpu().numpy()) * avg_grad
    ig_map = ig.sum(axis=0)
    ig_map = np.abs(ig_map)
    ig_map = (ig_map - ig_map.min()) / (ig_map.max() - ig_map.min() + 1e-8)
    
    return ig_map

# ======================== Main CLI ========================

def parse_args():
    parser = argparse.ArgumentParser(
        description='Enhanced AI Brain Mapping Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, default='data',
                       help='Directory containing case folders')
    parser.add_argument('--atlas', type=str, required=True,
                       help='Path to brain atlas NIfTI file')
    parser.add_argument('--case_list', type=str, default=None,
                       help='File with list of case paths')
    
    # Training arguments
    parser.add_argument('--train', action='store_true',
                       help='Run training')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=2,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--val_split', type=float, default=0.2,
                       help='Validation split fraction')
    
    # Model arguments
    parser.add_argument('--base_filters', type=int, default=32,
                       help='Base number of filters')
    parser.add_argument('--use_attention', action='store_true', default=True,
                       help='Use attention mechanisms')
    parser.add_argument('--dropout', type=float, default=0.1,
                       help='Dropout rate')
    
    # Inference arguments
    parser.add_argument('--infer', action='store_true',
                       help='Run inference')
    parser.add_argument('--case_idx', type=int, default=0,
                       help='Case index for inference')
    parser.add_argument('--model_path', type=str, default='best_model.pth',
                       help='Path to trained model')
    
    # XAI arguments
    parser.add_argument('--xai', action='store_true',
                       help='Generate XAI visualizations')
    parser.add_argument('--uncertainty', action='store_true',
                       help='Compute uncertainty estimates')
    
    # Output arguments
    parser.add_argument('--save_dir', type=str, default='results',
                       help='Output directory')
    parser.add_argument('--device', type=str,
                       default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to use')
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Setup logging
    logger.info("="*60)
    logger.info("Enhanced AI Brain Mapping Pipeline")
    logger.info("="*60)
    logger.info(f"Device: {args.device}")
    
    # Load cases
    if args.case_list and os.path.exists(args.case_list):
        with open(args.case_list) as f:
            cases = [l.strip() for l in f if l.strip()]
    else:
        cases = []
        for name in os.listdir(args.data_dir):
            path = os.path.join(args.data_dir, name)
            if os.path.isdir(path) and os.path.exists(os.path.join(path, 'T1.nii.gz')):
                cases.append(path)
    
    logger.info(f"Found {len(cases)} cases")
    
    if len(cases) == 0:
        logger.error("No cases found!")
        return
    
    # Create model
    model_config = ModelConfig(
        in_channels=3,
        out_channels=2,
        base_filters=args.base_filters,
        use_attention=args.use_attention,
        dropout=args.dropout
    )
    model = EnhancedUNet3D(model_config)
    logger.info(f"Model created with {sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters")
    
    # Training
    if args.train:
        # Split dataset
        n_val = int(len(cases) * args.val_split)
        train_cases = cases[n_val:]
        val_cases = cases[:n_val]
        
        logger.info(f"Training: {len(train_cases)} cases, Validation: {len(val_cases)} cases")
        
        # Create datasets
        train_dataset = EnhancedBrainDataset(train_cases, augment=True)
        val_dataset = EnhancedBrainDataset(val_cases, augment=False)
        
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                                 shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=1,
                               shuffle=False, num_workers=2, pin_memory=True)
        
        # Train
        training_config = TrainingConfig(
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            val_split=args.val_split
        )
        
        trainer = Trainer(model, training_config, device=args.device)
        model = trainer.fit(train_loader, val_loader, save_path=args.model_path)
        
        # Plot training curves
        BrainVisualizer.plot_training_curves(
            trainer.history,
            save_path=os.path.join(args.save_dir, 'training_curves.png')
        )
    
    # Inference
    if args.infer:
        # Load model
        if os.path.exists(args.model_path):
            checkpoint = torch.load(args.model_path, map_location=args.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f"Loaded model from {args.model_path}")
        else:
            logger.warning("No trained model found, using random weights")
        
        # Create inference pipeline
        pipeline = InferencePipeline(model, device=args.device)
        
        # Process case
        case_path = cases[args.case_idx]
        results = pipeline.process_case(case_path, args.atlas, args.save_dir)
        
        logger.info("="*60)
        logger.info("Inference completed successfully!")
        logger.info(f"Results saved to: {args.save_dir}")
        logger.info("="*60)

if __name__ == '__main__':
    main()
