"""3D attention U-Net architecture and the combined training loss.

Extracted from the original script (AttentionBlock3D, ResidualBlock3D,
EnhancedUNet3D, CombinedLoss). Logic is unchanged.

Note: ``ModelConfig.depth`` is accepted by the config but, as in the
original script, the encoder/decoder depth of EnhancedUNet3D is hardcoded
to 4 levels (enc1..enc4 + bottleneck) rather than being driven by
``config.depth``. This is a pre-existing inconsistency carried over from
the original single-file script, not something introduced by this
refactor.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig

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
        self.enc1 = ResidualBlock3D(
            config.in_channels, config.base_filters, config.use_attention, config.dropout
        )
        self.pool1 = nn.MaxPool3d(2)

        self.enc2 = ResidualBlock3D(
            config.base_filters, config.base_filters * 2, config.use_attention, config.dropout
        )
        self.pool2 = nn.MaxPool3d(2)

        self.enc3 = ResidualBlock3D(
            config.base_filters * 2, config.base_filters * 4, config.use_attention, config.dropout
        )
        self.pool3 = nn.MaxPool3d(2)

        self.enc4 = ResidualBlock3D(
            config.base_filters * 4, config.base_filters * 8, config.use_attention, config.dropout
        )
        self.pool4 = nn.MaxPool3d(2)

        # Bottleneck
        self.bottleneck = ResidualBlock3D(
            config.base_filters * 8, config.base_filters * 16, True, config.dropout
        )

        # Decoder
        self.up4 = nn.ConvTranspose3d(config.base_filters * 16, config.base_filters * 8, 2, 2)
        self.dec4 = ResidualBlock3D(
            config.base_filters * 16, config.base_filters * 8, config.use_attention, config.dropout
        )

        self.up3 = nn.ConvTranspose3d(config.base_filters * 8, config.base_filters * 4, 2, 2)
        self.dec3 = ResidualBlock3D(
            config.base_filters * 8, config.base_filters * 4, config.use_attention, config.dropout
        )

        self.up2 = nn.ConvTranspose3d(config.base_filters * 4, config.base_filters * 2, 2, 2)
        self.dec2 = ResidualBlock3D(
            config.base_filters * 4, config.base_filters * 2, config.use_attention, config.dropout
        )

        self.up1 = nn.ConvTranspose3d(config.base_filters * 2, config.base_filters, 2, 2)
        self.dec1 = ResidualBlock3D(
            config.base_filters * 2, config.base_filters, config.use_attention, config.dropout
        )

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

    def __init__(
        self,
        ce_weight: float = 1.0,
        dice_weight: float = 1.0,
        focal_weight: float = 0.5,
        alpha: float = 0.25,
        gamma: float = 2.0,
    ):
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
        return 1 - (2.0 * inter + smooth) / (union + smooth)

    def focal_loss(self, pred, target):
        """Focal loss"""
        ce = F.cross_entropy(pred, target, reduction="none")
        pt = torch.exp(-ce)
        focal = self.alpha * (1 - pt) ** self.gamma * ce
        return focal.mean()

    def forward(self, pred, target):
        ce_loss = self.ce(pred, target)
        dice = self.dice_loss(pred, target)
        focal = self.focal_loss(pred, target)

        total = self.ce_weight * ce_loss + self.dice_weight * dice + self.focal_weight * focal

        return total, {
            "ce": ce_loss.item(),
            "dice": dice.item(),
            "focal": focal.item(),
            "total": total.item(),
        }
