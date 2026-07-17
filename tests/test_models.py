"""Tests for brain_mapping.models: forward-pass shapes and loss computation.

Uses small synthetic tensors (no real NIfTI/BraTS data needed) and a
tiny ModelConfig (base_filters=4, small spatial size) so these run fast
on CPU.

NOTE: requires torch to be installed.
"""

import torch

from brain_mapping.config import ModelConfig
from brain_mapping.models import AttentionBlock3D, CombinedLoss, EnhancedUNet3D, ResidualBlock3D

# EnhancedUNet3D has 4 hardcoded pooling stages (2x each) regardless of
# ModelConfig.depth. Spatial dims must be divisible by 2**4 = 16 AND the
# bottleneck (input / 16) must be > 1 in each dimension, since InstanceNorm3d
# in the bottleneck's ResidualBlock3D requires more than one spatial element
# per instance/channel. 16 itself collapses to a 1x1x1 bottleneck and fails;
# 32 leaves a 2x2x2 bottleneck.
SPATIAL = 32


def _tiny_model_config(**overrides):
    kwargs = dict(
        in_channels=2,
        out_channels=2,
        # AttentionBlock3D does `channels // 8` for its query/key projection
        # (see models.py) - base_filters below 8 makes that 0 at the first
        # encoder stage and breaks the forward pass. 8 is the practical
        # floor for any config with use_attention=True.
        base_filters=8,
        use_attention=True,
        use_residual=True,
        dropout=0.0,
    )
    kwargs.update(overrides)
    return ModelConfig(**kwargs)


def test_attention_block_preserves_shape():
    block = AttentionBlock3D(channels=8)
    x = torch.randn(1, 8, 4, 4, 4)
    out = block(x)
    assert out.shape == x.shape


def test_residual_block_changes_channels_only():
    block = ResidualBlock3D(in_ch=4, out_ch=8, use_attention=False, dropout=0.0)
    x = torch.randn(2, 4, 8, 8, 8)
    out = block(x)
    assert out.shape == (2, 8, 8, 8, 8)


def test_unet_forward_pass_shape():
    config = _tiny_model_config()
    model = EnhancedUNet3D(config)
    model.eval()

    x = torch.randn(1, config.in_channels, SPATIAL, SPATIAL, SPATIAL)
    with torch.no_grad():
        out = model(x)

    assert out.shape == (1, config.out_channels, SPATIAL, SPATIAL, SPATIAL)


def test_unet_forward_pass_batch_and_no_attention():
    config = _tiny_model_config(use_attention=False)
    model = EnhancedUNet3D(config)
    model.eval()

    x = torch.randn(2, config.in_channels, SPATIAL, SPATIAL, SPATIAL)
    with torch.no_grad():
        out = model(x)

    assert out.shape == (2, config.out_channels, SPATIAL, SPATIAL, SPATIAL)


def test_combined_loss_returns_scalar_and_components():
    criterion = CombinedLoss()
    pred = torch.randn(2, 2, 4, 4, 4, requires_grad=True)
    target = torch.randint(0, 2, (2, 4, 4, 4))

    loss, components = criterion(pred, target)

    assert loss.dim() == 0
    assert torch.isfinite(loss)
    for key in ("ce", "dice", "focal", "total"):
        assert key in components
        assert components[key] == components[key]  # not NaN

    # Loss should be differentiable w.r.t. predictions
    loss.backward()
    assert pred.grad is not None


def test_combined_loss_perfect_prediction_has_low_dice_loss():
    criterion = CombinedLoss()
    target = torch.zeros(1, 4, 4, 4, dtype=torch.long)
    target[0, :2, :2, :2] = 1

    # Build near-perfect logits: large positive logit for correct class
    pred = torch.zeros(1, 2, 4, 4, 4)
    pred[:, 1] = torch.where(target == 1, torch.tensor(10.0), torch.tensor(-10.0))
    pred[:, 0] = -pred[:, 1]

    _, components = criterion(pred, target)
    assert components["dice"] < 0.05
