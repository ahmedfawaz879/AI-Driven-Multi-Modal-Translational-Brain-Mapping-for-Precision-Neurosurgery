"""Explainability and uncertainty methods: MC-Dropout, Layer-CAM, Integrated Gradients.

Extracted from the original script (UncertaintyEstimator, LayerCAM3D,
integrated_gradients). Logic is unchanged. SHAP is intentionally not
wired up here beyond the optional-import guard in cli.py/inference.py -
the original script never actually called shap anywhere despite
advertising it; see README Limitations.
"""

from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

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

    def estimate(self, x: torch.Tensor, device: str = "cuda") -> Tuple[np.ndarray, np.ndarray]:
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
        cam = F.interpolate(cam, size=x.shape[2:], mode="trilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()

        # Normalize
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        return cam


# ======================== Integrated Gradients ========================


def integrated_gradients(model, input_tensor, target_class=1, baseline=None, steps=50, device="cpu"):
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
