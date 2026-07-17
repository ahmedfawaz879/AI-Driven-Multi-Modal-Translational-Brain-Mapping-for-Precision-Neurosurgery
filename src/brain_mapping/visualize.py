"""Multi-panel matplotlib visualizations for tumor/XAI/uncertainty maps.

Extracted from the original script (BrainVisualizer). Logic is unchanged.
"""

import os
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np

from .utils import ensure_dir, logger

# ======================== Visualization ========================


class BrainVisualizer:
    """Advanced visualization utilities"""

    @staticmethod
    def create_multi_panel_figure(
        t1_vol: np.ndarray,
        tumor_prob: np.ndarray,
        xai_maps: Dict[str, np.ndarray],
        uncertainty: Optional[np.ndarray] = None,
        save_path: Optional[str] = None,
        dpi: int = 300,
    ):
        """Create comprehensive visualization"""
        n_xai = len(xai_maps)
        n_cols = 3 if uncertainty is None else 4
        n_rows = n_xai + 1

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 4))
        if n_rows == 1:
            axes = axes.reshape(1, -1)

        Z, Y, X = t1_vol.shape
        slice_idx = {"axial": Z // 2, "coronal": Y // 2, "sagittal": X // 2}
        views = ["axial", "coronal", "sagittal"]

        # First row: Original + tumor probability
        for col, view in enumerate(views):
            if view == "axial":
                im = t1_vol[slice_idx["axial"], :, :]
                tp = tumor_prob[slice_idx["axial"], :, :]
            elif view == "coronal":
                im = t1_vol[:, slice_idx["coronal"], :]
                tp = tumor_prob[:, slice_idx["coronal"], :]
            else:
                im = t1_vol[:, :, slice_idx["sagittal"]]
                tp = tumor_prob[:, :, slice_idx["sagittal"]]

            axes[0, col].imshow(np.rot90(im), cmap="gray")
            axes[0, col].contour(np.rot90(tp), levels=[0.5], colors="red", linewidths=2)
            axes[0, col].set_title(f"{view.capitalize()} - Tumor")
            axes[0, col].axis("off")

        # Uncertainty map if available
        if uncertainty is not None:
            unc_axial = uncertainty[slice_idx["axial"], :, :]
            im_unc = axes[0, 3].imshow(np.rot90(unc_axial), cmap="viridis")
            axes[0, 3].set_title("Uncertainty")
            axes[0, 3].axis("off")
            plt.colorbar(im_unc, ax=axes[0, 3], fraction=0.046)

        # XAI maps rows
        for row_idx, (xai_name, xai_map) in enumerate(xai_maps.items(), start=1):
            for col, view in enumerate(views):
                if view == "axial":
                    im = t1_vol[slice_idx["axial"], :, :]
                    xm = xai_map[slice_idx["axial"], :, :]
                elif view == "coronal":
                    im = t1_vol[:, slice_idx["coronal"], :]
                    xm = xai_map[:, slice_idx["coronal"], :]
                else:
                    im = t1_vol[:, :, slice_idx["sagittal"]]
                    xm = xai_map[:, :, slice_idx["sagittal"]]

                axes[row_idx, col].imshow(np.rot90(im), cmap="gray")
                axes[row_idx, col].imshow(np.rot90(xm), cmap="hot", alpha=0.6)
                axes[row_idx, col].set_title(f"{view} - {xai_name}")
                axes[row_idx, col].axis("off")

            if uncertainty is not None:
                # Show overlay on uncertainty
                unc_axial = uncertainty[slice_idx["axial"], :, :]
                xm_axial = xai_map[slice_idx["axial"], :, :]
                axes[row_idx, 3].imshow(np.rot90(unc_axial), cmap="viridis")
                axes[row_idx, 3].imshow(np.rot90(xm_axial), cmap="hot", alpha=0.4)
                axes[row_idx, 3].set_title(f"{xai_name} + Uncertainty")
                axes[row_idx, 3].axis("off")

        fig.suptitle("Brain Tumor Segmentation with XAI Analysis", fontsize=16)
        plt.tight_layout()

        if save_path:
            ensure_dir(os.path.dirname(save_path) or ".")
            plt.savefig(f"{save_path}.png", dpi=dpi, bbox_inches="tight")
            plt.savefig(f"{save_path}.pdf", dpi=300, bbox_inches="tight")
            logger.info(f"Saved visualization to {save_path}")

        return fig

    @staticmethod
    def plot_training_curves(history: Dict, save_path: Optional[str] = None):
        """Plot training and validation curves"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        epochs = range(1, len(history["train_loss"]) + 1)

        # Loss curves
        axes[0].plot(epochs, history["train_loss"], "b-", label="Train Loss")
        axes[0].plot(epochs, history["val_loss"], "r-", label="Val Loss")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].set_title("Training and Validation Loss")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Dice scores
        dice_scores = [m["dice"] for m in history["metrics"]]
        axes[1].plot(epochs, dice_scores, "g-", label="Dice Score")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Dice Score")
        axes[1].set_title("Validation Dice Score")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Saved training curves to {save_path}")

        return fig
