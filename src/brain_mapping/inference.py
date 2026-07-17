"""End-to-end single-case inference: predict, explain, visualize, report.

Extracted from the original script (InferencePipeline, incl. process_case).
Logic is unchanged.
"""

import os
from typing import Dict, Tuple

import nibabel as nib
import numpy as np
import torch
import torch.nn as nn

from .connectivity import ConnectivityAnalyzer
from .data import EnhancedBrainDataset, save_nifti
from .explain import LayerCAM3D, UncertaintyEstimator, integrated_gradients
from .report import ClinicalReportGenerator
from .utils import ensure_dir, logger
from .visualize import BrainVisualizer

# ======================== Inference Pipeline ========================


class InferencePipeline:
    """Complete inference pipeline with all enhancements"""

    def __init__(self, model: nn.Module, device: str = "cuda"):
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
                if isinstance(module, nn.Conv3d) and "dec" in name:
                    last_conv = module

            if last_conv:
                layer_cam = LayerCAM3D(self.model, last_conv)
                xai_maps["LayerCAM"] = layer_cam.generate(imgs.unsqueeze(0).to(self.device))
        except Exception as e:
            logger.warning(f"LayerCAM failed: {e}")

        # Integrated Gradients
        try:
            baseline = torch.zeros_like(imgs)
            ig_map = integrated_gradients(
                self.model, imgs, baseline=baseline, steps=50, device=self.device
            )
            xai_maps["IntegratedGradients"] = ig_map
        except Exception as e:
            logger.warning(f"Integrated Gradients failed: {e}")

        return xai_maps

    def process_case(self, case_path: str, atlas_path: str, save_dir: str = "results") -> Dict:
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
        t1_img = nib.load(os.path.join(case_path, "T1.nii.gz"))
        voxel_volume = np.prod(t1_img.header.get_zooms())
        tumor_volume = (tumor_prob > 0.5).sum() * voxel_volume

        # Uncertainty statistics
        uncertainty_stats = {
            "mean": float(uncertainty.mean()),
            "std": float(uncertainty.std()),
            "max": float(uncertainty.max()),
            "p95": float(np.percentile(uncertainty, 95)),
        }

        # Functional connectivity analysis
        fmri_path = os.path.join(case_path, "fmri.nii.gz")
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
                    mask = atlas_data == roi
                    if mask.sum() > 0:
                        overlap = tumor_prob[mask].mean()
                        roi_scores.append((int(roi), float(overlap)))

                affected_rois = sorted(roi_scores, key=lambda x: -x[1])

                # Compute connectivity disruption
                metrics = analyzer.network_metrics(connectivity)
                if metrics:
                    connectivity_disruption = {
                        roi: float(metrics["betweenness"][i])
                        for i, (roi, _) in enumerate(affected_rois[:10])
                    }
            except Exception as e:
                logger.warning(f"Connectivity analysis failed: {e}")

        # Save results
        save_nifti(
            tumor_prob, t1_img.affine, os.path.join(save_dir, f"{case_name}_tumor_prob.nii.gz")
        )
        save_nifti(
            uncertainty, t1_img.affine, os.path.join(save_dir, f"{case_name}_uncertainty.nii.gz")
        )

        # Generate visualization
        self.visualizer.create_multi_panel_figure(
            t1_vol,
            tumor_prob,
            xai_maps,
            uncertainty,
            save_path=os.path.join(save_dir, f"{case_name}_analysis"),
        )

        # Generate clinical report
        report = ClinicalReportGenerator.generate_report(
            case_name,
            tumor_volume,
            affected_rois,
            uncertainty_stats,
            connectivity_disruption,
            save_path=os.path.join(save_dir, f"{case_name}_report.txt"),
        )

        logger.info("\n" + report)

        return {
            "case_name": case_name,
            "tumor_volume": tumor_volume,
            "uncertainty_stats": uncertainty_stats,
            "affected_rois": affected_rois,
            "connectivity_disruption": connectivity_disruption,
        }
