# AI-Driven Multi-Modal Translational Brain Mapping for Precision Neurosurgery

This repository implements a **full end-to-end pipeline** for translational brain mapping using **AI and machine learning**. It integrates multi-modal MRI and fMRI data to generate individualized brain maps, supports tumor segmentation, and provides voxel-level interpretability using **XAI methods** such as Grad-CAM, Integrated Gradients, and SHAP.

## Features

* Multi-modal MRI (T1, T2, FLAIR) tumor segmentation with 3D U-Net
* Functional connectivity extraction from fMRI using ROI atlases
* Fusion of tumor maps with connectome to assess tumor impact on brain networks
* Explainable AI (XAI) methods:

  * 3D Grad-CAM
  * Integrated Gradients
  * SHAP voxel-level importance maps
* Publication-ready figures and 3D interactive Plotly visualizations
* CLI and modular Python functions for training, inference, and visualization

## Installation

Install Python dependencies:

```bash
pip install numpy scipy matplotlib plotly imageio scikit-image torch torchvision tqdm kaleido shap nibabel nilearn
```

## Dataset Structure

Expected directory structure:

```
data/
  case001/
    T1.nii.gz
    T2.nii.gz
    FLAIR.nii.gz
    fmri.nii.gz  # optional
    mask.nii.gz  # optional, ground truth
  case002/
    ...
```

* `T1/T2/FLAIR.nii.gz`: structural MRI volumes
* `fmri.nii.gz`: resting-state fMRI for connectivity
* `mask.nii.gz`: optional tumor segmentation for training

## Usage

### Training

```bash
python main.py --data_dir data --atlas atlas.nii.gz --run_train --epochs 10 --batch_size 2 --save_model unet3d.pth
```

### Inference + Visualization

```bash
python main.py --data_dir data --atlas atlas.nii.gz --run_infer --run_xai --case_idx 0 --save_dir results
```

### Outputs

* Tumor probability NIfTI volume: `caseXXX/tumor_prob.nii.gz`
* Publication-ready 3-panel PNG and PDF figures with XAI overlays
* Interactive 3D Plotly visualization of tumor voxels, ROI centroids, and connectivity edges

## Explainability (XAI)

* Grad-CAM 3D: highlights important regions in feature maps for tumor prediction
* Integrated Gradients: computes contribution of each voxel to the tumor prediction
* SHAP (optional): voxel-level attribution map if `shap` installed

