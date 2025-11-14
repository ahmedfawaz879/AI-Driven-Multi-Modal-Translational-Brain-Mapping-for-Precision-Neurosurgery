#  AI-Driven Multi-Modal Translational Brain Mapping for Precision Neurosurgery

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A state-of-the-art **deep learning pipeline** for translational brain mapping that combines multi-modal neuroimaging with explainable AI to support precision neurosurgery planning. This framework integrates structural MRI, functional connectivity, and advanced XAI methods to generate comprehensive, interpretable brain tumor analysis.

##  Overview

This repository implements a complete end-to-end pipeline featuring:

- **Advanced 3D U-Net architecture** with attention mechanisms and residual connections
- **Multi-modal MRI fusion** (T1, T2, FLAIR) for robust tumor segmentation
- **Uncertainty quantification** via Monte Carlo Dropout for prediction confidence
- **Functional connectivity analysis** from resting-state fMRI
- **State-of-the-art XAI methods**: Layer-CAM, Integrated Gradients, SHAP
- **Clinical report generation** with tumor characterization and affected brain regions
- **Production-ready training** with mixed precision, gradient accumulation, and early stopping

##  Key Features

###  **Advanced Architecture**
- **Enhanced 3D U-Net** with attention blocks for improved feature extraction
- **Residual connections** throughout encoder-decoder for better gradient flow
- **Multi-scale feature fusion** with skip connections
- **Instance normalization** for stable training across diverse MRI protocols

###  **Sophisticated Training Pipeline**
- **Mixed precision training (AMP)** - 2x faster with lower memory footprint
- **Gradient accumulation** - train with effectively larger batch sizes
- **Combined loss function** - Cross-Entropy + Dice + Focal Loss
- **Smart early stopping** with validation monitoring
- **Learning rate scheduling** with ReduceLROnPlateau
- **K-fold cross-validation** support

###  **Advanced Data Processing**
- **Multiple normalization methods**: Z-score, Percentile clipping, Nyúl standardization
- **Rich 3D augmentation**: Random flips, rotations, scaling, noise injection, gamma correction
- **Robust preprocessing** with automatic error handling and logging
- **Multi-modal registration** support

###  **Explainable AI (XAI)**
- **Layer-CAM** - More accurate than traditional Grad-CAM for 3D medical images
- **Integrated Gradients** - Voxel-level attribution with path integration
- **SHAP analysis** - Shapley-based feature importance (optional)
- **Uncertainty maps** - Epistemic uncertainty via Monte Carlo Dropout
- **Multi-method comparison** - Side-by-side XAI visualization

###  **Clinical Integration**
- **Automated tumor volume estimation** with proper voxel spacing
- **Affected ROI ranking** with overlap percentages
- **Functional connectivity impact** assessment
- **Structured clinical reports** in human-readable format
- **Publication-quality visualizations** (PNG, PDF, interactive HTML)

###  **Functional Connectivity**
- **ROI time-series extraction** using brain atlases (AAL, Harvard-Oxford, etc.)
- **Correlation and partial correlation** connectivity matrices
- **Graph theory metrics**: Degree centrality, betweenness, clustering coefficient
- **Tumor-connectivity fusion** to assess network disruption
- **3D network visualization** with affected connections

###  **Visualization & Reporting**
- **Multi-panel figures** showing axial, coronal, sagittal views
- **XAI overlay comparisons** across different methods
- **Uncertainty heatmaps** for prediction confidence
- **Interactive 3D plots** with Plotly for tumor and ROI exploration
- **Training curve analysis** with loss and metric tracking

##  Installation

### Prerequisites
- Python 3.8 or higher
- CUDA-capable GPU (recommended, 8GB+ VRAM)
- 16GB+ RAM

### Quick Install

```bash
# Clone the repository
git clone https://github.com/yourusername/brain-mapping.git
cd brain-mapping

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Requirements File (`requirements.txt`)

```txt
# Core dependencies
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.24.0
scipy>=1.10.0

# Medical imaging
nibabel>=5.0.0
nilearn>=0.10.0

# Visualization
matplotlib>=3.7.0
plotly>=5.14.0
kaleido>=0.2.1

# Machine learning utilities
scikit-learn>=1.2.0
scikit-image>=0.20.0
tqdm>=4.65.0

# Optional: Advanced features
shap>=0.41.0  # For SHAP analysis
monai>=1.1.0  # For additional losses and metrics
networkx>=3.0  # For graph theory metrics

# Development
pytest>=7.2.0
black>=23.0.0
```

### Optional Dependencies

For full functionality:

```bash
# For MONAI integration (advanced losses)
pip install monai

# For graph theory metrics
pip install networkx

# For SHAP explanations
pip install shap

# For development
pip install pytest black flake8
```

##  Dataset Structure

Organize your data as follows:

```
data/
├── patient_001/
│   ├── T1.nii.gz          # T1-weighted MRI
│   ├── T2.nii.gz          # T2-weighted MRI
│   ├── FLAIR.nii.gz       # FLAIR sequence
│   ├── fmri.nii.gz        # Resting-state fMRI (optional)
│   └── mask.nii.gz        # Ground truth segmentation (optional, for training)
├── patient_002/
│   └── ...
└── patient_N/
    └── ...

atlases/
├── AAL.nii.gz             # Automated Anatomical Labeling atlas
├── HarvardOxford.nii.gz   # Harvard-Oxford atlas
└── custom_atlas.nii.gz    # Your custom atlas
```

### Data Requirements

- **MRI Volumes**: NIfTI format (.nii or .nii.gz)
- **Required modalities**: At least T1-weighted images
- **Optional modalities**: T2, FLAIR improve segmentation accuracy
- **fMRI**: 4D volume for connectivity analysis (optional)
- **Masks**: Binary segmentation for training (0=background, 1=tumor)
- **Atlas**: 3D volume with integer labels for ROI analysis

##  Usage

### Basic Training

Train a model from scratch:

```bash
python main.py \
  --data_dir data \
  --atlas atlases/AAL.nii.gz \
  --train \
  --epochs 50 \
  --batch_size 2 \
  --lr 1e-4 \
  --base_filters 32 \
  --use_attention \
  --device cuda
```

### Training with Advanced Options

```bash
python main.py \
  --data_dir data \
  --atlas atlases/AAL.nii.gz \
  --train \
  --epochs 100 \
  --batch_size 2 \
  --lr 1e-4 \
  --val_split 0.2 \
  --base_filters 32 \
  --use_attention \
  --dropout 0.1 \
  --model_path checkpoints/best_model.pth \
  --save_dir results \
  --device cuda
```

### Inference on New Cases

Run prediction with uncertainty estimation:

```bash
python main.py \
  --data_dir data \
  --atlas atlases/AAL.nii.gz \
  --infer \
  --uncertainty \
  --case_idx 0 \
  --model_path checkpoints/best_model.pth \
  --save_dir results \
  --device cuda
```

### Full Pipeline with XAI

Generate complete analysis with all XAI methods:

```bash
python main.py \
  --data_dir data \
  --atlas atlases/AAL.nii.gz \
  --infer \
  --xai \
  --uncertainty \
  --case_idx 0 \
  --model_path checkpoints/best_model.pth \
  --save_dir results/case_001 \
  --device cuda
```

### Batch Processing

Process multiple cases:

```bash
# Create a case list file
echo "data/patient_001" > cases.txt
echo "data/patient_002" >> cases.txt
echo "data/patient_003" >> cases.txt

# Process all cases
for idx in {0..2}; do
  python main.py \
    --case_list cases.txt \
    --atlas atlases/AAL.nii.gz \
    --infer --xai --uncertainty \
    --case_idx $idx \
    --model_path checkpoints/best_model.pth \
    --save_dir results/batch_analysis \
    --device cuda
done
```

##  Output Files

After running the pipeline, you'll find:

```
results/
├── case_001/
│   ├── patient_001_tumor_prob.nii.gz       # Tumor probability map
│   ├── patient_001_uncertainty.nii.gz      # Uncertainty map
│   ├── patient_001_analysis.png            # Multi-panel visualization
│   ├── patient_001_analysis.pdf            # Publication-ready PDF
│   ├── patient_001_xai_gradcam.png         # Grad-CAM visualization
│   ├── patient_001_xai_ig.png              # Integrated Gradients
│   ├── patient_001_xai_shap.png            # SHAP visualization (if available)
│   └── patient_001_report.txt              # Clinical report
├── training_curves.png                      # Training history
└── best_model.pth                           # Trained model checkpoint
```

### Clinical Report Example

```
╔══════════════════════════════════════════════════════════════╗
║                Brain Tumor Analysis Report                   ║
╚══════════════════════════════════════════════════════════════╝

Case ID: patient_001
Analysis Date: 2025-01-15

─────────────────────────────────────────────────────────────

TUMOR CHARACTERISTICS:
  • Estimated Volume: 15234.56 mm³
  • Mean Uncertainty: 0.123
  • Max Uncertainty: 0.456
  • 95th Percentile Uncertainty: 0.389

─────────────────────────────────────────────────────────────

AFFECTED BRAIN REGIONS (Top 10):
   1. ROI  45:  87.3% overlap  (Left Temporal Lobe)
   2. ROI  46:  82.1% overlap  (Left Hippocampus)
   3. ROI  47:  65.4% overlap  (Left Parahippocampal)
   ...

─────────────────────────────────────────────────────────────

FUNCTIONAL CONNECTIVITY IMPACT:
  • ROI 45: 0.82 disruption score
  • ROI 46: 0.76 disruption score
  ...
```

##  XAI Methods Explained

### Layer-CAM (Recommended)
- **What**: Gradient-weighted class activation mapping at layer level
- **Advantage**: More accurate localization than standard Grad-CAM
- **Use case**: Identify which brain regions the model focuses on for tumor detection

### Integrated Gradients
- **What**: Attribution method that computes the integral of gradients along a path
- **Advantage**: Satisfies sensitivity and implementation invariance
- **Use case**: Understand voxel-level contributions to the prediction

### SHAP (Shapley Additive Explanations)
- **What**: Game-theory based feature attribution
- **Advantage**: Theoretically grounded with additive property
- **Use case**: Comprehensive feature importance with local and global interpretability
- **Note**: Computationally expensive, requires background samples

### Uncertainty Quantification
- **Method**: Monte Carlo Dropout (20 forward passes)
- **Output**: Mean prediction + standard deviation
- **Interpretation**: High uncertainty suggests the model is unsure, requiring manual review

##  Architecture Details

### Enhanced 3D U-Net

```
Input (3×128×128×128)
    ↓
[Encoder Block 1] → 32 filters + Attention
    ↓ MaxPool
[Encoder Block 2] → 64 filters + Attention
    ↓ MaxPool
[Encoder Block 3] → 128 filters + Attention
    ↓ MaxPool
[Encoder Block 4] → 256 filters + Attention
    ↓ MaxPool
[Bottleneck] → 512 filters + Attention
    ↓
[Decoder Block 4] ← Skip connection + 256 filters
    ↓
[Decoder Block 3] ← Skip connection + 128 filters
    ↓
[Decoder Block 2] ← Skip connection + 64 filters
    ↓
[Decoder Block 1] ← Skip connection + 32 filters
    ↓
Output (2×128×128×128) - Background + Tumor
```

**Key Components:**
- Residual blocks with pre-activation
- 3D attention mechanisms for feature refinement
- Instance normalization for protocol robustness
- Dropout for regularization and uncertainty

##  Training Best Practices

### Hyperparameter Recommendations

| Parameter | Recommended | Notes |
|-----------|-------------|-------|
| Batch Size | 2-4 | Limited by GPU memory |
| Learning Rate | 1e-4 | Use scheduler for adaptation |
| Epochs | 50-100 | With early stopping |
| Base Filters | 32 | Balance between capacity and memory |
| Dropout | 0.1 | Helps with uncertainty estimation |
| Accumulation Steps | 4 | Effective batch size = 8-16 |

### Data Augmentation Settings

```python
augmentation_config = {
    'random_flip': {'p': 0.5},
    'random_rotate': {'p': 0.5, 'max_angle': 15},
    'random_scale': {'p': 0.5, 'range': (0.9, 1.1)},
    'random_noise': {'p': 0.3, 'std': 0.05},
    'random_gamma': {'p': 0.3, 'range': (0.8, 1.2)}
}
```

### Loss Function Weights

```python
loss_weights = {
    'cross_entropy': 1.0,
    'dice': 1.0,
    'focal': 0.5  # Lower weight for focal loss
}
```

##  Advanced Configuration

### Custom Model Configuration

```python
from dataclasses import dataclass

@dataclass
class ModelConfig:
    in_channels: int = 3          # T1, T2, FLAIR
    out_channels: int = 2         # Background, Tumor
    base_filters: int = 32        # Starting number of filters
    depth: int = 4                # Number of downsampling layers
    use_attention: bool = True    # Enable attention blocks
    use_residual: bool = True     # Enable residual connections
    dropout: float = 0.1          # Dropout rate
```

### Training Configuration

```python
@dataclass
class TrainingConfig:
    epochs: int = 50
    batch_size: int = 2
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    patience: int = 10            # Early stopping patience
    use_amp: bool = True          # Mixed precision training
    accumulation_steps: int = 4   # Gradient accumulation
    val_split: float = 0.2        # Validation split
```

##  Evaluation Metrics

The pipeline computes:

- **Dice Coefficient**: Overlap between prediction and ground truth
- **IoU (Jaccard Index)**: Intersection over Union
- **Hausdorff Distance**: Maximum surface distance error
- **Sensitivity/Specificity**: True positive and negative rates
- **Volume Error**: Difference in tumor volume estimation

##  Troubleshooting

### Common Issues

**Out of Memory Error**
```bash
# Solution 1: Reduce batch size
--batch_size 1

# Solution 2: Use gradient accumulation
--batch_size 1 --accumulation_steps 8

# Solution 3: Reduce model size
--base_filters 16
```

**Training Diverges**
```bash
# Solution: Lower learning rate
--lr 5e-5

# Or use warm-up scheduling
```

**Poor Segmentation Quality**
- Ensure proper data normalization
- Check data augmentation isn't too aggressive
- Verify ground truth mask quality
- Train longer with early stopping

