# AI-Driven Multi-Modal Translational Brain Mapping for Precision Neurosurgery

A PyTorch pipeline for multi-modal (T1/T2/FLAIR) brain tumor segmentation with a 3D attention U-Net, Monte Carlo uncertainty estimation, voxel-level explainability (Layer-CAM, Integrated Gradients), and resting-state functional connectivity analysis - built as a template for the kind of tooling that would sit upstream of precision neurosurgery planning.

**Status: implementation only.** This code has not been trained or evaluated on real neuroimaging data. There are no benchmark results in this README, and none should be assumed. See [Results](#results) and [Limitations](#limitations).

## Why this exists

Neurosurgical planning for tumor resection has to balance two things that are in tension: removing as much tumor as possible, and preserving eloquent brain tissue and its functional connections. A model that only outputs a segmentation mask doesn't help with that trade-off - a surgeon also needs to know *how confident* the model is in a given region, *which* anatomical/functional structures the tumor overlaps, and *why* the model drew the boundary it did. That's the motivation for combining segmentation with uncertainty quantification, voxel-level attribution, and connectivity analysis in one pipeline rather than treating them as separate tools.

This repository is a from-scratch implementation of that combination. It is a portfolio/engineering artifact demonstrating the pipeline design, not a validated clinical tool, and it must not be used for any real clinical decision-making.

## Method

- **Segmentation**: a 3D U-Net (`EnhancedUNet3D`) with residual encoder/decoder blocks and self-attention blocks (`AttentionBlock3D`) at each stage, trained with a combined Cross-Entropy + Dice + Focal loss (`CombinedLoss`).
- **Uncertainty**: Monte Carlo Dropout (`UncertaintyEstimator`) - N stochastic forward passes at inference time with dropout left active, giving a per-voxel mean prediction and standard deviation (epistemic uncertainty proxy). This is a common, cheap approximation, not a calibrated confidence estimate - see [Limitations](#limitations).
- **Explainability**: Layer-CAM (`LayerCAM3D`) and Integrated Gradients (`integrated_gradients`) both run today and produce voxel-level attribution maps. SHAP is listed as an optional dependency and guarded with an `try/except` import, but **no SHAP-based explanation is actually wired into the inference pipeline** - installing `shap` alone does not add SHAP output. See [Limitations](#limitations).
- **Functional connectivity**: `ConnectivityAnalyzer` extracts ROI time series from a 4D resting-state fMRI volume via an atlas (`nilearn.input_data.NiftiLabelsMasker`), computes a correlation (or partial-correlation via `GraphicalLassoCV`) connectivity matrix, and derives graph-theory metrics (degree, betweenness, clustering) with `networkx` when it's installed.
- **Clinical report**: `ClinicalReportGenerator` renders tumor volume, uncertainty statistics, and the top-10 tumor-overlapping ROIs (by numeric atlas label) into a plain-text report. See the example below.

## Dataset

This pipeline expects data in the [BraTS](https://www.synapse.org/brats) (Brain Tumor Segmentation Challenge) format - co-registered, skull-stripped T1/T2/FLAIR MRI volumes plus an expert segmentation mask. **BraTS data requires registration at https://www.synapse.org/brats and is not redistributed here; no data is bundled with this repository.**

Expected directory layout:

```
data/
├── patient_001/
│   ├── T1.nii.gz          # T1-weighted MRI (required)
│   ├── T2.nii.gz          # T2-weighted MRI (optional, improves segmentation)
│   ├── FLAIR.nii.gz       # FLAIR sequence (optional, improves segmentation)
│   ├── fmri.nii.gz        # resting-state fMRI, 4D (optional, enables connectivity analysis)
│   └── mask.nii.gz        # ground-truth segmentation (optional, required only for training)
└── patient_002/
    └── ...

atlases/
└── atlas.nii.gz            # integer-labeled ROI atlas (e.g. AAL, Harvard-Oxford) for connectivity analysis
```

Any modality file that's missing is silently replaced with zeros by `EnhancedBrainDataset` and a warning is logged - the pipeline will run, but segmentation quality on a case missing T2/FLAIR is untested and likely to be poor.

## Installation

Requires Python 3.9+.

```bash
git clone https://github.com/ahmedfawaz879/AI-Driven-Multi-Modal-Translational-Brain-Mapping-for-Precision-Neurosurgery.git
cd AI-Driven-Multi-Modal-Translational-Brain-Mapping-for-Precision-Neurosurgery
pip install -e ".[dev]"
```

This installs the exact pins in `requirements.txt` (declared as `dependencies` in `pyproject.toml`) plus `pytest`. Two optional extras are available and are **not** required for the core pipeline to run:

```bash
pip install -e ".[xai]"   # shap, monai - see Limitations for what these do (and don't) enable today
pip install -r requirements-optional.txt   # equivalent, without editable install
```

Verify the install:

```bash
pytest tests/ -v
brain-mapping-train --help
```

## Usage

The CLI (`src/brain_mapping/cli.py`) is installed as two console-script aliases, `brain-mapping-train` and `brain-mapping-infer`, that both point at the same `main()` entry point - which of `--train` / `--infer` you pass decides what actually runs (this mirrors the original script's single-entrypoint, flag-driven design). You can also invoke it without installing, via `python -m brain_mapping.cli`.

### Train

```bash
brain-mapping-train \
  --data_dir data \
  --atlas atlases/atlas.nii.gz \
  --train \
  --config configs/default.yaml \
  --epochs 50 \
  --model_path checkpoints/best_model.pth \
  --device cuda
```

### Run inference (segmentation + uncertainty + XAI + connectivity + report) on one case

```bash
brain-mapping-infer \
  --data_dir data \
  --atlas atlases/atlas.nii.gz \
  --infer --xai --uncertainty \
  --case_idx 0 \
  --model_path checkpoints/best_model.pth \
  --save_dir results/case_000 \
  --device cuda
```

`--seed` (default `42`) controls a `set_seed()` call made at CLI startup that seeds `random`, `numpy`, and `torch`/`torch.cuda` - the original script had no seed control at all, so runs were not reproducible.

Run `brain-mapping-train --help` / `brain-mapping-infer --help` for the full flag list.

### Output files

```
results/case_000/
├── patient_001_tumor_prob.nii.gz     # tumor probability map (NIfTI)
├── patient_001_uncertainty.nii.gz    # MC-Dropout uncertainty map (NIfTI)
├── patient_001_analysis.png          # multi-panel axial/coronal/sagittal figure
├── patient_001_analysis.pdf          # same figure, publication-quality PDF
└── patient_001_report.txt            # clinical text report (see example below)
```

## Example output - illustrative, not real results

The block below shows the **exact text format** `ClinicalReportGenerator.generate_report()` actually produces today, with placeholder numbers standing in for a real inference run. It is not the output of a real model run on real data, and none of the code in this repository has ever been trained on patient data.

Note in particular that affected regions are listed by **numeric atlas ROI id only** (e.g. `ROI  45`). There is no atlas-label-to-anatomical-name lookup implemented anywhere in this codebase, so the report cannot currently print a region name like "Left Temporal Lobe" - resolving ROI ids to names is left to the user, against whichever atlas they supplied.

```
╔══════════════════════════════════════════════════════════════╗
║          AI-Driven Brain Tumor Analysis Report               ║
╚══════════════════════════════════════════════════════════════╝

Case ID: patient_001
Analysis Date: 2026-07-17

─────────────────────────────────────────────────────────────

TUMOR CHARACTERISTICS:
  • Estimated Volume: 15234.56 mm³
  • Mean Uncertainty: 0.123
  • Max Uncertainty: 0.456
  • 95th Percentile Uncertainty: 0.389

─────────────────────────────────────────────────────────────

AFFECTED BRAIN REGIONS (Top 10):
   1. ROI  45:  87.3% overlap
   2. ROI  46:  82.1% overlap
   3. ROI  47:  65.4% overlap

─────────────────────────────────────────────────────────────

FUNCTIONAL CONNECTIVITY IMPACT:
  • ROI 45: 0.82 disruption score
  • ROI 46: 0.76 disruption score

─────────────────────────────────────────────────────────────

INTERPRETATION NOTES:
  • High uncertainty regions require additional clinical review
  • Affected ROIs indicate potential functional impact zones
  • Connectivity analysis shows network-level implications

─────────────────────────────────────────────────────────────
```

(The original version of this README showed this same example with invented region names like "Left Temporal Lobe" and "Left Hippocampus" appended to the ROI lines. The code has never been able to produce those names; that example has been corrected here.)

## Results

Implementation only; not yet evaluated on benchmark data. No Dice, IoU, Hausdorff distance, sensitivity/specificity, uncertainty-calibration, or any other performance numbers exist for this pipeline, and none are claimed here. `Trainer.validate()` does compute a running Dice score during training as a training-time monitoring signal, but no training run has been performed and no resulting number is reported anywhere in this repository.

## Limitations

- **Never trained or evaluated on real data.** No BraTS run, no checkpoint, no benchmark numbers. Everything above is a description of what the code does, not what it has achieved.
- **ROI reporting is numeric-id-only.** `ClinicalReportGenerator` has no atlas-label-to-anatomical-name lookup; region identity must be resolved manually against whichever atlas was supplied.
- **Uncertainty estimates are uncalibrated and unvalidated.** MC-Dropout standard deviation is a common cheap proxy for epistemic uncertainty, but it has not been validated against expert disagreement or any ground truth on this task - treat it as a relative signal at best, not a calibrated probability.
- **SHAP is not actually wired up.** It's listed as an optional dependency and guarded with a `try/except ImportError`, but no code path calls it. For 3D medical volumes, SHAP is also compute-heavy (background-sample-based methods scale poorly with voxel count) and would need real engineering work, not just an `import shap`, before it produces anything.
- **`ModelConfig.depth` is not wired up.** `EnhancedUNet3D` always builds 4 encoder/decoder stages; the `depth` field exists in the config but changing it currently has no effect. Carried over unchanged from the original script.
- **`base_filters` has a practical floor of 8 when `use_attention=True`.** `AttentionBlock3D` projects `channels -> channels // 8` for its query/key convolutions; any encoder stage with fewer than 8 channels collapses that projection to 0 output channels and crashes the forward pass. This is a pre-existing constraint in the original script's attention mechanism, not something introduced here - it only surfaced during this refactor's test-writing (see `tests/test_models.py`), because the original script was never actually run with a small `base_filters` value.
- **A few imports in the original script were dead code** and were not carried into this refactor: `plotly.graph_objects` (imported, no interactive plot was ever generated) and `sklearn.model_selection.KFold` (imported, no cross-validation loop existed). `plotly` remains a pinned dependency since the module map from the original audit called for it; it is presently unused by any code path.
- **fMRI/connectivity path is unexercised.** `ConnectivityAnalyzer` has not been run against real resting-state data as part of this refactor.
- **No CI pipeline** runs these tests automatically on push; `pytest` was run locally (see below) but there is no GitHub Actions workflow yet.

## Reproduce

```bash
git clone https://github.com/ahmedfawaz879/AI-Driven-Multi-Modal-Translational-Brain-Mapping-for-Precision-Neurosurgery.git
cd AI-Driven-Multi-Modal-Translational-Brain-Mapping-for-Precision-Neurosurgery
pip install -e ".[dev]" && pytest tests/ -v
```

That installs the pinned dependencies and runs the full test suite (model forward-pass shapes, loss computation, crop/pad correctness, config loading, clinical report formatting) against synthetic data - no BraTS download required to verify the code itself works.

## Project layout

```
src/brain_mapping/
├── config.py        # ModelConfig, TrainingConfig, YAML loader
├── data.py           # NIfTI I/O, normalization, augmentation, Dataset
├── models.py         # AttentionBlock3D, ResidualBlock3D, EnhancedUNet3D, CombinedLoss
├── train.py           # Trainer (AMP, gradient accumulation, early stopping)
├── explain.py         # UncertaintyEstimator, LayerCAM3D, integrated_gradients
├── visualize.py        # BrainVisualizer (multi-panel figures, training curves)
├── connectivity.py      # ConnectivityAnalyzer (ROI time series, graph metrics)
├── report.py            # ClinicalReportGenerator
├── inference.py          # InferencePipeline (end-to-end single-case processing)
├── cli.py                 # parse_args, main - the real entry point
└── utils.py                # logging setup, set_seed, ensure_dir

configs/default.yaml   # externalized ModelConfig/TrainingConfig defaults
tests/                  # pytest suite, synthetic-data only
```

## Citation

If you reference this codebase, please cite it as:

```bibtex
@software{fawaz2026brainmapping,
  author = {Fawaz, Ahmed},
  title = {AI-Driven Multi-Modal Translational Brain Mapping for Precision Neurosurgery},
  year = {2026},
  url = {https://github.com/ahmedfawaz879/AI-Driven-Multi-Modal-Translational-Brain-Mapping-for-Precision-Neurosurgery}
}
```

## License

MIT - see [LICENSE](LICENSE).
