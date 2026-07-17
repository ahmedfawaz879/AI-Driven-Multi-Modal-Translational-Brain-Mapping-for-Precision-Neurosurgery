"""Command-line entry point: training and/or inference over a case directory.

Extracted from the original script (parse_args, main). Behavior is
unchanged except for two additions the original script lacked entirely:

- ``set_seed()`` is called at startup for reproducible runs (controlled by
  the new ``--seed`` flag).
- An optional ``--config`` flag to load ModelConfig/TrainingConfig from a
  YAML file (see configs/default.yaml) instead of only from CLI flags /
  hardcoded dataclass defaults.

This module is the real, working entry point referenced by the
``brain-mapping-train`` / ``brain-mapping-infer`` console scripts defined
in pyproject.toml (both simply call ``main()`` here; the original script's
single ``--train``/``--infer`` flag pair decides what actually runs).
"""

import argparse
import os

import torch
from torch.utils.data import DataLoader

from .config import ModelConfig, TrainingConfig, load_config
from .data import EnhancedBrainDataset
from .inference import InferencePipeline
from .models import EnhancedUNet3D
from .train import Trainer
from .utils import logger, set_seed
from .visualize import BrainVisualizer


def parse_args():
    parser = argparse.ArgumentParser(
        description="AI-Driven Multi-Modal Brain Mapping Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Data arguments
    parser.add_argument(
        "--data_dir", type=str, default="data", help="Directory containing case folders"
    )
    parser.add_argument("--atlas", type=str, required=True, help="Path to brain atlas NIfTI file")
    parser.add_argument(
        "--case_list", type=str, default=None, help="File with list of case paths"
    )

    # Config file (new: externalizes ModelConfig/TrainingConfig defaults)
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional path to a YAML file (see configs/default.yaml) with "
        "model/training settings. CLI flags below override values loaded "
        "from this file.",
    )

    # Training arguments
    parser.add_argument("--train", action="store_true", help="Run training")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--val_split", type=float, default=0.2, help="Validation split fraction")

    # Model arguments
    parser.add_argument("--base_filters", type=int, default=32, help="Base number of filters")
    parser.add_argument(
        "--use_attention", action="store_true", default=True, help="Use attention mechanisms"
    )
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")

    # Inference arguments
    parser.add_argument("--infer", action="store_true", help="Run inference")
    parser.add_argument("--case_idx", type=int, default=0, help="Case index for inference")
    parser.add_argument(
        "--model_path", type=str, default="best_model.pth", help="Path to trained model"
    )

    # XAI arguments
    parser.add_argument("--xai", action="store_true", help="Generate XAI visualizations")
    parser.add_argument("--uncertainty", action="store_true", help="Compute uncertainty estimates")

    # Output arguments
    parser.add_argument("--save_dir", type=str, default="results", help="Output directory")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use",
    )

    # Reproducibility (new)
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for numpy/torch/random"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    set_seed(args.seed)

    # Setup logging
    logger.info("=" * 60)
    logger.info("AI-Driven Brain Mapping Pipeline")
    logger.info("=" * 60)
    logger.info(f"Device: {args.device}")
    logger.info(f"Seed: {args.seed}")

    # Model/training config: start from YAML if given, then let explicit
    # CLI flags win (CLI flags always had defaults in the original script,
    # so we only treat a flag as "explicitly set" by the user for the
    # values that participate in ModelConfig/TrainingConfig overrides).
    if args.config:
        model_config, training_config = load_config(args.config)
    else:
        model_config, training_config = ModelConfig(), TrainingConfig()

    # Load cases
    if args.case_list and os.path.exists(args.case_list):
        with open(args.case_list) as f:
            cases = [l.strip() for l in f if l.strip()]
    else:
        cases = []
        for name in os.listdir(args.data_dir):
            path = os.path.join(args.data_dir, name)
            if os.path.isdir(path) and os.path.exists(os.path.join(path, "T1.nii.gz")):
                cases.append(path)

    logger.info(f"Found {len(cases)} cases")

    if len(cases) == 0:
        logger.error("No cases found!")
        return

    # Create model
    model_config.base_filters = args.base_filters
    model_config.use_attention = args.use_attention
    model_config.dropout = args.dropout
    model = EnhancedUNet3D(model_config)
    logger.info(
        f"Model created with {sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters"
    )

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

        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=1, shuffle=False, num_workers=2, pin_memory=True
        )

        # Train
        training_config.epochs = args.epochs
        training_config.batch_size = args.batch_size
        training_config.learning_rate = args.lr
        training_config.val_split = args.val_split

        trainer = Trainer(model, training_config, device=args.device)
        model = trainer.fit(train_loader, val_loader, save_path=args.model_path)

        # Plot training curves
        BrainVisualizer.plot_training_curves(
            trainer.history, save_path=os.path.join(args.save_dir, "training_curves.png")
        )

    # Inference
    if args.infer:
        # Load model
        if os.path.exists(args.model_path):
            checkpoint = torch.load(args.model_path, map_location=args.device)
            model.load_state_dict(checkpoint["model_state_dict"])
            logger.info(f"Loaded model from {args.model_path}")
        else:
            logger.warning("No trained model found, using random weights")

        # Create inference pipeline
        pipeline = InferencePipeline(model, device=args.device)

        # Process case
        case_path = cases[args.case_idx]
        results = pipeline.process_case(case_path, args.atlas, args.save_dir)

        logger.info("=" * 60)
        logger.info("Inference completed successfully!")
        logger.info(f"Results saved to: {args.save_dir}")
        logger.info("=" * 60)


if __name__ == "__main__":
    main()
