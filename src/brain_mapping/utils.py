"""Cross-cutting utilities: logging setup, directory helpers, seeding.

Extracted from the original single-file script (setup_logging, ensure_dir).
set_seed() is new: the original script had no seed control, so runs were
not reproducible.

Note: torch is imported lazily inside set_seed() rather than at module
scope. Nearly every module in this package imports ``logger`` from here,
so keeping this module's top-level import list light (no torch) means
config/report/logging-only code paths can be imported and unit-tested
without requiring a torch install.
"""

import logging
import os
import random
from pathlib import Path
from typing import Union

import numpy as np


def setup_logging(log_dir: str = "logs") -> logging.Logger:
    """Configure structured logging to file + stdout and return a logger.

    Idempotent: calling this more than once (e.g. once at import time and
    once from the CLI) will not duplicate log handlers.
    """
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(f"{log_dir}/brain_mapping.log"),
                logging.StreamHandler(),
            ],
        )
    return logging.getLogger("brain_mapping")


def ensure_dir(path: Union[str, Path]) -> None:
    """Create directory if it doesn't exist."""
    Path(path).mkdir(parents=True, exist_ok=True)


def set_seed(seed: int = 42) -> None:
    """Seed all relevant RNGs (random, numpy, torch, torch.cuda) for
    reproducible runs. The original script had no seeding at all.
    """
    import torch  # local import - see module docstring

    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


logger = setup_logging()
