"""Configuration dataclasses for the model and training loop.

Extracted verbatim from the original script. Defaults are unchanged; the
only addition is ``load_config``, which lets the magic numbers baked into
these dataclasses be overridden from ``configs/default.yaml`` (or any other
YAML file) instead of only being editable in source.
"""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Tuple, Union

import yaml


@dataclass
class ModelConfig:
    """Model configuration"""

    in_channels: int = 3
    out_channels: int = 2
    base_filters: int = 32
    depth: int = 4
    use_attention: bool = True
    use_residual: bool = True
    dropout: float = 0.1


@dataclass
class TrainingConfig:
    """Training configuration"""

    epochs: int = 50
    batch_size: int = 2
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    patience: int = 10
    use_amp: bool = True
    accumulation_steps: int = 4
    val_split: float = 0.2


def load_config(path: Union[str, Path]) -> Tuple[ModelConfig, TrainingConfig]:
    """Load ModelConfig and TrainingConfig from a YAML file.

    The file is expected to have top-level ``model:`` and ``training:``
    mappings (see configs/default.yaml). Any fields omitted fall back to
    the dataclass defaults above. Unknown keys raise a TypeError so typos
    in the YAML are caught early rather than silently ignored.
    """
    with open(path) as f:
        raw = yaml.safe_load(f) or {}

    model_cfg = ModelConfig(**(raw.get("model") or {}))
    training_cfg = TrainingConfig(**(raw.get("training") or {}))
    return model_cfg, training_cfg


def config_to_dict(config) -> dict:
    """Convert a ModelConfig/TrainingConfig dataclass instance to a dict."""
    return asdict(config)
