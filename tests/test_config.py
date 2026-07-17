"""Tests for brain_mapping.config: dataclass defaults and YAML loading.

These tests do not require torch - config.py only depends on stdlib
dataclasses and pyyaml.
"""

import textwrap

import pytest

from brain_mapping.config import ModelConfig, TrainingConfig, load_config


def test_model_config_defaults():
    cfg = ModelConfig()
    assert cfg.in_channels == 3
    assert cfg.out_channels == 2
    assert cfg.base_filters == 32
    assert cfg.depth == 4
    assert cfg.use_attention is True
    assert cfg.use_residual is True
    assert cfg.dropout == pytest.approx(0.1)


def test_training_config_defaults():
    cfg = TrainingConfig()
    assert cfg.epochs == 50
    assert cfg.batch_size == 2
    assert cfg.learning_rate == pytest.approx(1e-4)
    assert cfg.patience == 10
    assert cfg.use_amp is True
    assert cfg.accumulation_steps == 4
    assert cfg.val_split == pytest.approx(0.2)


def test_load_config_overrides_defaults(tmp_path):
    yaml_content = textwrap.dedent(
        """
        model:
          base_filters: 8
          use_attention: false
          dropout: 0.25
        training:
          epochs: 5
          batch_size: 1
          learning_rate: 0.001
        """
    )
    config_path = tmp_path / "test_config.yaml"
    config_path.write_text(yaml_content)

    model_cfg, training_cfg = load_config(config_path)

    # Overridden fields
    assert model_cfg.base_filters == 8
    assert model_cfg.use_attention is False
    assert model_cfg.dropout == pytest.approx(0.25)
    assert training_cfg.epochs == 5
    assert training_cfg.batch_size == 1
    assert training_cfg.learning_rate == pytest.approx(0.001)

    # Fields not present in the YAML fall back to dataclass defaults
    assert model_cfg.in_channels == 3
    assert training_cfg.patience == 10


def test_load_config_empty_file_uses_defaults(tmp_path):
    config_path = tmp_path / "empty.yaml"
    config_path.write_text("")

    model_cfg, training_cfg = load_config(config_path)

    assert model_cfg == ModelConfig()
    assert training_cfg == TrainingConfig()


def test_load_config_rejects_unknown_keys(tmp_path):
    yaml_content = textwrap.dedent(
        """
        model:
          base_filters: 8
          not_a_real_field: 123
        """
    )
    config_path = tmp_path / "bad.yaml"
    config_path.write_text(yaml_content)

    with pytest.raises(TypeError):
        load_config(config_path)
