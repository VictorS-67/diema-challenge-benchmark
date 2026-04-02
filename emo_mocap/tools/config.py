"""YAML configuration loader with validation.

Loads experiment configs from YAML files and provides structured access
to settings with validation for required fields and sensible defaults.
"""

from pathlib import Path
from types import SimpleNamespace

import yaml


def _to_namespace(d):
    """Recursively convert a dict to a SimpleNamespace."""
    if isinstance(d, dict):
        return SimpleNamespace(**{k: _to_namespace(v) for k, v in d.items()})
    if isinstance(d, list):
        return [_to_namespace(item) for item in d]
    return d


_REQUIRED_FIELDS = [
    ("data", "data_path"),
    ("model", "type"),
    ("model", "num_class"),
    ("skeleton", "num_nodes"),
    ("skeleton", "inward_edges"),
]

_DEFAULTS = {
    "model": {
        "in_channels": 3,
        "dropout": 0.5,
        "dual_loss": False,
    },
    "training": {
        "base_lr": 0.1,
        "optimizer": "SGD",
        "scheduler_type": "cosine",
        "scheduler_params": [],
        "weight_decay": 0.0001,
        "max_epochs": 100,
        "batch_size": 32,
        "clip_length": 64,
        "aux_loss_weights": {},
        "accelerator": "auto",
        "early_stopping": True,
        "early_stopping_monitor": "val_loss",
        "early_stopping_patience": 10,
    },
    "data": {
        "split_path": None,
        "num_workers": None,
        "seed": 255,
        "target_repr": "euler",
    },
    "augmentation": {
        "enabled": False,
        "rotate": False,
        "rotate_range": [-180, 180],
        "mirror": False,
        "mirror_prob": 0.5,
        "speed": False,
        "speed_range": [0.8, 1.2],
        "noise_sigma": 0.0,
    },
    "logging": {
        "experiment_name": None,
        "log_dir": "logs/",
    },
}


def _apply_defaults(raw: dict) -> dict:
    """Apply default values to a raw config dict (non-destructive)."""
    for section, defaults in _DEFAULTS.items():
        if section not in raw:
            raw[section] = {}
        for key, value in defaults.items():
            if key not in raw[section]:
                raw[section][key] = value
    return raw


def _validate(raw: dict, path: str) -> None:
    """Validate that all required fields are present."""
    for section, field in _REQUIRED_FIELDS:
        if section not in raw:
            raise ValueError(
                f"Config {path}: missing required section '{section}'"
            )
        if field not in raw[section]:
            raise ValueError(
                f"Config {path}: missing required field '{section}.{field}'"
            )


def load_config(path: str | Path) -> SimpleNamespace:
    """Load and validate a YAML config file.

    Args:
        path: path to the YAML config file

    Returns:
        A nested SimpleNamespace with config values accessible as attributes
        (e.g., config.model.type, config.training.base_lr)

    Raises:
        FileNotFoundError: if the config file doesn't exist
        ValueError: if required fields are missing
        yaml.YAMLError: if the YAML is malformed
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path) as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict):
        raise ValueError(f"Config {path}: expected a YAML mapping, got {type(raw).__name__}")

    _validate(raw, str(path))
    raw = _apply_defaults(raw)

    return _to_namespace(raw)


def _coerce_value(value_str):
    """Coerce a string value to int, float, bool, or leave as str."""
    if value_str.lower() in ("true", "false"):
        return value_str.lower() == "true"
    try:
        return int(value_str)
    except ValueError:
        pass
    try:
        return float(value_str)
    except ValueError:
        pass
    return value_str


def apply_overrides(raw: dict, overrides: list[str]) -> dict:
    """Apply dot-separated key=value overrides to a config dict.

    Args:
        raw: the raw config dict (modified in-place)
        overrides: list of strings like 'training.max_epochs=200'

    Returns:
        The modified config dict
    """
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"Invalid override (no '='): {override}")
        key, value = override.split("=", 1)
        parts = key.split(".")
        if len(parts) != 2:
            raise ValueError(
                f"Override key must be section.field (got {key})"
            )
        section, field = parts
        if section not in raw:
            raw[section] = {}
        raw[section][field] = _coerce_value(value)
    return raw


def load_config_with_overrides(path: str | Path, overrides: list[str] | None = None) -> SimpleNamespace:
    """Load a YAML config, apply overrides, validate, and return namespace.

    Convenience wrapper combining load_config and apply_overrides.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path) as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict):
        raise ValueError(f"Config {path}: expected a YAML mapping, got {type(raw).__name__}")

    if overrides:
        raw = apply_overrides(raw, overrides)

    _validate(raw, str(path))
    raw = _apply_defaults(raw)

    return _to_namespace(raw)
