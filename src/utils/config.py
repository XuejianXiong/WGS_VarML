# utils/config.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from logzero import logger


def load_config(config_path: Optional[str]) -> Dict[str, Any]:
    """
    Load YAML configuration file.

    Parameters
    ----------
    config_path : str or None
        Path to YAML config file

    Returns
    -------
    dict
        Parsed configuration dictionary (empty if no config provided)
    """
    if config_path is None:
        logger.info("No config file provided; using CLI/default values only")
        return {}

    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open() as f:
        config = yaml.safe_load(f) or {}

    logger.info(f"Loaded config from {path}")
    return config


def resolve(value, config_value, default):
    """
    Resolve final parameter value using precedence:
    CLI > YAML > default
    """
    if value is not None:
        return value
    if config_value is not None:
        return config_value
    return default
