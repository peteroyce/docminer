"""YAML configuration loader."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from docminer.config.schema import DocMinerConfig

logger = logging.getLogger(__name__)

_DEFAULT_CONFIG_PATH = Path(__file__).parent.parent.parent.parent / "configs" / "default.yml"


def load_config(path: Optional[str | Path] = None) -> DocMinerConfig:
    """Load configuration from a YAML file.

    Parameters
    ----------
    path:
        Path to a YAML configuration file.  If *None*, loads the
        bundled ``configs/default.yml`` if it exists; otherwise returns
        default settings.

    Returns
    -------
    DocMinerConfig
        Populated configuration object.
    """
    if path is None:
        path = _DEFAULT_CONFIG_PATH

    config_path = Path(path)
    if not config_path.exists():
        logger.info("Config file not found at %s; using defaults", config_path)
        return DocMinerConfig()

    try:
        import yaml
    except ImportError:
        logger.warning("pyyaml not installed; using default config")
        return DocMinerConfig()

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            raw: dict = yaml.safe_load(f) or {}
        config = DocMinerConfig.model_validate(raw)
        logger.info("Loaded configuration from %s", config_path)
        return config
    except Exception as exc:
        logger.error("Failed to parse config %s: %s; using defaults", config_path, exc)
        return DocMinerConfig()


def save_config(config: DocMinerConfig, path: str | Path) -> None:
    """Serialise *config* to a YAML file at *path*."""
    try:
        import yaml
    except ImportError:
        raise RuntimeError("pyyaml is required to save configuration files")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(config.model_dump(), f, default_flow_style=False, allow_unicode=True)
    logger.info("Configuration saved to %s", path)
