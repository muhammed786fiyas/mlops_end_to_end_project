"""
Centralized logger factory for the football-mlops project.

Usage:
    from src.utils.logger import get_logger
    logger = get_logger(__name__)
    logger.info("Starting feature engineering")
    logger.error("Database connection failed")

All loggers share the same format and write to both console and a rotating file.
Configuration is loaded from config.yaml at the project root.
"""

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Optional

import yaml

# Module-level cache so we don't re-create handlers if get_logger() is called twice
# This is a common gotcha — without caching, every call would add another handler,
# and you'd see your log messages duplicated 3, 4, 5 times.
_LOGGERS_CACHE: dict[str, logging.Logger] = {}
_CONFIG_CACHE: Optional[dict] = None


def _load_config() -> dict:
    """Load logging config from config.yaml. Cached after first call."""
    global _CONFIG_CACHE
    if _CONFIG_CACHE is not None:
        return _CONFIG_CACHE

    # Walk up from this file's location to find project root (where config.yaml lives)
    # This makes the logger work regardless of where the calling script is run from
    current = Path(__file__).resolve()
    for parent in [current, *current.parents]:
        candidate = parent / "config.yaml"
        if candidate.exists():
            with open(candidate, "r") as f:
                _CONFIG_CACHE = yaml.safe_load(f)
                return _CONFIG_CACHE

    raise FileNotFoundError("Could not find config.yaml in any parent directory")


def get_logger(name: str) -> logging.Logger:
    """
    Get a configured logger instance.

    Args:
        name: Logger name, typically __name__ of the calling module.
              Using __name__ gives you 'src.features.build_features' style names
              which makes it easy to filter logs by module later.

    Returns:
        Configured logging.Logger instance with console + rotating file handlers.
    """
    # Return cached logger if it exists — avoids duplicate handlers
    if name in _LOGGERS_CACHE:
        return _LOGGERS_CACHE[name]

    config = _load_config()
    log_cfg = config["logging"]

    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_cfg["level"].upper()))

    # Prevent log messages from bubbling up to the root logger and being printed twice
    logger.propagate = False

    # If logger somehow already has handlers (e.g., reloading in Jupyter), clear them
    if logger.hasHandlers():
        logger.handlers.clear()

    # Build the formatter once and reuse for both handlers
    formatter = logging.Formatter(
        fmt=log_cfg["format"],
        datefmt=log_cfg["date_format"],
    )

    # Console handler — writes to stdout (so it shows in terminal / docker logs / Jupyter)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Rotating file handler — writes to logs/football_mlops.log
    # Rotates when file hits max_bytes, keeps backup_count old files
    log_dir = _project_root() / log_cfg["log_dir"]
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / log_cfg["log_file"]

    file_handler = logging.handlers.RotatingFileHandler(
        filename=log_path,
        maxBytes=log_cfg["max_bytes"],
        backupCount=log_cfg["backup_count"],
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    _LOGGERS_CACHE[name] = logger
    return logger


def _project_root() -> Path:
    """Find project root by walking up to where config.yaml lives."""
    current = Path(__file__).resolve()
    for parent in [current, *current.parents]:
        if (parent / "config.yaml").exists():
            return parent
    raise FileNotFoundError("Could not find project root")