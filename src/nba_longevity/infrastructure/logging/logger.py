from __future__ import annotations

import sys
from pathlib import Path
from typing import Final
from loguru import logger
from nba_longevity.infrastructure.config.settings import InfraConfig


LOG_FILE_NAME: Final[str] = "nba_longevity.log"


def setup_logger(config: InfraConfig):
    """
    Configure the global Loguru logger.

    Design principles:
    - Single global logger
    - Configuration-driven (no hardcoded values)
    - No logging logic in the Domain layer
    - Compatible with batch jobs, APIs, and Spark

    Parameters
    ----------
    config : InfraConfig
        Infrastructure configuration object loaded from infra.yaml

    Returns
    -------
    logger
        Configured global Loguru logger instance
    """

    # Ensure logs directory exists
    logs_dir: Path = config.paths.logs_dir
    logs_dir.mkdir(parents=True, exist_ok=True)

    log_file: Path = logs_dir / LOG_FILE_NAME

    # Reset previous configuration (critical)
    logger.remove()

    # Bind static context (project-level metadata)
    logger.configure(
        extra={
            "project": config.project.name,
            "environment": config.project.environment,
        }
    )

    # Console sink (human-readable)
    logger.add(
        sys.stdout,
        level=config.runtime.log_level,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level}</level> | "
            "<cyan>{extra[project]}</cyan> | "
            "<magenta>{extra[environment]}</magenta> | "
            "{module}:{function}:{line} | "
            "{message}"
        ),
        enqueue=True,
    )

    # File sink (audit / production)
    logger.add(
        log_file,
        level="DEBUG",
        rotation="20 MB",
        retention="30 days",
        compression="zip",
        format=(
            "{time:YYYY-MM-DD HH:mm:ss} | "
            "{level} | "
            "{extra[project]} | "
            "{extra[environment]} | "
            "{module}:{function}:{line} | "
            "{message}"
        ),
        enqueue=True,
        backtrace=True,
        diagnose=False,
    )

    return logger
