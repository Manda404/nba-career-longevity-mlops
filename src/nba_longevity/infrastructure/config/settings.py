"""
INFRASTRUCTURE CONFIGURATION LOADER
==================================

Responsibilities
----------------
- Load YAML configuration files
- Load environment variables (.env)
- Validate configuration with Pydantic
- Resolve project-relative paths
- Ensure required directories exist

Rules
-----
- No business logic
- No ML logic
- No experiment logic
"""

from pathlib import Path
from typing import Literal

import os
import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, field_validator
from nba_longevity.infrastructure.system_utils.root_finder import get_repository_root

# -------------------------------------------------------------------------
# Project root (structural, never configurable)
# -------------------------------------------------------------------------
PROJECT_ROOT = get_repository_root() #Path(__file__).resolve().parents[4]


# -------------------------------------------------------------------------
# Pydantic models
# -------------------------------------------------------------------------
class ProjectConfig(BaseModel):
    name: str
    environment: Literal["local", "staging", "prod"]


class PathsConfig(BaseModel):
    """
    All paths are expected to be:
    - relative to PROJECT_ROOT (recommended)
    - or absolute (allowed, but discouraged in config)
    """
    data_dir: Path
    raw_data: Path
    train_data: Path
    test_data: Path
    artifacts_dir: Path
    logs_dir: Path

    @field_validator("data_dir", "artifacts_dir", "logs_dir")
    def create_directories(cls, v: Path) -> Path:
        v.mkdir(parents=True, exist_ok=True)
        return v


class MLflowConfig(BaseModel):
    tracking_uri: str
    experiment_name: str


class RuntimeConfig(BaseModel):
    random_state: int
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"]


class InfraConfig(BaseModel):
    project: ProjectConfig
    paths: PathsConfig
    mlflow: MLflowConfig
    runtime: RuntimeConfig

    model_config = {
        "frozen": True  # Immutable config (important)
    }


# -------------------------------------------------------------------------
# Loader
# -------------------------------------------------------------------------
def load_infra_config(config_path: Path) -> InfraConfig:
    """
    Load and validate infrastructure configuration.

    Steps:
    1. Load environment variables (.env)
    2. Load YAML configuration
    3. Resolve environment variables
    4. Resolve project-relative paths
    5. Validate with Pydantic
    """
    # Load .env (local only, ignored in prod/CI)
    load_dotenv()

    with open(config_path, "r") as f:
        raw_config = yaml.safe_load(f)

    # --------------------------------------------------
    # Resolve MLflow tracking URI from env
    # --------------------------------------------------
    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI")
    if mlflow_uri is None:
        raise RuntimeError("MLFLOW_TRACKING_URI is not defined in environment")

    raw_config["mlflow"]["tracking_uri"] = mlflow_uri

    # --------------------------------------------------
    # Resolve paths (relative → PROJECT_ROOT, absolute → unchanged)
    # --------------------------------------------------
    resolved_paths = {}
    for key, value in raw_config["paths"].items():
        path = Path(value)
        resolved_paths[key] = path if path.is_absolute() else PROJECT_ROOT / path

    raw_config["paths"] = resolved_paths

    return InfraConfig(**raw_config)