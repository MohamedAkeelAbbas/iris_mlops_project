"""
Configuration Management Module
===============================
Load and manage YAML configurations with validation.
"""

from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from loguru import logger
from pydantic import BaseModel, Field


class DataConfig(BaseModel):
    """Data configuration schema."""

    raw_path: str = "data/raw/iris.csv"
    processed_path: str = "data/processed"
    features_path: str = "data/features"
    test_size: float = Field(default=0.2, ge=0.1, le=0.5)
    validation_size: float = Field(default=0.1, ge=0.0, le=0.3)
    random_state: int = 42
    target_column: str = "species"


class FeaturesConfig(BaseModel):
    """Feature engineering configuration schema."""

    scaling_method: str = "standard"
    create_polynomial: bool = True
    polynomial_degree: int = Field(default=2, ge=1, le=4)
    create_interactions: bool = True
    feature_selection: bool = True
    n_features_to_select: int = Field(default=10, ge=1)


class ModelConfig(BaseModel):
    """Model configuration schema."""

    type: str = "random_forest"
    random_forest: Dict[str, Any] = {}
    logistic_regression: Dict[str, Any] = {}
    svm: Dict[str, Any] = {}
    gradient_boosting: Dict[str, Any] = {}


class TrainingConfig(BaseModel):
    """Training configuration schema."""

    cross_validation: bool = True
    cv_folds: int = Field(default=5, ge=2, le=10)
    early_stopping: bool = False
    hyperparameter_tuning: bool = False
    tuning_method: str = "grid_search"


class EvaluationConfig(BaseModel):
    """Evaluation configuration schema."""

    metrics: list = ["accuracy", "precision", "recall", "f1_score"]
    save_plots: bool = True
    plots_path: str = "experiments/plots"


class ExperimentConfig(BaseModel):
    """Experiment tracking configuration schema."""

    tracking_uri: str = "experiments/mlruns"
    experiment_name: str = "iris_classification"
    log_models: bool = True
    log_artifacts: bool = True


class RegistryConfig(BaseModel):
    """Model registry configuration schema."""

    path: str = "models/registry"
    promotion_threshold: Dict[str, float] = {"accuracy": 0.90, "f1_score": 0.88}


class ProjectConfig(BaseModel):
    """Complete project configuration."""

    project: Dict[str, str] = {}
    data: DataConfig = DataConfig()
    features: FeaturesConfig = FeaturesConfig()
    model: ModelConfig = ModelConfig()
    training: TrainingConfig = TrainingConfig()
    evaluation: EvaluationConfig = EvaluationConfig()
    experiment: ExperimentConfig = ExperimentConfig()
    registry: RegistryConfig = RegistryConfig()


def load_config(config_path: str = "configs/config.yaml") -> ProjectConfig:
    """
    Load and validate configuration from YAML file.

    Args:
        config_path: Path to the configuration file

    Returns:
        Validated ProjectConfig object
    """
    config_file = Path(config_path)

    if not config_file.exists():
        logger.warning(f"Config file not found: {config_path}. Using defaults.")
        return ProjectConfig()

    with open(config_file, "r") as f:
        config_dict = yaml.safe_load(f)

    config = ProjectConfig(**config_dict)
    logger.info(f"Configuration loaded from {config_path}")

    return config


def save_config(config: ProjectConfig, config_path: str):
    """
    Save configuration to YAML file.

    Args:
        config: ProjectConfig object
        config_path: Path to save the configuration
    """
    config_file = Path(config_path)
    config_file.parent.mkdir(parents=True, exist_ok=True)

    with open(config_file, "w") as f:
        yaml.dump(config.model_dump(), f, default_flow_style=False, sort_keys=False)

    logger.info(f"Configuration saved to {config_path}")
