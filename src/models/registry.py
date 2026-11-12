"""
Model Registry Module
=====================
Local model registry for versioning and managing models.
"""

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
from loguru import logger


class ModelRegistry:
    """
    Local model registry for versioning and managing trained models.
    """

    def __init__(
        self,
        registry_path: str = "models/registry",
        promotion_threshold: Dict[str, float] = None,
    ):
        """
        Initialize ModelRegistry.

        Args:
            registry_path: Path to the registry
            promotion_threshold: Minimum metrics for production promotion
        """
        self.registry_path = Path(registry_path)
        self.registry_path.mkdir(parents=True, exist_ok=True)

        self.promotion_threshold = promotion_threshold or {
            "accuracy": 0.90,
            "f1_score": 0.88,
        }

        self.metadata_file = self.registry_path / "registry_metadata.json"
        self._load_metadata()

    def _load_metadata(self):
        """Load registry metadata from file."""
        if self.metadata_file.exists():
            with open(self.metadata_file, "r") as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {
                "models": {},
                "production_model": None,
                "staging_model": None,
            }
            self._save_metadata()

    def _save_metadata(self):
        """Save registry metadata to file."""
        with open(self.metadata_file, "w") as f:
            json.dump(self.metadata, f, indent=2)

    def _generate_version(self, model_name: str) -> str:
        """Generate a new version number for a model."""
        if model_name not in self.metadata["models"]:
            return "1.0.0"

        versions = list(self.metadata["models"][model_name].keys())
        if not versions:
            return "1.0.0"

        # Get latest version and increment
        latest = max(versions, key=lambda v: [int(x) for x in v.split(".")])
        major, minor, patch = map(int, latest.split("."))
        return f"{major}.{minor}.{patch + 1}"

    def register_model(
        self,
        model,
        model_name: str,
        metrics: Dict[str, float],
        params: Dict[str, Any] = None,
        artifacts: Dict[str, str] = None,
        description: str = None,
    ) -> str:
        """
        Register a new model version.

        Args:
            model: The model object
            model_name: Name of the model
            metrics: Model performance metrics
            params: Model parameters
            artifacts: Additional artifact paths
            description: Model description

        Returns:
            Version string
        """
        version = self._generate_version(model_name)

        # Create model directory
        model_dir = self.registry_path / model_name / version
        model_dir.mkdir(parents=True, exist_ok=True)

        # Save model
        model_path = model_dir / "model.joblib"
        joblib.dump(model, model_path)

        # Copy artifacts if provided
        if artifacts:
            artifacts_dir = model_dir / "artifacts"
            artifacts_dir.mkdir(exist_ok=True)
            for name, path in artifacts.items():
                if Path(path).exists():
                    shutil.copy(path, artifacts_dir / Path(path).name)

        # Create model metadata
        model_metadata = {
            "version": version,
            "model_name": model_name,
            "registered_at": datetime.now().isoformat(),
            "metrics": metrics,
            "params": params or {},
            "description": description or "",
            "stage": "none",
            "model_path": str(model_path),
            "artifacts": artifacts or {},
        }

        # Update registry metadata
        if model_name not in self.metadata["models"]:
            self.metadata["models"][model_name] = {}

        self.metadata["models"][model_name][version] = model_metadata

        # Save metadata
        self._save_metadata()

        # Also save local metadata
        with open(model_dir / "metadata.json", "w") as f:
            json.dump(model_metadata, f, indent=2)

        logger.info(f"Registered model {model_name} version {version}")
        return version

    def promote_to_staging(self, model_name: str, version: str) -> bool:
        """
        Promote a model to staging.

        Args:
            model_name: Name of the model
            version: Version to promote

        Returns:
            True if promoted successfully
        """
        if model_name not in self.metadata["models"]:
            logger.error(f"Model {model_name} not found")
            return False

        if version not in self.metadata["models"][model_name]:
            logger.error(f"Version {version} not found for {model_name}")
            return False

        self.metadata["models"][model_name][version]["stage"] = "staging"
        self.metadata["staging_model"] = {"model_name": model_name, "version": version}
        self._save_metadata()

        logger.info(f"Promoted {model_name} v{version} to staging")
        return True

    def promote_to_production(
        self, model_name: str, version: str, force: bool = False
    ) -> bool:
        """
        Promote a model to production.

        Args:
            model_name: Name of the model
            version: Version to promote
            force: Force promotion even if threshold not met

        Returns:
            True if promoted successfully
        """
        if model_name not in self.metadata["models"]:
            logger.error(f"Model {model_name} not found")
            return False

        if version not in self.metadata["models"][model_name]:
            logger.error(f"Version {version} not found for {model_name}")
            return False

        model_meta = self.metadata["models"][model_name][version]
        metrics = model_meta["metrics"]

        # Check threshold
        if not force:
            for metric, threshold in self.promotion_threshold.items():
                if metric in metrics and metrics[metric] < threshold:
                    logger.warning(
                        f"Model does not meet threshold for {metric}: "
                        f"{metrics[metric]:.4f} < {threshold}"
                    )
                    return False

        # Archive current production model
        if self.metadata["production_model"]:
            old_name = self.metadata["production_model"]["model_name"]
            old_version = self.metadata["production_model"]["version"]
            self.metadata["models"][old_name][old_version]["stage"] = "archived"

        # Promote to production
        self.metadata["models"][model_name][version]["stage"] = "production"
        self.metadata["production_model"] = {
            "model_name": model_name,
            "version": version,
        }
        self._save_metadata()

        logger.info(f"Promoted {model_name} v{version} to PRODUCTION")
        return True

    def get_model(
        self, model_name: str, version: str = None, stage: str = None
    ) -> tuple:
        """
        Get a model from the registry.

        Args:
            model_name: Name of the model
            version: Specific version (optional)
            stage: Get model at specific stage ('production', 'staging')

        Returns:
            Tuple of (model, metadata)
        """
        if stage == "production":
            if self.metadata["production_model"]:
                model_name = self.metadata["production_model"]["model_name"]
                version = self.metadata["production_model"]["version"]
            else:
                logger.warning("No production model found")
                return None, None

        if stage == "staging":
            if self.metadata["staging_model"]:
                model_name = self.metadata["staging_model"]["model_name"]
                version = self.metadata["staging_model"]["version"]
            else:
                logger.warning("No staging model found")
                return None, None

        if version is None:
            # Get latest version
            versions = list(self.metadata["models"].get(model_name, {}).keys())
            if not versions:
                logger.error(f"No versions found for {model_name}")
                return None, None
            version = max(versions, key=lambda v: [int(x) for x in v.split(".")])

        model_meta = self.metadata["models"].get(model_name, {}).get(version)
        if not model_meta:
            logger.error(f"Model {model_name} v{version} not found")
            return None, None

        model_path = Path(model_meta["model_path"])
        if not model_path.exists():
            logger.error(f"Model file not found: {model_path}")
            return None, None

        model = joblib.load(model_path)
        logger.info(f"Loaded model {model_name} v{version}")

        return model, model_meta

    def list_models(self) -> List[Dict[str, Any]]:
        """
        List all registered models.

        Returns:
            List of model information
        """
        models = []
        for model_name, versions in self.metadata["models"].items():
            for version, meta in versions.items():
                models.append(
                    {
                        "model_name": model_name,
                        "version": version,
                        "stage": meta.get("stage", "none"),
                        "registered_at": meta.get("registered_at"),
                        "metrics": meta.get("metrics", {}),
                    }
                )
        return models

    def get_production_model(self) -> tuple:
        """Get the current production model."""
        return self.get_model(None, stage="production")

    def compare_models(
        self, model_name: str, version1: str, version2: str
    ) -> Dict[str, Any]:
        """
        Compare two model versions.

        Args:
            model_name: Name of the model
            version1: First version
            version2: Second version

        Returns:
            Comparison dictionary
        """
        meta1 = self.metadata["models"].get(model_name, {}).get(version1, {})
        meta2 = self.metadata["models"].get(model_name, {}).get(version2, {})

        if not meta1 or not meta2:
            logger.error("One or both versions not found")
            return {}

        comparison = {
            "version1": version1,
            "version2": version2,
            "metrics_comparison": {},
        }

        metrics1 = meta1.get("metrics", {})
        metrics2 = meta2.get("metrics", {})

        for metric in set(list(metrics1.keys()) + list(metrics2.keys())):
            val1 = metrics1.get(metric)
            val2 = metrics2.get(metric)
            comparison["metrics_comparison"][metric] = {
                "v1": val1,
                "v2": val2,
                "diff": (val2 - val1) if val1 and val2 else None,
            }

        return comparison
