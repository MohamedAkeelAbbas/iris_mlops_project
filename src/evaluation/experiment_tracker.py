
"""
Experiment Tracking Module
==========================
Local experiment tracking with MLflow integration.
"""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import mlflow
from loguru import logger
from mlflow.tracking import MlflowClient


class ExperimentTracker:
    """
    Experiment tracking using MLflow.
    """

    def __init__(
        self,
        tracking_uri: str = "sqlite:///experiments/mlflow.db",
        experiment_name: str = "iris_classification",
        log_models: bool = True,
        log_artifacts: bool = True,
    ):
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
        self.log_models = log_models
        self.log_artifacts = log_artifacts

        self.run_id: Optional[str] = None
        self.experiment_id: Optional[str] = None
        self.active_run = None

        self._setup_mlflow()

    def _setup_mlflow(self):
        """Set up MLflow tracking."""

        if self.tracking_uri.startswith("sqlite:///"):
            db_path = self.tracking_uri[len("sqlite:///"):]
            database_file = Path(db_path)
            database_file.parent.mkdir(
                parents=True,
                exist_ok=True
            )

        mlflow.set_tracking_uri(self.tracking_uri)

        experiment = mlflow.get_experiment_by_name(
            self.experiment_name
        )

        if experiment is None:
            try:
                self.experiment_id = mlflow.create_experiment(
                    self.experiment_name
                )

            except mlflow.exceptions.MlflowException:
                experiment = mlflow.get_experiment_by_name(
                    self.experiment_name
                )

                if experiment is None:
                    raise

                self.experiment_id = experiment.experiment_id

        else:
            self.experiment_id = experiment.experiment_id

        mlflow.set_experiment(self.experiment_name)

        logger.info(
            f"MLflow experiment '{self.experiment_name}' initialized"
        )

        logger.info(
            f"MLflow tracking URI: {self.tracking_uri}"
        )

    def start_run(
        self,
        run_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None
    ):
        """Start a new MLflow run."""

        if run_name is None:
            run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        self.active_run = mlflow.start_run(
            run_name=run_name
        )

        self.run_id = self.active_run.info.run_id

        if tags:
            for key, value in tags.items():
                mlflow.set_tag(key, value)

        logger.info(
            f"Started MLflow run: {run_name} (ID: {self.run_id})"
        )

    def end_run(self, status: str = "FINISHED"):
        """End the current MLflow run."""

        mlflow.end_run(status=status)

        logger.info(
            f"Ended MLflow run with status: {status}"
        )

    def log_params(self, params: Dict[str, Any]):
        """Log parameters."""

        flat_params = self._flatten_dict(params)

        for key, value in flat_params.items():
            try:
                mlflow.log_param(key, value)

            except Exception as e:
                logger.warning(
                    f"Could not log param {key}: {e}"
                )

        logger.info(
            f"Logged {len(flat_params)} parameters"
        )

    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: Optional[int] = None
    ):
        """Log metrics."""

        for key, value in metrics.items():

            if isinstance(value, (int, float)) and not isinstance(value, bool):
                try:
                    mlflow.log_metric(
                        key,
                        value,
                        step=step
                    )

                except Exception as e:
                    logger.warning(
                        f"Could not log metric {key}: {e}"
                    )

        logger.info(
            f"Logged {len(metrics)} metrics"
        )

    def log_artifact(
        self,
        artifact_path: str,
        artifact_dir: Optional[str] = None
    ):
        """Log an artifact."""

        if self.log_artifacts:
            mlflow.log_artifact(
                artifact_path,
                artifact_dir
            )

            logger.info(
                f"Logged artifact: {artifact_path}"
            )

    def log_model(
        self,
        model,
        model_name: str,
        signature=None
    ):
        """Log a scikit-learn model."""

        if self.log_models:
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path=model_name,
                signature=signature,
                skops_trusted_types=[
                    "sklearn.tree._tree.Tree"
                ]
            )

            logger.info(
                f"Logged model: {model_name}"
            )

    def log_figure(
        self,
        figure,
        artifact_file: str
    ):
        """Log a matplotlib figure."""

        mlflow.log_figure(
            figure,
            artifact_file
        )

        logger.info(
            f"Logged figure: {artifact_file}"
        )

    def log_dict(
        self,
        dictionary: Dict[str, Any],
        artifact_file: str
    ):
        """Log a dictionary as JSON artifact."""

        mlflow.log_dict(
            dictionary,
            artifact_file
        )

        logger.info(
            f"Logged dict: {artifact_file}"
        )

    def _flatten_dict(
        self,
        d: Dict[str, Any],
        parent_key: str = "",
        sep: str = "."
    ) -> Dict[str, Any]:
        """Flatten nested dictionary."""

        items = []

        for k, v in d.items():

            new_key = (
                f"{parent_key}{sep}{k}"
                if parent_key
                else k
            )

            if isinstance(v, dict):
                items.extend(
                    self._flatten_dict(
                        v,
                        new_key,
                        sep=sep
                    ).items()
                )

            else:
                items.append(
                    (new_key, v)
                )

        return dict(items)

    def get_run_info(self) -> Dict[str, Any]:
        """Get information about current run."""

        if self.run_id is None:
            return {}

        client = MlflowClient()

        run = client.get_run(
            self.run_id
        )

        return {
            "run_id": run.info.run_id,
            "run_name": run.info.run_name,
            "experiment_id": run.info.experiment_id,
            "status": run.info.status,
            "start_time": run.info.start_time,
            "end_time": run.info.end_time,
            "artifact_uri": run.info.artifact_uri,
        }

    def get_best_run(
        self,
        metric: str = "accuracy",
        ascending: bool = False
    ) -> Dict[str, Any]:
        """Get the best run based on a metric."""

        client = MlflowClient()

        experiment = client.get_experiment_by_name(
            self.experiment_name
        )

        if experiment is None:
            return {}

        runs = client.search_runs(
            experiment_ids=[
                experiment.experiment_id
            ],
            order_by=[
                f"metrics.{metric} {'ASC' if ascending else 'DESC'}"
            ],
            max_results=1,
        )

        if not runs:
            return {}

        best_run = runs[0]

        return {
            "run_id": best_run.info.run_id,
            "run_name": best_run.info.run_name,
            "metrics": best_run.data.metrics,
            "params": best_run.data.params,
            "artifact_uri": best_run.info.artifact_uri,
        }

    def list_experiments(self) -> List[Dict[str, str]]:
        """List all experiments."""

        client = MlflowClient()

        experiments = client.search_experiments()

        return [
            {
                "experiment_id": exp.experiment_id,
                "name": exp.name,
                "artifact_location": exp.artifact_location,
            }
            for exp in experiments
        ]
