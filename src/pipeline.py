"""
ML Pipeline Orchestrator
========================
Main pipeline that orchestrates all ML workflow steps.
"""

import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from data.data_loader import DataLoader
from data.preprocessor import DataPreprocessor
from evaluation.evaluator import ModelEvaluator
from evaluation.experiment_tracker import ExperimentTracker
from features.feature_engineer import FeatureEngineer
from models.registry import ModelRegistry
from models.trainer import ModelTrainer
from utils.config import ProjectConfig, load_config
from utils.logger import get_logger, setup_logger


class MLPipeline:
    """
    End-to-end ML pipeline orchestrator.
    """

    def __init__(self, config_path: str = "configs/config.yaml"):
        """
        Initialize MLPipeline.

        Args:
            config_path: Path to configuration file
        """
        self.config = load_config(config_path)
        self.logger = get_logger()
        self.run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Initialize components
        self.data_loader: Optional[DataLoader] = None
        self.preprocessor: Optional[DataPreprocessor] = None
        self.feature_engineer: Optional[FeatureEngineer] = None
        self.trainer: Optional[ModelTrainer] = None
        self.evaluator: Optional[ModelEvaluator] = None
        self.experiment_tracker: Optional[ExperimentTracker] = None
        self.model_registry: Optional[ModelRegistry] = None

        # Results
        self.results: Dict[str, Any] = {}

    def setup_logger(self):
        """Set up logging."""
        setup_logger(log_file=f"logs/pipeline_{self.run_timestamp}.log", level="INFO")
        self.logger = get_logger()
        self.logger.info("Pipeline logger initialized")

    def load_data(self) -> Dict[str, Any]:
        """
        Step 1: Load data.

        Returns:
            Data loading results
        """
        self.logger.info("=" * 60)
        self.logger.info("STEP 1: DATA LOADING")
        self.logger.info("=" * 60)

        self.data_loader = DataLoader(self.config.data.raw_path)
        df = self.data_loader.load()

        # Validate data
        is_valid, issues = self.data_loader.validate_data()
        if not is_valid:
            self.logger.warning(f"Data validation issues: {issues}")

        # Save raw data
        raw_path = f"{self.config.data.raw_path}"
        Path(raw_path).parent.mkdir(parents=True, exist_ok=True)
        self.data_loader.save_raw_data(raw_path)

        data_info = self.data_loader.get_data_info()
        self.results["data_loading"] = {
            "n_samples": data_info["n_samples"],
            "n_features": data_info["n_features"],
            "is_valid": is_valid,
            "issues": issues,
        }

        return self.results["data_loading"]

    def preprocess_data(self) -> Dict[str, Any]:
        """
        Step 2: Preprocess data.

        Returns:
            Preprocessing results
        """
        self.logger.info("=" * 60)
        self.logger.info("STEP 2: DATA PREPROCESSING")
        self.logger.info("=" * 60)

        self.preprocessor = DataPreprocessor(
            target_column=self.config.data.target_column,
            test_size=self.config.data.test_size,
            validation_size=self.config.data.validation_size,
            random_state=self.config.data.random_state,
        )

        # Process data
        self.splits = self.preprocessor.process(self.data_loader.raw_data)

        # Save preprocessing artifacts
        artifacts_path = f"{self.config.data.processed_path}/artifacts"
        self.preprocessor.save_artifacts(artifacts_path)

        self.results["preprocessing"] = {
            "train_size": len(self.splits["train"][0]),
            "val_size": len(self.splits["val"][0]),
            "test_size": len(self.splits["test"][0]),
            "feature_columns": self.preprocessor.feature_columns,
            "target_mapping": self.preprocessor.target_mapping,
        }

        return self.results["preprocessing"]

    def engineer_features(self) -> Dict[str, Any]:
        """
        Step 3: Feature engineering.

        Returns:
            Feature engineering results
        """
        self.logger.info("=" * 60)
        self.logger.info("STEP 3: FEATURE ENGINEERING")
        self.logger.info("=" * 60)

        self.feature_engineer = FeatureEngineer(
            scaling_method=self.config.features.scaling_method,
            create_polynomial=self.config.features.create_polynomial,
            polynomial_degree=self.config.features.polynomial_degree,
            create_interactions=self.config.features.create_interactions,
            feature_selection=self.config.features.feature_selection,
            n_features_to_select=self.config.features.n_features_to_select,
        )

        X_train, y_train = self.splits["train"]
        X_val, y_val = self.splits["val"]
        X_test, y_test = self.splits["test"]

        # Fit and transform
        self.X_train_fe = self.feature_engineer.fit_transform(X_train, y_train)
        self.y_train = y_train

        self.X_val_fe = (
            self.feature_engineer.transform(X_val) if len(X_val) > 0 else X_val
        )
        self.y_val = y_val

        self.X_test_fe = self.feature_engineer.transform(X_test)
        self.y_test = y_test

        # Save feature engineering artifacts
        fe_path = f"{self.config.data.features_path}/artifacts"
        self.feature_engineer.save_artifacts(fe_path)

        # Get feature importance
        feature_importance = self.feature_engineer.get_feature_importance()

        self.results["feature_engineering"] = {
            "n_original_features": len(self.preprocessor.feature_columns),
            "n_final_features": len(self.feature_engineer.final_feature_names),
            "final_features": self.feature_engineer.final_feature_names,
            "top_features": dict(list(feature_importance.items())[:5]),
        }

        return self.results["feature_engineering"]

    def train_model(self) -> Dict[str, Any]:
        """
        Step 4: Model training.

        Returns:
            Training results
        """
        self.logger.info("=" * 60)
        self.logger.info("STEP 4: MODEL TRAINING")
        self.logger.info("=" * 60)

        model_type = self.config.model.type
        model_params = getattr(self.config.model, model_type, {})

        # Convert Pydantic model to dict if needed
        if hasattr(model_params, "model_dump"):
            model_params = model_params
        elif isinstance(model_params, dict):
            model_params = model_params
        else:
            model_params = {}

        self.trainer = ModelTrainer(
            model_type=model_type,
            model_params=model_params,
            cross_validation=self.config.training.cross_validation,
            cv_folds=self.config.training.cv_folds,
        )

        # Train model
        self.trainer.train(
            self.X_train_fe,
            self.y_train,
            self.X_val_fe if len(self.X_val_fe) > 0 else None,
            self.y_val if len(self.y_val) > 0 else None,
        )

        # Save model
        model_path = f"models/trained/model_{self.run_timestamp}.joblib"
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        self.trainer.save_model(model_path)

        # Get feature importance
        self.feature_importance = self.trainer.get_feature_importance(
            self.feature_engineer.final_feature_names
        )

        self.results["training"] = {
            "model_type": model_type,
            "model_params": model_params,
            "training_history": self.trainer.training_history,
            "model_path": model_path,
        }

        return self.results["training"]

    def evaluate_model(self) -> Dict[str, Any]:
        """
        Step 5: Model evaluation.

        Returns:
            Evaluation results
        """
        self.logger.info("=" * 60)
        self.logger.info("STEP 5: MODEL EVALUATION")
        self.logger.info("=" * 60)

        # Get class names from target mapping
        class_names = [
            self.preprocessor.target_mapping[i]
            for i in sorted(self.preprocessor.target_mapping.keys())
        ]

        self.evaluator = ModelEvaluator(
            metrics=self.config.evaluation.metrics,
            save_plots=self.config.evaluation.save_plots,
            plots_path=f"{self.config.evaluation.plots_path}/{self.run_timestamp}",
            class_names=class_names,
        )

        # Make predictions
        y_pred = self.trainer.predict(self.X_test_fe)
        y_proba = self.trainer.predict_proba(self.X_test_fe)

        # Evaluate
        metrics = self.evaluator.evaluate(self.y_test.values, y_pred, y_proba)

        # Generate plots
        if self.config.evaluation.save_plots:
            self.evaluator.plot_confusion_matrix(self.y_test.values, y_pred)
            self.evaluator.plot_feature_importance(self.feature_importance)
            self.evaluator.plot_metrics_comparison(metrics)

        # Generate report
        report_path = self.evaluator.generate_report()

        # Print summary
        self.evaluator.print_summary()

        self.results["evaluation"] = {
            "metrics": {
                k: v for k, v in metrics.items() if isinstance(v, (int, float))
            },
            "report_path": report_path,
        }

        return self.results["evaluation"]

    def track_experiment(self):
        """
        Step 6: Track experiment with MLflow.
        """
        self.logger.info("=" * 60)
        self.logger.info("STEP 6: EXPERIMENT TRACKING")
        self.logger.info("=" * 60)

        self.experiment_tracker = ExperimentTracker(
            tracking_uri=self.config.experiment.tracking_uri,
            experiment_name=self.config.experiment.experiment_name,
            log_models=self.config.experiment.log_models,
            log_artifacts=self.config.experiment.log_artifacts,
        )

        # Start run
        self.experiment_tracker.start_run(
            run_name=f"run_{self.run_timestamp}",
            tags={"model_type": self.config.model.type},
        )

        # Log parameters
        params = {
            "model_type": self.config.model.type,
            "test_size": self.config.data.test_size,
            "scaling_method": self.config.features.scaling_method,
            "cv_folds": self.config.training.cv_folds,
            **getattr(self.config.model, self.config.model.type, {}),
        }
        self.experiment_tracker.log_params(params)

        # Log metrics
        metrics = self.results["evaluation"]["metrics"]
        self.experiment_tracker.log_metrics(metrics)

        # Log model
        if self.config.experiment.log_models:
            self.experiment_tracker.log_model(
                self.trainer.model, f"model_{self.config.model.type}"
            )

        # Log artifacts
        if self.config.experiment.log_artifacts:
            report_path = self.results["evaluation"]["report_path"]
            self.experiment_tracker.log_artifact(report_path)

        # End run
        self.experiment_tracker.end_run()

        self.logger.info("Experiment tracking completed")

    def register_model(self) -> str:
        """
        Step 7: Register model in the registry.

        Returns:
            Model version string
        """
        self.logger.info("=" * 60)
        self.logger.info("STEP 7: MODEL REGISTRATION")
        self.logger.info("=" * 60)

        self.model_registry = ModelRegistry(
            registry_path=self.config.registry.path,
            promotion_threshold=self.config.registry.promotion_threshold,
        )

        # Register model
        version = self.model_registry.register_model(
            model=self.trainer.model,
            model_name=f"iris_{self.config.model.type}",
            metrics=self.results["evaluation"]["metrics"],
            params=getattr(self.config.model, self.config.model.type, {}),
            description=f"Iris classifier using {self.config.model.type}",
        )

        # Check if model meets production threshold
        metrics = self.results["evaluation"]["metrics"]
        meets_threshold = all(
            metrics.get(m, 0) >= t
            for m, t in self.config.registry.promotion_threshold.items()
        )

        if meets_threshold:
            self.model_registry.promote_to_staging(
                f"iris_{self.config.model.type}", version
            )
            self.logger.info(f"Model promoted to staging (meets threshold)")

        self.results["registration"] = {
            "version": version,
            "meets_threshold": meets_threshold,
        }

        return version

    def run(self) -> Dict[str, Any]:
        """
        Run the complete pipeline.

        Returns:
            Complete pipeline results
        """
        self.setup_logger()

        self.logger.info("*" * 60)
        self.logger.info("STARTING ML PIPELINE")
        self.logger.info(f"Run ID: {self.run_timestamp}")
        self.logger.info("*" * 60)

        try:
            # Execute pipeline steps
            self.load_data()
            self.preprocess_data()
            self.engineer_features()
            self.train_model()
            self.evaluate_model()
            self.track_experiment()
            self.register_model()

            self.logger.info("*" * 60)
            self.logger.info("PIPELINE COMPLETED SUCCESSFULLY")
            self.logger.info("*" * 60)

            # Final summary
            self._print_final_summary()

        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}")
            raise

        return self.results

    def _print_final_summary(self):
        """Print final pipeline summary."""
        print("\n" + "=" * 60)
        print("PIPELINE EXECUTION SUMMARY")
        print("=" * 60)
        print(f"\nRun ID: {self.run_timestamp}")
        print(f"Model Type: {self.config.model.type}")
        print(f"\nData:")
        print(f"  - Training samples: {self.results['preprocessing']['train_size']}")
        print(f"  - Test samples: {self.results['preprocessing']['test_size']}")
        print(
            f"  - Features: {self.results['feature_engineering']['n_final_features']}"
        )
        print(f"\nMetrics:")
        for metric, value in self.results["evaluation"]["metrics"].items():
            print(f"  - {metric}: {value:.4f}")
        print(f"\nModel Version: {self.results['registration']['version']}")
        print(
            f"Meets Production Threshold: {self.results['registration']['meets_threshold']}"
        )
        print("=" * 60 + "\n")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Iris MLOps Pipeline")
    parser.add_argument(
        "--config", type=str, default="configs/config.yaml", help="Path to config file"
    )
    args = parser.parse_args()

    # Run pipeline
    pipeline = MLPipeline(config_path=args.config)
    results = pipeline.run()

    return results


if __name__ == "__main__":
    main()
