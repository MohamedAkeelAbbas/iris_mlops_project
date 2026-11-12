"""
Model Trainer Module
====================
Handles model training with support for multiple algorithms.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.svm import SVC


class ModelTrainer:
    """
    Model trainer supporting multiple classification algorithms.
    """

    SUPPORTED_MODELS = {
        "random_forest": RandomForestClassifier,
        "logistic_regression": LogisticRegression,
        "svm": SVC,
        "gradient_boosting": GradientBoostingClassifier,
    }

    def __init__(
        self,
        model_type: str = "random_forest",
        model_params: Dict[str, Any] = None,
        cross_validation: bool = True,
        cv_folds: int = 5,
    ):
        """
        Initialize ModelTrainer.

        Args:
            model_type: Type of model to train
            model_params: Model hyperparameters
            cross_validation: Whether to perform cross-validation
            cv_folds: Number of cross-validation folds
        """
        if model_type not in self.SUPPORTED_MODELS:
            raise ValueError(
                f"Unsupported model type: {model_type}. "
                f"Supported: {list(self.SUPPORTED_MODELS.keys())}"
            )

        self.model_type = model_type
        self.model_params = model_params or {}
        self.cross_validation = cross_validation
        self.cv_folds = cv_folds
        self.model = None
        self.cv_scores: List[float] = []
        self.training_history: Dict[str, Any] = {}

    def _create_model(self):
        """Create model instance with specified parameters."""
        model_class = self.SUPPORTED_MODELS[self.model_type]
        self.model = model_class(**self.model_params)
        logger.info(f"Created {self.model_type} model with params: {self.model_params}")

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
    ) -> "ModelTrainer":
        """
        Train the model.

        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features (optional)
            y_val: Validation target (optional)

        Returns:
            self
        """
        self._create_model()

        logger.info(f"Training {self.model_type} on {len(X_train)} samples...")

        # Cross-validation
        if self.cross_validation:
            logger.info(f"Running {self.cv_folds}-fold cross-validation...")
            self.cv_scores = cross_val_score(
                self.model, X_train, y_train, cv=self.cv_folds, scoring="accuracy"
            ).tolist()
            logger.info(f"CV Scores: {self.cv_scores}")
            logger.info(
                f"CV Mean: {np.mean(self.cv_scores):.4f} (+/- {np.std(self.cv_scores)*2:.4f})"
            )

        # Fit model
        self.model.fit(X_train, y_train)

        # Calculate training score
        train_score = self.model.score(X_train, y_train)
        logger.info(f"Training accuracy: {train_score:.4f}")

        # Calculate validation score if provided
        val_score = None
        if X_val is not None and y_val is not None and len(X_val) > 0:
            val_score = self.model.score(X_val, y_val)
            logger.info(f"Validation accuracy: {val_score:.4f}")

        # Store training history
        self.training_history = {
            "model_type": self.model_type,
            "model_params": self.model_params,
            "train_samples": len(X_train),
            "train_score": train_score,
            "val_score": val_score,
            "cv_scores": self.cv_scores,
            "cv_mean": np.mean(self.cv_scores) if self.cv_scores else None,
            "cv_std": np.std(self.cv_scores) if self.cv_scores else None,
        }

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions.

        Args:
            X: Features to predict

        Returns:
            Predictions
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Get prediction probabilities.

        Args:
            X: Features to predict

        Returns:
            Prediction probabilities
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)
        else:
            logger.warning(f"{self.model_type} does not support predict_proba")
            return None

    def get_feature_importance(self, feature_names: List[str]) -> Dict[str, float]:
        """
        Get feature importance.

        Args:
            feature_names: List of feature names

        Returns:
            Dictionary of feature names to importance scores
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        importance = {}
        if hasattr(self.model, "feature_importances_"):
            importance = dict(zip(feature_names, self.model.feature_importances_))
        elif hasattr(self.model, "coef_"):
            # For logistic regression, use absolute coefficient values
            if len(self.model.coef_.shape) > 1:
                importance = dict(
                    zip(feature_names, np.abs(self.model.coef_).mean(axis=0))
                )
            else:
                importance = dict(zip(feature_names, np.abs(self.model.coef_)))

        return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))

    def hyperparameter_tune(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        param_grid: Dict[str, list],
        method: str = "grid_search",
        n_iter: int = 20,
    ) -> Dict[str, Any]:
        """
        Perform hyperparameter tuning.

        Args:
            X_train: Training features
            y_train: Training target
            param_grid: Parameter grid for search
            method: 'grid_search' or 'random_search'
            n_iter: Number of iterations for random search

        Returns:
            Best parameters and score
        """
        self._create_model()

        logger.info(f"Starting hyperparameter tuning with {method}...")

        if method == "grid_search":
            search = GridSearchCV(
                self.model,
                param_grid,
                cv=self.cv_folds,
                scoring="accuracy",
                n_jobs=-1,
                verbose=1,
            )
        else:
            search = RandomizedSearchCV(
                self.model,
                param_grid,
                n_iter=n_iter,
                cv=self.cv_folds,
                scoring="accuracy",
                n_jobs=-1,
                verbose=1,
                random_state=42,
            )

        search.fit(X_train, y_train)

        self.model = search.best_estimator_
        self.model_params = search.best_params_

        results = {
            "best_params": search.best_params_,
            "best_score": search.best_score_,
            "cv_results": {
                "mean_test_score": search.cv_results_["mean_test_score"].tolist(),
                "std_test_score": search.cv_results_["std_test_score"].tolist(),
            },
        }

        logger.info(f"Best params: {search.best_params_}")
        logger.info(f"Best CV score: {search.best_score_:.4f}")

        return results

    def save_model(self, output_path: str) -> str:
        """
        Save trained model.

        Args:
            output_path: Path to save the model

        Returns:
            Path where model was saved
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        joblib.dump(self.model, path)
        logger.info(f"Model saved to {output_path}")

        return output_path

    def load_model(self, model_path: str):
        """
        Load a trained model.

        Args:
            model_path: Path to the saved model
        """
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        self.model = joblib.load(path)
        logger.info(f"Model loaded from {model_path}")
