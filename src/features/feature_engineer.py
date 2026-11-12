"""
Feature Engineering Module
==========================
Handles feature scaling, transformation, and selection.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.preprocessing import (
    MinMaxScaler,
    PolynomialFeatures,
    RobustScaler,
    StandardScaler,
)


class FeatureEngineer:
    """
    Feature engineering pipeline for ML models.
    """

    def __init__(
        self,
        scaling_method: str = "standard",
        create_polynomial: bool = True,
        polynomial_degree: int = 2,
        create_interactions: bool = True,
        feature_selection: bool = True,
        n_features_to_select: int = 10,
    ):
        """
        Initialize FeatureEngineer.

        Args:
            scaling_method: 'standard', 'minmax', or 'robust'
            create_polynomial: Whether to create polynomial features
            polynomial_degree: Degree of polynomial features
            create_interactions: Whether to create interaction features
            feature_selection: Whether to perform feature selection
            n_features_to_select: Number of features to select
        """
        self.scaling_method = scaling_method
        self.create_polynomial = create_polynomial
        self.polynomial_degree = polynomial_degree
        self.create_interactions = create_interactions
        self.feature_selection = feature_selection
        self.n_features_to_select = n_features_to_select

        # Initialize transformers
        self.scaler = self._get_scaler()
        self.poly_transformer: Optional[PolynomialFeatures] = None
        self.selector: Optional[SelectKBest] = None
        self.selected_feature_names: List[str] = []
        self.final_feature_names: List[str] = []

    def _get_scaler(self):
        """Get the appropriate scaler based on config."""
        scalers = {
            "standard": StandardScaler(),
            "minmax": MinMaxScaler(),
            "robust": RobustScaler(),
        }
        return scalers.get(self.scaling_method, StandardScaler())

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "FeatureEngineer":
        """
        Fit the feature engineering pipeline.

        Args:
            X: Training features
            y: Training target

        Returns:
            self
        """
        logger.info("Fitting feature engineering pipeline...")
        X_transformed = X.copy()
        current_feature_names = list(X.columns)

        # Step 1: Scaling
        logger.info(f"Applying {self.scaling_method} scaling...")
        X_scaled = self.scaler.fit_transform(X_transformed)
        X_transformed = pd.DataFrame(
            X_scaled, columns=current_feature_names, index=X.index
        )

        # Step 2: Polynomial Features
        if self.create_polynomial:
            logger.info(
                f"Creating polynomial features (degree={self.polynomial_degree})..."
            )
            self.poly_transformer = PolynomialFeatures(
                degree=self.polynomial_degree,
                include_bias=False,
                interaction_only=not self.create_interactions,
            )
            X_poly = self.poly_transformer.fit_transform(X_transformed)
            poly_feature_names = self.poly_transformer.get_feature_names_out(
                current_feature_names
            )
            X_transformed = pd.DataFrame(
                X_poly, columns=poly_feature_names, index=X.index
            )
            current_feature_names = list(poly_feature_names)
            logger.info(f"Created {len(current_feature_names)} features")

        # Step 3: Feature Selection
        if self.feature_selection:
            n_features = min(self.n_features_to_select, X_transformed.shape[1])
            logger.info(f"Selecting top {n_features} features...")
            self.selector = SelectKBest(score_func=f_classif, k=n_features)
            self.selector.fit(X_transformed, y)

            # Get selected feature names
            selected_mask = self.selector.get_support()
            self.selected_feature_names = [
                name
                for name, selected in zip(current_feature_names, selected_mask)
                if selected
            ]
            logger.info(f"Selected features: {self.selected_feature_names}")

        self.final_feature_names = (
            self.selected_feature_names
            if self.feature_selection
            else current_feature_names
        )

        logger.info(
            f"Feature engineering fit complete. Final feature count: {len(self.final_feature_names)}"
        )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data using fitted pipeline.

        Args:
            X: Features to transform

        Returns:
            Transformed features
        """
        X_transformed = X.copy()

        # Step 1: Scaling
        X_scaled = self.scaler.transform(X_transformed)
        X_transformed = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)

        # Step 2: Polynomial Features
        if self.create_polynomial and self.poly_transformer is not None:
            X_poly = self.poly_transformer.transform(X_transformed)
            poly_feature_names = self.poly_transformer.get_feature_names_out(X.columns)
            X_transformed = pd.DataFrame(
                X_poly, columns=poly_feature_names, index=X.index
            )

        # Step 3: Feature Selection
        if self.feature_selection and self.selector is not None:
            X_selected = self.selector.transform(X_transformed)
            X_transformed = pd.DataFrame(
                X_selected, columns=self.selected_feature_names, index=X.index
            )

        return X_transformed

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """
        Fit and transform in one step.

        Args:
            X: Training features
            y: Training target

        Returns:
            Transformed features
        """
        self.fit(X, y)
        return self.transform(X)

    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance scores from selector.

        Returns:
            Dictionary of feature names to importance scores
        """
        if self.selector is None:
            logger.warning("Feature selector not fitted. Run fit() first.")
            return {}

        scores = self.selector.scores_
        if self.create_polynomial and self.poly_transformer is not None:
            feature_names = self.poly_transformer.get_feature_names_out()
        else:
            feature_names = self.final_feature_names

        importance = dict(zip(feature_names, scores))
        return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))

    def save_artifacts(self, output_dir: str) -> Dict[str, str]:
        """
        Save feature engineering artifacts.

        Args:
            output_dir: Directory to save artifacts

        Returns:
            Dictionary with paths to saved artifacts
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        paths = {}

        # Save scaler
        scaler_path = output_path / "scaler.joblib"
        joblib.dump(self.scaler, scaler_path)
        paths["scaler"] = str(scaler_path)

        # Save polynomial transformer
        if self.poly_transformer is not None:
            poly_path = output_path / "poly_transformer.joblib"
            joblib.dump(self.poly_transformer, poly_path)
            paths["poly_transformer"] = str(poly_path)

        # Save selector
        if self.selector is not None:
            selector_path = output_path / "selector.joblib"
            joblib.dump(self.selector, selector_path)
            paths["selector"] = str(selector_path)

        # Save feature names
        names_path = output_path / "feature_names.joblib"
        joblib.dump(
            {
                "selected": self.selected_feature_names,
                "final": self.final_feature_names,
            },
            names_path,
        )
        paths["feature_names"] = str(names_path)

        logger.info(f"Feature engineering artifacts saved to {output_dir}")
        return paths

    def load_artifacts(self, input_dir: str):
        """
        Load feature engineering artifacts.

        Args:
            input_dir: Directory containing artifacts
        """
        input_path = Path(input_dir)

        # Load scaler
        scaler_path = input_path / "scaler.joblib"
        if scaler_path.exists():
            self.scaler = joblib.load(scaler_path)

        # Load polynomial transformer
        poly_path = input_path / "poly_transformer.joblib"
        if poly_path.exists():
            self.poly_transformer = joblib.load(poly_path)

        # Load selector
        selector_path = input_path / "selector.joblib"
        if selector_path.exists():
            self.selector = joblib.load(selector_path)

        # Load feature names
        names_path = input_path / "feature_names.joblib"
        if names_path.exists():
            names = joblib.load(names_path)
            self.selected_feature_names = names.get("selected", [])
            self.final_feature_names = names.get("final", [])

        logger.info(f"Feature engineering artifacts loaded from {input_dir}")
