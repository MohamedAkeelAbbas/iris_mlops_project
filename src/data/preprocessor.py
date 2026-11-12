"""
Data Preprocessor Module
========================
Handles data cleaning, splitting, and basic transformations.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


class DataPreprocessor:
    """
    Data preprocessor for cleaning and splitting data.
    """

    def __init__(
        self,
        target_column: str = "species",
        test_size: float = 0.2,
        validation_size: float = 0.1,
        random_state: int = 42,
    ):
        """
        Initialize DataPreprocessor.

        Args:
            target_column: Name of the target column
            test_size: Proportion of data for testing
            validation_size: Proportion of data for validation
            random_state: Random seed for reproducibility
        """
        self.target_column = target_column
        self.test_size = test_size
        self.validation_size = validation_size
        self.random_state = random_state
        self.label_encoder = LabelEncoder()
        self.feature_columns: List[str] = []

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the data by handling missing values and duplicates.

        Args:
            df: Input DataFrame

        Returns:
            Cleaned DataFrame
        """
        logger.info("Cleaning data...")
        df_clean = df.copy()

        # Remove duplicates
        n_before = len(df_clean)
        df_clean = df_clean.drop_duplicates()
        n_removed = n_before - len(df_clean)
        if n_removed > 0:
            logger.info(f"Removed {n_removed} duplicate rows")

        # Handle missing values (for numeric columns, fill with median)
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df_clean[col].isnull().any():
                median_val = df_clean[col].median()
                df_clean[col].fillna(median_val, inplace=True)
                logger.info(f"Filled missing values in {col} with median: {median_val}")

        # Handle missing values in categorical columns
        cat_cols = df_clean.select_dtypes(include=["object", "category"]).columns
        for col in cat_cols:
            if df_clean[col].isnull().any():
                mode_val = df_clean[col].mode()[0]
                df_clean[col].fillna(mode_val, inplace=True)
                logger.info(f"Filled missing values in {col} with mode: {mode_val}")

        logger.info(f"Data cleaned. Shape: {df_clean.shape}")
        return df_clean

    def encode_target(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[int, str]]:
        """
        Encode the target column to numeric values.

        Args:
            df: Input DataFrame

        Returns:
            Tuple of (DataFrame with encoded target, mapping dict)
        """
        logger.info(f"Encoding target column: {self.target_column}")
        df_encoded = df.copy()

        # Fit and transform target
        df_encoded[f"{self.target_column}_encoded"] = self.label_encoder.fit_transform(
            df_encoded[self.target_column]
        )

        # Create mapping
        classes = self.label_encoder.classes_
        mapping = {i: cls for i, cls in enumerate(classes)}

        logger.info(f"Target encoding mapping: {mapping}")
        return df_encoded, mapping

    def split_data(
        self, df: pd.DataFrame, target_col: str = None
    ) -> Dict[str, Tuple[pd.DataFrame, pd.Series]]:
        """
        Split data into train, validation, and test sets.

        Args:
            df: Input DataFrame
            target_col: Target column name (uses encoded target by default)

        Returns:
            Dictionary with train, val, test splits
        """
        if target_col is None:
            target_col = f"{self.target_column}_encoded"

        # Determine feature columns
        self.feature_columns = [
            col for col in df.columns if col not in [self.target_column, target_col]
        ]

        X = df[self.feature_columns]
        y = df[target_col]

        logger.info(
            f"Splitting data with test_size={self.test_size}, val_size={self.validation_size}"
        )

        # First split: train+val vs test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=y
        )

        # Second split: train vs val
        if self.validation_size > 0:
            val_ratio = self.validation_size / (1 - self.test_size)
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp,
                y_temp,
                test_size=val_ratio,
                random_state=self.random_state,
                stratify=y_temp,
            )
        else:
            X_train, y_train = X_temp, y_temp
            X_val, y_val = pd.DataFrame(), pd.Series(dtype=int)

        splits = {
            "train": (X_train, y_train),
            "val": (X_val, y_val),
            "test": (X_test, y_test),
        }

        logger.info(
            f"Split sizes - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}"
        )

        return splits

    def process(self, df: pd.DataFrame) -> Dict[str, Tuple[pd.DataFrame, pd.Series]]:
        """
        Run the complete preprocessing pipeline.

        Args:
            df: Raw input DataFrame

        Returns:
            Dictionary with processed train, val, test splits
        """
        # Clean data
        df_clean = self.clean_data(df)

        # Encode target
        df_encoded, self.target_mapping = self.encode_target(df_clean)

        # Split data
        splits = self.split_data(df_encoded)

        return splits

    def save_artifacts(self, output_dir: str) -> Dict[str, str]:
        """
        Save preprocessing artifacts (encoders, etc.).

        Args:
            output_dir: Directory to save artifacts

        Returns:
            Dictionary with paths to saved artifacts
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        paths = {}

        # Save label encoder
        encoder_path = output_path / "label_encoder.joblib"
        joblib.dump(self.label_encoder, encoder_path)
        paths["label_encoder"] = str(encoder_path)

        # Save feature columns
        features_path = output_path / "feature_columns.joblib"
        joblib.dump(self.feature_columns, features_path)
        paths["feature_columns"] = str(features_path)

        logger.info(f"Preprocessing artifacts saved to {output_dir}")
        return paths

    def load_artifacts(self, input_dir: str):
        """
        Load preprocessing artifacts.

        Args:
            input_dir: Directory containing artifacts
        """
        input_path = Path(input_dir)

        encoder_path = input_path / "label_encoder.joblib"
        if encoder_path.exists():
            self.label_encoder = joblib.load(encoder_path)

        features_path = input_path / "feature_columns.joblib"
        if features_path.exists():
            self.feature_columns = joblib.load(features_path)

        logger.info(f"Preprocessing artifacts loaded from {input_dir}")
