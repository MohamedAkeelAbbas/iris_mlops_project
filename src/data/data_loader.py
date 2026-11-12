"""
Data Loader Module
==================
Handles loading and initial validation of the Iris dataset.
"""

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.datasets import load_iris


class DataLoader:
    """
    Data loader for Iris dataset with support for local files or sklearn dataset.
    """

    def __init__(self, data_path: Optional[str] = None):
        """
        Initialize DataLoader.

        Args:
            data_path: Path to CSV file, or None to use sklearn's iris dataset
        """
        self.data_path = data_path
        self.raw_data: Optional[pd.DataFrame] = None

    def load_from_sklearn(self) -> pd.DataFrame:
        """
        Load Iris dataset from sklearn.

        Returns:
            DataFrame with iris data
        """
        logger.info("Loading Iris dataset from sklearn...")

        iris = load_iris()

        # Create DataFrame
        df = pd.DataFrame(data=iris.data, columns=iris.feature_names)

        # Add target column
        df["species"] = pd.Categorical.from_codes(
            iris.target, categories=iris.target_names
        )

        # Clean column names (remove spaces and special chars)
        df.columns = [
            col.replace(" ", "_").replace("(", "").replace(")", "")
            for col in df.columns
        ]

        logger.info(f"Loaded {len(df)} samples with {len(df.columns)} columns")
        self.raw_data = df
        return df

    def load_from_csv(self, file_path: str) -> pd.DataFrame:
        """
        Load data from CSV file.

        Args:
            file_path: Path to CSV file

        Returns:
            DataFrame with loaded data
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")

        logger.info(f"Loading data from {file_path}...")
        df = pd.read_csv(file_path)

        logger.info(f"Loaded {len(df)} samples with {len(df.columns)} columns")
        self.raw_data = df
        return df

    def load(self) -> pd.DataFrame:
        """
        Load data from specified source or sklearn.

        Returns:
            DataFrame with loaded data
        """
        if self.data_path and Path(self.data_path).exists():
            return self.load_from_csv(self.data_path)
        else:
            return self.load_from_sklearn()

    def save_raw_data(self, output_path: str) -> str:
        """
        Save raw data to CSV file.

        Args:
            output_path: Path to save the data

        Returns:
            Path where data was saved
        """
        if self.raw_data is None:
            raise ValueError("No data loaded. Call load() first.")

        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        self.raw_data.to_csv(path, index=False)
        logger.info(f"Raw data saved to {output_path}")

        return output_path

    def get_data_info(self) -> dict:
        """
        Get information about the loaded data.

        Returns:
            Dictionary with data statistics
        """
        if self.raw_data is None:
            raise ValueError("No data loaded. Call load() first.")

        info = {
            "n_samples": len(self.raw_data),
            "n_features": len(self.raw_data.columns) - 1,
            "columns": list(self.raw_data.columns),
            "dtypes": self.raw_data.dtypes.astype(str).to_dict(),
            "missing_values": self.raw_data.isnull().sum().to_dict(),
            "target_distribution": self.raw_data["species"].value_counts().to_dict(),
        }

        return info

    def validate_data(self) -> Tuple[bool, list]:
        """
        Validate the loaded data.

        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []

        if self.raw_data is None:
            return False, ["No data loaded"]

        # Check for missing values
        missing = self.raw_data.isnull().sum()
        if missing.any():
            cols_with_missing = missing[missing > 0].index.tolist()
            issues.append(f"Missing values in columns: {cols_with_missing}")

        # Check for duplicates
        n_duplicates = self.raw_data.duplicated().sum()
        if n_duplicates > 0:
            issues.append(f"Found {n_duplicates} duplicate rows")

        # Check target column exists
        if "species" not in self.raw_data.columns:
            issues.append("Target column 'species' not found")

        # Check minimum samples
        if len(self.raw_data) < 50:
            issues.append(f"Insufficient samples: {len(self.raw_data)} < 50")

        is_valid = len(issues) == 0

        if is_valid:
            logger.info("Data validation passed!")
        else:
            logger.warning(f"Data validation issues: {issues}")

        return is_valid, issues
