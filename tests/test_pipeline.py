"""
Unit Tests for Iris MLOps Pipeline
==================================
Comprehensive test suite for all pipeline components.
"""

import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data.data_loader import DataLoader
from data.preprocessor import DataPreprocessor
from evaluation.evaluator import ModelEvaluator
from features.feature_engineer import FeatureEngineer
from models.registry import ModelRegistry
from models.trainer import ModelTrainer

# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def sample_iris_data():
    """Create sample Iris-like data for testing."""
    np.random.seed(42)
    n_samples = 100

    data = {
        "sepal_length_cm": np.random.uniform(4.0, 8.0, n_samples),
        "sepal_width_cm": np.random.uniform(2.0, 4.5, n_samples),
        "petal_length_cm": np.random.uniform(1.0, 7.0, n_samples),
        "petal_width_cm": np.random.uniform(0.1, 2.5, n_samples),
        "species": np.random.choice(["setosa", "versicolor", "virginica"], n_samples),
    }
    return pd.DataFrame(data)


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test artifacts."""
    temp_path = tempfile.mkdtemp()
    yield temp_path
    shutil.rmtree(temp_path)


# ============================================================================
# DATA LOADER TESTS
# ============================================================================


class TestDataLoader:
    """Tests for DataLoader class."""

    def test_load_from_sklearn(self):
        """Test loading Iris data from sklearn."""
        loader = DataLoader()
        df = loader.load_from_sklearn()

        assert df is not None
        assert len(df) == 150
        assert "species" in df.columns
        assert df["species"].nunique() == 3

    def test_data_validation_pass(self, sample_iris_data):
        """Test data validation passes for valid data."""
        loader = DataLoader()
        loader.raw_data = sample_iris_data

        is_valid, issues = loader.validate_data()
        assert is_valid
        assert len(issues) == 0

    def test_data_validation_missing_target(self, sample_iris_data):
        """Test validation catches missing target column."""
        loader = DataLoader()
        loader.raw_data = sample_iris_data.drop("species", axis=1)

        is_valid, issues = loader.validate_data()
        assert not is_valid
        assert any("species" in issue for issue in issues)

    def test_get_data_info(self):
        """Test getting data information."""
        loader = DataLoader()
        loader.load_from_sklearn()

        info = loader.get_data_info()
        assert info["n_samples"] == 150
        assert info["n_features"] == 4
        assert "species" in info["columns"]


# ============================================================================
# PREPROCESSOR TESTS
# ============================================================================


class TestDataPreprocessor:
    """Tests for DataPreprocessor class."""

    def test_clean_data(self, sample_iris_data):
        """Test data cleaning."""
        preprocessor = DataPreprocessor()
        df_clean = preprocessor.clean_data(sample_iris_data)

        assert len(df_clean) <= len(sample_iris_data)
        assert not df_clean.isnull().any().any()

    def test_encode_target(self, sample_iris_data):
        """Test target encoding."""
        preprocessor = DataPreprocessor()
        df_encoded, mapping = preprocessor.encode_target(sample_iris_data)

        assert "species_encoded" in df_encoded.columns
        assert len(mapping) == 3
        assert all(isinstance(k, int) for k in mapping.keys())

    def test_split_data(self, sample_iris_data):
        """Test data splitting."""
        preprocessor = DataPreprocessor(test_size=0.2, validation_size=0.1)
        df_encoded, _ = preprocessor.encode_target(sample_iris_data)
        splits = preprocessor.split_data(df_encoded)

        assert "train" in splits
        assert "val" in splits
        assert "test" in splits

        total = len(splits["train"][0]) + len(splits["val"][0]) + len(splits["test"][0])
        assert total == len(sample_iris_data)

    def test_process_pipeline(self, sample_iris_data):
        """Test complete preprocessing pipeline."""
        preprocessor = DataPreprocessor()
        splits = preprocessor.process(sample_iris_data)

        assert len(splits["train"][0]) > 0
        assert len(splits["test"][0]) > 0


# ============================================================================
# FEATURE ENGINEER TESTS
# ============================================================================


class TestFeatureEngineer:
    """Tests for FeatureEngineer class."""

    def test_scaling(self, sample_iris_data):
        """Test feature scaling."""
        preprocessor = DataPreprocessor()
        splits = preprocessor.process(sample_iris_data)
        X_train, y_train = splits["train"]

        fe = FeatureEngineer(create_polynomial=False, feature_selection=False)
        X_scaled = fe.fit_transform(X_train, y_train)

        # Check scaling (standard scaler should have ~0 mean, ~1 std)
        assert X_scaled.mean().abs().max() < 0.5
        assert X_scaled.std().max() < 2.0

    def test_polynomial_features(self, sample_iris_data):
        """Test polynomial feature creation."""
        preprocessor = DataPreprocessor()
        splits = preprocessor.process(sample_iris_data)
        X_train, y_train = splits["train"]

        fe = FeatureEngineer(
            create_polynomial=True, polynomial_degree=2, feature_selection=False
        )
        X_poly = fe.fit_transform(X_train, y_train)

        # Polynomial degree 2 creates more features
        assert X_poly.shape[1] > X_train.shape[1]

    def test_feature_selection(self, sample_iris_data):
        """Test feature selection."""
        preprocessor = DataPreprocessor()
        splits = preprocessor.process(sample_iris_data)
        X_train, y_train = splits["train"]

        n_select = 3
        fe = FeatureEngineer(
            create_polynomial=False,
            feature_selection=True,
            n_features_to_select=n_select,
        )
        X_selected = fe.fit_transform(X_train, y_train)

        assert X_selected.shape[1] == n_select

    def test_transform_consistency(self, sample_iris_data):
        """Test transform gives consistent results."""
        preprocessor = DataPreprocessor()
        splits = preprocessor.process(sample_iris_data)
        X_train, y_train = splits["train"]
        X_test, _ = splits["test"]

        fe = FeatureEngineer(feature_selection=False)
        fe.fit(X_train, y_train)

        X_train_t = fe.transform(X_train)
        X_test_t = fe.transform(X_test)

        assert X_train_t.shape[1] == X_test_t.shape[1]


# ============================================================================
# MODEL TRAINER TESTS
# ============================================================================


class TestModelTrainer:
    """Tests for ModelTrainer class."""

    @pytest.fixture
    def training_data(self, sample_iris_data):
        """Prepare training data."""
        preprocessor = DataPreprocessor()
        splits = preprocessor.process(sample_iris_data)

        fe = FeatureEngineer(feature_selection=False)
        X_train, y_train = splits["train"]
        X_test, y_test = splits["test"]

        X_train_fe = fe.fit_transform(X_train, y_train)
        X_test_fe = fe.transform(X_test)

        return X_train_fe, y_train, X_test_fe, y_test

    def test_random_forest_training(self, training_data):
        """Test Random Forest training."""
        X_train, y_train, X_test, y_test = training_data

        trainer = ModelTrainer(
            model_type="random_forest",
            model_params={"n_estimators": 10, "random_state": 42},
            cross_validation=True,
            cv_folds=3,
        )
        trainer.train(X_train, y_train)

        assert trainer.model is not None
        assert len(trainer.cv_scores) == 3

    def test_logistic_regression_training(self, training_data):
        """Test Logistic Regression training."""
        X_train, y_train, _, _ = training_data

        trainer = ModelTrainer(
            model_type="logistic_regression",
            model_params={"max_iter": 1000, "random_state": 42},
        )
        trainer.train(X_train, y_train)

        assert trainer.model is not None

    def test_predict(self, training_data):
        """Test model prediction."""
        X_train, y_train, X_test, y_test = training_data

        trainer = ModelTrainer(model_type="random_forest")
        trainer.train(X_train, y_train)
        predictions = trainer.predict(X_test)

        assert len(predictions) == len(X_test)

    def test_save_load_model(self, training_data, temp_dir):
        """Test model save and load."""
        X_train, y_train, X_test, _ = training_data

        trainer = ModelTrainer(model_type="random_forest")
        trainer.train(X_train, y_train)

        model_path = f"{temp_dir}/model.joblib"
        trainer.save_model(model_path)

        # Load in new trainer
        trainer2 = ModelTrainer(model_type="random_forest")
        trainer2.load_model(model_path)

        # Predictions should be identical
        pred1 = trainer.predict(X_test)
        pred2 = trainer2.predict(X_test)
        np.testing.assert_array_equal(pred1, pred2)


# ============================================================================
# EVALUATOR TESTS
# ============================================================================


class TestModelEvaluator:
    """Tests for ModelEvaluator class."""

    def test_evaluate_metrics(self):
        """Test metric calculation."""
        y_true = np.array([0, 0, 1, 1, 2, 2])
        y_pred = np.array([0, 0, 1, 1, 2, 1])

        evaluator = ModelEvaluator(class_names=["a", "b", "c"])
        metrics = evaluator.evaluate(y_true, y_pred)

        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1_score" in metrics
        assert metrics["accuracy"] == pytest.approx(5 / 6, rel=0.01)

    def test_confusion_matrix_plot(self, temp_dir):
        """Test confusion matrix plot generation."""
        y_true = np.array([0, 0, 1, 1, 2, 2])
        y_pred = np.array([0, 0, 1, 1, 2, 1])

        evaluator = ModelEvaluator(
            save_plots=True, plots_path=temp_dir, class_names=["a", "b", "c"]
        )

        plot_path = evaluator.plot_confusion_matrix(y_true, y_pred)
        assert Path(plot_path).exists()


# ============================================================================
# MODEL REGISTRY TESTS
# ============================================================================


class TestModelRegistry:
    """Tests for ModelRegistry class."""

    @pytest.fixture
    def trained_model(self, sample_iris_data):
        """Create a trained model."""
        from sklearn.ensemble import RandomForestClassifier

        preprocessor = DataPreprocessor()
        splits = preprocessor.process(sample_iris_data)
        X_train, y_train = splits["train"]

        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)
        return model

    def test_register_model(self, trained_model, temp_dir):
        """Test model registration."""
        registry = ModelRegistry(registry_path=temp_dir)

        version = registry.register_model(
            model=trained_model,
            model_name="test_model",
            metrics={"accuracy": 0.95, "f1_score": 0.94},
        )

        assert version == "1.0.0"

    def test_multiple_versions(self, trained_model, temp_dir):
        """Test registering multiple versions."""
        registry = ModelRegistry(registry_path=temp_dir)

        v1 = registry.register_model(trained_model, "test_model", {"accuracy": 0.90})
        v2 = registry.register_model(trained_model, "test_model", {"accuracy": 0.92})
        v3 = registry.register_model(trained_model, "test_model", {"accuracy": 0.95})

        assert v1 == "1.0.0"
        assert v2 == "1.0.1"
        assert v3 == "1.0.2"

    def test_get_model(self, trained_model, temp_dir):
        """Test retrieving a model."""
        registry = ModelRegistry(registry_path=temp_dir)

        registry.register_model(trained_model, "test_model", {"accuracy": 0.95})

        loaded_model, metadata = registry.get_model("test_model", "1.0.0")

        assert loaded_model is not None
        assert metadata["version"] == "1.0.0"

    def test_promote_to_production(self, trained_model, temp_dir):
        """Test promoting model to production."""
        registry = ModelRegistry(
            registry_path=temp_dir, promotion_threshold={"accuracy": 0.90}
        )

        registry.register_model(trained_model, "test_model", {"accuracy": 0.95})

        success = registry.promote_to_production("test_model", "1.0.0", force=False)

        assert success
        model, _ = registry.get_production_model()
        assert model is not None


# ============================================================================
# INTEGRATION TESTS
# ============================================================================


class TestIntegration:
    """Integration tests for the full pipeline."""

    def test_full_pipeline_flow(self, temp_dir):
        """Test the complete pipeline flow with real Iris data."""
        # 1. Load data - use real Iris dataset for meaningful test
        loader = DataLoader()
        real_iris_data = loader.load_from_sklearn()

        # 2. Preprocess
        preprocessor = DataPreprocessor(test_size=0.2, validation_size=0.1)
        splits = preprocessor.process(real_iris_data)

        # 3. Feature engineering
        X_train, y_train = splits["train"]
        X_test, y_test = splits["test"]

        fe = FeatureEngineer(
            create_polynomial=True,
            polynomial_degree=2,
            feature_selection=True,
            n_features_to_select=8,
        )
        X_train_fe = fe.fit_transform(X_train, y_train)
        X_test_fe = fe.transform(X_test)

        # 4. Train model
        trainer = ModelTrainer(
            model_type="random_forest",
            model_params={"n_estimators": 50, "random_state": 42},
        )
        trainer.train(X_train_fe, y_train)

        # 5. Evaluate
        y_pred = trainer.predict(X_test_fe)

        evaluator = ModelEvaluator(
            plots_path=temp_dir, class_names=["setosa", "versicolor", "virginica"]
        )
        metrics = evaluator.evaluate(y_test.values, y_pred)

        # 6. Register
        registry = ModelRegistry(registry_path=f"{temp_dir}/registry")
        version = registry.register_model(trainer.model, "iris_classifier", metrics)

        # Assertions
        assert metrics["accuracy"] > 0.5  # Should be better than random
        assert version == "1.0.0"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
