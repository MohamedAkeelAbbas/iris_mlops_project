# Iris MLOps Project

[![CI Pipeline](https://github.com/VenkateswarluPudur/iris_mlops_project/actions/workflows/ci.yml/badge.svg)](https://github.com/VenkateswarluPudur/iris_mlops_project/actions/workflows/ci.yml)
[![CD Pipeline](https://github.com/VenkateswarluPudur/iris_mlops_project/actions/workflows/cd.yml/badge.svg)](https://github.com/VenkateswarluPudur/iris_mlops_project/actions/workflows/cd.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A complete end-to-end MLOps pipeline demonstrating industry-standard practices for machine learning workflows using the classic Iris dataset.

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Pipeline Architecture](#pipeline-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Pipeline Steps](#pipeline-steps)
- [Experiment Tracking](#experiment-tracking)
- [Model Registry](#model-registry)
- [Testing](#testing)
- [CI/CD Pipeline](#cicd-pipeline)
- [Docker](#docker)

---

## Overview

This project implements a production-ready ML pipeline with:

- **Data Loading & Validation** - Load from sklearn or CSV with automatic validation
- **Data Preprocessing** - Cleaning, encoding, and train/val/test splitting
- **Feature Engineering** - Scaling, polynomial features, and feature selection
- **Model Training** - Support for multiple algorithms with cross-validation
- **Model Evaluation** - Comprehensive metrics and visualization
- **Experiment Tracking** - MLflow integration for reproducibility
- **Model Registry** - Version control and promotion workflow

---

## Project Structure

```
iris_mlops_project/
│
├── configs/                          # Configuration files
│   └── config.yaml                   # Main pipeline configuration
│
├── src/                              # Source code
│   ├── __init__.py
│   │
│   ├── data/                         # Data handling modules
│   │   ├── __init__.py
│   │   ├── data_loader.py            # Load data from sklearn/CSV
│   │   └── preprocessor.py           # Clean, encode, split data
│   │
│   ├── features/                     # Feature engineering
│   │   ├── __init__.py
│   │   └── feature_engineer.py       # Scaling, polynomial, selection
│   │
│   ├── models/                       # Model training & management
│   │   ├── __init__.py
│   │   ├── trainer.py                # Train ML models
│   │   └── registry.py               # Model versioning & registry
│   │
│   ├── evaluation/                   # Model evaluation
│   │   ├── __init__.py
│   │   ├── evaluator.py              # Metrics & visualizations
│   │   └── experiment_tracker.py     # MLflow tracking
│   │
│   ├── utils/                        # Utilities
│   │   ├── __init__.py
│   │   ├── config.py                 # Configuration management
│   │   └── logger.py                 # Logging setup
│   │
│   └── pipeline.py                   # Main pipeline orchestrator
│
├── tests/                            # Unit tests
│   ├── __init__.py
│   └── test_pipeline.py              # Comprehensive test suite
│
├── data/                             # Data storage
│   ├── raw/                          # Raw data files
│   ├── processed/                    # Processed data & artifacts
│   └── features/                     # Feature engineering artifacts
│
├── models/                           # Model storage
│   ├── trained/                      # Trained model files
│   └── registry/                     # Model registry with versions
│
├── experiments/                      # Experiment artifacts
│   ├── plots/                        # Generated visualizations
│   └── mlruns/                       # MLflow tracking data
│
├── logs/                             # Pipeline logs
│
├── notebooks/                        # Jupyter notebooks (optional)
│
├── run_pipeline.py                   # Main entry point
├── requirements.txt                  # Python dependencies
└── README.md                         # This file
```

---

## Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           IRIS MLOPS PIPELINE                                │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STEP 1: DATA LOADING                                                        │
│  ├── Load from sklearn or CSV                                                │
│  ├── Validate data quality                                                   │
│  └── Save raw data                                                           │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STEP 2: DATA PREPROCESSING                                                  │
│  ├── Clean data (remove duplicates, handle missing)                          │
│  ├── Encode target labels                                                    │
│  └── Split into train/validation/test sets                                   │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STEP 3: FEATURE ENGINEERING                                                 │
│  ├── Scale features (Standard/MinMax/Robust)                                 │
│  ├── Create polynomial features                                              │
│  └── Select top K features                                                   │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STEP 4: MODEL TRAINING                                                      │
│  ├── Train selected algorithm                                                │
│  ├── Perform cross-validation                                                │
│  └── Save trained model                                                      │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STEP 5: MODEL EVALUATION                                                    │
│  ├── Calculate metrics (accuracy, precision, recall, F1, ROC-AUC)            │
│  ├── Generate plots (confusion matrix, feature importance)                   │
│  └── Create evaluation report                                                │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STEP 6: EXPERIMENT TRACKING                                                 │
│  ├── Log parameters to MLflow                                                │
│  ├── Log metrics to MLflow                                                   │
│  └── Log model artifacts                                                     │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STEP 7: MODEL REGISTRATION                                                  │
│  ├── Register model with version                                             │
│  ├── Check promotion thresholds                                              │
│  └── Promote to staging/production                                           │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Installation

### 1. Clone/Navigate to the project

```bash
cd /path/to/iris_mlops_project
```

### 2. Create virtual environment

```bash
python3 -m venv venv
```

### 3. Activate virtual environment

```bash
# On macOS/Linux
source venv/bin/activate

# On Windows
venv\Scripts\activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Usage

### Run the Complete Pipeline

```bash
# Run with default configuration
python run_pipeline.py

# Run with custom config file
python run_pipeline.py --config configs/config.yaml
```

### Run with Different Models

```bash
# Random Forest (default)
python run_pipeline.py --model random_forest

# Logistic Regression
python run_pipeline.py --model logistic_regression

# Support Vector Machine
python run_pipeline.py --model svm

# Gradient Boosting
python run_pipeline.py --model gradient_boosting
```

### View Help

```bash
python run_pipeline.py --help
```

### Run Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=src

# Run specific test class
pytest tests/test_pipeline.py::TestDataLoader -v
```

### Launch MLflow UI

```bash
# View experiment tracking dashboard
mlflow ui --backend-store-uri experiments/mlruns

# Open browser at http://localhost:5000
```

---

## Configuration

All pipeline settings are controlled via `configs/config.yaml`:

### Key Configuration Sections

```yaml
# Data Configuration
data:
  test_size: 0.2              # 20% for testing
  validation_size: 0.1        # 10% for validation
  random_state: 42            # Reproducibility seed

# Feature Engineering
features:
  scaling_method: "standard"  # standard, minmax, robust
  create_polynomial: true     # Create polynomial features
  polynomial_degree: 2        # Degree of polynomials
  n_features_to_select: 10    # Number of features to keep

# Model Selection
model:
  type: "random_forest"       # Model algorithm
  random_forest:
    n_estimators: 100
    max_depth: 10

# Training
training:
  cross_validation: true
  cv_folds: 5

# Model Registry Thresholds
registry:
  promotion_threshold:
    accuracy: 0.90
    f1_score: 0.88
```

---

## Pipeline Steps

### Step 1: Data Loading (`src/data/data_loader.py`)

```python
from src.data.data_loader import DataLoader

loader = DataLoader()
df = loader.load()  # Loads from sklearn
info = loader.get_data_info()
is_valid, issues = loader.validate_data()
```

### Step 2: Data Preprocessing (`src/data/preprocessor.py`)

```python
from src.data.preprocessor import DataPreprocessor

preprocessor = DataPreprocessor(
    target_column="species",
    test_size=0.2,
    validation_size=0.1
)
splits = preprocessor.process(df)
# Returns: {'train': (X, y), 'val': (X, y), 'test': (X, y)}
```

### Step 3: Feature Engineering (`src/features/feature_engineer.py`)

```python
from src.features.feature_engineer import FeatureEngineer

fe = FeatureEngineer(
    scaling_method="standard",
    create_polynomial=True,
    polynomial_degree=2,
    feature_selection=True,
    n_features_to_select=10
)
X_train_fe = fe.fit_transform(X_train, y_train)
X_test_fe = fe.transform(X_test)
```

### Step 4: Model Training (`src/models/trainer.py`)

```python
from src.models.trainer import ModelTrainer

trainer = ModelTrainer(
    model_type='random_forest',
    model_params={'n_estimators': 100},
    cross_validation=True,
    cv_folds=5
)
trainer.train(X_train, y_train)
predictions = trainer.predict(X_test)
```

### Step 5: Model Evaluation (`src/evaluation/evaluator.py`)

```python
from src.evaluation.evaluator import ModelEvaluator

evaluator = ModelEvaluator(
    metrics=['accuracy', 'precision', 'recall', 'f1_score'],
    save_plots=True
)
metrics = evaluator.evaluate(y_true, y_pred, y_proba)
evaluator.plot_confusion_matrix(y_true, y_pred)
evaluator.print_summary()
```

### Step 6: Experiment Tracking (`src/evaluation/experiment_tracker.py`)

```python
from src.evaluation.experiment_tracker import ExperimentTracker

tracker = ExperimentTracker(
    experiment_name="iris_classification"
)
tracker.start_run(run_name="experiment_1")
tracker.log_params({"model": "random_forest"})
tracker.log_metrics({"accuracy": 0.95})
tracker.log_model(model, "model")
tracker.end_run()
```

### Step 7: Model Registry (`src/models/registry.py`)

```python
from src.models.registry import ModelRegistry

registry = ModelRegistry(
    registry_path="models/registry",
    promotion_threshold={'accuracy': 0.90}
)
version = registry.register_model(model, "iris_rf", metrics)
registry.promote_to_staging("iris_rf", version)
registry.promote_to_production("iris_rf", version)
model, metadata = registry.get_production_model()
```

---

## Experiment Tracking

MLflow is used for experiment tracking. After running the pipeline:

```bash
# Launch MLflow UI
mlflow ui --backend-store-uri experiments/mlruns
```

### What Gets Tracked

| Category | Items |
|----------|-------|
| **Parameters** | model_type, n_estimators, max_depth, test_size, cv_folds |
| **Metrics** | accuracy, precision, recall, f1_score, roc_auc |
| **Artifacts** | trained model, evaluation report, plots |

---

## Model Registry

The local model registry manages model versions and promotions:

### Registry Structure

```
models/registry/
├── registry_metadata.json       # Central registry index
└── iris_random_forest/
    └── 1.0.0/
        ├── model.joblib         # Serialized model
        └── metadata.json        # Model metadata
```

### Model Stages

| Stage | Description |
|-------|-------------|
| `none` | Newly registered, not promoted |
| `staging` | Ready for testing |
| `production` | Live model |
| `archived` | Previous production model |

---

## Testing

The test suite covers all pipeline components:

```bash
# Run all tests
pytest tests/ -v

# Expected output: 22 passed
```

### Test Coverage

| Module | Tests |
|--------|-------|
| DataLoader | 4 tests |
| DataPreprocessor | 4 tests |
| FeatureEngineer | 4 tests |
| ModelTrainer | 4 tests |
| ModelEvaluator | 2 tests |
| ModelRegistry | 4 tests |
| Integration | 1 test |

---


### 1. **Modular Design**
Each component (data, features, models, evaluation) is independent and testable.

### 2. **Configuration-Driven**
All parameters are externalized in YAML, making experiments reproducible.

### 3. **Artifact Management**
Every pipeline run generates versioned artifacts (models, plots, logs).

### 4. **Experiment Tracking**
MLflow logs all experiments for comparison and reproducibility.

### 5. **Model Versioning**
The registry maintains model versions with promotion workflow.

### 6. **Testing**
Comprehensive unit tests ensure code quality.

### 7. **Logging**
Structured logging with Loguru for debugging and monitoring.

---

## Output Examples

### Pipeline Run Output

```
============================================================
PIPELINE EXECUTION SUMMARY
============================================================

Run ID: 20260121_202724
Model Type: random_forest

Data:
  - Training samples: 104
  - Test samples: 30
  - Features: 10

Metrics:
  - accuracy: 0.9333
  - precision: 0.9333
  - recall: 0.9333
  - f1_score: 0.9333
  - roc_auc: 0.9900

Model Version: 1.0.0
Meets Production Threshold: True
============================================================
```

### Generated Artifacts

| Artifact | Location |
|----------|----------|
| Trained Model | `models/trained/model_*.joblib` |
| Confusion Matrix | `experiments/plots/*/confusion_matrix.png` |
| Feature Importance | `experiments/plots/*/feature_importance.png` |
| Metrics Plot | `experiments/plots/*/metrics_comparison.png` |
| Evaluation Report | `experiments/plots/*/evaluation_report.json` |
| Pipeline Log | `logs/pipeline_*.log` |

---

## Supported Models

| Model | Config Key | Description |
|-------|------------|-------------|
| Random Forest | `random_forest` | Ensemble of decision trees |
| Logistic Regression | `logistic_regression` | Linear classifier |
| SVM | `svm` | Support Vector Machine |
| Gradient Boosting | `gradient_boosting` | Boosted decision trees |

---

## Dependencies

| Package | Purpose |
|---------|---------|
| pandas | Data manipulation |
| numpy | Numerical computing |
| scikit-learn | ML algorithms |
| mlflow | Experiment tracking |
| pydantic | Configuration validation |
| pyyaml | YAML parsing |
| matplotlib | Plotting |
| seaborn | Statistical visualization |
| loguru | Logging |
| pytest | Testing |

---

## CI/CD Pipeline

This project uses **GitHub Actions** for continuous integration and deployment.

### CI Pipeline (`.github/workflows/ci.yml`)

Triggered on every push to `main`/`develop` and pull requests:

| Job | Description |
|-----|-------------|
| **Lint** | Code formatting (Black), import sorting (isort), linting (flake8) |
| **Test** | Run pytest across Python 3.10, 3.11, 3.12 with coverage |
| **Pipeline Validation** | Smoke test - runs ML pipeline end-to-end |
| **Build Docker** | Builds Docker image on main branch |

### CD Pipeline (`.github/workflows/cd.yml`)

Triggered on releases and manual dispatch:

| Job | Description |
|-----|-------------|
| **Build & Push** | Build and push Docker image to GitHub Container Registry |
| **Test Image** | Run tests inside the built container |
| **Deploy** | (Placeholder) Deploy to staging/production |

### Running CI Locally

```bash
# Lint checks
pip install black isort flake8
black --check src/ tests/
isort --check-only src/ tests/
flake8 src/ tests/

# Run tests with coverage
pytest tests/ -v --cov=src --cov-report=term-missing
```

---

## Docker

### Build and Run

```bash
# Build the image
docker build -t iris-mlops .

# Run the ML pipeline
docker run --rm iris-mlops

# Run with specific model
docker run --rm iris-mlops python run_pipeline.py --model svm

# Run tests
docker run --rm iris-mlops pytest tests/ -v
```

### Docker Compose

```bash
# Start MLflow server and run pipeline
docker-compose up

# Run only tests
docker-compose --profile testing up test

# Start development environment
docker-compose --profile development up dev

# Access MLflow UI at http://localhost:5000
```

### Services

| Service | Description | Port |
|---------|-------------|------|
| `ml-pipeline` | Main ML pipeline execution | - |
| `mlflow` | MLflow tracking server | 5000 |
| `test` | Run test suite | - |
| `dev` | Development container | - |

---




