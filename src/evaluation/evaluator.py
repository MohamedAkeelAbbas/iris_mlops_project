"""
Model Evaluator Module
======================
Handles model evaluation with comprehensive metrics and visualizations.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from loguru import logger
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


class ModelEvaluator:
    """
    Comprehensive model evaluation with metrics and visualizations.
    """

    def __init__(
        self,
        metrics: List[str] = None,
        save_plots: bool = True,
        plots_path: str = "experiments/plots",
        class_names: List[str] = None,
    ):
        """
        Initialize ModelEvaluator.

        Args:
            metrics: List of metrics to calculate
            save_plots: Whether to save plots
            plots_path: Path to save plots
            class_names: Names of target classes
        """
        self.metrics = metrics or [
            "accuracy",
            "precision",
            "recall",
            "f1_score",
            "confusion_matrix",
            "classification_report",
        ]
        self.save_plots = save_plots
        self.plots_path = plots_path
        self.class_names = class_names
        self.results: Dict[str, Any] = {}

    def evaluate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate model predictions.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Prediction probabilities (optional)

        Returns:
            Dictionary of evaluation metrics
        """
        logger.info("Evaluating model performance...")

        results = {}

        # Accuracy
        if "accuracy" in self.metrics:
            results["accuracy"] = accuracy_score(y_true, y_pred)
            logger.info(f"Accuracy: {results['accuracy']:.4f}")

        # Precision (weighted for multiclass)
        if "precision" in self.metrics:
            results["precision"] = precision_score(
                y_true, y_pred, average="weighted", zero_division=0
            )
            logger.info(f"Precision: {results['precision']:.4f}")

        # Recall (weighted for multiclass)
        if "recall" in self.metrics:
            results["recall"] = recall_score(
                y_true, y_pred, average="weighted", zero_division=0
            )
            logger.info(f"Recall: {results['recall']:.4f}")

        # F1 Score (weighted for multiclass)
        if "f1_score" in self.metrics:
            results["f1_score"] = f1_score(
                y_true, y_pred, average="weighted", zero_division=0
            )
            logger.info(f"F1 Score: {results['f1_score']:.4f}")

        # Confusion Matrix
        if "confusion_matrix" in self.metrics:
            cm = confusion_matrix(y_true, y_pred)
            results["confusion_matrix"] = cm.tolist()

        # Classification Report
        if "classification_report" in self.metrics:
            report = classification_report(
                y_true,
                y_pred,
                target_names=self.class_names,
                output_dict=True,
                zero_division=0,
            )
            results["classification_report"] = report

        # Per-class metrics
        results["per_class"] = {
            "precision": precision_score(
                y_true, y_pred, average=None, zero_division=0
            ).tolist(),
            "recall": recall_score(
                y_true, y_pred, average=None, zero_division=0
            ).tolist(),
            "f1": f1_score(y_true, y_pred, average=None, zero_division=0).tolist(),
        }

        # ROC AUC (if probabilities available)
        if y_proba is not None:
            try:
                if len(np.unique(y_true)) == 2:
                    results["roc_auc"] = roc_auc_score(y_true, y_proba[:, 1])
                else:
                    results["roc_auc"] = roc_auc_score(
                        y_true, y_proba, multi_class="ovr", average="weighted"
                    )
                logger.info(f"ROC AUC: {results['roc_auc']:.4f}")
            except Exception as e:
                logger.warning(f"Could not calculate ROC AUC: {e}")

        self.results = results
        return results

    def plot_confusion_matrix(
        self, y_true: np.ndarray, y_pred: np.ndarray, save_path: Optional[str] = None
    ) -> str:
        """
        Plot confusion matrix.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            save_path: Path to save the plot

        Returns:
            Path to saved plot
        """
        cm = confusion_matrix(y_true, y_pred)

        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=self.class_names or range(cm.shape[1]),
            yticklabels=self.class_names or range(cm.shape[0]),
        )
        plt.title("Confusion Matrix")
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.tight_layout()

        if save_path is None:
            save_path = f"{self.plots_path}/confusion_matrix.png"

        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()

        logger.info(f"Confusion matrix plot saved to {save_path}")
        return save_path

    def plot_feature_importance(
        self,
        feature_importance: Dict[str, float],
        top_n: int = 15,
        save_path: Optional[str] = None,
    ) -> str:
        """
        Plot feature importance.

        Args:
            feature_importance: Dictionary of feature names to importance scores
            top_n: Number of top features to show
            save_path: Path to save the plot

        Returns:
            Path to saved plot
        """
        # Sort and get top features
        sorted_features = sorted(
            feature_importance.items(), key=lambda x: x[1], reverse=True
        )[:top_n]

        features, importances = zip(*sorted_features)

        plt.figure(figsize=(12, 8))
        colors = plt.cm.viridis(np.linspace(0, 0.8, len(features)))
        plt.barh(range(len(features)), importances, color=colors)
        plt.yticks(range(len(features)), features)
        plt.xlabel("Importance Score")
        plt.ylabel("Features")
        plt.title(f"Top {top_n} Feature Importances")
        plt.gca().invert_yaxis()
        plt.tight_layout()

        if save_path is None:
            save_path = f"{self.plots_path}/feature_importance.png"

        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()

        logger.info(f"Feature importance plot saved to {save_path}")
        return save_path

    def plot_metrics_comparison(
        self, metrics: Dict[str, float], save_path: Optional[str] = None
    ) -> str:
        """
        Plot metrics comparison bar chart.

        Args:
            metrics: Dictionary of metric names to values
            save_path: Path to save the plot

        Returns:
            Path to saved plot
        """
        # Filter numeric metrics
        numeric_metrics = {
            k: v for k, v in metrics.items() if isinstance(v, (int, float))
        }

        plt.figure(figsize=(10, 6))
        colors = ["#2ecc71", "#3498db", "#9b59b6", "#e74c3c", "#f39c12"]
        bars = plt.bar(
            numeric_metrics.keys(),
            numeric_metrics.values(),
            color=colors[: len(numeric_metrics)],
        )

        # Add value labels on bars
        for bar, val in zip(bars, numeric_metrics.values()):
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=10,
            )

        plt.ylim(0, 1.1)
        plt.ylabel("Score")
        plt.title("Model Performance Metrics")
        plt.tight_layout()

        if save_path is None:
            save_path = f"{self.plots_path}/metrics_comparison.png"

        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()

        logger.info(f"Metrics comparison plot saved to {save_path}")
        return save_path

    def generate_report(self, output_path: Optional[str] = None) -> str:
        """
        Generate a comprehensive evaluation report.

        Args:
            output_path: Path to save the report

        Returns:
            Path to saved report
        """
        if not self.results:
            raise ValueError("No evaluation results. Run evaluate() first.")

        report = {
            "summary": {
                "accuracy": self.results.get("accuracy"),
                "precision": self.results.get("precision"),
                "recall": self.results.get("recall"),
                "f1_score": self.results.get("f1_score"),
                "roc_auc": self.results.get("roc_auc"),
            },
            "per_class_metrics": self.results.get("per_class"),
            "classification_report": self.results.get("classification_report"),
            "confusion_matrix": self.results.get("confusion_matrix"),
        }

        if output_path is None:
            output_path = f"{self.plots_path}/evaluation_report.json"

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)

        logger.info(f"Evaluation report saved to {output_path}")
        return output_path

    def print_summary(self):
        """Print evaluation summary to console."""
        if not self.results:
            logger.warning("No evaluation results. Run evaluate() first.")
            return

        print("\n" + "=" * 60)
        print("MODEL EVALUATION SUMMARY")
        print("=" * 60)
        print(f"  Accuracy:  {self.results.get('accuracy', 'N/A'):.4f}")
        print(f"  Precision: {self.results.get('precision', 'N/A'):.4f}")
        print(f"  Recall:    {self.results.get('recall', 'N/A'):.4f}")
        print(f"  F1 Score:  {self.results.get('f1_score', 'N/A'):.4f}")
        if "roc_auc" in self.results:
            print(f"  ROC AUC:   {self.results['roc_auc']:.4f}")
        print("=" * 60 + "\n")

        if "classification_report" in self.results:
            print("\nClassification Report:")
            print("-" * 60)
            report = self.results["classification_report"]
            for cls, metrics in report.items():
                if isinstance(metrics, dict):
                    print(f"  {cls}:")
                    for metric, value in metrics.items():
                        if isinstance(value, float):
                            print(f"    {metric}: {value:.4f}")
