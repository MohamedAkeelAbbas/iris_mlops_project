#!/usr/bin/env python3
"""
Iris MLOps Pipeline - Main Entry Point
======================================

Run the complete ML pipeline with:
    python run_pipeline.py

Run with custom config:
    python run_pipeline.py --config configs/config_dev.yaml

Show help:
    python run_pipeline.py --help
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

# Change to project directory
os.chdir(project_root)


def main():
    """Main entry point for the pipeline."""
    import argparse
    from src.pipeline import MLPipeline

    parser = argparse.ArgumentParser(
        description="Iris MLOps Pipeline - Complete ML workflow",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_pipeline.py                    # Run with default config
  python run_pipeline.py --config custom    # Run with custom config

Pipeline Steps:
  1. Data Loading       - Load Iris dataset
  2. Preprocessing      - Clean and split data
  2a. visualition       - optional notebook has been added
  3. Feature Engineering- Scale and select features
  4. Model Training     - Train classifier with CV
  5. Evaluation         - Generate metrics and plots
  6. Experiment Tracking - Log to MLflow
  7. Model Registry     - Version and register model
        """
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to configuration file (default: configs/config.yaml)"
    )

    parser.add_argument(
        "--model",
        type=str,
        choices=["random_forest", "logistic_regression", "svm", "gradient_boosting"],
        help="Override model type from config"
    )

    parser.add_argument(
        "--no-tracking",
        action="store_true",
        help="Disable MLflow experiment tracking"
    )

    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("    IRIS MLOPS PIPELINE")
    print("    Complete ML Workflow Demo")
    print("=" * 60 + "\n")

    try:
        # Initialize pipeline
        pipeline = MLPipeline(config_path=args.config)

        # Override model type if specified
        if args.model:
            pipeline.config.model.type = args.model
            print(f"Model type overridden to: {args.model}")

        # Run pipeline
        results = pipeline.run()

        print("\n" + "=" * 60)
        print("    PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 60)

        # Show key results
        print(f"\nKey Results:")
        print(f"  - Accuracy: {results['evaluation']['metrics']['accuracy']:.4f}")
        print(f"  - F1 Score: {results['evaluation']['metrics']['f1_score']:.4f}")
        print(f"  - Model Version: {results['registration']['version']}")

        print(f"\nArtifacts saved to:")
        print(f"  - Models: models/trained/")
        print(f"  - Plots: experiments/plots/")
        print(f"  - Logs: logs/")
        print(f"  - MLflow: experiments/mlruns/")

        return 0

    except Exception as e:
        print(f"\n ERROR: Pipeline failed!")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
