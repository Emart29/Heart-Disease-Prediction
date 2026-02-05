"""Heart disease model training with MLflow experiment tracking.

This module provides functions for training the heart disease prediction
model with comprehensive MLflow logging of hyperparameters, metrics,
and model artifacts.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.config import settings
from src.features.engineering import (
    create_risk_features,
    encode_categorical,
    get_feature_columns,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_and_prepare_data(
    data_path: Optional[Path] = None,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Load and prepare the heart disease dataset.

    Args:
        data_path: Path to the CSV data file. If None, uses settings.data_path.

    Returns:
        Tuple of (features DataFrame, target Series).

    Raises:
        FileNotFoundError: If the data file doesn't exist.
    """
    path = data_path or settings.data_path
    logger.info(f"Loading data from {path}")

    df = pd.read_csv(path)

    # Separate target
    target = df["target"]
    features = df.drop("target", axis=1)

    # Apply feature engineering
    features = create_risk_features(features)
    features = encode_categorical(features)

    # Ensure all expected columns exist
    expected_cols = get_feature_columns()
    for col in expected_cols:
        if col not in features.columns:
            features[col] = 0

    # Select only expected columns in correct order
    features = features[expected_cols]

    logger.info(f"Loaded {len(features)} samples with {len(expected_cols)} features")

    return features, target


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
) -> Dict[str, float]:
    """Compute classification metrics.

    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        y_prob: Predicted probabilities for positive class.

    Returns:
        Dictionary of metric names to values.
    """
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_prob),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }


def train_model(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    hyperparameters: Optional[Dict[str, Any]] = None,
) -> Tuple[RandomForestClassifier, StandardScaler]:
    """Train the heart disease prediction model.

    Args:
        X_train: Training features as DataFrame with feature names.
        y_train: Training labels.
        hyperparameters: Model hyperparameters. If None, uses defaults.

    Returns:
        Tuple of (trained model, fitted scaler).
    """
    # Default hyperparameters
    default_params = {
        "n_estimators": 100,
        "max_depth": 10,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "random_state": settings.random_state,
    }

    params = {**default_params, **(hyperparameters or {})}

    # Scale features - fit on DataFrame to preserve feature names
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Train model
    model = RandomForestClassifier(**params)
    model.fit(X_train_scaled, y_train)

    return model, scaler


def save_model_artifacts(
    model: RandomForestClassifier,
    scaler: StandardScaler,
    feature_names: list,
    output_path: Optional[Path] = None,
) -> Path:
    """Save model artifacts to disk.

    Args:
        model: Trained model.
        scaler: Fitted scaler.
        feature_names: List of feature column names.
        output_path: Path to save artifacts. If None, uses settings.model_path.

    Returns:
        Path where artifacts were saved.
    """
    path = output_path or settings.model_path

    artifacts = {
        "model": model,
        "scaler": scaler,
        "feature_names": feature_names,
    }

    # Ensure directory exists
    path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(artifacts, path)
    logger.info(f"Model artifacts saved to {path}")

    return path


def run_training_experiment(
    experiment_name: str = "heart_disease_prediction",
    run_name: Optional[str] = None,
    hyperparameters: Optional[Dict[str, Any]] = None,
    data_path: Optional[Path] = None,
    model_output_path: Optional[Path] = None,
    register_model: bool = True,
    model_registry_name: str = "heart_disease_model",
) -> Dict[str, Any]:
    """Run a complete training experiment with MLflow tracking.

    This function:
    1. Loads and prepares the data
    2. Splits into train/test sets
    3. Trains the model with specified hyperparameters
    4. Logs all hyperparameters and metrics to MLflow
    5. Saves and registers model artifacts

    Args:
        experiment_name: Name of the MLflow experiment.
        run_name: Name for this specific run. If None, auto-generated.
        hyperparameters: Model hyperparameters to use.
        data_path: Path to training data.
        model_output_path: Path to save model artifacts.
        register_model: Whether to register model in MLflow registry.
        model_registry_name: Name for registered model.

    Returns:
        Dictionary containing:
            - run_id: MLflow run ID
            - metrics: Dictionary of computed metrics
            - model_path: Path to saved model artifacts
            - model_version: Version tag if registered

    Raises:
        FileNotFoundError: If data file doesn't exist.
    """
    # Set up MLflow experiment
    mlflow.set_experiment(experiment_name)

    # Generate run name if not provided
    if run_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"training_run_{timestamp}"

    # Default hyperparameters
    default_params = {
        "n_estimators": 100,
        "max_depth": 10,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "random_state": settings.random_state,
        "test_size": settings.test_size,
    }

    params = {**default_params, **(hyperparameters or {})}

    logger.info(f"Starting training experiment: {run_name}")

    with mlflow.start_run(run_name=run_name) as run:
        run_id = run.info.run_id
        logger.info(f"MLflow run ID: {run_id}")

        # Log hyperparameters
        mlflow.log_params(params)
        logger.info(f"Logged hyperparameters: {params}")

        # Load and prepare data
        X, y = load_and_prepare_data(data_path)
        feature_names = get_feature_columns()

        # Log dataset info
        mlflow.log_param("n_samples", len(X))
        mlflow.log_param("n_features", len(feature_names))

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=params["test_size"],
            random_state=params["random_state"],
            stratify=y,
        )

        mlflow.log_param("train_samples", len(X_train))
        mlflow.log_param("test_samples", len(X_test))

        # Extract model-specific hyperparameters
        model_params = {k: v for k, v in params.items() if k not in ["test_size"]}

        # Train model - pass DataFrame to preserve feature names in scaler
        model, scaler = train_model(X_train, y_train.values, model_params)

        # Make predictions
        X_test_scaled = scaler.transform(X_test)
        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:, 1]

        # Compute and log metrics
        metrics = compute_metrics(y_test.values, y_pred, y_prob)
        mlflow.log_metrics(metrics)
        logger.info(f"Metrics: {metrics}")

        # Save model artifacts locally
        model_path = save_model_artifacts(
            model, scaler, feature_names, model_output_path
        )

        # Log model to MLflow
        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            registered_model_name=model_registry_name if register_model else None,
        )

        # Log scaler as artifact
        scaler_path = Path("mlflow_artifacts/scaler.pkl")
        scaler_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(scaler, scaler_path)
        mlflow.log_artifact(str(scaler_path), artifact_path="artifacts")

        # Log feature names
        feature_path = Path("mlflow_artifacts/feature_names.txt")
        with open(feature_path, "w") as f:
            f.write("\n".join(feature_names))
        mlflow.log_artifact(str(feature_path), artifact_path="artifacts")

        # Add tags
        mlflow.set_tag("model_type", "RandomForestClassifier")
        mlflow.set_tag("model_version", settings.model_version)
        mlflow.set_tag("training_date", datetime.now().isoformat())

        # Get model version if registered
        model_version = None
        if register_model:
            try:
                client = mlflow.tracking.MlflowClient()
                versions = client.search_model_versions(f"name='{model_registry_name}'")
                if versions:
                    model_version = max(v.version for v in versions)
                    logger.info(f"Registered model version: {model_version}")
            except Exception as e:
                logger.warning(f"Could not get model version: {e}")

        result = {
            "run_id": run_id,
            "metrics": metrics,
            "model_path": str(model_path),
            "model_version": model_version,
        }

        logger.info(f"Training experiment completed: {result}")

        return result


def compare_experiments(
    experiment_name: str = "heart_disease_prediction",
    metric: str = "accuracy",
) -> pd.DataFrame:
    """Compare runs within an experiment.

    Args:
        experiment_name: Name of the MLflow experiment.
        metric: Metric to sort by (descending).

    Returns:
        DataFrame with run information sorted by specified metric.
    """
    experiment = mlflow.get_experiment_by_name(experiment_name)

    if experiment is None:
        logger.warning(f"Experiment '{experiment_name}' not found")
        return pd.DataFrame()

    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=[f"metrics.{metric} DESC"],
    )

    return runs


def get_best_run(
    experiment_name: str = "heart_disease_prediction",
    metric: str = "accuracy",
) -> Optional[Dict[str, Any]]:
    """Get the best run from an experiment based on a metric.

    Args:
        experiment_name: Name of the MLflow experiment.
        metric: Metric to optimize (higher is better).

    Returns:
        Dictionary with best run info, or None if no runs found.
    """
    runs = compare_experiments(experiment_name, metric)

    if runs.empty:
        return None

    best_run = runs.iloc[0]

    return {
        "run_id": best_run["run_id"],
        "metrics": {
            "accuracy": best_run.get("metrics.accuracy"),
            "roc_auc": best_run.get("metrics.roc_auc"),
            "precision": best_run.get("metrics.precision"),
            "recall": best_run.get("metrics.recall"),
            "f1": best_run.get("metrics.f1"),
        },
        "params": {
            k.replace("params.", ""): v
            for k, v in best_run.items()
            if k.startswith("params.")
        },
    }


if __name__ == "__main__":
    # Example usage: run a training experiment
    import argparse

    parser = argparse.ArgumentParser(description="Train heart disease model")
    parser.add_argument(
        "--experiment-name",
        default="heart_disease_prediction",
        help="MLflow experiment name",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Name for this training run",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=100,
        help="Number of trees in random forest",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=10,
        help="Maximum tree depth",
    )
    parser.add_argument(
        "--no-register",
        action="store_true",
        help="Don't register model in MLflow registry",
    )

    args = parser.parse_args()

    hyperparameters = {
        "n_estimators": args.n_estimators,
        "max_depth": args.max_depth,
    }

    result = run_training_experiment(
        experiment_name=args.experiment_name,
        run_name=args.run_name,
        hyperparameters=hyperparameters,
        register_model=not args.no_register,
    )

    print(f"\nTraining completed!")
    print(f"Run ID: {result['run_id']}")
    print(f"Metrics: {result['metrics']}")
    print(f"Model saved to: {result['model_path']}")
    if result["model_version"]:
        print(f"Model version: {result['model_version']}")
