#!/usr/bin/env python3
"""
06_train_model.py

Train a baseline ML model for ClinVar pathogenicity prediction
using VEP-derived features.

Pipeline
--------
1. Load feature matrix
2. Filter valid CLNSIG labels (0/1)
3. Select ML features
4. Impute missing values
5. Train / validation split (stratified)
6. Train baseline RandomForest
7. Evaluate model
8. Save model, metrics, and feature importance

Usage:
-----
    python3 src/06_train_model.py <features_csv>
    python3 src/06_train_model.py <features_csv> --outdir results/models
    python3 src/06_train_model.py <features_csv> --config config/config.yaml

Example
-----
    python src/06_train_model.py data/processed/clinvar.vep.features.csv --config config/config.yaml

Notes
-----
- Unknown CLNSIG (-1) is excluded from training
- Designed for large datasets (millions of rows)
- Support an OPTIONAL YAML config file to control train/validation split and model hyperparameters. 
  If no YAML is provided, default values are used.

"""

from __future__ import annotations

import argparse
from pathlib import Path
import joblib
import yaml

import pandas as pd
from logzero import logger, setup_logger

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score


# ------------------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------------------
setup_logger()


# ------------------------------------------------------------------------------
# Argument parsing
# ------------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Train ML model on ClinVar VEP features"
    )

    parser.add_argument("features_csv", type=str)

    parser.add_argument(
        "--outdir", type=str, default="results/models"
    )

    parser.add_argument(
        "--config", type=str, default=None,
        help="Optional YAML configuration file"
    )

    return parser.parse_args()


# ------------------------------------------------------------------------------
# Config loading
# ------------------------------------------------------------------------------
def load_config(config_path: Path | None) -> dict:
    """
    Load YAML configuration file.

    Returns empty dict if config is not provided.
    """
    if config_path is None:
        return {}

    logger.info(f"Loading config: {config_path}")
    with open(config_path) as f:
        return yaml.safe_load(f)


# ------------------------------------------------------------------------------
# Data loading & filtering
# ------------------------------------------------------------------------------
def load_features(features_csv: Path) -> pd.DataFrame:
    logger.info(f"Loading feature matrix: {features_csv}")
    df = pd.read_csv(features_csv, low_memory=False)
    logger.info(f"Initial dataset shape: {df.shape}")
    return df


def filter_valid_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only rows with valid binary CLNSIG labels (0/1).
    """
    logger.info("Filtering valid CLNSIG labels (0/1)")
    df = df[df["clnsig_label"].isin([0, 1])].copy()
    logger.info(f"After filtering: {df.shape}")
    return df


# ------------------------------------------------------------------------------
# Feature selection
# ------------------------------------------------------------------------------
def select_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Separate feature matrix X and label vector y.
    """
    logger.info("Selecting ML features")

    drop_cols = {"chr", "pos", "ref", "alt", "clnsig"}

    feature_cols = [
        c for c in df.columns
        if c not in drop_cols and c != "clnsig_label"
    ]

    X = df[feature_cols]
    y = df["clnsig_label"]

    logger.info(f"Selected {X.shape[1]} features")
    return X, y


# ------------------------------------------------------------------------------
# Missing value handling
# ------------------------------------------------------------------------------
def handle_missing_values(
    X: pd.DataFrame,
) -> tuple[pd.DataFrame, SimpleImputer]:
    """
    Median-impute numeric features; drop all-missing columns.
    """
    logger.info("Handling missing values")

    num_cols = X.select_dtypes(include="number").columns.tolist()

    empty_cols = [c for c in num_cols if X[c].notna().sum() == 0]
    if empty_cols:
        logger.warning(
            f"Dropping numeric columns with no observed values: {empty_cols}"
        )
        X = X.drop(columns=empty_cols)
        num_cols = [c for c in num_cols if c not in empty_cols]

    imputer = SimpleImputer(strategy="median")
    X[num_cols] = imputer.fit_transform(X[num_cols])

    return X, imputer


# ------------------------------------------------------------------------------
# Train / validation split
# ------------------------------------------------------------------------------
def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float,
    random_state: int,
):
    """
    Stratified train/validation split.
    """
    logger.info("Splitting train / validation sets")

    return train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )


# ------------------------------------------------------------------------------
# Model training
# ------------------------------------------------------------------------------
def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_cfg: dict,
    random_state: int,
) -> RandomForestClassifier:
    """
    Train RandomForest classifier using YAML-driven hyperparameters.
    """
    logger.info("Training RandomForest model")

    model = RandomForestClassifier(
        n_estimators=model_cfg.get("n_estimators", 200),
        max_depth=model_cfg.get("max_depth", 12),
        n_jobs=model_cfg.get("n_jobs", -1),
        class_weight=model_cfg.get("class_weight", "balanced"),
        random_state=random_state,
    )

    model.fit(X_train, y_train)
    return model


# ------------------------------------------------------------------------------
# Evaluation
# ------------------------------------------------------------------------------
def evaluate_model(
    model: RandomForestClassifier,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    outdir: Path,
):
    """
    Evaluate model performance on validation set.
    """
    logger.info("Evaluating model")

    y_pred = model.predict(X_val)
    y_prob = model.predict_proba(X_val)[:, 1]

    report = classification_report(y_val, y_pred, output_dict=True)
    roc_auc = roc_auc_score(y_val, y_prob)

    logger.info(f"ROC-AUC: {roc_auc:.4f}")

    metrics_df = pd.DataFrame(report).T
    metrics_df["roc_auc"] = roc_auc

    metrics_path = outdir / "metrics.csv"
    metrics_df.to_csv(metrics_path)
    logger.info(f"Metrics written to {metrics_path}")


def save_feature_importance(
    model: RandomForestClassifier,
    feature_names: list[str],
    outdir: Path,
):
    """
    Save RandomForest feature importance.
    """
    fi = (
        pd.Series(model.feature_importances_, index=feature_names)
        .sort_values(ascending=False)
    )

    fi_path = outdir / "feature_importance.csv"
    fi.to_csv(fi_path, header=["importance"])
    logger.info(f"Feature importance written to {fi_path}")


def save_artifacts(
    model: RandomForestClassifier,
    imputer: SimpleImputer,
    outdir: Path,
):
    """
    Persist trained model and preprocessing artifacts.
    """
    joblib.dump(model, outdir / "random_forest_model.joblib")
    joblib.dump(imputer, outdir / "imputer.joblib")


# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------
def main() -> None:
    args = parse_args()

    config = load_config(Path(args.config)) if args.config else {}

    train_cfg = config.get("train", {})
    model_cfg = train_cfg.get("model", {})

    test_size = train_cfg.get("test_size", 0.2)
    random_state = train_cfg.get("random_state", 42)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Load & prepare data
    df = load_features(Path(args.features_csv))

    # Filter valide labels
    df = filter_valid_labels(df)
    
    # Select features
    X, y = select_features(df)

    # Handle missing values
    X, imputer = handle_missing_values(X)

    # Split train / validation data
    X_train, X_val, y_train, y_val = split_data(
        X, y, test_size, random_state
    )

    # Train model
    model = train_model(
        X_train, y_train, model_cfg, random_state
    )

    # Evaluate model
    evaluate_model(model, X_val, y_val, outdir)

    # Save results
    save_feature_importance(model, X.columns.tolist(), outdir)
    save_artifacts(model, imputer, outdir)

    logger.info("Training pipeline completed successfully")


# ------------------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    main()
