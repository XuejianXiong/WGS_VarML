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
------
    python src/06_train_model.py <features_csv> <output_dir> <config.yaml>
"""

from __future__ import annotations

import argparse
from pathlib import Path
import joblib
import pandas as pd
from logzero import logger, setup_logger
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score

# Shared config utilities
from utils.config import load_config
from utils.ml_utils import select_features, handle_missing_values, save_artifacts

# ------------------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------------------
setup_logger()


# ------------------------------------------------------------------------------
# Argument parsing
# ------------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train ML model on ClinVar VEP features"
    )
    parser.add_argument("features", type=str, help="Feature CSV/Parquet")
    parser.add_argument("--outdir", type=str, default="results/models")
    parser.add_argument("--config", type=str, default=None, help="Optional YAML config")
    return parser.parse_args()


# ------------------------------------------------------------------------------
# Data loading & filtering
# ------------------------------------------------------------------------------
def load_features(path: Path) -> pd.DataFrame:
    logger.info(f"Loading feature matrix: {path}")
    suffix = path.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(path, low_memory=False)
    elif suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        raise ValueError(f"Unsupported feature format: {suffix}")
    logger.info(f"Initial dataset shape: {df.shape}")
    return df


def filter_valid_labels(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Filtering valid CLNSIG labels (0/1)")
    df = df[df["clnsig_label"].isin([0, 1])].copy()
    logger.info(f"After filtering: {df.shape}")
    return df


# ------------------------------------------------------------------------------
# Train / validation split
# ------------------------------------------------------------------------------
def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    logger.info("Splitting train / validation sets")
    return train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )


# ------------------------------------------------------------------------------
# Model training
# ------------------------------------------------------------------------------
def train_model(X_train: pd.DataFrame, y_train: pd.Series, model_cfg: dict, random_state: int) -> RandomForestClassifier:
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
# Evaluate model
# ------------------------------------------------------------------------------
def evaluate_model(model: RandomForestClassifier, X_val: pd.DataFrame, y_val: pd.Series, outdir: Path):
    logger.info("Evaluating model")
    y_pred = model.predict(X_val)
    y_prob = model.predict_proba(X_val)[:, 1]

    report = classification_report(y_val, y_pred, output_dict=True)
    roc_auc = roc_auc_score(y_val, y_prob)
    pr_auc = average_precision_score(y_val, y_prob)

    logger.info(f"ROC-AUC: {roc_auc:.4f}")
    logger.info(f"PR-AUC:  {pr_auc:.4f}")

    metrics_df = pd.DataFrame(report).T
    metrics_df["roc_auc"] = roc_auc
    metrics_df["pr_auc"] = pr_auc
    metrics_path = outdir / "metrics.csv"
    metrics_df.to_csv(metrics_path)
    logger.info(f"Metrics written to {metrics_path}")


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
    df = load_features(Path(args.features))
    df = filter_valid_labels(df)
    X, y = select_features(df)
    X, imputer = handle_missing_values(X)

    X_train, X_val, y_train, y_val = split_data(X, y, test_size, random_state)
    model = train_model(X_train, y_train, model_cfg, random_state)

    evaluate_model(model, X_val, y_val, outdir)
    save_artifacts(model, imputer, X.columns.tolist(), outdir)

    logger.info("Training pipeline completed successfully")


# ------------------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    main()
