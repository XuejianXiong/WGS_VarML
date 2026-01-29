#!/usr/bin/env python3
"""
07_predict.py

Run inference using a trained ML model for ClinVar pathogenicity prediction.

Pipeline
--------
1. Load trained model artifacts (model, imputer, feature order)
2. Load feature matrix
3. Align features to training feature order
4. Impute missing values using trained imputer
5. Generate predictions (labels + probabilities)
6. Save predictions to disk

Usage
-----
    python3 src/07_predict.py <features_csv> <model_dir> --outdir <outdir>

Example
-------
    python3 src/07_predict.py data/processed/new_variants.vep.features.csv results/models --outdir results/predictions
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from logzero import logger, setup_logger

from utils.ml_utils import (
    load_artifacts,
    align_features,
    handle_missing_values,
)

# ------------------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------------------
setup_logger()


# ------------------------------------------------------------------------------
# Argument parsing
# ------------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict pathogenicity using a trained ClinVar ML model"
    )

    parser.add_argument(
        "features",
        type=str,
        help="Feature matrix (CSV or Parquet)"
    )

    parser.add_argument(
        "model_dir",
        type=str,
        help="Directory containing trained model artifacts"
    )

    parser.add_argument(
        "--outdir",
        type=str,
        default="results/predictions",
        help="Output directory for predictions"
    )

    return parser.parse_args()


# ------------------------------------------------------------------------------
# Data loading
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

    logger.info(f"Feature matrix shape: {df.shape}")
    return df


# ------------------------------------------------------------------------------
# Prediction
# ------------------------------------------------------------------------------
def run_prediction(
    df: pd.DataFrame,
    model,
    imputer,
    feature_order: list[str],
) -> pd.DataFrame:
    """
    Align features, impute missing values, and generate predictions.
    """
    logger.info("Preparing features for inference")

    X = align_features(df, feature_order)
    X, _ = handle_missing_values(X, imputer=imputer)

    logger.info("Running model inference")
    y_prob = model.predict_proba(X)[:, 1]
    y_pred = model.predict(X)

    result_df = df.copy()
    result_df["pred_label"] = y_pred
    result_df["pred_prob"] = y_prob

    return result_df


# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------
def main() -> None:
    args = parse_args()

    features_path = Path(args.features)
    model_dir = Path(args.model_dir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Load inputs
    df = load_features(features_path)
    model, imputer, feature_order = load_artifacts(model_dir)

    # Predict
    predictions = run_prediction(
        df=df,
        model=model,
        imputer=imputer,
        feature_order=feature_order,
    )

    # Save results
    out_path = outdir / "predictions.csv"
    predictions.to_csv(out_path, index=False)
    logger.info(f"Predictions written to {out_path}")

    logger.info("Prediction pipeline completed successfully")


# ------------------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    main()
