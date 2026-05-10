#!/usr/bin/env python3
"""
08_model_inference.py

Genomic ML Inference Engine for Variant Pathogenicity Prediction.
Used after 07_train_model to generate pathogenicity scores for unobserved 
variants using serialized artifacts (model, imputer, feature order).

Workflow (in order)
-------------------
1. Artifact Loading: Retrieves model.joblib, imputer.joblib, and feature_order.txt.
2. Feature Alignment: Reindexes input data to match the exact training schema.
3. Imputation: Applies the fitted imputer state to resolve missing genomic values.
4. Prediction: Generates pathogenicity probabilities and binary classifications.
5. Export: Saves results to a model-specific CSV (e.g., xgb_predictions.csv).

Config (config.yaml under "inference")
--------------------------------------
Supports CLI overrides for output directory and model naming prefixes. 
Ensures zero-skew between training and production environments.

Usage
-----
    python3 src/08_model_inference.py <features.parquet> <model_dir> [--outdir DIR]

Example
-------
    python3 src/08_model_inference.py data/splits/clinvar.infer.parquet results/models/xgb/ --outdir results/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Tuple, Any

import pandas as pd
from logzero import logger, setup_logger

# Project Utilities
try:
    from utils.ml_utils import (
        load_artifacts,
        align_features,
        handle_missing_values,
    )
except ImportError as e:
    logger.error(f"Critical Error: Missing project utilities in 'utils/ml_utils.py': {e}")
    sys.exit(1)

# Initialize production logging
setup_logger(name="Inference", level=20)


def parse_args() -> argparse.Namespace:
    """
    Parses and validates command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="WGS_VarML Pathogenicity Inference Engine",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "features", 
        type=str, 
        help="Input feature matrix (CSV or Parquet format)"
    )
    parser.add_argument(
        "model_dir", 
        type=str, 
        help="Directory containing model.joblib, imputer.joblib, etc."
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="results",
        help="Directory to save the prediction CSV"
    )
    parser.add_argument(
        "--id-cols",
        nargs="+",
        default=["chr", "pos", "ref", "alt"],
        help="Columns to preserve for variant identification"
    )

    return parser.parse_args()


def load_dataset(path: Path) -> pd.DataFrame:
    """
    Loads data from disk, supporting Parquet and CSV formats.
    """
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    logger.info(f"Reading input features: {path.name}")
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, low_memory=False)


def run_pipeline(
    df: pd.DataFrame,
    model: Any,
    imputer: Any,
    feature_order: list[str],
    id_cols: list[str]
) -> pd.DataFrame:
    """
    Executes the inference logic: Align -> Impute -> Predict.
    """
    # 1. Align features to training order (Crucial for correct math)
    X = align_features(df, feature_order)
    
    # 2. Handle missing values using the fitted training imputer
    X_clean, _ = handle_missing_values(X, imputer=imputer)

    # 3. Generate Scores
    logger.info("Generating model predictions...")
    probs = model.predict_proba(X_clean)[:, 1]
    labels = model.predict(X_clean)

    # 4. Consolidate results with variant IDs
    existing_ids = [c for c in id_cols if c in df.columns]
    results = df[existing_ids].copy()
    results["pathogenicity_score"] = probs
    results["predicted_label"] = labels

    return results


def main() -> None:
    """
    Main execution block for the inference pipeline.
    """
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    try:
        # 1. Load trained artifacts from the specified subfolder
        # Note: Since you use 'model.joblib', this points to model_dir/model.joblib
        logger.info(f"Loading artifacts from: {args.model_dir}")
        model, imputer, feature_order = load_artifacts(Path(args.model_dir))

        # 2. Load feature data
        df = load_dataset(Path(args.features))

        # 3. Process
        output_df = run_pipeline(
            df=df,
            model=model,
            imputer=imputer,
            feature_order=feature_order,
            id_cols=args.id_cols
        )

        # 4. Save results (using the model subfolder name for the filename)
        model_name = Path(args.model_dir).name
        output_path = outdir / f"{model_name}_predictions.csv"
        output_df.to_csv(output_path, index=False)
        
        logger.info(f"Inference complete. {len(output_df)} variants scored.")
        logger.info(f"Predictions saved to: {output_path}")

    except Exception as e:
        logger.error(f"Inference Pipeline Failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()