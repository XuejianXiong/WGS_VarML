#!/usr/bin/env python3
"""
ml_utils.py

Shared utilities for training and prediction ML pipelines:
- Feature selection
- Missing value handling
- Train/validation split
- Model artifact loading/saving
"""

from pathlib import Path
import pandas as pd
from sklearn.impute import SimpleImputer
import joblib
from logzero import logger


# -----------------------------
# Feature selection
# -----------------------------
def select_features(df: pd.DataFrame, drop_cols=None) -> tuple[pd.DataFrame, pd.Series]:
    drop_cols = drop_cols or {"chr", "pos", "ref", "alt", "clnsig"}

    feature_cols = [c for c in df.columns if c not in drop_cols and c != "clnsig_label"]
    feature_cols = sorted(feature_cols)

    X = df[feature_cols]
    y = df["clnsig_label"] if "clnsig_label" in df.columns else None

    logger.info(f"Selected {X.shape[1]} features")
    return X, y


# -----------------------------
# Handle missing values
# -----------------------------
def handle_missing_values(X: pd.DataFrame, imputer: SimpleImputer = None) -> tuple[pd.DataFrame, SimpleImputer]:
    num_cols = X.select_dtypes(include="number").columns.tolist()

    empty_cols = [c for c in num_cols if X[c].notna().sum() == 0]
    if empty_cols:
        logger.warning(f"Dropping numeric columns with no observed values: {empty_cols}")
        X = X.drop(columns=empty_cols)
        num_cols = [c for c in num_cols if c not in empty_cols]

    if imputer is None:
        imputer = SimpleImputer(strategy="median")
        X[num_cols] = imputer.fit_transform(X[num_cols])
    else:
        X[num_cols] = imputer.transform(X[num_cols])

    return X, imputer


# -----------------------------
# Align features for prediction
# -----------------------------
def align_features(df: pd.DataFrame, feature_order: list[str], fill_value=0) -> pd.DataFrame:
    missing_features = [f for f in feature_order if f not in df.columns]
    if missing_features:
        logger.warning(f"Missing features in input data: {missing_features}")
        for f in missing_features:
            df[f] = fill_value

    X = df[feature_order].copy()
    return X


# -----------------------------
# Model artifact loading/saving
# -----------------------------
def load_artifacts(model_dir: Path):
    model = joblib.load(model_dir / "random_forest_model.joblib")
    imputer = joblib.load(model_dir / "imputer.joblib")
    feature_order_path = model_dir / "feature_order.txt"
    with open(feature_order_path) as f:
        feature_order = [line.strip() for line in f]
    return model, imputer, feature_order


def save_artifacts(model, imputer, feature_names, outdir: Path):
    import joblib
    joblib.dump(model, outdir / "random_forest_model.joblib")
    joblib.dump(imputer, outdir / "imputer.joblib")
    (outdir / "feature_order.txt").write_text("\n".join(feature_names))
