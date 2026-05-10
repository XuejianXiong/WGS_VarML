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
def select_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    # Technical drops only
    technical_drops = {"chr", "pos", "ref", "alt", "clnsig"}
    
    # Target only the "Circular" VEP impact ratings
    circular_drops = {c for c in df.columns if "impact_" in c}
    
    # We KEEP "cons_" (consequences) because they represent raw biology
    effective_drops = technical_drops | circular_drops

    feature_cols = [c for c in df.columns if c not in effective_drops and c != "clnsig_label"]
    feature_cols = sorted(feature_cols)

    X = df[feature_cols]
    y = df["clnsig_label"] if "clnsig_label" in df.columns else None

    logger.info(f"Training with {X.shape[1]} features (including consequences and scores)")
    return X, y

def select_features_v1(df: pd.DataFrame, drop_cols=None) -> tuple[pd.DataFrame, pd.Series]:
    drop_cols = drop_cols or {"chr", "pos", "ref", "alt", "clnsig"}

    feature_cols = [c for c in df.columns if c not in drop_cols and c != "clnsig_label"]
    feature_cols = sorted(feature_cols)

    X = df[feature_cols]
    y = df["clnsig_label"] if "clnsig_label" in df.columns else None

    logger.info(f"Selected {X.shape[1]} features")
    return X, y


def select_features_v2(df: pd.DataFrame, drop_cols=None) -> tuple[pd.DataFrame, pd.Series]:
    """
    Refined feature selection. 
    To force models to learn from nuanced scores (SIFT, etc.), 
    we add impact/consequence categories to the drop list.
    """
    # 1. Base technical drops
    base_drops = {"chr", "pos", "ref", "alt", "clnsig"}
    
    # 2. Add "Obvious" features to the drop list for the experiment
    # These are the columns currently dominating your 0.94 ROC-AUC
    obvious_drops = {c for c in df.columns if "impact_" in c or "cons_" in c}
    
    # Combine lists
    effective_drops = (drop_cols or base_drops) | obvious_drops

    feature_cols = [c for c in df.columns if c not in effective_drops and c != "clnsig_label"]
    feature_cols = sorted(feature_cols)

    X = df[feature_cols]
    y = df["clnsig_label"] if "clnsig_label" in df.columns else None

    logger.info(f"Selected {X.shape[1]} features (Dropped {len(obvious_drops)} impact/cons features)")
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
    model = joblib.load(model_dir / "model.joblib")
    imputer = joblib.load(model_dir / "imputer.joblib")
    feature_order_path = model_dir / "feature_order.txt"
    with open(feature_order_path) as f:
        feature_order = [line.strip() for line in f]
    return model, imputer, feature_order


def save_artifacts(model, imputer, feature_names, outdir: Path):
    import joblib
    joblib.dump(model, outdir / "model.joblib")
    joblib.dump(imputer, outdir / "imputer.joblib")
    (outdir / "feature_order.txt").write_text("\n".join(feature_names))
