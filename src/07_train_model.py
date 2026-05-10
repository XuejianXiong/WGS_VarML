#!/usr/bin/env python3
"""
07_train_model.py

A robust, factory-pattern training suite for genomic variant pathogenicity.
Benchmarking engine for multiple classifier architectures (RF, XGBoost, 
LightGBM, Logistic Regression) against vectorized genomic features.

Execution Logic (in order)
-------------------------
1. Load training/test partitions and resolve experiment configuration.
2. Initialize model architecture based on target model_type.
3. Fit median-based numerical imputer on Train set (preventing data leakage).
4. Apply imputation and strict feature schema enforcement to Test set.
5. Execute training with support for 'balanced' class weighting.
6. Evaluate performance (Accuracy, F1, ROC-AUC) on validation and test sets.
7. Serialize artifacts: Model binary, fitted imputer, and feature metadata.

Config (config.yaml under "training"): model_params, random_state, 
test_size, imputation_strategy, class_weight.
Precedence: CLI > YAML > defaults.

Usage
-----
    python3 src/07_train_model.py <train_parquet> <test_parquet> <--config CONFIG> [--model_type <type>] [--outdir DIR]

Example
-------
    python3 src/07_train_model.py data/splits/clinvar.train.parquet --test-set data/splits/clinvar.test.parquet --config input.json --model-type xgb --outdir results/models/xgb/
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple, Dict, Any, List, Optional

import joblib
import pandas as pd
from logzero import logger, setup_logger
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    average_precision_score,
)

# Optional Advanced Gradient Boosting Frameworks
try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None

try:
    from lightgbm import LGBMClassifier
except ImportError:
    LGBMClassifier = None

# Project Utilities
from utils.config import load_config
from utils.ml_utils import (
    select_features,
    handle_missing_values,
    save_artifacts,
    align_features,
)

# ------------------------------------------------------------------------------
# Logging & Configuration
# ------------------------------------------------------------------------------
setup_logger()

class ModelTrainer:
    """
    Main execution class for model training, validation, and artifact generation.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = load_config(Path(config_path)) if config_path else {}
        self.train_cfg = self.config.get("train", {})
        self.model_cfg = self.train_cfg.get("model", {})
        self.random_state = self.train_cfg.get("random_state", 42)
        self.imputer = None
        self.feature_order = []

    def load_data(self, path: Path) -> pd.DataFrame:
        """Loads and prepares dataset; enforces binary clnsig labels."""
        logger.info(f"Accessing dataset: {path.name}")
        df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
        
        # ML Focus: Benign (0) vs Pathogenic (1)
        valid_mask = df["clnsig_label"].isin([0, 1])
        df_filtered = df[valid_mask].copy()
        
        logger.debug(f"Filter complete. Retained {len(df_filtered)} of {len(df)} variants.")
        return df_filtered

    def get_classifier(self, model_type: str) -> Any:
        """
        Factory method to initialize specific classifier architectures based on 
        the --model-type flag.
        """
        logger.info(f"Initializing {model_type.upper()} classifier architecture.")
        
        if model_type == "rf":
            return RandomForestClassifier(
                n_estimators=self.model_cfg.get("n_estimators", 500),
                max_depth=self.model_cfg.get("max_depth", 15),
                class_weight="balanced",
                n_jobs=-1,
                random_state=self.random_state
            )
        
        elif model_type == "xgb":
            if not XGBClassifier:
                raise ImportError("XGBoost is required for model-type 'xgb'.")
            return XGBClassifier(
                n_estimators=self.model_cfg.get("n_estimators", 500),
                max_depth=self.model_cfg.get("max_depth", 6),
                learning_rate=self.model_cfg.get("learning_rate", 0.05),
                random_state=self.random_state,
                eval_metric='logloss'
            )
        
        elif model_type == "lgbm":
            if not LGBMClassifier:
                raise ImportError("LightGBM is required for model-type 'lgbm'.")
            return LGBMClassifier(
                n_estimators=self.model_cfg.get("n_estimators", 500),
                num_leaves=self.model_cfg.get("num_leaves", 31),
                learning_rate=self.model_cfg.get("learning_rate", 0.05),
                class_weight="balanced",
                random_state=self.random_state,
                importance_type='gain'
            )
        
        elif model_type == "lr":
            return LogisticRegression(
                max_iter=2000, 
                class_weight="balanced", 
                random_state=self.random_state
            )
        
        else:
            raise ValueError(f"Unsupported model_type: {model_type}")

    def evaluate(self, model: Any, X: pd.DataFrame, y: pd.Series, outdir: Path, prefix: str):
        """Generates classification metrics and saves to CSV."""
        y_pred = model.predict(X)
        y_prob = model.predict_proba(X)[:, 1]

        report = classification_report(y, y_pred, output_dict=True)
        roc_auc = roc_auc_score(y, y_prob)
        pr_auc = average_precision_score(y, y_prob)

        logger.info(f"[{prefix.upper()}] ROC-AUC: {roc_auc:.4f} | PR-AUC: {pr_auc:.4f}")

        # Persistence
        m_df = pd.DataFrame(report).T
        m_df["roc_auc"] = roc_auc
        m_df["pr_auc"] = pr_auc
        m_df.to_csv(outdir / f"{prefix}_metrics.csv")

    def save_feature_importance(self, model: Any, outdir: Path):
        """Persists ranked feature contributions if supported by the model."""
        importance = None
        if hasattr(model, "feature_importances_"):
            importance = model.feature_importances_
        elif hasattr(model, "coef_"):
            importance = model.coef_[0]

        if importance is not None:
            imp_df = pd.DataFrame({
                "feature": self.feature_order,
                "importance": importance
            }).sort_values("importance", ascending=False)
            imp_df.to_csv(outdir / "feature_importance.csv", index=False)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Genomic ML Training Suite")
    parser.add_argument("features", type=str, help="Path to training features (Parquet/CSV)")
    parser.add_argument("--model-type", type=str, default="rf", choices=["rf", "xgb", "lgbm", "lr"])
    parser.add_argument("--test-set", type=str, default=None, help="Held-out evaluation set")
    parser.add_argument("--outdir", type=str, default="results/models")
    parser.add_argument("--config", type=str, default=None, help="Experiment YAML config")
    return parser.parse_args()

def main():
    args = parse_args()
    trainer = ModelTrainer(args.config)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # --- 1. Pipeline: Data Ingestion & Preprocessing ---
    df_train_full = trainer.load_data(Path(args.features))
    X, y = select_features(df_train_full)
    trainer.feature_order = X.columns.tolist()
    
    # Fit imputer once on Train to avoid 'Look-ahead' bias
    X_clean, trainer.imputer = handle_missing_values(X)

    # --- 2. Pipeline: Model Optimization ---
    X_train, X_val, y_train, y_val = train_test_split(
        X_clean, y, 
        test_size=trainer.train_cfg.get("val_split", 0.2), 
        stratify=y, 
        random_state=trainer.random_state
    )
    
    clf = trainer.get_classifier(args.model_type)
    clf.fit(X_train, y_train)

    # --- 3. Pipeline: Evaluation & Artifact Export ---
    trainer.evaluate(clf, X_val, y_val, outdir, "val")
    trainer.save_feature_importance(clf, outdir)
    save_artifacts(clf, trainer.imputer, trainer.feature_order, outdir)

    # --- 4. Pipeline: Final Test Set (Independent Hold-out) ---
    if args.test_set:
        logger.info("Evaluating on independent hold-out set.")
        df_test = trainer.load_data(Path(args.test_set))
        X_test = align_features(df_test, trainer.feature_order)
        X_test_clean, _ = handle_missing_values(X_test, imputer=trainer.imputer)
        trainer.evaluate(clf, X_test_clean, df_test["clnsig_label"], outdir, "test")

    logger.info(f"Workflow complete. Artifacts stored in {outdir}")

if __name__ == "__main__":
    main()