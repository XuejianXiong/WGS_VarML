#!/usr/bin/env python3
"""
train_model.py
--------------------------------------
Train a supervised ML model (RandomForest) to classify high-confidence variants
using extracted features.
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
import joblib

FEATURE_FILE = "features/variant_features.csv"
MODEL_FILE = "models/random_forest_variantqc.pkl"
RESULTS_DIR = "results/"

def load_data():
    if not os.path.exists(FEATURE_FILE):
        raise FileNotFoundError(f"Feature file not found: {FEATURE_FILE}")
    
    df = pd.read_csv(FEATURE_FILE)        
    print(f"[INFO] Loaded {df.shape[0]} variants with {df.shape[1]} features.")

    return df


def train_model(df):
    
    # Select only numeric features
    feature_cols = ["QUAL", "DP", "MQ", "QD", "FS", "SOR", "ReadPosRankSum", "MQRankSum"]
    X = df[feature_cols]

    # Convert FILTER column to binary label: PASS = 1, others = 0
    y = (df["FILTER"] == "PASS").astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
        )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, y_proba)
    print(f"[INFO] Model AUC: {auc:.3f}")
    print(classification_report(y_test, y_pred))

    # Save model
    os.makedirs(os.path.dirname(MODEL_FILE), exist_ok=True)
    joblib.dump(model, MODEL_FILE)
    print(f"[INFO] Saved model → {MODEL_FILE}")


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    df = load_data()
    train_model(df)

if __name__ == "__main__":
    main()
