"""
train.py – Conversion Prediction Model Training Script
======================================================
Loads campaign CSVs from data/campaign/, trains an XGBoost classifier,
evaluates it against a Logistic Regression baseline, and saves the artifact 
to models/conversion_model.pkl ONLY if it outperforms the baseline.

Usage:
    python services/conversion/train.py

Output:
    models/conversion_model.pkl    – trained model + metadata
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    f1_score,
    roc_auc_score,
    average_precision_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data" / "campaign"
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "conversion_model.pkl"
METRICS_PATH = MODEL_DIR / "train_metrics.json"

TARGET_COLUMN = os.getenv("TARGET_COLUMN", "converted")
NUMERIC_FEATURES = os.getenv(
    "NUMERIC_FEATURES",
    "age,income,num_campaigns,num_clicks,num_opens,recency_days,tenure_days",
).split(",")
CATEGORICAL_FEATURES = os.getenv(
    "CATEGORICAL_FEATURES",
    "gender,channel,product_category,region",
).split(",")


def load_data(data_dir: Path) -> pd.DataFrame:
    csv_files = list(data_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}.")
    frames = [pd.read_csv(f) for f in csv_files]
    df = pd.concat(frames, ignore_index=True)
    return df


def engineer_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list, list, dict]:
    num_cols = [c for c in NUMERIC_FEATURES if c in df.columns]
    cat_cols = [c for c in CATEGORICAL_FEATURES if c in df.columns]

    X_num = df[num_cols].apply(lambda col: col.fillna(col.median()))

    X_cat = df[cat_cols].copy()
    encoders: dict[str, LabelEncoder] = {}
    for col in cat_cols:
        le = LabelEncoder()
        X_cat[col] = X_cat[col].astype(str).fillna("__missing__")
        X_cat[col] = le.fit_transform(X_cat[col])
        encoders[col] = le

    X = pd.concat([X_num, X_cat], axis=1)
    y = df[TARGET_COLUMN].astype(int)

    return X, y, num_cols, cat_cols, encoders


def train(test_size: float = 0.2, random_state: int = 42) -> dict[str, Any]:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data(DATA_DIR)
    X, y, num_cols, cat_cols, encoders = engineer_features(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # 1. Baseline Model (Logistic Regression)
    log.info("Training Logistic Regression Baseline...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    baseline = LogisticRegression(max_iter=1000, random_state=random_state)
    baseline.fit(X_train_scaled, y_train)
    
    lr_prob = baseline.predict_proba(X_test_scaled)[:, 1]
    lr_pred = (lr_prob >= 0.5).astype(int)
    lr_auc = roc_auc_score(y_test, lr_prob)
    lr_pr_auc = average_precision_score(y_test, lr_prob)
    
    log.info("Baseline ROC-AUC: %.4f | PR-AUC: %.4f", lr_auc, lr_pr_auc)

    # 2. Advanced Model (XGBoost)
    log.info("Training XGBoost...")
    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=random_state,
        n_jobs=-1,
        early_stopping_rounds=30,
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    xgb_prob = model.predict_proba(X_test)[:, 1]
    xgb_pred = (xgb_prob >= 0.5).astype(int)
    xgb_auc = roc_auc_score(y_test, xgb_prob)
    xgb_pr_auc = average_precision_score(y_test, xgb_prob)
    xgb_f1 = f1_score(y_test, xgb_pred)

    log.info("XGBoost ROC-AUC : %.4f | PR-AUC: %.4f", xgb_auc, xgb_pr_auc)

    # 3. Relative Promotion Gate (R10)
    if xgb_auc <= lr_auc:
        log.error("PROMOTION FAILED: XGBoost (%.4f) did not beat baseline (%.4f).", xgb_auc, lr_auc)
        sys.exit(1)
        
    log.info("PROMOTION SUCCESS: XGBoost beats baseline. Saving model.")

    artifact: dict[str, Any] = {
        "model": model,
        "feature_columns": num_cols + cat_cols,
        "numeric_features": num_cols,
        "categorical_features": cat_cols,
        "label_encoders": encoders,
        "model_version": "v1.1",
        "metrics": {"roc_auc": xgb_auc, "pr_auc": xgb_pr_auc, "f1": xgb_f1, "baseline_roc_auc": lr_auc},
    }

    with open(MODEL_PATH, "wb") as fh:
        pickle.dump(artifact, fh)

    metrics = artifact["metrics"]
    with open(METRICS_PATH, "w") as fh:
        json.dump(metrics, fh, indent=2)

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train conversion prediction model")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    metrics = train(test_size=args.test_size, random_state=args.seed)
    print(json.dumps(metrics, indent=2))
