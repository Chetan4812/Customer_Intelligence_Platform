"""
verify.py  –  Standalone verification script for the Customer Intelligence Platform
====================================================================================
Runs all validation checks in a single Python process (no subprocess, no pytest)
to avoid repeated DLL scanning on Windows Defender.

Usage:
    python scripts/verify.py
"""

from __future__ import annotations

import os
import pickle
import sys
import traceback
from pathlib import Path
from unittest.mock import MagicMock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PASS = "[OK]"
FAIL = "[FAIL]"
results: list[tuple[str, bool, str]] = []


def check(name: str, fn):
    """Run fn(), record PASS or FAIL with any exception message."""
    try:
        fn()
        results.append((name, True, ""))
        print(f"  {PASS}  {name}")
    except Exception as exc:
        msg = f"{type(exc).__name__}: {exc}"
        results.append((name, False, msg))
        print(f"  {FAIL}  {name}")
        print(f"       {msg}")


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Import numpy / pandas / sklearn  (AV scan happens here, once)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[1/5] Importing scientific stack…")
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
print("      numpy, pandas, sklearn  OK")

# ─────────────────────────────────────────────────────────────────────────────
# 2.  Import FastAPI / Starlette (AV scan for C extensions)
# ─────────────────────────────────────────────────────────────────────────────
print("[2/5] Importing FastAPI…")
from fastapi import FastAPI
from fastapi.testclient import TestClient
print("      FastAPI, TestClient  OK")

# ─────────────────────────────────────────────────────────────────────────────
# 3.  Mock xgboost  +  import service modules
# ─────────────────────────────────────────────────────────────────────────────
print("[3/5] Importing service modules (xgboost mocked)…")

_mock_xgb = MagicMock()

# Provide a working XGBClassifier stub
class _FakeXGB:
    def __init__(self, **kw): pass
    def fit(self, X, y, **kw): pass
    def predict_proba(self, X): return np.array([[0.3, 0.7]] * len(X))

_mock_xgb.XGBClassifier = _FakeXGB
sys.modules["xgboost"] = _mock_xgb

import services.conversion.train as _train_mod
import services.conversion.main as _main_mod
print("      service modules  OK")

# ─────────────────────────────────────────────────────────────────────────────
# 4.  Build fixtures
# ─────────────────────────────────────────────────────────────────────────────
print("[4/5] Building test fixtures…")

import tempfile

tmp = Path(tempfile.mkdtemp())
csv_dir = tmp / "campaign"
csv_dir.mkdir()

rng = np.random.default_rng(42)
n = 20
df_fixture = pd.DataFrame({
    "age":              rng.integers(20, 65, n).tolist(),
    "income":           rng.integers(25000, 100000, n).tolist(),
    "num_campaigns":    rng.integers(1, 6, n).tolist(),
    "num_clicks":       rng.integers(0, 20, n).tolist(),
    "num_opens":        rng.integers(0, 30, n).tolist(),
    "recency_days":     rng.integers(1, 120, n).tolist(),
    "tenure_days":      rng.integers(30, 1000, n).tolist(),
    "gender":           ["M", "F"] * (n // 2),
    "channel":          ["email", "sms", "email", "push", "email"] * (n // 5),
    "product_category": ["electronics", "fashion"] * (n // 2),
    "region":           ["north", "south", "east", "west", "north"] * (n // 5),
    "converted":        [1] * (n // 2) + [0] * (n // 2),
})
df_fixture.to_csv(csv_dir / "campaign.csv", index=False)

# Build a lightweight artifact (mock model inside)
NUMERIC  = ["age", "income", "num_campaigns", "num_clicks",
            "num_opens", "recency_days", "tenure_days"]
CATEG    = ["gender", "channel", "product_category", "region"]
ALL_FEAT = NUMERIC + CATEG

encoders = {}
for col, vals in {
    "gender": ["F", "M"], "channel": ["email", "push", "sms"],
    "product_category": ["electronics", "fashion"],
    "region": ["east", "north", "south", "west"],
}.items():
    le = LabelEncoder(); le.fit(vals); encoders[col] = le

mock_model = MagicMock()
mock_model.predict_proba.return_value = np.array([[0.3, 0.7]])

# In-memory artifact for unit checks (not pickled – MagicMock can't be pickled)
artifact = {
    "model": mock_model,
    "feature_columns": ALL_FEAT,
    "numeric_features": NUMERIC,
    "categorical_features": CATEG,
    "label_encoders": encoders,
    "model_version": "v1.0-test",
    "metrics": {"roc_auc": 0.88, "f1": 0.91},
}

# For API tests use the real trained model artifact (already on disk from train.py)
real_artifact_path = ROOT / "models" / "conversion_model.pkl"
if not real_artifact_path.exists():
    print("  WARNING: models/conversion_model.pkl not found – run train.py first")
    real_artifact_path = None

print("      fixtures ready  OK")

# ─────────────────────────────────────────────────────────────────────────────
# 5.  Run checks
# ─────────────────────────────────────────────────────────────────────────────
print("[5/5] Running checks…\n")

if real_artifact_path:
    # Restore real xgboost for pickle deserialization of the trained model
    for k in list(sys.modules.keys()):
        if k == "xgboost" or k.startswith("xgboost."):
            del sys.modules[k]
    import xgboost as _real_xgb  # noqa: F401

def _test_artifact_keys():
    for k in ("model", "feature_columns", "numeric_features",
              "categorical_features", "label_encoders", "model_version"):
        assert k in artifact, f"Missing key: {k}"
check("artifact has required keys", _test_artifact_keys)

def _test_pickle_roundtrip():
    assert real_artifact_path is not None, "models/conversion_model.pkl missing"
    with open(real_artifact_path, "rb") as fh:
        loaded = pickle.load(fh)
    assert "model_version" in loaded
    assert "feature_columns" in loaded
    assert loaded["model_version"] == "v1.1"
check("real artifact pickle round-trip", _test_pickle_roundtrip)

def _test_label_encoders():
    le = artifact["label_encoders"]["gender"]
    vals = sorted(le.transform(["M", "F"]).tolist())
    assert vals == [0, 1]
check("label encoders transform correctly", _test_label_encoders)

def _test_mock_predict():
    row = pd.DataFrame([{f: 0 for f in ALL_FEAT}])
    prob = float(artifact["model"].predict_proba(row)[0, 1])
    assert 0.0 <= prob <= 1.0
check("mock model predict_proba in range", _test_mock_predict)

# ── Feature-engineering checks ────────────────────────────────────────────
print("\n  [Feature Engineering]")

def _test_load_data():
    df = _train_mod.load_data(csv_dir)
    assert len(df) == n
    assert "converted" in df.columns
check("load_data reads CSV correctly", _test_load_data)

def _test_engineer_shape():
    df = _train_mod.load_data(csv_dir)
    X, y, num_cols, cat_cols, enc = _train_mod.engineer_features(df)
    assert X.shape[0] == n
    assert len(y) == n
check("engineer_features output shape correct", _test_engineer_shape)

def _test_engineer_no_nulls():
    df = _train_mod.load_data(csv_dir)
    X, *_ = _train_mod.engineer_features(df)
    assert not X.isnull().any().any()
check("engineer_features produces no nulls", _test_engineer_no_nulls)

def _test_train_split_stratified():
    df = _train_mod.load_data(csv_dir)
    X, y, *_ = _train_mod.engineer_features(df)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2,
                                               random_state=42, stratify=y)
    assert len(X_tr) == 16
    assert len(X_te) == 4
    assert y_te.sum() == 2  # 2 positives in test set (balanced)
check("stratified train/test split works on 20 rows", _test_train_split_stratified)

# ── FastAPI endpoint checks ────────────────────────────────────────────────
print("\n  [API Endpoints]")

if real_artifact_path:
    _main_mod.MODEL_PATH = real_artifact_path
    client = TestClient(_main_mod.app, raise_server_exceptions=True)
    # Trigger lifespan to load the model for tests
    client.__enter__()
else:
    client = None
    print("  SKIP - real model not found, skipping endpoint tests")

def _test_health():
    assert client is not None, "No client - model missing"
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"
check("GET /health returns 200 ok", _test_health)

def _test_predict_full():
    assert client is not None, "No client - model missing"
    resp = client.post("/predict", json={
        "age": 35, "income": 55000, "num_campaigns": 3,
        "num_clicks": 7, "num_opens": 12, "recency_days": 14,
        "tenure_days": 365, "gender": "M", "channel": "email",
        "product_category": "electronics", "region": "north",
    })
    assert resp.status_code == 200
    data = resp.json()
    assert "conversion_prob" in data
    assert 0.0 <= data["conversion_prob"] <= 1.0
    assert data["prediction"] in (0, 1)
check("POST /predict full payload -> 200", _test_predict_full)

def _test_predict_minimal():
    assert client is not None, "No client - model missing"
    resp = client.post("/predict", json={"age": 28})
    assert resp.status_code == 200
    assert "conversion_prob" in resp.json()
check("POST /predict minimal payload -> 200", _test_predict_minimal)

def _test_root():
    assert client is not None, "No client - model missing"
    resp = client.get("/")
    assert resp.status_code == 200
    assert "service" in resp.json()
check("GET / returns service info", _test_root)

# ── Model metrics check (from saved train_metrics.json) ───────────────────
print("\n  [Trained Model Metrics]")

def _test_real_metrics():
    metrics_path = ROOT / "models" / "train_metrics.json"
    assert metrics_path.exists(), "Run train.py first to generate metrics"
    import json
    m = json.loads(metrics_path.read_text())
    assert m["roc_auc"] >= 0.7, f"ROC-AUC too low: {m['roc_auc']}"
    assert m["f1"] >= 0.7, f"F1 too low: {m['f1']}"
    print(f"       ROC-AUC={m['roc_auc']}  F1={m['f1']}")
check("real trained model metrics >= 0.7", _test_real_metrics)

# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────
passed = sum(1 for _, ok, _ in results if ok)
failed = sum(1 for _, ok, _ in results if not ok)
total  = len(results)

print(f"\n{'='*60}")
print(f"  Results: {passed}/{total} passed", end="")
if failed:
    print(f"  |  {failed} FAILED")
    for name, ok, msg in results:
        if not ok:
            print(f"    {FAIL} {name}: {msg}")
else:
    print("  -  ALL PASSED OK")
print(f"{'='*60}\n")

sys.exit(0 if failed == 0 else 1)
