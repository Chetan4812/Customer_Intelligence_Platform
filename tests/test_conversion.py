"""
test_conversion.py – Unit tests for the conversion prediction service.

Strategy: fully mock xgboost so we never import the DLL (avoids AV scanning delay).
The real model training is tested via integration (train.py script); here we validate
the API contract, feature engineering, and endpoint logic.

Run with:
    python -m pytest tests/test_conversion.py -v
"""

from __future__ import annotations

import os
import pickle
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import LabelEncoder

# ── Project root on path ──────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

NUMERIC_FEATURES = ["age", "income", "num_campaigns", "num_clicks",
                    "num_opens", "recency_days", "tenure_days"]
CATEGORICAL_FEATURES = ["gender", "channel", "product_category", "region"]
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def _make_encoders():
    """Build label encoders matching the sample fixture values."""
    specs = {
        "gender":           ["F", "M"],
        "channel":          ["email", "push", "sms"],
        "product_category": ["electronics", "fashion"],
        "region":           ["east", "north", "south", "west"],
    }
    encoders = {}
    for col, classes in specs.items():
        le = LabelEncoder()
        le.fit(classes)
        encoders[col] = le
    return encoders


class MockXGBoostModel:
    def predict_proba(self, X):
        return np.array([[0.3, 0.7]] * len(X))


def _make_mock_model():
    """A lightweight sklearn-compatible mock that always predicts 0.7 probability."""
    return MockXGBoostModel()


def _make_artifact(tmp_path: Path) -> tuple[Path, dict]:
    """Write a fake model artifact to disk and return (path, artifact)."""
    artifact = {
        "model": _make_mock_model(),
        "feature_columns": ALL_FEATURES,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "label_encoders": _make_encoders(),
        "model_version": "v1.0-test",
        "metrics": {"roc_auc": 0.88, "f1": 0.91},
    }
    path = tmp_path / "conversion_model.pkl"
    with open(path, "wb") as fh:
        pickle.dump(artifact, fh)
    return path, artifact


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def artifact_path(tmp_path):
    path, _ = _make_artifact(tmp_path)
    return path


@pytest.fixture
def artifact(tmp_path):
    _, art = _make_artifact(tmp_path)
    return art


@pytest.fixture
def sample_csv_dir(tmp_path):
    """20-row balanced campaign CSV – no xgboost needed."""
    rng = np.random.default_rng(42)
    n = 20
    df = pd.DataFrame({
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
    csv_dir = tmp_path / "campaign"
    csv_dir.mkdir()
    df.to_csv(csv_dir / "campaign.csv", index=False)
    return csv_dir


# ─────────────────────────────────────────────────────────────────────────────
# Artifact structure tests  (no xgboost import needed)
# ─────────────────────────────────────────────────────────────────────────────

class TestArtifactStructure:
    def test_artifact_keys(self, artifact):
        for key in ("model", "feature_columns", "numeric_features",
                    "categorical_features", "label_encoders", "model_version"):
            assert key in artifact, f"Missing key '{key}'"

    def test_artifact_roundtrip(self, artifact_path):
        """Pickle round-trip must preserve all keys."""
        with open(artifact_path, "rb") as fh:
            loaded = pickle.load(fh)
        assert loaded["model_version"] == "v1.0-test"
        assert loaded["feature_columns"] == ALL_FEATURES

    def test_label_encoders_work(self, artifact):
        le = artifact["label_encoders"]["gender"]
        assert set(le.transform(["M", "F"])) == {0, 1}

    def test_mock_model_predicts(self, artifact):
        row = pd.DataFrame([{f: 0 for f in ALL_FEATURES}])
        prob = float(artifact["model"].predict_proba(row)[0, 1])
        assert 0.0 <= prob <= 1.0


# ─────────────────────────────────────────────────────────────────────────────
# Feature-engineering tests  (imports sklearn only – no xgboost)
# ─────────────────────────────────────────────────────────────────────────────

class TestFeatureEngineering:
    """
    Test engineer_features() by mocking the xgboost import so train.py
    can be imported without touching xgboost.dll.
    """

    @pytest.fixture(autouse=True)
    def mock_xgboost(self):
        """Prevent xgboost from being imported during train module load."""
        mock_xgb = MagicMock()
        mock_xgb.XGBClassifier.return_value = _make_mock_model()
        with patch.dict("sys.modules", {"xgboost": mock_xgb}):
            # Force re-import of train module with mock in place
            if "services.conversion.train" in sys.modules:
                del sys.modules["services.conversion.train"]
            yield
            if "services.conversion.train" in sys.modules:
                del sys.modules["services.conversion.train"]

    def test_engineer_features_shape(self, sample_csv_dir):
        from services.conversion.train import engineer_features, load_data
        df = load_data(sample_csv_dir)
        X, y, num_cols, cat_cols, encoders = engineer_features(df)
        assert X.shape[0] == len(df)
        assert set(num_cols).issubset(set(df.columns))
        assert len(y) == len(df)

    def test_engineer_features_no_nulls(self, sample_csv_dir):
        from services.conversion.train import engineer_features, load_data
        df = load_data(sample_csv_dir)
        X, y, *_ = engineer_features(df)
        assert not X.isnull().any().any(), "Feature matrix contains nulls"

    def test_load_data_reads_csv(self, sample_csv_dir):
        from services.conversion.train import load_data
        df = load_data(sample_csv_dir)
        assert len(df) == 20
        assert "converted" in df.columns


# ─────────────────────────────────────────────────────────────────────────────
# FastAPI endpoint tests  (xgboost mocked, artifact loaded from pickle)
# ─────────────────────────────────────────────────────────────────────────────

class TestConversionEndpoints:
    """Uses FastAPI TestClient. The pickle artifact contains a mock model object."""

    @pytest.fixture(autouse=True)
    def setup_client(self, artifact_path):
        from fastapi.testclient import TestClient

        # Mock xgboost before importing main so the module loads clean
        mock_xgb = MagicMock()
        with patch.dict("sys.modules", {"xgboost": mock_xgb}):
            if "services.conversion.main" in sys.modules:
                del sys.modules["services.conversion.main"]
            import services.conversion.main as mod
            with patch.object(mod, "MODEL_PATH", artifact_path):
                with TestClient(mod.app, raise_server_exceptions=True) as client:
                    self.client = client
                    yield

    def test_health_ok(self):
        resp = self.client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_predict_full_payload(self):
        payload = {
            "age": 35, "income": 55000, "num_campaigns": 3,
            "num_clicks": 7, "num_opens": 12, "recency_days": 14,
            "tenure_days": 365, "gender": "M", "channel": "email",
            "product_category": "electronics", "region": "north",
        }
        resp = self.client.post("/predict", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert "conversion_prob" in data
        assert 0.0 <= data["conversion_prob"] <= 1.0
        assert data["prediction"] in (0, 1)

    def test_predict_minimal_payload(self):
        """Missing features should be imputed; endpoint must not error."""
        resp = self.client.post("/predict", json={"age": 28})
        assert resp.status_code == 200
        assert "conversion_prob" in resp.json()

    def test_root_redirect(self):
        resp = self.client.get("/")
        assert resp.status_code == 200
        assert "service" in resp.json()
