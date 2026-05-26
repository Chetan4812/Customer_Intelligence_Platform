"""
main.py – Conversion Prediction FastAPI Service
================================================
Loads the trained XGBoost model and serves POST /predict.

Start with:
    uvicorn services.conversion.main:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

import logging
import os
import pickle
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional, List

import asyncio

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


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
MODEL_PATH = BASE_DIR / "models" / "conversion_model.pkl"

# ---------------------------------------------------------------------------
# Global model store (loaded once at startup)
# ---------------------------------------------------------------------------
_MODEL_ARTIFACT: dict[str, Any] = {}


# ---------------------------------------------------------------------------
# Lifespan – load model before accepting requests
# ---------------------------------------------------------------------------
def _load_model_sync() -> dict:
    """Synchronous model load – runs in a thread to avoid blocking event loop."""
    with open(MODEL_PATH, "rb") as fh:
        return pickle.load(fh)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model artifact on startup; clean up on shutdown."""
    global _MODEL_ARTIFACT
    if not MODEL_PATH.exists():
        log.warning(
            "Model not found at %s. "
            "Run services/conversion/train.py first. "
            "Serving with a 'no model' placeholder.",
            MODEL_PATH,
        )
    else:
        log.info("Loading model from %s (this may take a moment)…", MODEL_PATH)
        _MODEL_ARTIFACT = await asyncio.to_thread(_load_model_sync)
        log.info(
            "Loaded model version=%s  features=%s",
            _MODEL_ARTIFACT.get("model_version"),
            _MODEL_ARTIFACT.get("feature_columns"),
        )
    yield
    _MODEL_ARTIFACT.clear()
    log.info("Conversion service shut down.")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Campaign Conversion Prediction Service",
    description=(
        "Predicts the probability that a marketing campaign contact "
        "will convert to a customer."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------
class PredictRequest(BaseModel):
    """Feature payload for a single campaign contact."""

    # Numeric features (all optional; missing values are imputed with 0)
    age: Optional[float] = Field(None, example=35.0)
    income: Optional[float] = Field(None, example=55000.0)
    num_campaigns: Optional[float] = Field(None, example=3.0)
    num_clicks: Optional[float] = Field(None, example=7.0)
    num_opens: Optional[float] = Field(None, example=12.0)
    recency_days: Optional[float] = Field(None, example=14.0)
    tenure_days: Optional[float] = Field(None, example=365.0)

    # Categorical features
    gender: Optional[str] = Field(None, example="M")
    channel: Optional[str] = Field(None, example="email")
    product_category: Optional[str] = Field(None, example="electronics")
    region: Optional[str] = Field(None, example="north")

    class Config:
        extra = "allow"  # Accept additional fields without error


class BatchPredictRequest(BaseModel):
    requests: List[PredictRequest]


class PredictResponse(BaseModel):
    conversion_prob: float = Field(..., description="Probability of conversion [0, 1]")
    prediction: int = Field(..., description="Binary prediction (1=convert, 0=not)")
    model_version: str
    feature_count: int


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_feature_row(payload: PredictRequest) -> pd.DataFrame:
    """
    Convert a PredictRequest into a single-row DataFrame aligned with
    the feature columns the model was trained on.
    """
    if not _MODEL_ARTIFACT:
        raise RuntimeError("Model not loaded.")

    raw = payload.model_dump()
    num_cols: list[str] = _MODEL_ARTIFACT["numeric_features"]
    cat_cols: list[str] = _MODEL_ARTIFACT["categorical_features"]
    encoders: dict = _MODEL_ARTIFACT["label_encoders"]

    row: dict[str, Any] = {}

    # Numerics – impute missing with 0
    for col in num_cols:
        row[col] = float(raw.get(col) or 0.0)

    # Categoricals – encode with fitted LabelEncoder; unknown → -1
    for col in cat_cols:
        val = str(raw.get(col) or "__missing__")
        le = encoders.get(col)
        if le is not None:
            try:
                row[col] = int(le.transform([val])[0])
            except ValueError:
                row[col] = -1  # unseen category
        else:
            row[col] = -1

    return pd.DataFrame([row])[num_cols + cat_cols]


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health", tags=["ops"], summary="Health check")
async def health() -> dict:
    """Returns service health and model load status."""
    return {
        "status": "ok",
        "model_loaded": bool(_MODEL_ARTIFACT),
        "model_version": _MODEL_ARTIFACT.get("model_version", "none"),
    }


@app.post(
    "/batch-score",
    response_model=List[PredictResponse],
    tags=["inference"],
    summary="Batch Predict conversion probability",
)
async def batch_score(payload: BatchPredictRequest) -> List[PredictResponse]:
    if not _MODEL_ARTIFACT:
        raise HTTPException(status_code=503, detail="Model not loaded.")
        
    responses = []
    for req in payload.requests:
        try:
            X = _build_feature_row(req)
            prob = float(_MODEL_ARTIFACT["model"].predict_proba(X)[0, 1])
            pred = int(prob >= 0.5)
            responses.append(PredictResponse(
                conversion_prob=round(prob, 6),
                prediction=pred,
                model_version=_MODEL_ARTIFACT.get("model_version", "unknown"),
                feature_count=len(X.columns)
            ))
        except Exception as exc:
            log.exception("Batch Prediction failed: %s", exc)
            
    return responses

@app.post(
    "/customer-intel",
    tags=["intelligence"],
    summary="Get combined ML conversion band and complaint themes",
)
async def customer_intel(request: PredictRequest) -> dict:
    if not _MODEL_ARTIFACT:
        raise HTTPException(status_code=503, detail="Model not loaded.")
        
    X = _build_feature_row(request)
    prob = float(_MODEL_ARTIFACT["model"].predict_proba(X)[0, 1])
    
    # R11: ML conversion band
    if prob > 0.7:
        band = "High"
    elif prob > 0.4:
        band = "Medium"
    else:
        band = "Low"
        
    # Mocking RAG service integration for top complaint themes with cited IDs
    complaints = [
        {"theme": "Billing Overcharge", "cited_id": "Complaint #1204"},
        {"theme": "Service Outage", "cited_id": "Complaint #1102"}
    ]
    
    return {
        "conversion_probability": round(prob, 4),
        "conversion_band": band,
        "top_complaint_themes": complaints,
        "model_version": _MODEL_ARTIFACT.get("model_version", "unknown")
    }

@app.post(
    "/predict",
    response_model=PredictResponse,
    tags=["inference"],
    summary="Predict campaign conversion probability",
)
async def predict(request: PredictRequest) -> PredictResponse:
    """
    Given campaign contact features, return the probability of conversion
    and a binary prediction (threshold = 0.5).
    """
    if not _MODEL_ARTIFACT:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Run train.py first then restart the service.",
        )

    try:
        X = _build_feature_row(request)
        model = _MODEL_ARTIFACT["model"]
        prob = float(model.predict_proba(X)[0, 1])
        pred = int(prob >= 0.5)
    except Exception as exc:
        log.exception("Prediction failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return PredictResponse(
        conversion_prob=round(prob, 6),
        prediction=pred,
        model_version=_MODEL_ARTIFACT.get("model_version", "unknown"),
        feature_count=len(X.columns),
    )


@app.get("/", include_in_schema=False)
async def root():
    return JSONResponse({"service": "conversion-prediction", "docs": "/docs"})
