"""
FastAPI service exposing health & prediction endpoints (Phase 1).
Run with:
    uvicorn api.main:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(title="Crypto Volatility Watcher", version="0.1.0")

DATA_DIR = Path("data")


class PredictionItem(BaseModel):
    coin: str = Field(..., example="bitcoin")
    volatility_probability: float = Field(..., example=0.672312)


class PredictionResponse(BaseModel):
    prediction_date: str = Field(..., example="2025-05-04")
    most_volatile_coin: str = Field(..., example="bitcoin")
    predictions: List[PredictionItem]


@app.get("/", tags=["root"])
def read_root() -> dict[str, str]:
    """Welcome endpoint."""
    return {
        "message": "Welcome to the Crypto‑Volatility‑Watcher MVP (Phase 1). See /predict for latest prediction."
    }


@app.get("/health", tags=["health"])
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/predict", response_model=PredictionResponse, tags=["prediction"])
def get_prediction() -> PredictionResponse:
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    today_file = DATA_DIR / f"predictions_{today}.json"

    if today_file.exists():
        with today_file.open() as fp:
            return json.load(fp)

    # Fallback: most recent prediction file
    files = sorted(DATA_DIR.glob("predictions_*.json"), reverse=True)
    if not files:
        raise HTTPException(
            status_code=404,
            detail="No predictions available. Run the prediction pipeline first.",
        )

    with files[0].open() as fp:
        return json.load(fp)