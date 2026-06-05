"""FastAPI application serving the fine-tuned intent classifier."""

from __future__ import annotations

import os
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(title="intent-classifier-service", version="0.1.0")

# Set CHECKPOINT_DIR env var before launching to point at a trained model.
_CHECKPOINT_DIR: Optional[str] = os.getenv("CHECKPOINT_DIR")
_model = None
_tokenizer = None


class PredictRequest(BaseModel):
    """Request body for the /predict endpoint."""

    text: str = Field(..., min_length=1, description="Utterance to classify.")


class PredictResponse(BaseModel):
    """Response body for the /predict endpoint."""

    intent: str = Field(..., description="Top predicted intent name.")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Softmax probability of the top class.")
    text: str = Field(..., description="Echo of the input utterance.")


@app.on_event("startup")
def _load_model() -> None:
    """Load model and tokenizer from CHECKPOINT_DIR at startup.

    Skips loading when CHECKPOINT_DIR is not set (useful during testing).
    """
    global _model, _tokenizer
    if _CHECKPOINT_DIR is None:
        return
    from src.model import load_checkpoint

    _model, _tokenizer = load_checkpoint(_CHECKPOINT_DIR)


@app.get("/health")
def health() -> dict:
    """Return service health and whether the model is loaded."""
    return {
        "status": "ok",
        "model_loaded": _model is not None,
    }


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    """Classify an utterance and return the top intent with confidence.

    Parameters
    ----------
    req:
        JSON body with a ``text`` field.

    Returns
    -------
    PredictResponse with ``intent``, ``confidence``, and ``text``.

    Raises
    ------
    HTTPException 503 if the model has not been loaded.
    """
    if _model is None or _tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Set CHECKPOINT_DIR.")
    from src.model import predict_intent

    intent, confidence = predict_intent(req.text, _model, _tokenizer)
    return PredictResponse(intent=intent, confidence=confidence, text=req.text)
