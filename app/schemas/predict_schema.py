"""
schemas/predict_schema.py
=========================
Pydantic models for the /predict endpoint.
"""

from pydantic import BaseModel
from typing import Any, Dict, List, Optional


class PredictRequest(BaseModel):
    """Body sent to POST /predict."""
    model_id: str
    data: List[Dict[str, Any]]   # list of row-dicts


class PredictResponse(BaseModel):
    """Returned after prediction."""
    predictions: List[Any]
    probabilities: Optional[List[List[float]]] = None  # classification only
