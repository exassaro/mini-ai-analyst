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
    predictions: List[Dict[str, Any]]

class ModelInfoResponse(BaseModel):
    """Returned by GET /model-info"""
    features: List[str]
    target: str
    problem_type: str
