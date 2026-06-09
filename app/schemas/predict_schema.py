"""
schemas/predict_schema.py
=========================
Pydantic models for the /predict and /model-info endpoints.
"""

from pydantic import BaseModel, ConfigDict
from typing import Any, Dict, List, Optional


class PredictRequest(BaseModel):
    """Body sent to POST /predict."""
    model_id: str
    data: List[Dict[str, Any]]   # list of row-dicts

    model_config = ConfigDict(protected_namespaces=())


class PredictResponse(BaseModel):
    """Returned after prediction."""
    predictions: List[Dict[str, Any]]


class ModelInfoResponse(BaseModel):
    """Returned by GET /model-info"""
    features: List[str]
    target: str
    problem_type: str
    feature_importances: Optional[Dict[str, float]] = None
