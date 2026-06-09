"""
schemas/train_schema.py
=======================
Pydantic models for the /train endpoint.
"""

from pydantic import BaseModel, ConfigDict
from typing import Dict, Optional, List


class TrainRequest(BaseModel):
    """Body sent to POST /train."""
    file_id: str
    target_column: Optional[str] = None  # auto-inferred if not provided
    features: Optional[List[str]] = None


class TrainResponse(BaseModel):
    """Returned after model training completes."""
    model_config = ConfigDict(protected_namespaces=())

    model_id: str
    problem_type: str          # "classification" or "regression"
    metrics: Dict[str, float]
    features: List[str]
    target: str
    feature_importances: Optional[Dict[str, float]] = None
    message: str = "Model trained successfully"
