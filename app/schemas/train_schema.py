"""
schemas/train_schema.py
=======================
Pydantic models for the /train endpoint.
"""

from pydantic import BaseModel
from typing import Dict, Optional, List

class TrainRequest(BaseModel):
    """Body sent to POST /train."""
    file_id: str
    target_column: str
    features: Optional[List[str]] = None

class TrainResponse(BaseModel):
    """Returned after model training completes."""
    model_id: str
    problem_type: str          # "classification" or "regression"
    metrics: Dict[str, float]
    features: List[str]
    target: str
    message: str = "Model trained successfully"
