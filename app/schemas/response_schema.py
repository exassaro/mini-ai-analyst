"""
schemas/response_schema.py
==========================
Generic / shared Pydantic response models.
"""

from pydantic import BaseModel
from typing import Any, Dict, List, Optional


class ProfileResponse(BaseModel):
    """Returned by GET /profile."""
    file_id: str
    shape: List[int]                          # [rows, cols]
    columns: List[str]
    column_types: Dict[str, str]
    null_percentage: Dict[str, float]
    unique_counts: Dict[str, int]
    numeric_correlations: Dict[str, Dict[str, float]]
    outlier_counts: Dict[str, int]
    high_cardinality_columns: List[str]
    constant_columns: List[str]


class SummaryResponse(BaseModel):
    """Returned by GET /summary."""
    file_id: str
    model_id: str
    dataset_shape: List[int]
    target_column: str
    problem_type: str
    top_correlated_features: List[Dict[str, Any]]
    model_performance: Dict[str, float]
    summary_text: str


class ErrorResponse(BaseModel):
    """Standard error body."""
    detail: str
