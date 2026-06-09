"""
schemas/response_schema.py
==========================
Generic / shared Pydantic response models.
"""

from pydantic import BaseModel, ConfigDict
from typing import Any, Dict, List, Optional


class Insight(BaseModel):
    severity: str  # "info", "warning", "danger"
    title: str
    message: str
    affected_columns: List[str]
    recommendation: Optional[str] = None

class PlotConfig(BaseModel):
    type: str
    title: str
    x_axis: str
    y_axis: Optional[str] = None
    data: Optional[Any] = None

class ProfileResponse(BaseModel):
    """Returned by GET /profile."""
    file_id: str
    task_type: Optional[str] = None
    schema_info: Dict[str, Any]
    numeric_stats: Dict[str, Any]
    categorical_stats: Dict[str, Any]
    datetime_stats: Dict[str, Any]
    target_analysis: Optional[Dict[str, Any]] = None
    insights: List[Insight]
    plot_recommendations: List[PlotConfig]


class SummaryResponse(BaseModel):
    """Returned by GET /summary."""
    model_config = ConfigDict(protected_namespaces=())

    file_id: str
    model_id: str
    dataset_shape: List[int]
    target_column: str
    problem_type: str
    top_correlated_features: List[Dict[str, Any]]
    feature_importances: Optional[Dict[str, float]] = None
    model_performance: Dict[str, float]
    summary_text: str
    warnings: List[str] = []


class ErrorResponse(BaseModel):
    """Standard error body."""
    detail: str
