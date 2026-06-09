"""
schemas/upload_schema.py
========================
Pydantic models for the /upload endpoint.

The response includes lightweight schema inference so the frontend
can display column info immediately after upload.
"""

from pydantic import BaseModel
from typing import Dict, List


class ColumnInfo(BaseModel):
    """Per-column metadata returned after upload."""
    dtype: str
    semantic_type: str
    null_percentage: float
    unique_count: int
    is_high_cardinality: bool = False
    is_constant: bool = False


class UploadResponse(BaseModel):
    """Returned after a successful CSV upload."""
    file_id: str
    filename: str
    message: str = "File uploaded successfully"
    shape: List[int] = []
    columns: List[str] = []
    column_info: Dict[str, ColumnInfo] = {}
    high_cardinality_columns: List[str] = []
    constant_columns: List[str] = []
