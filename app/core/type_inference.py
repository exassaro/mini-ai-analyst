"""
core/type_inference.py
======================
Semantic column-type inference for uploaded datasets.

Maps raw pandas dtypes to human-readable categories:
  categorical, numerical, datetime, boolean, text
"""

import pandas as pd
import numpy as np
from typing import Dict

from app.core.config import settings
from app.core.logger import get_logger

log = get_logger(__name__)

# Column names that hint at datetime content
_DATETIME_NAME_HINTS = {
    "date", "datetime", "timestamp", "time", "created", "updated",
    "created_at", "updated_at", "start_date", "end_date", "dob",
    "birth_date", "registration_date",
}


def infer_semantic_type(series: pd.Series) -> str:
    """
    Infer the semantic type of a single column.

    Returns
    -------
    str
        One of: ``"boolean"``, ``"datetime"``, ``"numerical"``,
        ``"categorical"``, ``"text"``.
    """
    col_name = (series.name or "").lower().strip().replace(" ", "_")
    non_null = series.dropna()

    if len(non_null) == 0:
        return "categorical"  # safe fallback for all-null

    dtype = series.dtype

    # ── Boolean ──────────────────────────────────────────────────────
    if dtype == "bool" or dtype.name == "boolean":
        return "boolean"
    # Object columns with exactly 2 unique values that look boolean
    if dtype == "object" and non_null.nunique() == 2:
        vals = set(non_null.astype(str).str.strip().str.lower().unique())
        if vals <= {"true", "false", "yes", "no", "0", "1", "t", "f", "y", "n"}:
            return "boolean"

    # ── Datetime ─────────────────────────────────────────────────────
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    # Heuristic: try parsing object columns that look like dates
    if dtype == "object" and col_name in _DATETIME_NAME_HINTS:
        try:
            sample = non_null.head(20)
            pd.to_datetime(sample)
            return "datetime"
        except (ValueError, TypeError, OverflowError):
            pass

    # ── Numerical ────────────────────────────────────────────────────
    if pd.api.types.is_numeric_dtype(series):
        return "numerical"

    # ── High-cardinality text vs categorical ─────────────────────────
    n_unique = non_null.nunique()
    n_total = len(non_null)
    avg_len = non_null.astype(str).str.len().mean()

    # Long strings with high cardinality → likely free-text
    if avg_len > 50 and n_unique / max(n_total, 1) > 0.5:
        return "text"

    return "categorical"


def infer_all_column_types(df: pd.DataFrame) -> Dict[str, str]:
    """
    Infer semantic types for every column in a DataFrame.

    Returns
    -------
    Dict[str, str]
        Mapping of column name → semantic type.
    """
    result = {}
    for col in df.columns:
        result[col] = infer_semantic_type(df[col])
    return result
