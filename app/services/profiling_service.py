"""
services/profiling_service.py
=============================
Analyse an uploaded CSV and return profiling statistics.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any

from app.core.config import settings
from app.core.utils import load_csv
from app.core.logger import get_logger

log = get_logger(__name__)


def profile_data(file_id: str) -> Dict[str, Any]:
    """
    Build a profiling report for the CSV identified by *file_id*.

    Returns
    -------
    dict
        shape, column_types, null_percentage, unique_counts,
        numeric_correlations, outlier_counts,
        high_cardinality_columns, constant_columns
    """
    df = load_csv(file_id)
    n_rows = len(df)

    # ── Column types ─────────────────────────────────────────────────
    column_types = {col: str(dtype) for col, dtype in df.dtypes.items()}

    # ── Null percentage ──────────────────────────────────────────────
    null_pct = {
        col: round(float(df[col].isnull().sum() / n_rows * 100), 2)
        for col in df.columns
    }

    # ── Unique counts ────────────────────────────────────────────────
    unique_counts = {col: int(df[col].nunique()) for col in df.columns}

    # ── Numeric correlations ─────────────────────────────────────────
    num_df = df.select_dtypes(include="number")
    if num_df.shape[1] > 1:
        corr = num_df.corr().round(4)
        # convert NaN → null-safe
        correlations = {
            col: {k: (None if pd.isna(v) else v) for k, v in row.items()}
            for col, row in corr.to_dict().items()
        }
    else:
        correlations = {}

    # ── Outlier detection (IQR) ──────────────────────────────────────
    outlier_counts: Dict[str, int] = {}
    for col in num_df.columns:
        q1 = num_df[col].quantile(0.25)
        q3 = num_df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - settings.IQR_MULTIPLIER * iqr
        upper = q3 + settings.IQR_MULTIPLIER * iqr
        outlier_counts[col] = int(((num_df[col] < lower) | (num_df[col] > upper)).sum())

    # ── High-cardinality columns ─────────────────────────────────────
    high_card = [
        col for col in df.columns
        if df[col].nunique() / n_rows > settings.HIGH_CARDINALITY_RATIO
    ]

    # ── Constant columns ────────────────────────────────────────────
    constant = [
        col for col in df.columns
        if df[col].nunique() <= settings.CONSTANT_COLUMN_NUNIQUE
    ]

    log.info("Profiled file_id=%s  shape=%s", file_id, df.shape)

    return {
        "file_id": file_id,
        "shape": list(df.shape),
        "columns": list(df.columns),
        "column_types": column_types,
        "null_percentage": null_pct,
        "unique_counts": unique_counts,
        "numeric_correlations": correlations,
        "outlier_counts": outlier_counts,
        "high_cardinality_columns": high_card,
        "constant_columns": constant,
    }
