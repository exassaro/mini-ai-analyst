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


from typing import Dict, List, Any, Optional

def profile_data(file_id: str, target_column: Optional[str] = None) -> Dict[str, Any]:
    """
    Build a profiling report for the CSV identified by *file_id*.
    """
    df = load_csv(file_id)
    n_rows = len(df)

    if n_rows == 0:
        raise ValueError("The uploaded CSV is empty.")

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
            col: {k: (None if pd.isna(v) else float(v)) for k, v in row.items()}
            for col, row in corr.to_dict().items()
        }
    else:
        correlations = {}

    # ── Skewness for numerical columns ───────────────────────────────
    skewness = {}
    for col in num_df.columns:
        val = num_df[col].skew()
        skewness[col] = None if pd.isna(val) else round(float(val), 4)

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
        if df[col].nunique() <= 1
    ]

    # ── Imbalanced columns ───────────────────────────────────────────
    # Categorical columns where one class is > 90%
    imbalanced_columns = []
    cat_df = df.select_dtypes(exclude="number")
    for col in cat_df.columns:
        if cat_df[col].nunique() > 0:
            top_freq = cat_df[col].value_counts(normalize=True).iloc[0]
            if top_freq > 0.9:
                imbalanced_columns.append(col)

    # ── Histograms for numerical columns ────────────────────────────
    histograms = {}
    for col in num_df.columns:
        counts, bins = np.histogram(num_df[col].dropna(), bins=10)
        histograms[col] = {
            "counts": counts.tolist(),
            "bins": bins.tolist()
        }

    # ── Data Leakage Warnings ────────────────────────────────────────
    data_leakage_warnings = []
    if target_column and target_column in df.columns:
        # Check numerical correlations to target
        if target_column in correlations:
            for feature, cr in correlations[target_column].items():
                if feature != target_column and cr is not None and abs(cr) > 0.9:
                    data_leakage_warnings.append(f"Highly correlated with target (>0.9): {feature}")
        # Note: if target is categorical, we could encode and check, 
        # but for simplicity we rely on the numerical correlation matrix if target is numerical.

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
        "skewness": skewness,
        "imbalanced_columns": imbalanced_columns,
        "histograms": histograms,
        "data_leakage_warnings": data_leakage_warnings,
    }
