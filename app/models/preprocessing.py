"""
models/preprocessing.py
=======================
Data preprocessing utilities for ML training and prediction.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from typing import Dict, List, Tuple, Any

from app.core.logger import get_logger

log = get_logger(__name__)


def detect_problem_type(series: pd.Series) -> str:
    """
    Determine whether the target column represents a
    classification or regression problem.

    Rules
    -----
    * object / bool / category dtype  →  classification
    * numeric with ≤ 20 unique values →  classification
    * numeric with > 20 unique values →  regression
    """
    if series.dtype in ("object", "bool", "category"):
        return "classification"
    if series.nunique() <= 20:
        return "classification"
    return "regression"


def preprocess_dataframe(
    df: pd.DataFrame,
    target_column: str,
    label_encoders: Dict[str, LabelEncoder] | None = None,
    fit: bool = True,
) -> Tuple[pd.DataFrame, pd.Series, Dict[str, LabelEncoder]]:
    """
    Clean and encode a DataFrame for training or prediction.

    Parameters
    ----------
    df : pd.DataFrame
        Raw data (including the target column for training).
    target_column : str
        Name of the target column.
    label_encoders : dict, optional
        Pre-fitted encoders (used at prediction time).
    fit : bool
        If True, fit new encoders. If False, use the supplied ones.

    Returns
    -------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Target vector.
    label_encoders : dict
        Mapping of column name → fitted LabelEncoder.
    """
    if label_encoders is None:
        label_encoders = {}

    df = df.copy()

    y = df.pop(target_column)

    # ── Fill missing values ──────────────────────────────────────────
    for col in df.columns:
        if df[col].dtype in ("float64", "int64", "float32", "int32"):
            df[col] = df[col].fillna(df[col].mean())
        else:
            df[col] = df[col].fillna(df[col].mode().iloc[0] if not df[col].mode().empty else "UNKNOWN")

    # ── Encode categorical features ─────────────────────────────────
    for col in df.select_dtypes(include=["object", "category", "bool"]).columns:
        if fit:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le
        else:
            le = label_encoders.get(col)
            if le is not None:
                # Handle unseen labels gracefully
                mapping = {label: idx for idx, label in enumerate(le.classes_)}
                df[col] = df[col].astype(str).map(
                    lambda val, m=mapping: m.get(val, -1)
                )
            else:
                df[col] = 0  # unknown column — safe fallback

    # ── Encode target if categorical ─────────────────────────────────
    if y.dtype == "object" or y.dtype.name == "category":
        if fit:
            le_target = LabelEncoder()
            y = pd.Series(le_target.fit_transform(y.astype(str)), name=target_column)
            label_encoders["__target__"] = le_target
        else:
            le_target = label_encoders.get("__target__")
            if le_target is not None:
                y = pd.Series(le_target.transform(y.astype(str)), name=target_column)

    log.info("Preprocessing complete – X shape: %s, y shape: %s", df.shape, y.shape)
    return df, y, label_encoders


def preprocess_input(
    data: List[Dict[str, Any]],
    feature_columns: List[str],
    label_encoders: Dict[str, LabelEncoder],
) -> pd.DataFrame:
    """
    Preprocess raw JSON rows for prediction (no target column).

    Parameters
    ----------
    data : list[dict]
        Rows of feature values.
    feature_columns : list[str]
        Expected column order from training.
    label_encoders : dict
        Fitted encoders from training.

    Returns
    -------
    pd.DataFrame ready for model.predict()
    """
    df = pd.DataFrame(data)

    # Ensure same columns in same order; fill missing cols with 0
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0

    df = df[feature_columns]

    # Fill missing values
    for col in df.columns:
        if df[col].dtype in ("float64", "int64", "float32", "int32"):
            df[col] = df[col].fillna(0)
        else:
            df[col] = df[col].fillna("UNKNOWN")

    # Encode categoricals
    for col in df.select_dtypes(include=["object", "category", "bool"]).columns:
        le = label_encoders.get(col)
        if le is not None:
            mapping = {label: idx for idx, label in enumerate(le.classes_)}
            df[col] = df[col].astype(str).map(lambda val, m=mapping: m.get(val, -1))
        else:
            df[col] = 0

    return df
