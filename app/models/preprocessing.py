"""
models/preprocessing.py
=======================
Data preprocessing utilities for ML training and prediction.

Handles: missing values, categorical encoding, datetime feature
extraction, all-null column removal, and edge-case guards.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from typing import Dict, List, Optional, Tuple, Any

from app.core.logger import get_logger

log = get_logger(__name__)


def detect_problem_type(series: pd.Series) -> str:
    """
    Determine whether the target column represents a
    classification or regression problem.

    Rules
    -----
    * object / bool / category dtype  →  classification
    * float dtype                     →  regression (continuous)
    * integer with high uniqueness ratio (> 50% of rows) → regression
    * integer with ≤ 20 unique values AND low ratio      → classification

    Parameters
    ----------
    series : pd.Series
        The target column.

    Returns
    -------
    str
        ``"classification"`` or ``"regression"``.
    """
    non_null = series.dropna()
    if len(non_null) == 0:
        return "classification"  # safe default for empty

    if series.dtype in ("object", "bool", "category"):
        return "classification"

    # Float columns are almost always continuous / regression
    if pd.api.types.is_float_dtype(series):
        # But floats with very few unique values may be class labels (e.g. 0.0, 1.0)
        if non_null.nunique() <= 10:
            return "classification"
        return "regression"

    # For integer columns, use both absolute and ratio-based checks
    n_unique = non_null.nunique()
    ratio = n_unique / len(non_null) if len(non_null) > 0 else 0

    # If more than half the values are unique, likely regression
    if ratio > 0.5:
        return "regression"

    # Low cardinality integer → classification
    if n_unique <= 20:
        return "classification"

    return "regression"


def _extract_datetime_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Detect datetime columns and expand them into numeric features.

    For each datetime column, extracts: year, month, day, weekday, hour
    (hour only if any time component is non-midnight).

    Returns
    -------
    df : pd.DataFrame
        DataFrame with datetime columns replaced by numeric features.
    extracted_cols : list[str]
        Names of the original datetime columns that were extracted.
    """
    extracted_cols: List[str] = []

    for col in list(df.columns):
        series = df[col]

        # Already datetime type
        if pd.api.types.is_datetime64_any_dtype(series):
            dt_series = series
        elif series.dtype == "object":
            try:
                dt_series = pd.to_datetime(series)
            except (ValueError, TypeError, OverflowError):
                continue
        else:
            continue

        # Extract features
        prefix = col
        df[f"{prefix}_year"] = dt_series.dt.year.fillna(0).astype(int)
        df[f"{prefix}_month"] = dt_series.dt.month.fillna(0).astype(int)
        df[f"{prefix}_day"] = dt_series.dt.day.fillna(0).astype(int)
        df[f"{prefix}_weekday"] = dt_series.dt.weekday.fillna(0).astype(int)

        # Only add hour if there's actual time info (not all midnight)
        if dt_series.dropna().dt.hour.sum() > 0:
            df[f"{prefix}_hour"] = dt_series.dt.hour.fillna(0).astype(int)

        df = df.drop(columns=[col])
        extracted_cols.append(col)
        log.info("Extracted datetime features from column '%s'", col)

    return df, extracted_cols


def preprocess_dataframe(
    df: pd.DataFrame,
    target_column: str,
    label_encoders: Optional[Dict[str, LabelEncoder]] = None,
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

    Raises
    ------
    ValueError
        If the target column has no valid (non-null) values, or if
        a classification target has fewer than 2 classes.
    """
    if label_encoders is None:
        label_encoders = {}

    df = df.copy()

    # ── Separate target ──────────────────────────────────────────────
    y = df.pop(target_column)

    # ── Validate target ──────────────────────────────────────────────
    y_non_null = y.dropna()
    if len(y_non_null) == 0:
        raise ValueError(
            f"Target column '{target_column}' has no valid (non-null) values."
        )

    problem_type = detect_problem_type(y)
    if problem_type == "classification" and y_non_null.nunique() < 2:
        raise ValueError(
            f"Target column '{target_column}' has only "
            f"{y_non_null.nunique()} unique class(es). "
            f"Classification requires at least 2 classes."
        )

    # ── Drop all-null feature columns ────────────────────────────────
    all_null_cols = [col for col in df.columns if df[col].isnull().all()]
    if all_null_cols:
        log.warning("Dropping all-null columns: %s", all_null_cols)
        df = df.drop(columns=all_null_cols)

    # ── Extract datetime features ────────────────────────────────────
    if fit:
        df, dt_cols = _extract_datetime_features(df)
        label_encoders["__datetime_cols__"] = dt_cols  # type: ignore[assignment]
    else:
        dt_cols = label_encoders.get("__datetime_cols__", [])
        if dt_cols:
            df, _ = _extract_datetime_features(df)

    # ── Fill missing values ──────────────────────────────────────────
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            col_mean = df[col].mean()
            if pd.isna(col_mean):
                df[col] = df[col].fillna(0)
            else:
                df[col] = df[col].fillna(col_mean)
        else:
            mode_vals = df[col].mode()
            fill_val = mode_vals.iloc[0] if len(mode_vals) > 0 else "UNKNOWN"
            df[col] = df[col].fillna(fill_val)

    # ── Encode categorical features ──────────────────────────────────
    for col in df.select_dtypes(include=["object", "category", "bool"]).columns:
        if fit:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le
        else:
            le = label_encoders.get(col)
            if le is not None:
                mapping = {label: idx for idx, label in enumerate(le.classes_)}
                df[col] = df[col].astype(str).map(
                    lambda val, m=mapping: m.get(val, -1)
                )
            else:
                df[col] = 0  # unknown column — safe fallback

    # ── Fill target NaNs ─────────────────────────────────────────────
    if y.isnull().any():
        # Drop rows where target is null (can't train on them)
        valid_mask = y.notna()
        y = y[valid_mask]
        df = df.loc[valid_mask]
        log.warning("Dropped %d rows with null target values.",
                     int((~valid_mask).sum()))

    # ── Encode target if categorical ─────────────────────────────────
    if y.dtype == "object" or y.dtype.name == "category":
        if fit:
            le_target = LabelEncoder()
            y = pd.Series(
                le_target.fit_transform(y.astype(str)),
                name=target_column,
                index=y.index,
            )
            label_encoders["__target__"] = le_target
        else:
            le_target = label_encoders.get("__target__")
            if le_target is not None:
                y = pd.Series(
                    le_target.transform(y.astype(str)),
                    name=target_column,
                    index=y.index,
                )

    log.info("Preprocessing complete – X shape: %s, y shape: %s", df.shape, y.shape)
    return df, y, label_encoders


def preprocess_input(
    data: List[Dict[str, Any]],
    feature_columns: List[str],
    label_encoders: Dict[str, Any],
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
    pd.DataFrame
        Ready for model.predict().

    Raises
    ------
    ValueError
        If the input data is empty.
    """
    if not data:
        raise ValueError("Prediction input data is empty.")

    df = pd.DataFrame(data)

    # ── Extract datetime features if needed ──────────────────────────
    dt_cols = label_encoders.get("__datetime_cols__", [])
    if dt_cols:
        df, _ = _extract_datetime_features(df)

    # ── Ensure same columns in same order ────────────────────────────
    missing_cols = [col for col in feature_columns if col not in df.columns]
    extra_cols = [col for col in df.columns if col not in feature_columns]

    for col in missing_cols:
        df[col] = 0  # fill missing columns with safe default

    if extra_cols:
        log.warning("Ignoring extra columns not seen during training: %s", extra_cols)

    df = df[feature_columns]

    # ── Fill missing values ──────────────────────────────────────────
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(0)
        else:
            df[col] = df[col].fillna("UNKNOWN")

    # ── Encode categoricals ──────────────────────────────────────────
    for col in df.select_dtypes(include=["object", "category", "bool"]).columns:
        le = label_encoders.get(col)
        if le is not None:
            mapping = {label: idx for idx, label in enumerate(le.classes_)}
            df[col] = df[col].astype(str).map(lambda val, m=mapping: m.get(val, -1))
        else:
            df[col] = 0

    return df
