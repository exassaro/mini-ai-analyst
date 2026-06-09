"""
services/training_service.py
============================
Orchestrates the full train pipeline:
load → validate → preprocess → split → train → evaluate → persist.

Includes target auto-inference, minimum data guards, and
feature importance extraction.
"""

import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from typing import Dict, Any, Optional, List

from app.core.config import settings
from app.core.utils import load_csv, generate_uuid, get_model_path
from app.core.logger import get_logger
from app.models.preprocessing import detect_problem_type, preprocess_dataframe
from app.models.model_factory import get_model
from app.models.evaluation import evaluate_model

log = get_logger(__name__)

# Common names for target columns (case-insensitive matching)
_TARGET_NAME_HINTS = [
    "target", "label", "class", "y", "output",
    "outcome", "result", "prediction", "category",
]

# Minimum requirements for training
_MIN_ROWS_TRAIN = 5
_MIN_FEATURES = 1


def _auto_infer_target(columns: List[str]) -> Optional[str]:
    """
    Attempt to auto-detect the target column by matching common names.

    Parameters
    ----------
    columns : list[str]
        Column names from the dataset.

    Returns
    -------
    str or None
        The matched column name, or None if no match found.
    """
    col_lower_map = {col.lower().strip(): col for col in columns}
    for hint in _TARGET_NAME_HINTS:
        if hint in col_lower_map:
            return col_lower_map[hint]
    return None


def _extract_feature_importances(
    model: Any,
    feature_columns: List[str],
) -> Optional[Dict[str, float]]:
    """
    Extract feature importances from a trained model.

    Supports:
    - Tree-based models (`.feature_importances_`)
    - Linear models (`.coef_`)

    Returns
    -------
    dict or None
        Feature name → importance (normalized to sum=1), sorted
        descending. None if extraction is not possible.
    """
    importances = None

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        coef = model.coef_
        if coef.ndim > 1:
            # Multi-class: average absolute coefficients across classes
            importances = np.mean(np.abs(coef), axis=0)
        else:
            importances = np.abs(coef)

    if importances is None:
        return None

    # Normalize to sum = 1
    total = importances.sum()
    if total == 0:
        return {col: 0.0 for col in feature_columns}

    normalized = importances / total
    result = {
        col: round(float(val), 4)
        for col, val in zip(feature_columns, normalized)
    }
    # Sort descending
    return dict(sorted(result.items(), key=lambda x: x[1], reverse=True))


def train_model(
    file_id: str,
    target_column: Optional[str] = None,
    features: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    End-to-end training pipeline.

    Parameters
    ----------
    file_id : str
        UUID of the uploaded CSV.
    target_column : str, optional
        Name of the target column. If None, will attempt auto-inference.
    features : list[str], optional
        Subset of feature columns to use. If None, uses all non-target columns.

    Returns
    -------
    dict
        model_id, problem_type, metrics, features, target,
        feature_importances, message

    Raises
    ------
    FileNotFoundError
        If the CSV does not exist on disk.
    ValueError
        If validation fails (missing target, insufficient data, etc.).
    """
    # 1. Load data
    df = load_csv(file_id)

    # 2. Target column resolution
    if not target_column:
        target_column = _auto_infer_target(list(df.columns))
        if target_column is None:
            available = ", ".join(df.columns.tolist()[:20])
            raise ValueError(
                f"No target column specified and auto-inference failed. "
                f"Available columns: [{available}]. "
                f"Please specify a target_column."
            )
        log.info("Auto-inferred target column: '%s'", target_column)

    if target_column not in df.columns:
        raise ValueError(
            f"Target column '{target_column}' not found in dataset. "
            f"Available columns: {list(df.columns)}"
        )

    # 3. Minimum data guards
    if len(df) < _MIN_ROWS_TRAIN:
        raise ValueError(
            f"Insufficient training data: need at least {_MIN_ROWS_TRAIN} rows, "
            f"have {len(df)}."
        )

    non_null_target = df[target_column].dropna()
    if len(non_null_target) < _MIN_ROWS_TRAIN:
        raise ValueError(
            f"Insufficient non-null target values: need at least "
            f"{_MIN_ROWS_TRAIN}, have {len(non_null_target)}."
        )

    # 4. Detect problem type
    problem_type = detect_problem_type(df[target_column])
    log.info("Detected problem type: %s", problem_type)

    # 5. Feature selection (if specified)
    if features is not None and len(features) > 0:
        columns_to_keep = list(set(features + [target_column]))
        columns_to_keep = [col for col in columns_to_keep if col in df.columns]
        missing_feats = [f for f in features if f not in df.columns]
        if missing_feats:
            log.warning("Requested features not found in data: %s", missing_feats)
        if target_column not in columns_to_keep:
            raise ValueError(
                f"Target column '{target_column}' is missing after feature filtering."
            )
        df = df[columns_to_keep]

    # Check minimum features
    n_feature_cols = len(df.columns) - 1  # minus target
    if n_feature_cols < _MIN_FEATURES:
        raise ValueError(
            f"Insufficient feature columns: need at least {_MIN_FEATURES}, "
            f"have {n_feature_cols}."
        )

    # 6. Preprocess
    X, y, label_encoders = preprocess_dataframe(df, target_column, fit=True)

    if len(X) < _MIN_ROWS_TRAIN:
        raise ValueError(
            f"Insufficient training data after preprocessing: "
            f"have {len(X)} rows (need ≥{_MIN_ROWS_TRAIN})."
        )

    # 7. Train/test split
    test_size = settings.TEST_SIZE
    # Ensure at least 1 sample in both train and test
    min_test = max(1, int(len(X) * test_size))
    min_train = len(X) - min_test
    if min_train < 2 or min_test < 1:
        log.warning(
            "Dataset too small for standard split (%d rows). "
            "Using full dataset for both train and test.", len(X)
        )
        X_train, X_test, y_train, y_test = X, X, y, y
    else:
        stratify = y if problem_type == "classification" and y.nunique() >= 2 else None
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=test_size,
                random_state=settings.RANDOM_STATE,
                stratify=stratify,
            )
        except ValueError:
            # Stratification can fail if a class has too few samples
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=test_size,
                random_state=settings.RANDOM_STATE,
            )

    # 8. Select and train model
    model = get_model(problem_type, n_samples=len(X_train))
    model.fit(X_train, y_train)

    # 9. Evaluate
    metrics = evaluate_model(model, X_test, y_test, problem_type)

    # 10. Extract feature importances
    feature_importances = _extract_feature_importances(model, list(X.columns))

    # 11. Persist model artefact
    model_id = generate_uuid()
    artefact = {
        "model": model,
        "label_encoders": label_encoders,
        "feature_columns": list(X.columns),
        "problem_type": problem_type,
        "target_column": target_column,
        "metrics": metrics,
        "feature_importances": feature_importances,
        "file_id": file_id,
    }
    joblib.dump(artefact, get_model_path(model_id))
    log.info("Model saved  model_id=%s", model_id)

    return {
        "model_id": model_id,
        "problem_type": problem_type,
        "metrics": metrics,
        "features": list(X.columns),
        "target": target_column,
        "feature_importances": feature_importances,
        "message": "Model trained successfully",
    }
