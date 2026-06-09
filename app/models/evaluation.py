"""
models/evaluation.py
====================
Evaluate trained models and return metrics.

Includes edge-case guards for empty test sets and single-sample tests.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
    max_error,
    r2_score,
)
from typing import Dict

from app.core.logger import get_logger

log = get_logger(__name__)


def evaluate_model(model, X_test, y_test, problem_type: str) -> Dict[str, float]:
    """
    Compute evaluation metrics for a trained model.

    Classification → accuracy, balanced_accuracy, precision, recall, f1_weighted, f1_macro
    Regression     → mse, rmse, mae, mape, max_error, r2

    Parameters
    ----------
    model : sklearn estimator
        The trained model.
    X_test : pd.DataFrame
        Test feature matrix.
    y_test : pd.Series
        Test target vector.
    problem_type : str
        ``"classification"`` or ``"regression"``.

    Returns
    -------
    Dict[str, float]
        Evaluation metrics. Returns safe fallbacks if the test set
        is too small for reliable evaluation.
    """
    if len(X_test) == 0:
        log.warning("Empty test set — returning default metrics.")
        if problem_type == "classification":
            return {"accuracy": 0.0, "balanced_accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1_weighted": 0.0, "f1_macro": 0.0}
        return {"mse": 0.0, "rmse": 0.0, "mae": 0.0, "mape": 0.0, "max_error": 0.0, "r2": 0.0}

    try:
        y_pred = model.predict(X_test)
    except Exception as exc:
        log.error("Prediction failed during evaluation: %s", exc)
        if problem_type == "classification":
            return {"accuracy": 0.0, "balanced_accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1_weighted": 0.0, "f1_macro": 0.0}
        return {"mse": 0.0, "rmse": 0.0, "mae": 0.0, "mape": 0.0, "max_error": 0.0, "r2": 0.0}

    if problem_type == "classification":
        metrics = {
            "accuracy": round(accuracy_score(y_test, y_pred), 4),
            "balanced_accuracy": round(balanced_accuracy_score(y_test, y_pred), 4),
            "precision": round(
                precision_score(y_test, y_pred, average="weighted", zero_division=0), 4
            ),
            "recall": round(
                recall_score(y_test, y_pred, average="weighted", zero_division=0), 4
            ),
            "f1_weighted": round(
                f1_score(y_test, y_pred, average="weighted", zero_division=0), 4
            ),
            "f1_macro": round(
                f1_score(y_test, y_pred, average="macro", zero_division=0), 4
            ),
        }
    else:
        mse = float(mean_squared_error(y_test, y_pred))
        rmse = float(np.sqrt(mse))
        mae = float(mean_absolute_error(y_test, y_pred))
        
        try:
            mape = float(mean_absolute_percentage_error(y_test, y_pred))
        except Exception:
            mape = 0.0
            
        try:
            max_err = float(max_error(y_test, y_pred))
        except Exception:
            max_err = 0.0

        # R² can be meaningless with 1 sample
        if len(y_test) > 1:
            r2 = float(r2_score(y_test, y_pred))
        else:
            r2 = 0.0
            log.warning("R² not reliable with only 1 test sample.")

        metrics = {
            "mse": round(mse, 4),
            "rmse": round(rmse, 4),
            "mae": round(mae, 4),
            "mape": round(mape, 4),
            "max_error": round(max_err, 4),
            "r2": round(r2, 4),
        }

    log.info("Evaluation metrics: %s", metrics)
    return metrics
