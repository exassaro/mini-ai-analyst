"""
models/evaluation.py
====================
Evaluate trained models and return metrics.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    mean_squared_error,
    r2_score,
)
from typing import Dict

from app.core.logger import get_logger

log = get_logger(__name__)


def evaluate_model(model, X_test, y_test, problem_type: str) -> Dict[str, float]:
    """
    Compute evaluation metrics for a trained model.

    Classification → accuracy, precision, recall, f1_weighted
    Regression     → rmse, r2
    """
    y_pred = model.predict(X_test)

    if problem_type == "classification":
        metrics = {
            "accuracy": round(accuracy_score(y_test, y_pred), 4),
            "precision": round(
                precision_score(y_test, y_pred, average="weighted", zero_division=0), 4
            ),
            "recall": round(
                recall_score(y_test, y_pred, average="weighted", zero_division=0), 4
            ),
            "f1_weighted": round(
                f1_score(y_test, y_pred, average="weighted", zero_division=0), 4
            ),
        }
    else:
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        metrics = {
            "rmse": round(rmse, 4),
            "r2": round(r2_score(y_test, y_pred), 4),
        }

    log.info("Evaluation metrics: %s", metrics)
    return metrics
