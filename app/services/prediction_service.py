"""
services/prediction_service.py
==============================
Load a persisted model and run predictions on new data.

Includes schema validation, JSON serialization safety, and
edge-case handling.
"""

import joblib
import numpy as np
from typing import Any, Dict, List

from app.core.utils import get_model_path, model_exists
from app.core.logger import get_logger
from app.models.preprocessing import preprocess_input

log = get_logger(__name__)


def _safe_jsonify(val: Any) -> Any:
    """
    Convert a value to a JSON-safe Python type.

    Handles numpy scalars, NaN, Inf, and other non-serializable types.
    """
    if val is None:
        return None
    if isinstance(val, (np.integer,)):
        return int(val)
    if isinstance(val, (np.floating,)):
        f = float(val)
        if np.isnan(f) or np.isinf(f):
            return None
        return round(f, 4)
    if isinstance(val, np.ndarray):
        return val.tolist()
    if isinstance(val, float):
        if np.isnan(val) or np.isinf(val):
            return None
        return round(val, 4)
    if hasattr(val, "item"):
        return val.item()
    return val


def predict(model_id: str, data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Run inference with a saved model.

    Parameters
    ----------
    model_id : str
        UUID of the persisted model.
    data : list[dict]
        Rows of feature values.

    Returns
    -------
    dict
        predictions: list of dicts with target value (and confidence
        for classification).

    Raises
    ------
    FileNotFoundError
        If the model does not exist.
    ValueError
        If input data is empty or has incompatible schema.
    """
    if not model_exists(model_id):
        raise FileNotFoundError(f"No model found for model_id: {model_id}")

    if not data:
        raise ValueError("Prediction input data is empty.")

    artefact = joblib.load(get_model_path(model_id))
    model = artefact["model"]
    label_encoders = artefact["label_encoders"]
    feature_columns = artefact["feature_columns"]
    problem_type = artefact["problem_type"]
    target_col = artefact.get("target_column", "prediction")

    # ── Schema validation ────────────────────────────────────────────
    input_cols = set()
    for row in data:
        input_cols.update(row.keys())

    missing_features = [c for c in feature_columns if c not in input_cols]
    if missing_features:
        log.warning(
            "Missing features in prediction input (will use defaults): %s",
            missing_features,
        )

    # ── Preprocess ───────────────────────────────────────────────────
    X = preprocess_input(data, feature_columns, label_encoders)

    # ── Predict ──────────────────────────────────────────────────────
    preds = model.predict(X)

    # ── Decode target labels ─────────────────────────────────────────
    le_target = label_encoders.get("__target__")
    if le_target is not None:
        try:
            preds = le_target.inverse_transform(preds.astype(int))
        except (ValueError, IndexError) as exc:
            log.warning("Target decoding failed: %s. Returning raw values.", exc)

    # ── Format predictions ───────────────────────────────────────────
    formatted_preds: List[Dict[str, Any]] = []

    if problem_type == "classification" and hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba(X)
            for i, val in enumerate(preds):
                confidence = _safe_jsonify(np.max(proba[i]))
                formatted_preds.append({
                    target_col: _safe_jsonify(val),
                    "confidence": confidence,
                })
        except Exception as exc:
            log.warning("predict_proba failed: %s. Returning predictions without confidence.", exc)
            for val in preds:
                formatted_preds.append({
                    target_col: _safe_jsonify(val),
                })
    else:
        for val in preds:
            formatted_preds.append({
                target_col: _safe_jsonify(val),
            })

    log.info("Predicted %d rows with model_id=%s", len(data), model_id)
    return {"predictions": formatted_preds}
