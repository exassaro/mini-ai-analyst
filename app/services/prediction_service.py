"""
services/prediction_service.py
==============================
Load a persisted model and run predictions on new data.
"""

import joblib
import numpy as np
from typing import Any, Dict, List

from app.core.utils import get_model_path, model_exists
from app.core.logger import get_logger
from app.models.preprocessing import preprocess_input

log = get_logger(__name__)


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
        predictions, probabilities (classification only)
    """
    if not model_exists(model_id):
        raise FileNotFoundError(f"No model found for model_id: {model_id}")

    artefact = joblib.load(get_model_path(model_id))
    model = artefact["model"]
    label_encoders = artefact["label_encoders"]
    feature_columns = artefact["feature_columns"]
    problem_type = artefact["problem_type"]

    # Preprocess incoming rows
    X = preprocess_input(data, feature_columns, label_encoders)

    # Predict
    preds = model.predict(X)

    # Decode target labels back if they were encoded
    le_target = label_encoders.get("__target__")
    if le_target is not None:
        preds = le_target.inverse_transform(preds.astype(int))

    result: Dict[str, Any] = {"predictions": preds.tolist()}

    # Probabilities for classification
    if problem_type == "classification" and hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        result["probabilities"] = np.round(proba, 4).tolist()

    log.info("Predicted %d rows with model_id=%s", len(data), model_id)
    return result
