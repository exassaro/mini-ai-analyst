"""
services/training_service.py
============================
Orchestrates the full train pipeline:
load → preprocess → split → train → evaluate → persist.
"""

import joblib
from sklearn.model_selection import train_test_split
from typing import Dict, Any

from app.core.config import settings
from app.core.utils import load_csv, generate_uuid, get_model_path
from app.core.logger import get_logger
from app.models.preprocessing import detect_problem_type, preprocess_dataframe
from app.models.model_factory import get_model
from app.models.evaluation import evaluate_model

log = get_logger(__name__)


def train_model(file_id: str, target_column: str) -> Dict[str, Any]:
    """
    End-to-end training pipeline.

    Returns
    -------
    dict
        model_id, problem_type, metrics, message
    """
    # 1. Load data
    df = load_csv(file_id)

    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataset.")

    # 2. Detect problem type
    problem_type = detect_problem_type(df[target_column])
    log.info("Detected problem type: %s", problem_type)

    # 3. Preprocess
    X, y, label_encoders = preprocess_dataframe(df, target_column, fit=True)

    # 4. Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=settings.TEST_SIZE,
        random_state=settings.RANDOM_STATE,
    )

    # 5. Select and train model
    model = get_model(problem_type, n_samples=len(X_train))
    model.fit(X_train, y_train)

    # 6. Evaluate
    metrics = evaluate_model(model, X_test, y_test, problem_type)

    # 7. Persist model artefact
    model_id = generate_uuid()
    artefact = {
        "model": model,
        "label_encoders": label_encoders,
        "feature_columns": list(X.columns),
        "problem_type": problem_type,
        "target_column": target_column,
        "metrics": metrics,
        "file_id": file_id,
    }
    joblib.dump(artefact, get_model_path(model_id))
    log.info("Model saved  model_id=%s", model_id)

    return {
        "model_id": model_id,
        "problem_type": problem_type,
        "metrics": metrics,
        "message": "Model trained successfully",
    }
