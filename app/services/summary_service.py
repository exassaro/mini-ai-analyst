"""
services/summary_service.py
===========================
Generate a rule-based natural-language summary combining
dataset profile and model performance.
"""

import joblib
import pandas as pd
from typing import Any, Dict, List

from app.core.utils import load_csv, get_model_path, model_exists, file_exists
from app.core.logger import get_logger

log = get_logger(__name__)


def generate_summary(file_id: str, model_id: str) -> Dict[str, Any]:
    """
    Build a combined dataset + model summary.

    Returns
    -------
    dict
        dataset_shape, target_column, problem_type,
        top_correlated_features, model_performance, summary_text
    """
    if not file_exists(file_id):
        raise FileNotFoundError(f"No file found for file_id: {file_id}")
    if not model_exists(model_id):
        raise FileNotFoundError(f"No model found for model_id: {model_id}")

    df = load_csv(file_id)
    artefact = joblib.load(get_model_path(model_id))

    target_column = artefact["target_column"]
    problem_type = artefact["problem_type"]
    metrics = artefact["metrics"]

    # ── Top correlated features with target ──────────────────────────
    num_df = df.select_dtypes(include="number")
    top_corr: List[Dict[str, Any]] = []
    if target_column in num_df.columns and num_df.shape[1] > 1:
        corr_series = num_df.corr()[target_column].drop(target_column, errors="ignore").abs()
        corr_series = corr_series.sort_values(ascending=False).head(5)
        top_corr = [
            {"feature": feat, "correlation": round(float(val), 4)}
            for feat, val in corr_series.items()
        ]

    from app.services.profiling_service import profile_data
    # Run a quick profiling to get insights
    profile = profile_data(file_id, target_column=target_column)

    # ── Build human-readable summary ─────────────────────────────────
    lines = [
        f"Dataset has {df.shape[0]:,} rows and {df.shape[1]:,} columns.",
        f"Target column: '{target_column}' ({problem_type}).",
    ]

    # Model metrics
    if problem_type == "classification":
        n_classes = df[target_column].nunique()
        lines.append(f"Number of classes: {n_classes}.")
        lines.append(
            f"Model accuracy: {metrics.get('accuracy', 'N/A')}  |  "
            f"Precision: {metrics.get('precision', 'N/A')}  |  "
            f"Recall: {metrics.get('recall', 'N/A')}  |  "
            f"F1 (weighted): {metrics.get('f1_weighted', 'N/A')}."
        )
    else:
        lines.append(
            f"Model RMSE: {metrics.get('rmse', 'N/A')}  |  "
            f"R²: {metrics.get('r2', 'N/A')}."
        )

    # Key insights
    if top_corr:
        feat_str = ", ".join(f"{c['feature']} ({c['correlation']})" for c in top_corr[:3])
        lines.append(f"Top correlated features: {feat_str}.")

    insights = []
    if profile.get("imbalanced_columns"):
        insights.append(f"Imbalanced classes found in: {', '.join(profile['imbalanced_columns'])}.")
    if profile.get("high_cardinality_columns"):
        insights.append(f"High cardinality in: {', '.join(profile['high_cardinality_columns'])}.")
    if profile.get("constant_columns"):
        insights.append(f"Constant columns detected: {', '.join(profile['constant_columns'])}.")
    if profile.get("data_leakage_warnings"):
        insights.append("Warning Data Leakage possible: " + " ".join(profile["data_leakage_warnings"]))
    
    if insights:
        lines.append("Key Insights from Profiling: " + " ".join(insights))

    summary_text = " ".join(lines)
    log.info("Summary generated for file_id=%s  model_id=%s", file_id, model_id)

    return {
        "file_id": file_id,
        "model_id": model_id,
        "dataset_shape": list(df.shape),
        "target_column": target_column,
        "problem_type": problem_type,
        "top_correlated_features": top_corr,
        "model_performance": metrics,
        "summary_text": summary_text,
    }
