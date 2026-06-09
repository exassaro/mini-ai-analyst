"""
services/summary_service.py
===========================
Generate a rule-based natural-language summary combining
dataset profile and model performance.

Includes feature importance integration and edge-case guards.
"""

import joblib
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional

from app.core.utils import load_csv, get_model_path, model_exists, file_exists
from app.core.logger import get_logger

log = get_logger(__name__)


def _safe_float(val: Any) -> Optional[float]:
    """Convert to float safely, returning None for NaN/Inf."""
    if val is None:
        return None
    try:
        f = float(val)
        if np.isnan(f) or np.isinf(f):
            return None
        return round(f, 4)
    except (TypeError, ValueError):
        return None


def generate_summary(file_id: str, model_id: str) -> Dict[str, Any]:
    """
    Build a combined dataset + model summary.

    Parameters
    ----------
    file_id : str
        UUID of the uploaded CSV.
    model_id : str
        UUID of the trained model.

    Returns
    -------
    dict
        Compatible with ``SummaryResponse`` schema.

    Raises
    ------
    FileNotFoundError
        If the file or model does not exist.
    """
    if not file_exists(file_id):
        raise FileNotFoundError(f"No file found for file_id: {file_id}")
    if not model_exists(model_id):
        raise FileNotFoundError(f"No model found for model_id: {model_id}")

    df = load_csv(file_id)
    artefact = joblib.load(get_model_path(model_id))

    target_column: str = artefact["target_column"]
    problem_type: str = artefact["problem_type"]
    metrics: Dict[str, float] = artefact["metrics"]
    feature_importances: Optional[Dict[str, float]] = artefact.get("feature_importances")

    warnings: List[str] = []

    # ── Top correlated features with target ──────────────────────────
    num_df = df.select_dtypes(include="number")
    top_corr: List[Dict[str, Any]] = []

    if target_column in num_df.columns and num_df.shape[1] > 1 and len(df) >= 5:
        try:
            corr_series = num_df.corr()[target_column].drop(target_column, errors="ignore").abs()
            # Remove NaN correlations
            corr_series = corr_series.dropna()
            corr_series = corr_series.sort_values(ascending=False).head(5)
            top_corr = [
                {"feature": feat, "correlation": _safe_float(val)}
                for feat, val in corr_series.items()
                if _safe_float(val) is not None
            ]
        except Exception as exc:
            log.warning("Correlation computation failed: %s", exc)
            warnings.append("Could not compute feature correlations.")
    elif target_column not in num_df.columns:
        warnings.append(
            f"Target '{target_column}' is not numeric — "
            f"correlation analysis not applicable."
        )

    # ── Build human-readable summary ─────────────────────────────────
    lines: List[str] = [
        f"Dataset has {df.shape[0]:,} rows and {df.shape[1]:,} columns.",
        f"Target column: '{target_column}' ({problem_type}).",
    ]

    # Model metrics
    if problem_type == "classification":
        n_classes = df[target_column].nunique() if target_column in df.columns else 0
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
            f"MAE: {metrics.get('mae', 'N/A')}  |  "
            f"R²: {metrics.get('r2', 'N/A')}."
        )

    # Top correlated features
    if top_corr:
        feat_str = ", ".join(
            f"{c['feature']} ({c['correlation']})" for c in top_corr[:3]
        )
        lines.append(f"Top correlated features: {feat_str}.")

    # Feature importances from model
    if feature_importances:
        top_feats = list(feature_importances.items())[:5]
        feat_str = ", ".join(
            f"{name} ({imp:.1%})" for name, imp in top_feats
        )
        lines.append(f"Top predictors (by model importance): {feat_str}.")

    # Profiling insights
    from app.services.profiling_service import profile_data
    try:
        profile = profile_data(file_id, target_column=target_column)
        insights: List[str] = []

        if profile.get("imbalanced_columns"):
            insights.append(
                f"Imbalanced classes found in: "
                f"{', '.join(profile['imbalanced_columns'])}."
            )
        if profile.get("high_cardinality_columns"):
            insights.append(
                f"High cardinality in: "
                f"{', '.join(profile['high_cardinality_columns'])}."
            )
        if profile.get("constant_columns"):
            insights.append(
                f"Constant columns detected: "
                f"{', '.join(profile['constant_columns'])}."
            )
        if profile.get("data_leakage_warnings"):
            insights.append(
                "⚠ Data leakage possible: "
                + " ".join(profile["data_leakage_warnings"])
            )

        if insights:
            lines.append("Key insights: " + " ".join(insights))

        # Propagate profiling warnings
        if profile.get("warnings"):
            warnings.extend(profile["warnings"])

    except Exception as exc:
        log.warning("Profiling during summary failed: %s", exc)
        warnings.append("Could not generate profiling insights.")

    summary_text = " ".join(lines)
    log.info("Summary generated for file_id=%s  model_id=%s", file_id, model_id)

    return {
        "file_id": file_id,
        "model_id": model_id,
        "dataset_shape": list(df.shape),
        "target_column": target_column,
        "problem_type": problem_type,
        "top_correlated_features": top_corr,
        "feature_importances": feature_importances,
        "model_performance": metrics,
        "summary_text": summary_text,
        "warnings": warnings,
    }
