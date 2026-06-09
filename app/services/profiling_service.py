"""
services/profiling_service.py
=============================
Analyse an uploaded CSV and return profiling statistics with
confidence guards, dynamic insights, and plot recommendations.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional

from app.core.config import settings
from app.core.utils import load_csv
from app.core.logger import get_logger
from app.core.type_inference import infer_all_column_types
from app.schemas.response_schema import Insight, PlotConfig

log = get_logger(__name__)

# ── Minimum sample thresholds for reliable statistics ────────────────
_MIN_ROWS_CORRELATION = 5
_MIN_ROWS_SKEWNESS = 8
_MIN_ROWS_OUTLIER = 10
_MIN_ROWS_IMBALANCE = 10


def _safe_float(val: Any) -> Optional[float]:
    """Convert a value to float, returning None for NaN/Inf."""
    if val is None:
        return None
    try:
        f = float(val)
        if np.isnan(f) or np.isinf(f):
            return None
        return round(f, 4)
    except (TypeError, ValueError):
        return None


def profile_data(file_id: str, target_column: Optional[str] = None) -> Dict[str, Any]:
    """
    Build a dynamic profiling report for the CSV identified by *file_id*.
    """
    df = load_csv(file_id)
    n_rows, n_cols = df.shape
    if n_rows == 0:
        raise ValueError("The uploaded CSV is empty.")

    insights: List[Insight] = []
    plot_recommendations: List[PlotConfig] = []
    column_types = infer_all_column_types(df)

    if target_column and target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataset.")

    # ── 1. Schema-level output ───────────────────────────────────────
    schema_info = {
        "shape": [n_rows, n_cols],
        "columns": list(df.columns),
        "column_types": column_types,
        "null_percentage": {col: round(float(df[col].isnull().sum() / n_rows * 100), 2) for col in df.columns},
        "unique_counts": {col: int(df[col].nunique()) for col in df.columns},
    }

    high_cardinality = [c for c in df.columns if df[c].nunique() / max(n_rows, 1) > settings.HIGH_CARDINALITY_RATIO and column_types[c] == "categorical"]
    schema_info["high_cardinality"] = high_cardinality
    if high_cardinality:
        insights.append(Insight(
            severity="warning", title="High Cardinality", 
            message="Columns have a very high number of unique categories.", 
            affected_columns=high_cardinality, recommendation="Consider dropping these or encoding them properly."
        ))

    constant_cols = [c for c in df.columns if df[c].nunique() <= 1]
    schema_info["constant_columns"] = constant_cols
    if constant_cols:
        insights.append(Insight(
            severity="danger", title="Constant Columns", 
            message="These columns have a single constant value and provide no information.", 
            affected_columns=constant_cols, recommendation="Drop these columns before modeling."
        ))

    # ── Task inference ───────────────────────────────────────────────
    task_type = None
    if target_column:
        target_type = column_types.get(target_column)
        n_unique_target = df[target_column].nunique()
        if n_unique_target <= 1:
            insights.append(Insight(
                severity="danger", title="Invalid Target", 
                message="Target column has 1 or fewer unique values.", 
                affected_columns=[target_column], recommendation="Select a valid target with multiple classes or continuous values."
            ))
        elif target_type in ("categorical", "boolean", "text"):
            task_type = "classification"
        elif target_type == "numerical":
            if pd.api.types.is_float_dtype(df[target_column]) and n_unique_target > 2:
                task_type = "regression"
            elif n_unique_target <= 20:
                task_type = "classification"
            else:
                task_type = "regression"

    # ── 2. Numeric profiling ─────────────────────────────────────────
    numeric_cols = [c for c, t in column_types.items() if t == "numerical"]
    numeric_stats: Dict[str, Any] = {"columns": numeric_cols}
    
    if numeric_cols:
        num_df = df[numeric_cols]
        numeric_stats["zero_percentage"] = {c: round(float((num_df[c] == 0).sum() / n_rows * 100), 2) for c in numeric_cols}
        
        # summary stats, replace NaN with None for JSON serialization safety
        desc = num_df.describe().to_dict()
        safe_desc = {}
        for col, stats in desc.items():
            safe_desc[col] = {k: _safe_float(v) for k, v in stats.items()}
        numeric_stats["summary"] = safe_desc
        
        skewness = {}
        if n_rows >= _MIN_ROWS_SKEWNESS:
            for c in numeric_cols:
                val = _safe_float(num_df[c].skew())
                skewness[c] = val
                if val is not None and abs(val) > 1.5:
                    insights.append(Insight(
                        severity="info", title="High Skewness", 
                        message=f"Column is highly skewed (skew={val}).", 
                        affected_columns=[c], recommendation="Consider log-transform or scaling."
                    ))
        numeric_stats["skewness"] = skewness
        
        outliers = {}
        if n_rows >= _MIN_ROWS_OUTLIER:
            for c in numeric_cols:
                s = num_df[c].dropna()
                if len(s) < _MIN_ROWS_OUTLIER:
                    outliers[c] = 0
                    continue
                q1, q3 = s.quantile(0.25), s.quantile(0.75)
                iqr = q3 - q1
                if iqr == 0:
                    outliers[c] = 0
                    continue
                lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
                outliers[c] = int(((s < lower) | (s > upper)).sum())
                if outliers[c] > 0 and outliers[c] / len(s) > 0.05:
                    insights.append(Insight(
                        severity="warning", title="Many Outliers", 
                        message=f"Column has >5% outliers ({outliers[c]}).", 
                        affected_columns=[c], recommendation="Inspect outliers. Robust scaling might be needed."
                    ))
        numeric_stats["outliers"] = outliers
        
        correlations = {}
        if len(numeric_cols) > 1 and n_rows >= _MIN_ROWS_CORRELATION:
            corr = num_df.corr().round(4)
            correlations = {col: {k: _safe_float(v) for k, v in row.items()} for col, row in corr.to_dict().items()}
        numeric_stats["pairwise_correlations"] = correlations

    # ── 3. Categorical profiling ─────────────────────────────────────
    cat_cols = [c for c, t in column_types.items() if t in ("categorical", "boolean", "text")]
    categorical_stats: Dict[str, Any] = {"columns": cat_cols}
    if cat_cols:
        top_values = {}
        rare_categories = {}
        imbalanced = []
        for c in cat_cols:
            vc = df[c].value_counts()
            if vc.empty:
                continue
            top_values[c] = vc.head(5).to_dict()
            
            if n_rows >= _MIN_ROWS_IMBALANCE:
                top_freq = vc.iloc[0] / len(df[c].dropna())
                if top_freq > 0.9:
                    imbalanced.append(c)
                    
            if len(vc) > 1:
                rare = vc[vc < 5].index.tolist()
                if rare:
                    rare_categories[c] = [str(x) for x in rare[:5]]
                    
        categorical_stats["top_values"] = top_values
        categorical_stats["rare_categories"] = rare_categories
        categorical_stats["imbalanced"] = imbalanced
        if imbalanced:
             insights.append(Insight(
                 severity="warning", title="Imbalanced Categories", 
                 message="These categorical columns are highly imbalanced (>90% one value).", 
                 affected_columns=imbalanced, recommendation="May have low predictive power or require handling."
             ))

    # ── 4. Datetime profiling ────────────────────────────────────────
    dt_cols = [c for c, t in column_types.items() if t == "datetime"]
    datetime_stats: Dict[str, Any] = {"columns": dt_cols}
    if dt_cols:
        min_max = {}
        for c in dt_cols:
            try:
                s = pd.to_datetime(df[c], errors="coerce").dropna()
                if not s.empty:
                    min_max[c] = {"min": str(s.min()), "max": str(s.max())}
            except Exception:
                pass
        datetime_stats["min_max"] = min_max

    # ── 5. Target-aware behavior and Leakage ─────────────────────────
    target_analysis: Dict[str, Any] = {}
    if target_column and task_type:
        target_analysis["task_type"] = task_type
        if task_type == "classification":
            vc = df[target_column].value_counts()
            target_analysis["class_distribution"] = vc.to_dict()
            if len(vc) > 1 and vc.iloc[0] / len(df[target_column].dropna()) > 0.8:
                insights.append(Insight(
                    severity="warning", title="Imbalanced Target", 
                    message="Target class distribution is highly imbalanced.", 
                    affected_columns=[target_column], recommendation="Use stratified splitting and proper metrics (e.g., F1)."
                ))
            
            for nc in numeric_cols:
                 if nc == target_column: continue
                 grouped = df.groupby(target_column)[nc].mean().dropna()
                 plot_recommendations.append(PlotConfig(
                     type="bar", title=f"Avg {nc} by {target_column}", 
                     x_axis=target_column, y_axis=nc,
                     data={"labels": [str(k) for k in grouped.keys()], "values": [_safe_float(v) for v in grouped.values()]}
                 ))
                 
        elif task_type == "regression":
            desc = df[target_column].describe().to_dict()
            target_analysis["target_distribution"] = {k: _safe_float(v) for k, v in desc.items()}
            target_analysis["skewness"] = _safe_float(df[target_column].skew())
            
            if numeric_cols:
                 corrs = []
                 for nc in numeric_cols:
                     if nc == target_column: continue
                     try:
                         val = _safe_float(df[target_column].corr(df[nc]))
                         if val is not None: 
                             corrs.append({"feature": nc, "correlation": val})
                     except Exception:
                         pass
                 corrs.sort(key=lambda x: abs(x["correlation"]), reverse=True)
                 target_analysis["correlation_ranking"] = corrs
                 
                 for cr in corrs[:3]:
                     sample = df[[cr['feature'], target_column]].dropna()
                     if len(sample) > 200: sample = sample.sample(200, random_state=42)
                     scatter_data = [{"x": _safe_float(row[cr['feature']]), "y": _safe_float(row[target_column])} for _, row in sample.iterrows()]
                     plot_recommendations.append(PlotConfig(
                         type="scatter", title=f"{cr['feature']} vs {target_column}", 
                         x_axis=cr['feature'], y_axis=target_column,
                         data=scatter_data
                     ))
                     
        # Leakage checks
        for c in df.columns:
            if c == target_column:
                continue
            
            if df[c].equals(df[target_column]):
                insights.append(Insight(
                    severity="danger", title="Exact Target Duplicate", 
                    message="Column is identical to target.", 
                    affected_columns=[c], recommendation="Drop this column to prevent data leakage."
                ))
                
            elif task_type == "regression" and c in numeric_cols:
                try:
                    cr = _safe_float(df[target_column].corr(df[c]))
                    if cr and abs(cr) > 0.95:
                         insights.append(Insight(
                             severity="danger", title="Near-Perfect Correlation", 
                             message=f"Highly correlated with target (r={cr}).", 
                             affected_columns=[c], recommendation="May be target leakage. Investigate."
                         ))
                except Exception:
                    pass
                
            elif task_type == "classification" and column_types[c] in ("categorical", "boolean"):
                try:
                    grouped = df.groupby(c)[target_column].nunique()
                    if (grouped == 1).all() and df[c].nunique() > 1:
                        insights.append(Insight(
                            severity="danger", title="Perfect Predictor", 
                            message="Each category perfectly maps to a single target class.", 
                            affected_columns=[c], recommendation="Highly likely to be data leakage. Consider dropping."
                        ))
                except Exception:
                    pass
                
    else:
        # No target - EDA
        if numeric_cols:
            for nc in numeric_cols[:2]:
                s = df[nc].dropna()
                if len(s) > 0:
                    counts, bins = np.histogram(s, bins=min(10, len(s.unique())))
                    labels = [f"{bins[i]:.1f} - {bins[i+1]:.1f}" for i in range(len(bins)-1)]
                    plot_recommendations.append(PlotConfig(
                        type="histogram", title=f"Distribution of {nc}", x_axis=nc,
                        data={"labels": labels, "values": [int(x) for x in counts]}
                    ))
        if cat_cols:
            for cc in cat_cols[:2]:
                vc = df[cc].value_counts().head(5)
                plot_recommendations.append(PlotConfig(
                    type="bar", title=f"Top 5 Counts of {cc}", x_axis=cc,
                    data={"labels": [str(x) for x in vc.index], "values": [int(x) for x in vc.values]}
                ))
        
        potential_targets = [c for c in cat_cols if 1 < df[c].nunique() <= 10]
        if potential_targets:
             insights.append(Insight(
                 severity="info", title="Candidate Targets", 
                 message="These categorical columns have a low number of unique values and could be good classification targets.", 
                 affected_columns=potential_targets
             ))

    if not insights:
        insights.append(Insight(
            severity="info", title="Clean Dataset", 
            message="No significant issues found in the dataset.", 
            affected_columns=[]
        ))

    return {
        "file_id": file_id,
        "task_type": task_type,
        "schema_info": schema_info,
        "numeric_stats": numeric_stats,
        "categorical_stats": categorical_stats,
        "datetime_stats": datetime_stats,
        "target_analysis": target_analysis or {},
        "insights": insights,
        "plot_recommendations": plot_recommendations,
    }
