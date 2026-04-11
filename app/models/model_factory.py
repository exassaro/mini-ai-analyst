"""
models/model_factory.py
=======================
Select and instantiate the appropriate scikit-learn estimator
based on the detected problem type.
"""

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from app.core.config import settings
from app.core.logger import get_logger

log = get_logger(__name__)


def get_model(problem_type: str, n_samples: int):
    """
    Return an sklearn estimator.

    Strategy
    --------
    * Small datasets (< 5 000 rows):
        - classification → LogisticRegression
        - regression     → LinearRegression
    * Larger datasets:
        - classification → RandomForestClassifier
        - regression     → RandomForestRegressor

    Parameters
    ----------
    problem_type : str
        ``"classification"`` or ``"regression"``.
    n_samples : int
        Number of training rows (used to pick model complexity).
    """
    if problem_type == "classification":
        if n_samples < 5_000:
            log.info("Selected LogisticRegression (n=%d)", n_samples)
            return LogisticRegression(max_iter=1000, random_state=settings.RANDOM_STATE)
        log.info("Selected RandomForestClassifier (n=%d)", n_samples)
        return RandomForestClassifier(
            n_estimators=100,
            random_state=settings.RANDOM_STATE,
            n_jobs=-1,
        )
    else:
        if n_samples < 5_000:
            log.info("Selected LinearRegression (n=%d)", n_samples)
            return LinearRegression()
        log.info("Selected RandomForestRegressor (n=%d)", n_samples)
        return RandomForestRegressor(
            n_estimators=100,
            random_state=settings.RANDOM_STATE,
            n_jobs=-1,
        )
