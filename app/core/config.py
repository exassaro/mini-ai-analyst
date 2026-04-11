"""
core/config.py
==============
Centralised application settings.
All paths are relative to the project root (one level above `app/`).
"""

import os

# Project root = parent of the `app/` package directory
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class _Settings:
    """Simple settings container (no external deps required)."""

    APP_NAME: str = "Mini AI Analyst"
    APP_VERSION: str = "1.0.0"

    # ── Upload limits ────────────────────────────────────────────────
    MAX_UPLOAD_SIZE_MB: int = 50  # megabytes

    # ── Storage paths ────────────────────────────────────────────────
    UPLOAD_DIR: str = os.path.join(_PROJECT_ROOT, "app", "storage", "uploads")
    MODEL_DIR: str = os.path.join(_PROJECT_ROOT, "app", "storage", "models")

    # ── ML defaults ──────────────────────────────────────────────────
    TEST_SIZE: float = 0.2
    RANDOM_STATE: int = 42

    # ── Outlier detection ────────────────────────────────────────────
    IQR_MULTIPLIER: float = 1.5

    # ── High-cardinality / constant column thresholds ────────────────
    HIGH_CARDINALITY_RATIO: float = 0.8   # unique/total > 80 %
    CONSTANT_COLUMN_NUNIQUE: int = 1      # only 1 unique value


# Singleton-style instance used everywhere via `from app.core.config import settings`
settings = _Settings()
