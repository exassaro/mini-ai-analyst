"""
core/utils.py
=============
Shared utility helpers used across the application.
"""

import os
import uuid

import pandas as pd

from app.core.config import settings


def generate_uuid() -> str:
    """Return a new random UUID as a string."""
    return str(uuid.uuid4())


def get_upload_path(file_id: str) -> str:
    """Return the absolute path for a stored CSV given its file_id."""
    return os.path.join(settings.UPLOAD_DIR, f"{file_id}.csv")


def get_model_path(model_id: str) -> str:
    """Return the absolute path for a stored model given its model_id."""
    return os.path.join(settings.MODEL_DIR, f"{model_id}.pkl")


def load_csv(file_id: str) -> pd.DataFrame:
    """
    Load a CSV from disk by file_id.

    Raises
    ------
    FileNotFoundError
        If the CSV does not exist on disk.
    """
    path = get_upload_path(file_id)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"No uploaded file found for file_id: {file_id}")
    return pd.read_csv(path)


def file_exists(file_id: str) -> bool:
    """Check whether a CSV with the given file_id exists on disk."""
    return os.path.isfile(get_upload_path(file_id))


def model_exists(model_id: str) -> bool:
    """Check whether a model with the given model_id exists on disk."""
    return os.path.isfile(get_model_path(model_id))
