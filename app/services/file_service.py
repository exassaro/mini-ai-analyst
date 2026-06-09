"""
services/file_service.py
========================
Handles CSV upload persistence and post-upload schema inference.
"""

import os
import aiofiles
import pandas as pd
from fastapi import HTTPException

from app.core.config import settings
from app.core.utils import generate_uuid, get_upload_path
from app.core.logger import get_logger
from app.core.type_inference import infer_all_column_types

log = get_logger(__name__)

MAX_FILE_SIZE: int = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024  # bytes


async def save_upload(file) -> dict:
    """
    Stream an uploaded file to disk, validate it as CSV, and return
    metadata including lightweight schema inference.

    Parameters
    ----------
    file : fastapi.UploadFile

    Returns
    -------
    dict
        Keys: file_id, filename, shape, columns, column_info,
              high_cardinality_columns, constant_columns

    Raises
    ------
    HTTPException 400
        If the file is too large or not a valid CSV.
    """
    file_id = generate_uuid()
    dest = get_upload_path(file_id)

    # ── Stream to disk with size enforcement ─────────────────────────
    total_size = 0
    try:
        async with aiofiles.open(dest, "wb") as out:
            while chunk := await file.read(1024 * 1024):  # 1 MB chunks
                total_size += len(chunk)
                if total_size > MAX_FILE_SIZE:
                    await out.close()
                    _remove_file(dest)
                    raise HTTPException(
                        status_code=400,
                        detail=f"File too large. Maximum size is {settings.MAX_UPLOAD_SIZE_MB}MB.",
                    )
                await out.write(chunk)
    except HTTPException:
        raise
    except Exception as exc:
        _remove_file(dest)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to save uploaded file: {exc}",
        )

    # ── Validate CSV is parseable ────────────────────────────────────
    try:
        df = pd.read_csv(dest)
    except Exception as exc:
        _remove_file(dest)
        raise HTTPException(
            status_code=400,
            detail=f"Uploaded file is not a valid CSV: {exc}",
        )

    if df.empty or len(df.columns) == 0:
        _remove_file(dest)
        raise HTTPException(
            status_code=400,
            detail="Uploaded CSV is empty or has no columns.",
        )

    # ── Schema inference ─────────────────────────────────────────────
    schema_info = _infer_schema(df)

    log.info("Saved upload %s → %s (%d rows, %d cols)",
             file.filename, dest, df.shape[0], df.shape[1])

    return {
        "file_id": file_id,
        "filename": file.filename,
        **schema_info,
    }


def _infer_schema(df: pd.DataFrame) -> dict:
    """
    Compute lightweight schema inference for the uploaded CSV.

    Returns
    -------
    dict
        shape, columns, column_info, high_cardinality_columns,
        constant_columns
    """
    n_rows = len(df)
    semantic_types = infer_all_column_types(df)

    column_info = {}
    high_cardinality: list = []
    constant: list = []

    for col in df.columns:
        n_unique = int(df[col].nunique())
        null_pct = round(float(df[col].isnull().sum() / max(n_rows, 1) * 100), 2)
        is_high_card = (n_unique / max(n_rows, 1)) > settings.HIGH_CARDINALITY_RATIO
        is_const = n_unique <= settings.CONSTANT_COLUMN_NUNIQUE

        column_info[col] = {
            "dtype": str(df[col].dtype),
            "semantic_type": semantic_types.get(col, "categorical"),
            "null_percentage": null_pct,
            "unique_count": n_unique,
            "is_high_cardinality": is_high_card,
            "is_constant": is_const,
        }

        if is_high_card:
            high_cardinality.append(col)
        if is_const:
            constant.append(col)

    return {
        "shape": list(df.shape),
        "columns": list(df.columns),
        "column_info": column_info,
        "high_cardinality_columns": high_cardinality,
        "constant_columns": constant,
    }


def _remove_file(path: str) -> None:
    """Silently remove a file if it exists."""
    try:
        if os.path.isfile(path):
            os.remove(path)
    except OSError:
        pass
