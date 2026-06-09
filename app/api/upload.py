"""
api/upload.py
=============
POST /upload — accept a CSV file (max 50 MB), validate, infer schema.
"""

from fastapi import APIRouter, File, UploadFile, HTTPException

from app.core.config import settings
from app.schemas.upload_schema import UploadResponse
from app.services.file_service import save_upload

router = APIRouter()


@router.post("", response_model=UploadResponse)
async def upload_csv(file: UploadFile = File(...)):
    """
    Upload a CSV file.

    * Validates content type and filename extension.
    * Streams file to ``storage/uploads/{file_id}.csv``.
    * Validates the file is a parseable CSV.
    * Returns the generated ``file_id`` with schema inference.
    """
    # ── Validate content type ────────────────────────────────────────
    allowed_types = (
        "text/csv",
        "application/vnd.ms-excel",
        "application/octet-stream",
    )
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type: {file.content_type}. Please upload a CSV.",
        )

    # ── Validate filename extension ──────────────────────────────────
    filename = (file.filename or "").lower()
    if not filename.endswith(".csv"):
        raise HTTPException(
            status_code=400,
            detail="Invalid file extension. Please upload a .csv file.",
        )

    result = await save_upload(file)
    return UploadResponse(**result)
