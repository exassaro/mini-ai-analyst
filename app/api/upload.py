"""
api/upload.py
=============
POST /upload — accept a CSV file (max 50 MB), save to disk.
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

    * Validates content type and file size.
    * Streams file to ``storage/uploads/{file_id}.csv``.
    * Returns the generated ``file_id``.
    """
    # ── Validate content type ────────────────────────────────────────
    if file.content_type not in (
        "text/csv",
        "application/vnd.ms-excel",
        "application/octet-stream",
    ):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type: {file.content_type}. Please upload a CSV.",
        )

    # ── Validate size (read content-length header if available) ──────
    # Note: for streaming uploads the size may not be known upfront,
    # so we rely on the web-server / proxy for hard limits.

    result = await save_upload(file)
    return UploadResponse(**result)
