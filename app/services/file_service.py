"""
services/file_service.py
========================
Handles CSV upload persistence.
"""

import os
import aiofiles
from fastapi import HTTPException

from app.core.config import settings
from app.core.utils import generate_uuid, get_upload_path
from app.core.logger import get_logger

log = get_logger(__name__)

MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB

async def save_upload(file) -> dict:
    """
    Stream an uploaded file to disk and return metadata.
    Enforces a 50MB size limit.

    Parameters
    ----------
    file : fastapi.UploadFile

    Returns
    -------
    dict  with keys: file_id, filename
    """
    file_id = generate_uuid()
    dest = get_upload_path(file_id)

    total_size = 0
    async with aiofiles.open(dest, "wb") as out:
        while chunk := await file.read(1024 * 1024):  # 1 MB chunks
            total_size += len(chunk)
            if total_size > MAX_FILE_SIZE:
                # Close the file, then remove it
                await out.close()
                os.remove(dest)
                raise HTTPException(status_code=400, detail="File too large. Maximum size is 50MB.")
            await out.write(chunk)

    log.info("Saved upload %s → %s", file.filename, dest)
    return {"file_id": file_id, "filename": file.filename}
