"""
services/file_service.py
========================
Handles CSV upload persistence.
"""

import os
import aiofiles

from app.core.config import settings
from app.core.utils import generate_uuid, get_upload_path
from app.core.logger import get_logger

log = get_logger(__name__)


async def save_upload(file) -> dict:
    """
    Stream an uploaded file to disk and return metadata.

    Parameters
    ----------
    file : fastapi.UploadFile

    Returns
    -------
    dict  with keys: file_id, filename
    """
    file_id = generate_uuid()
    dest = get_upload_path(file_id)

    async with aiofiles.open(dest, "wb") as out:
        while chunk := await file.read(1024 * 1024):  # 1 MB chunks
            await out.write(chunk)

    log.info("Saved upload %s → %s", file.filename, dest)
    return {"file_id": file_id, "filename": file.filename}
