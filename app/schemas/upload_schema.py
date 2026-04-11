"""
schemas/upload_schema.py
========================
Pydantic models for the /upload endpoint.
"""

from pydantic import BaseModel


class UploadResponse(BaseModel):
    """Returned after a successful CSV upload."""
    file_id: str
    filename: str
    message: str = "File uploaded successfully"
