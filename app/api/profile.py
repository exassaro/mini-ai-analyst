"""
api/profile.py
==============
GET /profile?file_id=... — return data profiling statistics.
"""

from fastapi import APIRouter, HTTPException, Query, Header
from typing import Optional

from app.schemas.response_schema import ProfileResponse
from app.services.profiling_service import profile_data

router = APIRouter()

@router.get("", response_model=ProfileResponse)
def get_profile(
    file_id: str = Query(..., description="UUID of the uploaded CSV"),
    target_column: Optional[str] = Query(None, description="Optional target column for leakage detection"),
    session_token: str = Header(..., description="Session token for authentication")
):
    """
    Profile the uploaded CSV.
    """
    if not session_token or session_token == "expired":
        raise HTTPException(status_code=401, detail="Invalid or expired session token")

    try:
        return profile_data(file_id, target_column)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
