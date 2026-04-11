"""
api/profile.py
==============
GET /profile?file_id=... — return data profiling statistics.
"""

from fastapi import APIRouter, HTTPException, Query

from app.schemas.response_schema import ProfileResponse
from app.services.profiling_service import profile_data

router = APIRouter()


@router.get("", response_model=ProfileResponse)
def get_profile(file_id: str = Query(..., description="UUID of the uploaded CSV")):
    """
    Profile the uploaded CSV.

    Returns column types, null %, unique counts, correlations,
    outlier counts, high-cardinality flags, and constant-column flags.
    """
    try:
        return profile_data(file_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
