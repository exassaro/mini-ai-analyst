"""
api/profile.py
==============
GET /profile?file_id=... — return data profiling statistics.
"""

from fastapi import APIRouter, HTTPException, Query

from app.schemas.response_schema import ProfileResponse
from app.services.profiling_service import profile_data

router = APIRouter()


from typing import Optional

@router.get("", response_model=ProfileResponse)
def get_profile(
    file_id: str = Query(..., description="UUID of the uploaded CSV"),
    target_column: Optional[str] = Query(None, description="Optional target column for leakage detection")
):
    """
    Profile the uploaded CSV.

    Returns column types, null %, unique counts, correlations,
    outlier counts, high-cardinality flags, constant-column flags,
    skewness, imbalanced columns, and data leakage warnings.
    """
    try:
        return profile_data(file_id, target_column)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
