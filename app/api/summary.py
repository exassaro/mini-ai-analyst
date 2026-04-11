"""
api/summary.py
==============
GET /summary?file_id=...&model_id=... — combined dataset + model summary.
"""

from fastapi import APIRouter, HTTPException, Query

from app.schemas.response_schema import SummaryResponse
from app.services.summary_service import generate_summary

router = APIRouter()


@router.get("", response_model=SummaryResponse)
def get_summary(
    file_id: str = Query(..., description="UUID of the uploaded CSV"),
    model_id: str = Query(..., description="UUID of the trained model"),
):
    """
    Generate a rule-based summary of dataset and model performance.
    """
    try:
        return generate_summary(file_id, model_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
