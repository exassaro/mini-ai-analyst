from fastapi import APIRouter, HTTPException, Query
import joblib

from app.schemas.predict_schema import ModelInfoResponse
from app.core.utils import get_model_path, model_exists

router = APIRouter()

@router.get("", response_model=ModelInfoResponse)
def get_model_info(model_id: str = Query(..., description="UUID of the trained model")):
    """
    Get metadata for a trained model.
    """
    if not model_exists(model_id):
        raise HTTPException(status_code=404, detail=f"No model found for model_id: {model_id}")

    try:
        artefact = joblib.load(get_model_path(model_id))
        return {
            "features": artefact.get("feature_columns", []),
            "target": artefact.get("target_column", ""),
            "problem_type": artefact.get("problem_type", "")
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
