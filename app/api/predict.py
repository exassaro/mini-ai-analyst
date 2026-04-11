"""
api/predict.py
==============
POST /predict — run predictions with a saved model.
"""

from fastapi import APIRouter, HTTPException

from app.schemas.predict_schema import PredictRequest, PredictResponse
from app.services.prediction_service import predict

router = APIRouter()


@router.post("", response_model=PredictResponse)
def run_predict(req: PredictRequest):
    """
    Predict using a trained model.

    * Accepts ``model_id`` and a list of row dicts.
    * Returns predictions (and probabilities for classification).
    """
    try:
        result = predict(req.model_id, req.data)
        return PredictResponse(**result)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
