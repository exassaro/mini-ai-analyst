"""
api/train.py
============
POST /train — train a model on an uploaded CSV.
"""

from fastapi import APIRouter, HTTPException

from app.schemas.train_schema import TrainRequest, TrainResponse
from app.services.training_service import train_model

router = APIRouter()


@router.post("", response_model=TrainResponse)
def train(req: TrainRequest):
    """
    Train a model.

    * Accepts ``file_id`` and ``target_column``.
    * Auto-detects classification vs regression.
    * Returns ``model_id`` and evaluation metrics.
    """
    try:
        result = train_model(req.file_id, req.target_column, req.features)
        return TrainResponse(**result)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
