"""
tests/test_predict.py
=====================
Tests for /predict and /summary endpoints.
"""

import io
import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

CLASSIFICATION_CSV = (
    "sepal_length,sepal_width,petal_length,petal_width,species\n"
    "5.1,3.5,1.4,0.2,setosa\n"
    "4.9,3.0,1.4,0.2,setosa\n"
    "7.0,3.2,4.7,1.4,versicolor\n"
    "6.4,3.2,4.5,1.5,versicolor\n"
    "6.3,3.3,6.0,2.5,virginica\n"
    "5.8,2.7,5.1,1.9,virginica\n"
    "5.0,3.4,1.5,0.2,setosa\n"
    "6.7,3.1,4.4,1.4,versicolor\n"
    "6.0,2.9,4.5,1.5,versicolor\n"
    "7.2,3.6,6.1,2.5,virginica\n"
)


def _upload_and_train(csv_text: str, target: str):
    """Upload, train, and return (file_id, model_id)."""
    resp = client.post(
        "/upload",
        files={"file": ("data.csv", io.BytesIO(csv_text.encode()), "text/csv")},
    )
    file_id = resp.json()["file_id"]

    resp = client.post("/train", json={"file_id": file_id, "target_column": target})
    model_id = resp.json()["model_id"]

    return file_id, model_id


class TestPredict:
    def test_predict_classification(self):
        fid, mid = _upload_and_train(CLASSIFICATION_CSV, "species")
        resp = client.post("/predict", json={
            "model_id": mid,
            "data": [
                {"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2},
                {"sepal_length": 6.3, "sepal_width": 3.3, "petal_length": 6.0, "petal_width": 2.5},
            ],
        })
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["predictions"]) == 2
        assert data["probabilities"] is not None

    def test_predict_missing_model(self):
        resp = client.post("/predict", json={
            "model_id": "no-such-model",
            "data": [{"a": 1}],
        })
        assert resp.status_code == 404


class TestSummary:
    def test_summary_success(self):
        fid, mid = _upload_and_train(CLASSIFICATION_CSV, "species")
        resp = client.get(f"/summary?file_id={fid}&model_id={mid}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["problem_type"] == "classification"
        assert "summary_text" in data
        assert data["dataset_shape"][0] == 10

    def test_summary_missing_file(self):
        resp = client.get("/summary?file_id=bad&model_id=bad")
        assert resp.status_code == 404
