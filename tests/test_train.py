"""
tests/test_train.py
===================
Tests for /profile and /train endpoints.
"""

import io
import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

# A small classification dataset
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

# A small regression dataset
REGRESSION_CSV = (
    "size,bedrooms,price\n"
    "1400,3,245000\n"
    "1600,3,312000\n"
    "1700,4,279000\n"
    "1875,4,308000\n"
    "1100,2,199000\n"
    "1550,3,250000\n"
    "2350,5,405000\n"
    "2450,5,324000\n"
    "1425,3,219000\n"
    "1700,3,255000\n"
)


def _upload(csv_text: str) -> str:
    """Upload CSV and return file_id."""
    resp = client.post(
        "/upload",
        files={"file": ("data.csv", io.BytesIO(csv_text.encode()), "text/csv")},
    )
    assert resp.status_code == 200
    return resp.json()["file_id"]


class TestProfile:
    def test_profile_success(self):
        fid = _upload(CLASSIFICATION_CSV)
        resp = client.get(f"/profile?file_id={fid}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["shape"][0] == 10
        assert "species" in data["columns"]
        assert "column_types" in data

    def test_profile_missing_file(self):
        resp = client.get("/profile?file_id=nonexistent-id")
        assert resp.status_code == 404


class TestTrain:
    def test_train_classification(self):
        fid = _upload(CLASSIFICATION_CSV)
        resp = client.post("/train", json={"file_id": fid, "target_column": "species"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["problem_type"] == "classification"
        assert "accuracy" in data["metrics"]
        assert "model_id" in data

    def test_train_regression(self):
        fid = _upload(REGRESSION_CSV)
        resp = client.post("/train", json={"file_id": fid, "target_column": "price"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["problem_type"] == "regression"
        assert "rmse" in data["metrics"]

    def test_train_missing_target(self):
        fid = _upload(CLASSIFICATION_CSV)
        resp = client.post("/train", json={"file_id": fid, "target_column": "nonexistent"})
        assert resp.status_code == 400

    def test_train_missing_file(self):
        resp = client.post("/train", json={"file_id": "bad-id", "target_column": "x"})
        assert resp.status_code == 404
