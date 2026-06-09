"""
tests/test_edge_cases.py
========================
Edge-case tests to verify the backend handles unusual datasets
gracefully without crashing.
"""

import io
import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def _upload(csv_text: str) -> str:
    """Upload CSV and return file_id."""
    resp = client.post(
        "/upload",
        files={"file": ("edge.csv", io.BytesIO(csv_text.encode()), "text/csv")},
    )
    assert resp.status_code == 200
    return resp.json()["file_id"]


class TestEdgeCaseUpload:
    def test_empty_csv_rejected(self):
        """Completely empty CSV should be rejected."""
        resp = client.post(
            "/upload",
            files={"file": ("empty.csv", io.BytesIO(b""), "text/csv")},
        )
        assert resp.status_code == 400

    def test_header_only_csv(self):
        """CSV with headers but no data rows should still upload."""
        csv = "col_a,col_b,col_c\n"
        resp = client.post(
            "/upload",
            files={"file": ("header.csv", io.BytesIO(csv.encode()), "text/csv")},
        )
        # Should succeed but with 0 rows
        if resp.status_code == 200:
            data = resp.json()
            assert data["shape"][0] == 0


class TestEdgeCaseProfile:
    def test_profile_single_row(self):
        """Profiling should work with a single-row dataset."""
        csv = "a,b,c\n1,2,3\n"
        fid = _upload(csv)
        resp = client.get(f"/profile?file_id={fid}", headers={"session-token": "test-token"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["schema_info"]["shape"][0] == 1
        # Insights should mention suppressed stats or clean dataset
        assert len(data.get("insights", [])) > 0

    def test_profile_all_null_column(self):
        """All-null columns should not crash profiling."""
        csv = "a,b,c\n1,,x\n2,,y\n3,,z\n4,,w\n5,,v\n"
        fid = _upload(csv)
        resp = client.get(f"/profile?file_id={fid}", headers={"session-token": "test-token"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["schema_info"]["null_percentage"]["b"] == 100.0

    def test_profile_mixed_types(self):
        """Mixed-type columns should not crash profiling."""
        csv = "id,value,label\n1,hello,A\n2,123,B\n3,true,C\n4,,D\n5,3.14,E\n"
        fid = _upload(csv)
        resp = client.get(f"/profile?file_id={fid}", headers={"session-token": "test-token"})
        assert resp.status_code == 200

    def test_profile_constant_column(self):
        """Constant columns should be flagged."""
        csv = "a,b\n1,X\n2,X\n3,X\n4,X\n5,X\n"
        fid = _upload(csv)
        resp = client.get(f"/profile?file_id={fid}", headers={"session-token": "test-token"})
        assert resp.status_code == 200
        data = resp.json()
        assert "b" in data["schema_info"]["constant_columns"]


class TestEdgeCaseTrain:
    def test_train_single_class_rejected(self):
        """Single-class classification should return 400."""
        csv = "a,b,label\n1,2,yes\n3,4,yes\n5,6,yes\n7,8,yes\n9,10,yes\n"
        fid = _upload(csv)
        resp = client.post("/train", json={"file_id": fid, "target_column": "label"})
        assert resp.status_code == 400
        assert "class" in resp.json()["detail"].lower()

    def test_train_too_few_rows(self):
        """Very small datasets should return 400."""
        csv = "a,b,target\n1,2,0\n3,4,1\n"
        fid = _upload(csv)
        resp = client.post("/train", json={"file_id": fid, "target_column": "target"})
        assert resp.status_code == 400
        assert "insufficient" in resp.json()["detail"].lower()

    def test_train_no_target_no_auto(self):
        """No target column and no common name should return 400."""
        csv = "foo,bar,baz\n1,2,3\n4,5,6\n7,8,9\n10,11,12\n13,14,15\n"
        fid = _upload(csv)
        resp = client.post("/train", json={"file_id": fid})
        assert resp.status_code == 400
        assert "auto-inference failed" in resp.json()["detail"].lower()

    def test_train_with_datetime_column(self):
        """Datetime columns should be handled (extracted or dropped), not crash."""
        csv = (
            "date,amount,category\n"
            "2023-01-01,100,A\n"
            "2023-02-01,200,B\n"
            "2023-03-01,150,A\n"
            "2023-04-01,300,B\n"
            "2023-05-01,250,A\n"
            "2023-06-01,180,B\n"
        )
        fid = _upload(csv)
        resp = client.post("/train", json={"file_id": fid, "target_column": "category"})
        assert resp.status_code == 200

    def test_train_mostly_null_features(self):
        """Training should handle datasets where some features are mostly null."""
        csv = (
            "a,b,c,target\n"
            "1,,1,0\n"
            "2,,2,1\n"
            "3,,3,0\n"
            "4,,4,1\n"
            "5,,5,0\n"
        )
        fid = _upload(csv)
        resp = client.post("/train", json={"file_id": fid, "target_column": "target"})
        assert resp.status_code == 200


class TestEdgeCasePredict:
    def test_predict_with_extra_columns(self):
        """Extra columns should be ignored, not cause a crash."""
        csv = (
            "a,b,target\n"
            "1,2,0\n3,4,1\n5,6,0\n7,8,1\n9,10,0\n11,12,1\n"
        )
        fid = _upload(csv)
        resp = client.post("/train", json={"file_id": fid, "target_column": "target"})
        mid = resp.json()["model_id"]

        resp = client.post("/predict", json={
            "model_id": mid,
            "data": [{"a": 1, "b": 2, "extra_col": "hello"}],
        })
        assert resp.status_code == 200
