"""
tests/test_upload.py
====================
Tests for the /upload endpoint.
"""

import io
import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def _make_csv(content: str = "a,b,c\n1,2,3\n4,5,6\n"):
    """Return a CSV-like in-memory file for upload."""
    return io.BytesIO(content.encode("utf-8"))


class TestUpload:
    def test_upload_success(self):
        csv = _make_csv()
        resp = client.post(
            "/upload",
            files={"file": ("test.csv", csv, "text/csv")},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "file_id" in data
        assert data["filename"] == "test.csv"
        assert data["message"] == "File uploaded successfully"

    def test_upload_returns_schema_inference(self):
        csv = _make_csv("name,age,active\nAlice,30,true\nBob,25,false\n")
        resp = client.post(
            "/upload",
            files={"file": ("data.csv", csv, "text/csv")},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["shape"] == [2, 3]
        assert "name" in data["columns"]
        assert "column_info" in data
        assert "name" in data["column_info"]
        info = data["column_info"]["name"]
        assert "semantic_type" in info
        assert "null_percentage" in info
        assert "unique_count" in info

    def test_upload_invalid_type(self):
        resp = client.post(
            "/upload",
            files={"file": ("test.json", io.BytesIO(b"{}"), "application/json")},
        )
        assert resp.status_code == 400

    def test_upload_invalid_extension(self):
        resp = client.post(
            "/upload",
            files={"file": ("test.txt", io.BytesIO(b"a,b\n1,2\n"), "text/csv")},
        )
        assert resp.status_code == 400

    def test_upload_invalid_csv(self):
        resp = client.post(
            "/upload",
            files={"file": ("test.csv", io.BytesIO(b"\x00\x01\x02"), "text/csv")},
        )
        # Should fail with 400 (not a valid CSV) or 200 if pandas can parse it
        # The key point is it should NOT return 500
        assert resp.status_code in (200, 400)

    def test_upload_returns_unique_ids(self):
        csv1 = _make_csv()
        csv2 = _make_csv()
        r1 = client.post("/upload", files={"file": ("a.csv", csv1, "text/csv")}).json()
        r2 = client.post("/upload", files={"file": ("b.csv", csv2, "text/csv")}).json()
        assert r1["file_id"] != r2["file_id"]
