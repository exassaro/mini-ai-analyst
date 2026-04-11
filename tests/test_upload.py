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

    def test_upload_invalid_type(self):
        resp = client.post(
            "/upload",
            files={"file": ("test.json", io.BytesIO(b"{}"), "application/json")},
        )
        assert resp.status_code == 400

    def test_upload_returns_unique_ids(self):
        csv1 = _make_csv()
        csv2 = _make_csv()
        r1 = client.post("/upload", files={"file": ("a.csv", csv1, "text/csv")}).json()
        r2 = client.post("/upload", files={"file": ("b.csv", csv2, "text/csv")}).json()
        assert r1["file_id"] != r2["file_id"]
