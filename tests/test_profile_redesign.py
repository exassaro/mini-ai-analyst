"""
tests/test_profile_redesign.py
"""

import io
import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

def _upload(csv_text: str) -> str:
    resp = client.post(
        "/upload",
        files={"file": ("test.csv", io.BytesIO(csv_text.encode()), "text/csv")},
    )
    assert resp.status_code == 200
    return resp.json()["file_id"]

def test_profile_no_target():
    csv = "a,b,c\n1,2,x\n2,3,y\n3,4,z\n4,5,x\n5,6,y\n"
    fid = _upload(csv)
    resp = client.get(f"/profile?file_id={fid}", headers={"session-token": "valid-token"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["task_type"] is None
    assert "schema_info" in data
    assert "insights" in data

def test_profile_classification():
    csv = "a,b,label\n1,2,cls1\n3,4,cls2\n5,6,cls1\n7,8,cls2\n9,10,cls1\n"
    fid = _upload(csv)
    resp = client.get(f"/profile?file_id={fid}&target_column=label", headers={"session-token": "valid-token"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["task_type"] == "classification"
    assert "class_distribution" in data["target_analysis"]

def test_profile_regression():
    csv = "a,b,target\n1,2,10.5\n3,4,12.1\n5,6,11.3\n7,8,15.2\n9,10,14.0\n1,2,10.1\n3,4,11.9\n"
    fid = _upload(csv)
    resp = client.get(f"/profile?file_id={fid}&target_column=target", headers={"session-token": "valid-token"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["task_type"] == "regression"
    assert "target_distribution" in data["target_analysis"]

def test_profile_invalid_token():
    csv = "a,b\n1,2\n"
    fid = _upload(csv)
    resp = client.get(f"/profile?file_id={fid}", headers={"session-token": "expired"})
    assert resp.status_code == 401
    
    resp = client.get(f"/profile?file_id={fid}")
    assert resp.status_code == 422 # missing header

def test_profile_invalid_target():
    csv = "a,b\n1,2\n"
    fid = _upload(csv)
    resp = client.get(f"/profile?file_id={fid}&target_column=not_exist", headers={"session-token": "valid-token"})
    assert resp.status_code == 400
