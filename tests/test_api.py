"""
API tests. extract/chunk_embed/retrieve/generate are all mocked so tests
run offline - no real PDF, no Qdrant Cloud call, no Gemini call.
"""

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    with patch("src.retrieve.Retriever.__init__", return_value=None), \
         patch("src.generate.Generator.__init__", return_value=None):
        from src.api.main import app
        with TestClient(app) as c:
            yield c


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "healthy"


def test_ask_empty_question_returns_422(client):
    resp = client.post("/ask", json={"question": ""})
    assert resp.status_code == 422


def test_ingest_rejects_non_pdf(client):
    resp = client.post(
        "/ingest",
        files={"file": ("not_a_pdf.txt", b"hello", "text/plain")},
    )
    assert resp.status_code == 422
