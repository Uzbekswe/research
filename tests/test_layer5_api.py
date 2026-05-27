"""Layer 5 API tests — no real LLM or HTTP calls.

Uses FastAPI's TestClient for synchronous tests.
Background research tasks are fire-and-forget; tests only verify
the HTTP layer and task lifecycle state machine.
"""

import pytest
from fastapi.testclient import TestClient

from backend.main import app

client = TestClient(app)


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "researcher_ready" in data


def test_start_research_returns_202():
    response = client.post("/research", json={
        "query": "What is machine learning and how does it work?",
        "max_sections": 2
    })
    assert response.status_code == 202
    data = response.json()
    assert "task_id" in data
    assert data["status"] == "queued"
    assert "/research/" in data["poll_url"]


def test_get_nonexistent_task_returns_404():
    response = client.get("/research/nonexistent-id-12345")
    assert response.status_code == 404


def test_list_tasks_returns_list():
    response = client.get("/research")
    assert response.status_code == 200
    assert isinstance(response.json(), list)


def test_start_research_short_query_rejected():
    response = client.post("/research", json={
        "query": "AI"   # too short — min_length=10
    })
    assert response.status_code == 422   # validation error


def test_full_task_lifecycle():
    # Create task
    response = client.post("/research", json={
        "query": "What is machine learning and how does it work?"
    })
    assert response.status_code == 202
    task_id = response.json()["task_id"]

    # Immediately poll — should be queued or running (or failed if LLM not available)
    response = client.get(f"/research/{task_id}")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ["queued", "running", "failed"]
    assert data["task_id"] == task_id
    assert data["query"] == "What is machine learning and how does it work?"

    # Delete task
    response = client.delete(f"/research/{task_id}")
    assert response.status_code == 204

    # Verify deleted
    response = client.get(f"/research/{task_id}")
    assert response.status_code == 404
