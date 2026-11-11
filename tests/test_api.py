"""
API tests for OptiSelect FastAPI server.
"""

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from server.main import app

client = TestClient(app)


def test_list_datasets():
    resp = client.get("/api/datasets")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 5
    slugs = {d["slug"] for d in data}
    assert slugs == {"iris", "wine", "breast-cancer", "heart-disease", "wine-quality"}


def test_get_dataset_detail():
    resp = client.get("/api/datasets/heart-disease")
    assert resp.status_code == 200
    data = resp.json()
    assert data["slug"] == "heart-disease"
    assert data["features"] == 13
    assert "class_distribution" in data["charts"]
    assert "correlation_heatmap" in data["charts"]


def test_get_dataset_not_found():
    resp = client.get("/api/datasets/unknown-ds")
    assert resp.status_code == 404


def test_unknown_api_route_returns_404():
    resp = client.get("/api/does-not-exist")
    assert resp.status_code == 404


def test_run_experiment_valid():
    resp = client.post(
        "/api/experiment",
        json={
            "dataset": "iris",
            "classifier": "logistic_regression",
            "method": "selectkbest",
            "k": 3,
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["selected"]["selected_feature_names"]) == 3
    assert "figures" in data
    for key in [
        "accuracy_vs_features",
        "baseline_vs_optimized",
        "feature_importance",
        "confusion_matrix",
        "accuracy_vs_runtime",
    ]:
        assert key in data["figures"]
        fig = data["figures"][key]
        assert fig["width"] > 0
        assert fig["height"] > 0
        assert len(fig["alt"]) > 10


@pytest.mark.parametrize("bad_k", [1, 5, 0, -1])
def test_run_experiment_invalid_k_returns_422(bad_k: int):
    resp = client.post(
        "/api/experiment",
        json={
            "dataset": "iris",
            "classifier": "logistic_regression",
            "method": "selectkbest",
            "k": bad_k,
        },
    )
    assert resp.status_code == 422


def test_final_results():
    resp = client.get("/api/results/final")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["rows"]) == 15
    assert data["csv_url"] == "/results/final_table.csv"
