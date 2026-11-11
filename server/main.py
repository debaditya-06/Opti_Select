"""
FastAPI server for OptiSelect.

Serves the React frontend as an SPA (with fallback for client-side routing),
exposes the experiment API, and serves generated chart images from results/.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator, model_validator

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import experiments

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="OptiSelect API",
    description="Feature selection experiment runner",
    version="1.0.0",
)

app.add_middleware(GZipMiddleware, minimum_size=500)


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class ExperimentRequest(BaseModel):
    dataset: str = Field(..., description="Dataset slug")
    classifier: str = Field(default="logistic_regression", description="Classifier key")
    method: str = Field(..., description="Feature selection method")
    k: int = Field(..., ge=2, description="Number of features to select")

    @field_validator("dataset")
    @classmethod
    def valid_dataset(cls, v: str) -> str:
        if v not in experiments.DATASET_META:
            raise ValueError(f"Unknown dataset: {v}. Valid: {list(experiments.DATASET_META.keys())}")
        return v

    @field_validator("classifier")
    @classmethod
    def valid_classifier(cls, v: str) -> str:
        if v not in experiments.CLASSIFIERS:
            raise ValueError(f"Unknown classifier: {v}. Valid: {list(experiments.CLASSIFIERS.keys())}")
        return v

    @field_validator("method")
    @classmethod
    def valid_method(cls, v: str) -> str:
        if v not in experiments.METHODS:
            raise ValueError(f"Unknown method: {v}. Valid: {list(experiments.METHODS.keys())}")
        return v

    @model_validator(mode="after")
    def valid_k_for_dataset(self) -> "ExperimentRequest":
        max_k = experiments.DATASET_META[self.dataset]["n_features"]
        if self.k < 2 or self.k > max_k:
            raise ValueError(f"k must be between 2 and {max_k} for {self.dataset}, got {self.k}")
        return self


# ---------------------------------------------------------------------------
# API routes
# ---------------------------------------------------------------------------

@app.get("/api/datasets")
def list_datasets() -> list[dict[str, Any]]:
    """List all available datasets with real row, feature, and class counts."""
    result = []
    for slug in experiments.DATASET_META:
        try:
            info = experiments.get_dataset_info(slug)
            result.append({
                "slug": slug,
                "name": info["name"],
                "rows": info["rows"],
                "n_features": info["features"],
                "n_classes": info["n_classes"],
                "class_names": info["class_names"],
                "source": info["source"],
                "description": info["description"],
                "example_sizes": info["example_sizes"],
                "missing_values": info["missing_values"],
            })
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to load dataset {slug}: {e}")
    return result


@app.get("/api/datasets/{slug}")
def get_dataset(slug: str) -> dict[str, Any]:
    """Get detailed info and charts for a specific dataset."""
    if slug not in experiments.DATASET_META:
        raise HTTPException(status_code=404, detail=f"Dataset not found: {slug}")
    try:
        info = experiments.get_dataset_info(slug)
        class_dist = experiments.generate_class_distribution(slug)
        corr_heatmap = experiments.generate_correlation_heatmap(slug)
        info["charts"] = {
            "class_distribution": class_dist,
            "correlation_heatmap": corr_heatmap,
        }
        return info
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/experiment")
def run_experiment_endpoint(req: ExperimentRequest) -> dict[str, Any]:
    """Run a feature selection experiment and generate the five mandatory figures."""
    try:
        result = experiments.run_experiment(
            dataset_slug=req.dataset,
            classifier_key=req.classifier,
            method_key=req.method,
            k=req.k,
        )
        figures = experiments.generate_all_figures(result)
        result["figures"] = figures
        return result
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/results/final")
def get_final_results() -> dict[str, Any]:
    """Get the final comparison table from results/final_table.csv."""
    csv_path = experiments.RESULTS_DIR / "final_table.csv"
    if not csv_path.exists():
        try:
            experiments.run_all_experiments()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to generate results: {e}")

    df = pd.read_csv(csv_path, dtype=str)
    return {
        "columns": list(df.columns),
        "rows": df.to_dict(orient="records"),
        "csv_url": "/results/final_table.csv",
    }


@app.api_route("/api/{rest:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
def api_not_found(rest: str) -> JSONResponse:
    """Return proper HTTP 404 for unknown API paths."""
    return JSONResponse(status_code=404, content={"detail": f"API endpoint not found: /api/{rest}"})


# ---------------------------------------------------------------------------
# Static files: results/ images and CSV
# ---------------------------------------------------------------------------

app.mount("/results", StaticFiles(directory=str(experiments.RESULTS_DIR)), name="results")


# ---------------------------------------------------------------------------
# SEO metadata files
# ---------------------------------------------------------------------------

@app.get("/robots.txt")
def robots_txt(request: Request) -> Response:
    base = str(request.base_url).rstrip("/")
    content = f"User-agent: *\nAllow: /\n\nSitemap: {base}/sitemap.xml\n"
    return Response(content=content, media_type="text/plain")


@app.get("/sitemap.xml")
def sitemap_xml(request: Request) -> Response:
    base = str(request.base_url).rstrip("/")
    pages = [
        "/",
        "/experiment",
        "/datasets",
        "/datasets/iris",
        "/datasets/wine",
        "/datasets/breast-cancer",
        "/datasets/heart-disease",
        "/datasets/wine-quality",
        "/methodology",
        "/results",
    ]
    entries = "\n".join(
        f"  <url><loc>{base}{p}</loc></url>"
        for p in pages
    )
    xml = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n'
        f"{entries}\n"
        "</urlset>\n"
    )
    return Response(content=xml, media_type="application/xml")


# ---------------------------------------------------------------------------
# SPA fallback — serve the built React frontend
# ---------------------------------------------------------------------------

DIST_DIR = (Path(__file__).resolve().parent.parent / "web" / "dist").resolve()


@app.get("/{full_path:path}")
async def spa_fallback(full_path: str) -> Response:
    """Serve static assets from dist/ with cache headers or fall back to index.html."""
    if full_path:
        try:
            candidate = (DIST_DIR / full_path).resolve()
            if candidate.is_relative_to(DIST_DIR) and candidate.is_file():
                headers = {}
                if full_path.startswith("assets/"):
                    headers["Cache-Control"] = "public, max-age=31536000, immutable"
                else:
                    headers["Cache-Control"] = "public, max-age=3600"
                return FileResponse(candidate, headers=headers)
        except (ValueError, OSError):
            pass

    index_path = DIST_DIR / "index.html"
    if index_path.exists():
        return FileResponse(
            index_path,
            headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
        )

    return JSONResponse(
        status_code=503,
        content={"detail": "Frontend not built. Run: cd web && npm run build"},
    )
