from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, HTTPException

from backend.config import get_settings
from middleware.schemas.models import ModelMetrics, ModelVersion

router = APIRouter(prefix="/api/models", tags=["models"])


def _load_registry() -> list[dict]:
    settings = get_settings()
    registry = settings.output_abs_path / "model_registry.json"
    if not registry.exists():
        return []
    return json.loads(registry.read_text())


@router.get("/list", response_model=list[ModelVersion])
async def list_models():
    entries = _load_registry()
    return [
        ModelVersion(
            model_id=e["model_id"],
            model_type=e["model_type"],
            algorithm=e["algorithm"],
            champion=e.get("champion", False),
            created_at=e.get("created_at", ""),
            metrics=e.get("metrics", {}),
        )
        for e in entries
    ]


@router.get("/{model_id}/metrics", response_model=ModelMetrics)
async def model_metrics(model_id: str):
    entries = _load_registry()
    for e in entries:
        if e["model_id"] == model_id:
            return ModelMetrics(
                model_id=e["model_id"],
                model_type=e["model_type"],
                algorithm=e["algorithm"],
                hyperparameters=e.get("hyperparameters", {}),
                metrics=e.get("metrics", {}),
                feature_count=e.get("feature_count", 0),
                training_time_s=e.get("training_time_s"),
            )
    raise HTTPException(404, f"Model {model_id} not found")
