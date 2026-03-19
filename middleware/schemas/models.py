from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel


class ModelVersion(BaseModel):
    model_id: str
    model_type: str  # pd | lgd_stage1 | lgd_stage2 | ead
    algorithm: str
    champion: bool
    created_at: str
    metrics: dict[str, float]


class ModelMetrics(BaseModel):
    model_id: str
    model_type: str
    algorithm: str
    hyperparameters: dict[str, Any]
    metrics: dict[str, float]
    feature_count: int
    training_time_s: Optional[float] = None
