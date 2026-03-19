from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class PipelineStartRequest(BaseModel):
    """POST /api/pipeline/start body."""

    models: list[str] = Field(
        ...,
        description='Models to build: ["PD"], ["LGD"], ["EAD"], or ["PD","LGD","EAD","EL"]',
    )
    db_path: Optional[str] = Field(None, description="Override database path")
    config_overrides: Optional[dict[str, Any]] = Field(
        None, description="Optional agent parameter overrides"
    )


class PipelineStartResponse(BaseModel):
    run_id: str
    sse_url: str


class PipelineStatus(BaseModel):
    run_id: str
    status: str  # running | completed | failed
    current_agent: Optional[str] = None
    completed_agents: list[str] = []
    failed_agent: Optional[str] = None
    progress_pct: float = 0.0
    error: Optional[str] = None


class DatasetInfo(BaseModel):
    db_path: str
    table_name: str
    row_count: int
    column_count: int
    columns: list[str]
    sample_rows: list[dict[str, Any]] = []
