from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, BackgroundTasks, HTTPException
from fastapi.responses import StreamingResponse

from backend.config import get_settings
from middleware.schemas.pipeline import (
    PipelineStartRequest,
    PipelineStartResponse,
    PipelineStatus,
)

router = APIRouter(prefix="/api/pipeline", tags=["pipeline"])

# In-memory stores keyed by run_id
_queues: dict[str, asyncio.Queue] = {}
_statuses: dict[str, PipelineStatus] = {}


async def _run_pipeline(run_id: str, request: PipelineStartRequest) -> None:
    """Background task that drives the orchestrator and feeds the SSE queue."""
    from backend.orchestrator import PipelineOrchestrator

    queue = _queues[run_id]
    settings = get_settings()
    orchestrator = PipelineOrchestrator(
        run_id=run_id,
        models=request.models,
        event_queue=queue,
        settings=settings,
        db_path_override=request.db_path,
        config_overrides=request.config_overrides,
    )
    try:
        await orchestrator.run()
    except Exception as exc:
        await queue.put({"event": "pipeline_error", "data": {"error": str(exc)}})
    finally:
        await queue.put(None)  # sentinel to close SSE stream


@router.post("/start", response_model=PipelineStartResponse)
async def start_pipeline(
    request: PipelineStartRequest, background_tasks: BackgroundTasks
):
    valid_models = {"PD", "LGD", "EAD", "EL"}
    for m in request.models:
        if m.upper() not in valid_models:
            raise HTTPException(400, f"Invalid model: {m}. Must be one of {valid_models}")

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_id = f"pipeline_run_{ts}"

    _queues[run_id] = asyncio.Queue()
    _statuses[run_id] = PipelineStatus(run_id=run_id, status="running")

    background_tasks.add_task(_run_pipeline, run_id, request)

    return PipelineStartResponse(
        run_id=run_id,
        sse_url=f"/api/pipeline/stream/{run_id}",
    )


@router.get("/stream/{run_id}")
async def stream_pipeline(run_id: str):
    if run_id not in _queues:
        raise HTTPException(404, "Run not found")

    async def _event_generator():
        queue = _queues[run_id]
        while True:
            try:
                event = await asyncio.wait_for(queue.get(), timeout=15.0)
            except asyncio.TimeoutError:
                yield f"event: heartbeat\ndata: {{}}\n\n"
                continue

            if event is None:
                yield f"event: stream_end\ndata: {{}}\n\n"
                break

            event_type = event.get("event", "agent_log")
            data = json.dumps(event.get("data", event), default=str)
            yield f"event: {event_type}\ndata: {data}\n\n"

    return StreamingResponse(
        _event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/status/{run_id}", response_model=PipelineStatus)
async def pipeline_status(run_id: str):
    if run_id not in _statuses:
        raise HTTPException(404, "Run not found")
    return _statuses[run_id]
