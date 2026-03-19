from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from backend.config import get_settings
from middleware.schemas.reports import ReportMeta

router = APIRouter(prefix="/api/reports", tags=["reports"])


def _reports_dir(run_id: str) -> Path:
    settings = get_settings()
    return settings.output_abs_path / run_id / "07_reports"


@router.get("/{run_id}", response_model=list[ReportMeta])
async def list_reports(run_id: str):
    d = _reports_dir(run_id)
    if not d.exists():
        raise HTTPException(404, "Run not found or reports not yet generated")

    results = []
    for f in sorted(d.glob("*.docx")):
        # Derive type from filename prefix
        name_lower = f.stem.lower()
        if "data_quality" in name_lower or "dq" in name_lower:
            rtype = "dq"
        elif "el" in name_lower or "expected_loss" in name_lower:
            rtype = "el"
        elif "lgd" in name_lower:
            rtype = "lgd"
        elif "ead" in name_lower:
            rtype = "ead"
        elif "pd" in name_lower:
            rtype = "pd"
        else:
            rtype = "unknown"
        results.append(
            ReportMeta(filename=f.name, report_type=rtype, size_bytes=f.stat().st_size)
        )
    return results


@router.get("/{run_id}/download/{filename}")
async def download_report(run_id: str, filename: str):
    d = _reports_dir(run_id)
    filepath = d / filename
    if not filepath.exists():
        raise HTTPException(404, "Report file not found")
    return FileResponse(
        filepath,
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
