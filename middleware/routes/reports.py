from __future__ import annotations

import logging
from pathlib import Path

import mammoth
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, HTMLResponse

from backend.config import get_settings

logger = logging.getLogger(__name__)
from middleware.schemas.reports import ReportMeta

router = APIRouter(prefix="/api/reports", tags=["reports"])

# CSS for rendered HTML reports
_REPORT_CSS = """
<style>
  body { font-family: 'Segoe UI', system-ui, -apple-system, sans-serif; max-width: 960px; margin: 0 auto; padding: 2rem; background: #0f172a; color: #e2e8f0; line-height: 1.7; }
  h1 { color: #60a5fa; border-bottom: 2px solid #1e3a5f; padding-bottom: 0.5rem; font-size: 1.8rem; }
  h2 { color: #93c5fd; margin-top: 2rem; font-size: 1.4rem; }
  h3 { color: #bfdbfe; font-size: 1.1rem; }
  table { border-collapse: collapse; width: 100%; margin: 1rem 0; font-size: 0.9rem; }
  th { background: #1e293b; color: #93c5fd; padding: 0.6rem 0.8rem; text-align: left; border: 1px solid #334155; font-weight: 600; }
  td { padding: 0.5rem 0.8rem; border: 1px solid #334155; }
  tr:nth-child(even) { background: #1e293b40; }
  tr:hover { background: #1e293b80; }
  p { margin: 0.5rem 0; }
  strong { color: #fbbf24; }
  img { max-width: 100%; border-radius: 8px; margin: 1rem 0; }
  code { background: #1e293b; padding: 0.15rem 0.4rem; border-radius: 4px; font-size: 0.85em; }
  ul, ol { padding-left: 1.5rem; }
  li { margin: 0.3rem 0; }
</style>
"""

# In-memory cache: filepath → (mtime, html_string)
_html_cache: dict[str, tuple[float, str]] = {}


def _reports_dir(run_id: str) -> Path:
    settings = get_settings()
    return settings.output_abs_path / run_id / "07_reports"


def _docx_to_html(filepath: Path) -> str:
    """Convert .docx to styled HTML, with caching."""
    key = str(filepath)
    mtime = filepath.stat().st_mtime
    if key in _html_cache and _html_cache[key][0] == mtime:
        return _html_cache[key][1]

    with open(filepath, "rb") as f:
        result = mammoth.convert_to_html(f)
    html = f"<!DOCTYPE html><html><head><meta charset='utf-8'><title>{filepath.stem}</title>{_REPORT_CSS}</head><body>{result.value}</body></html>"
    _html_cache[key] = (mtime, html)
    return html


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
    logger.info("Download report %s/%s", run_id, filename)
    return FileResponse(
        filepath,
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.get("/{run_id}/html/{filename}")
async def view_report_html(run_id: str, filename: str):
    """Return the .docx report converted to styled HTML for in-browser viewing."""
    d = _reports_dir(run_id)
    # Accept either .docx or .html extension — always resolve to the .docx source
    stem = filename.rsplit(".", 1)[0]
    filepath = d / f"{stem}.docx"
    if not filepath.exists():
        raise HTTPException(404, "Report file not found")
    html = _docx_to_html(filepath)
    return HTMLResponse(content=html)


@router.get("/{run_id}/download-html/{filename}")
async def download_report_html(run_id: str, filename: str):
    """Download the .docx report as an .html file."""
    d = _reports_dir(run_id)
    stem = filename.rsplit(".", 1)[0]
    filepath = d / f"{stem}.docx"
    if not filepath.exists():
        raise HTTPException(404, "Report file not found")
    html = _docx_to_html(filepath)
    return HTMLResponse(
        content=html,
        headers={"Content-Disposition": f'attachment; filename="{stem}.html"'},
    )
