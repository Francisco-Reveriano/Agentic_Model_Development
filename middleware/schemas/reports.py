from __future__ import annotations

from pydantic import BaseModel


class ReportMeta(BaseModel):
    filename: str
    report_type: str  # dq | pd | lgd | ead | el
    size_bytes: int
