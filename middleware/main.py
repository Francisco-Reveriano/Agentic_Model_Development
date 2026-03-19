import sqlite3
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.config import get_settings
from middleware.routes.models import router as models_router
from middleware.routes.pipeline import router as pipeline_router
from middleware.routes.reports import router as reports_router
from middleware.schemas.pipeline import DatasetInfo

app = FastAPI(title="Credit Risk Modeling Platform", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(pipeline_router)
app.include_router(reports_router)
app.include_router(models_router)


@app.get("/api/health")
async def health():
    settings = get_settings()
    db_exists = settings.db_abs_path.exists()
    return {"status": "ok", "db_connected": db_exists}


@app.get("/api/dataset/info", response_model=DatasetInfo)
async def dataset_info():
    settings = get_settings()
    db_path = settings.db_abs_path
    table = settings.db_table

    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        cur = conn.cursor()
        cur.execute(f"SELECT COUNT(*) FROM \"{table}\"")
        row_count = cur.fetchone()[0]

        cur.execute(f"PRAGMA table_info(\"{table}\")")
        cols = [row[1] for row in cur.fetchall()]
    finally:
        conn.close()

    return DatasetInfo(
        db_path=str(db_path),
        table_name=table,
        row_count=row_count,
        column_count=len(cols),
        columns=cols,
    )
