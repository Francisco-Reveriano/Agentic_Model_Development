# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

End-to-end Credit Risk Modeling Platform that automates PD (Probability of Default), LGD (Loss Given Default), EAD (Exposure at Default) model development following the Basel II/III Internal Ratings-Based (IRB) Advanced Approach. The system computes Expected Loss (EL = PD x LGD x EAD) at loan-level and portfolio-level granularity.

Uses a pipeline of 7 AI agents (AWS Strands framework + Anthropic Claude) that execute data quality assessment, feature engineering, model training tournaments, evaluation, and regulatory-grade report generation. A React frontend streams real-time progress via SSE.

## Build & Run

```bash
# Backend dependencies
pip install -r requirements.txt

# Frontend dependencies
cd frontend && npm install

# Start FastAPI backend (port 8000)
uvicorn middleware.main:app --reload --port 8000

# Start React dev server (port 5173, proxies /api to backend)
cd frontend && npm run dev
```

## Environment

Requires a `.env` file at the project root. Key variables:

- `ANTHROPIC_API_KEY` — required for Claude agent calls
- `MODEL_ID` — Claude model identifier (default: `claude-opus-4-6`)
- `DB_PATH` — path to SQLite database (default: `Data/Raw/RCM_Controls.db`)
- `DB_TABLE` — table name (default: `my_table`)
- `OUTPUT_DIR` — pipeline output root (default: `Data/Output`)
- `PIPELINE_MD_PATH` — pipeline playbook markdown (default: `docs/credit_risk_pd_lgd_ead_pipeline.md`)
- `TOURNAMENT_*` — tournament engine settings (top_k, max_iterations, convergence_threshold, etc.)

## Architecture

### Three-Tier Layout

```
backend/          Python — 7 Strands agents, tools, tournament engine, report generator
middleware/       FastAPI — REST endpoints, SSE streaming, Pydantic schemas
frontend/         React + TypeScript + Vite + Tailwind — UI
```

### Agent Pipeline (sequential, handoff-based)

| # | Agent | Responsibility | Key Output |
|---|-------|---------------|------------|
| 1 | Data_Agent | Data quality tests (DQ-01–DQ-10), 6-step cleaning pipeline, target construction | cleaned_features.parquet, targets.parquet |
| 2 | Feature_Agent | WoE/IV, correlation, VIF, ratio features, feature selection | feature_matrix.parquet |
| 3 | PD_Agent | 12-model tournament (4 phases), statsmodels regulatory output | champion .joblib, tournament_results.json |
| 4 | LGD_Agent | Two-stage tournament (binary + severity regression) | stage1 + stage2 .joblib |
| 5 | EAD_Agent | Regression tournament, CCF analysis | champion .joblib |
| 6 | EL_Agent | PD x LGD x EAD, stress testing (Base/Adverse/Severe) | el_results.parquet |
| 7 | Report_Agent | python-docx generation of 5 regulatory reports | .docx files |

Agents communicate via `handoff.json` files in `Data/Output/pipeline_run_{timestamp}/{stage_dir}/`.

### Model Tournament (backend/tournament.py)

The core engine used by PD, LGD, and EAD agents:
1. **Phase 1 — Broad Sweep**: Train all candidates, score on validation set
2. **Phase 2 — Feature Importance Consensus**: Weighted cross-model importance, tier assignment
3. **Phase 3 — Refinement Loop**: RandomizedSearchCV on top-K, feature set variation, convergence check
4. **Phase 4 — Champion Selection**: Weighted scoring rubric (regulatory vs performance mode)

### API Endpoints (middleware/)

| Method | Endpoint | Purpose |
|--------|----------|---------|
| GET | `/api/health` | Health check |
| GET | `/api/dataset/info` | Dataset metadata (row/column counts) |
| POST | `/api/pipeline/start` | Start pipeline run, returns run_id + SSE URL |
| GET | `/api/pipeline/stream/{run_id}` | SSE real-time streaming |
| GET | `/api/pipeline/status/{run_id}` | Polling fallback |
| GET | `/api/reports/{run_id}` | List generated .docx reports |
| GET | `/api/reports/{run_id}/download/{filename}` | Download report file |
| GET | `/api/models/list` | List trained model versions |
| GET | `/api/models/{model_id}/metrics` | Model-specific metrics |

### Agent Code Pattern

All agents follow the factory function pattern:

```python
def create_*_agent(settings: Settings | None = None, output_dir: Path | None = None) -> Agent:
```

Tools are decorated with `@tool` from strands. Agents use `AnthropicModel` with claude-opus-4-6, 128k max tokens, adaptive thinking, max effort.

### Key Files

- `backend/config.py` — Centralized `Settings` via pydantic-settings, `create_anthropic_model()` factory
- `backend/tournament.py` — 4-phase model tournament engine (shared by PD/LGD/EAD)
- `backend/orchestrator.py` — Pipeline sequencing, handoff protocol, async SSE bridging via `asyncio.to_thread()`
- `backend/callbacks.py` — `SSECallbackHandler` bridging sync Strands agents to async queue
- `backend/report_generator.py` — python-docx report assembly (DQ, C1 template, EL summary)
- `backend/model_registry.py` — joblib model persistence + `model_registry.json`
- `docs/credit_risk_pd_lgd_ead_pipeline.md` — Pipeline playbook injected into agent system prompts

## Data

- `Data/Raw/RCM_Controls.db` — LendingClub 2007–2018, 2,260,701 loans, 151 columns, table `my_table`
- `Data/RAW_FILES/LCDataDictionary.csv` — 117 field definitions
- `Data/Output/` — Pipeline run outputs (gitignored)

Vintage-based splits: Train (issue_year <= 2015), Validation (2016), Test (>= 2017).

## Key Dependencies

| Category | Packages |
|----------|----------|
| Agent Framework | strands-agents[otel,anthropic] |
| ML Training | scikit-learn, xgboost, lightgbm, statsmodels, optbinning |
| Data | pandas, numpy, scipy, sqlalchemy |
| Reports | python-docx, matplotlib, seaborn |
| API | FastAPI, uvicorn, sse-starlette, pydantic |
| Frontend | React, TypeScript, Vite, Tailwind CSS, react-router-dom, lucide-react |
