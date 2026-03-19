# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

End-to-end Credit Risk Modeling Platform that automates PD/LGD/EAD model development following Basel II/III IRB Advanced Approach. Uses a pipeline of 7 AI agents (Strands framework + Claude Opus) that execute data quality, feature engineering, model training tournaments, and report generation. React frontend with real-time SSE streaming.

## Setup

```bash
# Backend
pip install -r requirements.txt

# Frontend
cd frontend && npm install
```

Requires `.env` with `ANTHROPIC_API_KEY` (and optionally `OPENAI_API_KEY`). See `.env` for all config variables including tournament settings.

## Running

```bash
# Start FastAPI backend
uvicorn middleware.main:app --reload --port 8000

# Start React frontend (proxies /api to backend)
cd frontend && npm run dev
```

## Architecture

### Three-Tier Design

**backend/** — Python: 7 Strands agents, tools, tournament engine, report generator
**middleware/** — FastAPI: REST endpoints, SSE streaming, Pydantic schemas
**frontend/** — React + TypeScript + Vite + Tailwind: model selection, pipeline dashboard, report download

### Agent Pipeline (pipeline-of-agents pattern)

1. **Data_Agent** → data quality tests, cleaning, target construction
2. **Feature_Agent** → WoE/IV, correlation, VIF, feature selection
3. **PD_Agent** → 12-model tournament, champion selection, statsmodels output
4. **LGD_Agent** → two-stage (binary + severity) tournament
5. **EAD_Agent** → regression tournament, CCF analysis
6. **EL_Agent** → PD×LGD×EAD combination, stress testing
7. **Report_Agent** → python-docx report generation (5 .docx reports)

Each agent writes `handoff.json` to `Data/Output/pipeline_run_{timestamp}/{stage_dir}/`.

### Key Files

- `backend/config.py` — Centralized Settings (pydantic-settings), `create_anthropic_model()` factory
- `backend/tournament.py` — 4-phase model tournament engine (broad sweep → feature consensus → refinement loop → champion selection)
- `backend/orchestrator.py` — Pipeline sequencing, handoff protocol, SSE event bridging
- `backend/callbacks.py` — SSE callback handler bridging sync agents to async queue
- `backend/tools/data_tools.py` — SQL-based data tools with read-only enforcement
- `docs/credit_risk_pd_lgd_ead_pipeline.md` — Pipeline playbook injected into agent system prompts

### API Endpoints

- `GET /api/health` — health check
- `GET /api/dataset/info` — dataset metadata
- `POST /api/pipeline/start` — start pipeline run
- `GET /api/pipeline/stream/{run_id}` — SSE streaming
- `GET /api/reports/{run_id}` — list reports
- `GET /api/reports/{run_id}/download/{filename}` — download .docx

### Agent Pattern

All agents use factory functions: `create_*_agent(settings, output_dir) -> Agent`. Strands `@tool` decorators for tools. `AnthropicModel` with claude-opus-4-6, 128k tokens, adaptive thinking, max effort.

## Data

- `Data/Raw/RCM_Controls.db` — LendingClub 2007-2018, 2.26M loans, 151 columns, table `my_table`
- `Data/RAW_FILES/LCDataDictionary.csv` — 117 field definitions
- `Data/Output/` — Pipeline run outputs (gitignored)

## Key Dependencies

- **strands-agents[otel,anthropic]** — Agent framework
- **scikit-learn, xgboost, lightgbm, statsmodels** — ML model training
- **optbinning** — WoE/IV binning
- **python-docx** — Report generation
- **FastAPI + sse-starlette** — REST API + SSE streaming
- **React + Vite + Tailwind** — Frontend
