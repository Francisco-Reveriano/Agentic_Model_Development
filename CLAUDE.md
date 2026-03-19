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

### Project Layout

```
backend/              Python — 7 Strands agents, tools, tournament engine, report generator
backend/enhancements/ Optional performance, resilience, and operational modules
middleware/           FastAPI — REST endpoints, SSE streaming, Pydantic schemas
frontend/             React + TypeScript + Vite + Tailwind — UI
tests/                pytest suite — test cases, e2e runs, Basel III model exams
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

### Enhancement Modules (backend/enhancements/)

Optional modules for performance, resilience, and operational improvements:

| Module | Purpose |
|--------|---------|
| `agent_timeout.py` | Per-agent timeout limits (e.g., DATA_AGENT: 5min, PD_AGENT: 30min) via Unix signals |
| `sse_heartbeat.py` | SSE keep-alive heartbeats to prevent client disconnection during long pipelines |
| `parallel_training.py` | joblib-based multi-core model candidate training with memory limits |
| `early_stopping.py` | XGBoost/LightGBM early stopping callbacks to halt training at plateau |
| `smote_handler.py` | Automatic SMOTE when minority class < 5%, plus class weight alternatives |
| `winsorization_config.py` | Outlier clipping via percentile/std/IQR with pre-configured credit risk thresholds |
| `scoring_mode.py` | Dual rubric weights for "regulatory" (interpretability) vs "performance" (AUC) modes |
| `model_comparison.py` | Transform tournament results into visualization-ready formats (bar, radar, heatmap) |
| `export_leaderboard.py` | Export tournament leaderboards to CSV/Excel/JSON |
| `run_history.py` | Scan, compare, and rank completed pipeline runs by metric |

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

## Testing

```bash
# Run all tests
pytest tests/

# Run by suite
pytest tests/test_cases/       # TC-01–TC-10: unit-level model property checks
pytest tests/e2e_runs/         # E2E-01–E2E-05: full/partial pipeline execution
pytest tests/model_exams/      # ME-01–ME-10: Basel III compliance exams

# Run a single test module
pytest tests/test_cases/test_tc01_pd_auc_discrimination.py -v
```

### Test Suites

**test_cases/ (TC-01–TC-10)** — Model property validation: PD discrimination (AUC > 0.75, Gini, KS), calibration (Hosmer-Lemeshow, Brier), PSI stability, LGD accuracy (MAE < 0.10, R² > 0.65), EAD/CCF bounds, EL calculation correctness, tournament phases, feature consensus, refinement convergence, report generation.

**e2e_runs/ (E2E-01–E2E-05)** — End-to-end pipeline scenarios: full regulatory-mode pipeline, PD-only run, performance-mode run, LGD/EAD regression focus, resilience/error-recovery.

**model_exams/ (ME-01–ME-10)** — Basel III IRB compliance per C1 Standalone Use Case Template: data appropriateness (I.DA), data quality (I.DQ), feature engineering (M.FE), methodology selection (M.MS), parameter estimation (M.PE), stability (O.S), robustness (O.R), explainability (O.E), business integration (B.EE), documentation (TI.D). 58 test methods total.

Shared fixtures in `tests/conftest.py` provide sample data, mock tournament results, artifact loaders, and auto-discovery of the latest pipeline run directory.

## Key Dependencies

| Category | Packages |
|----------|----------|
| Agent Framework | strands-agents[otel,anthropic] |
| ML Training | scikit-learn, xgboost, lightgbm, statsmodels, optbinning |
| Data | pandas, numpy, scipy, sqlalchemy |
| Reports | python-docx, matplotlib, seaborn |
| API | FastAPI, uvicorn, sse-starlette, pydantic |
| Testing | pytest, imbalanced-learn (SMOTE) |
| Frontend | React, TypeScript, Vite, Tailwind CSS, react-router-dom, lucide-react |
