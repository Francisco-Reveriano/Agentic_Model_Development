# Credit Risk Modeling Platform

An end-to-end platform for automated credit risk model development (PD, LGD, EAD) following the Basel II/III Internal Ratings-Based (IRB) Advanced Approach.

The system is driven by **AI agents** (AWS Strands + Anthropic Claude) that interpret a pipeline playbook and autonomously execute each stage of the data science workflow — from data quality assessment through model training, evaluation, and regulatory report generation.

## What It Does

1. **Upload a dataset** — LendingClub loan data (2.26M loans, 2007–2018) stored in SQLite
2. **Select models to build** — PD only, LGD only, EAD only, or the full pipeline with Expected Loss
3. **Watch agents work in real time** — SSE-streamed pipeline progress in the browser
4. **Download regulatory-grade reports** — 5 Word documents conforming to the C1 Standalone Use Case Template

### Generated Reports

| Report | Description |
|--------|-------------|
| Data Quality Report | Complete DQ assessment, cleaning log, feature profiling, fit-for-modeling decision |
| PD Model Report | Probability of Default — full C1 template with methodology, evaluation, coefficients |
| LGD Model Report | Loss Given Default — two-stage model documentation (binary + severity) |
| EAD Model Report | Exposure at Default — regression model with CCF analysis |
| EL Summary Report | Expected Loss (PD x LGD x EAD), stress testing, portfolio roll-up |

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    React Frontend                        │
│   Model Selection → Pipeline Dashboard → Report Download │
│                    (SSE streaming)                        │
├─────────────────────────────────────────────────────────┤
│                  FastAPI Middleware                       │
│   REST API  ·  SSE Streaming  ·  File Serving            │
├─────────────────────────────────────────────────────────┤
│                   Python Backend                         │
│                                                          │
│   Data_Agent ──► Feature_Agent ──► PD_Agent ──┐          │
│                                    LGD_Agent ─┤          │
│                                    EAD_Agent ─┤          │
│                                    EL_Agent ──┤          │
│                                    Report_Agent          │
│                                                          │
│   Tournament Engine  ·  Report Generator  ·  Registry    │
└─────────────────────────────────────────────────────────┘
```

### Agent Pipeline

| Agent | What It Does |
|-------|-------------|
| **Data_Agent** | Runs 10 data quality tests (DQ-01 – DQ-10), applies a 6-step cleaning pipeline (leakage removal, filtering, type coercion, imputation, Winsorization, encoding), constructs PD/LGD/EAD targets |
| **Feature_Agent** | Engineers ratio features, computes WoE/IV via optimal binning, runs correlation analysis and VIF checks, selects final feature set |
| **PD_Agent** | Trains 12 candidate models across 4 libraries in a 4-phase tournament, selects champion, always produces statsmodels regulatory output |
| **LGD_Agent** | Two-stage modeling — Stage 1 (binary: any loss?) + Stage 2 (severity regression) — each with its own tournament |
| **EAD_Agent** | Regression tournament with 9 candidates, validates against amortization schedule, computes Credit Conversion Factor |
| **EL_Agent** | Combines PD x LGD x EAD at loan level, runs 3 stress scenarios (Base / Adverse / Severe), generates portfolio roll-up |
| **Report_Agent** | Assembles all artifacts into formatted Word documents using python-docx |

### Model Tournament Engine

Each modeling agent runs a 4-phase tournament:

1. **Broad Sweep** — Train ALL candidates (12 PD / 13 LGD / 9 EAD) with baseline configs
2. **Feature Importance Consensus** — Aggregate importance signals across all models, assign feature tiers
3. **Refinement Loop** — Hyperparameter tuning (RandomizedSearchCV) on top-K models with convergence check
4. **Champion Selection** — Weighted scoring rubric (regulatory or performance mode)

## Getting Started

### Prerequisites

- Python 3.10+
- Node.js 18+
- An Anthropic API key

### Setup

```bash
# Clone the repo
git clone git@github.com:Francisco-Reveriano/Agentic_Model_Development.git
cd Agentic_Model_Development

# Create .env file
cat > .env << EOF
ANTHROPIC_API_KEY="your-key-here"
MODEL_ID="claude-opus-4-6"
DB_PATH="Data/Raw/RCM_Controls.db"
DB_TABLE="my_table"
OUTPUT_DIR="Data/Output"
PIPELINE_MD_PATH="docs/credit_risk_pd_lgd_ead_pipeline.md"
MAX_TOKENS=128000
LOG_LEVEL="INFO"
EOF

# Install backend dependencies
pip install -r requirements.txt

# Install frontend dependencies
cd frontend && npm install && cd ..
```

### Run

```bash
# Terminal 1 — Start the backend
uvicorn middleware.main:app --reload --port 8000

# Terminal 2 — Start the frontend
cd frontend && npm run dev
```

Open http://localhost:5173, select models to build, and click **Start Pipeline**.

## Project Structure

```
├── backend/
│   ├── agents/             # 7 Strands agent definitions
│   │   ├── data_agent.py
│   │   ├── feature_agent.py
│   │   ├── pd_agent.py
│   │   ├── lgd_agent.py
│   │   ├── ead_agent.py
│   │   ├── el_agent.py
│   │   └── report_agent.py
│   ├── tools/              # @tool-decorated functions for each agent
│   │   ├── data_tools.py       # SQL, DQ tests, cleaning pipeline
│   │   ├── feature_tools.py    # WoE/IV, correlation, VIF
│   │   ├── model_tools.py      # Shared evaluation, vintage split
│   │   ├── pd_tools.py         # 12 PD candidates, tournament runner
│   │   ├── lgd_tools.py        # Two-stage LGD candidates
│   │   ├── ead_tools.py        # EAD candidates, amortization
│   │   ├── el_tools.py         # EL computation, stress testing
│   │   └── report_tools.py     # Chart generation, report assembly
│   ├── tournament.py       # 4-phase model tournament engine
│   ├── orchestrator.py     # Pipeline sequencing + handoff protocol
│   ├── callbacks.py        # SSE callback handler
│   ├── report_generator.py # python-docx report assembly
│   ├── model_registry.py   # Model persistence + registry
│   └── config.py           # Centralized settings (pydantic-settings)
├── middleware/
│   ├── main.py             # FastAPI app, CORS, health check
│   ├── routes/             # Pipeline, reports, models endpoints
│   └── schemas/            # Pydantic request/response models
├── frontend/
│   └── src/
│       ├── pages/          # HomePage, PipelinePage, ReportsPage
│       ├── components/     # ModelSelector, Stepper, Log, Metrics, ReportCard
│       ├── hooks/          # useSSE, usePipelineState
│       ├── types/          # TypeScript interfaces
│       └── api/            # API client
├── docs/
│   └── credit_risk_pd_lgd_ead_pipeline.md   # Pipeline playbook
├── Data/
│   ├── Raw/                # Source SQLite database + CSV
│   ├── RAW_FILES/          # LendingClub data dictionary
│   └── Output/             # Pipeline run outputs (gitignored)
├── Notebooks/
│   └── 01. Create SQL Database.ipynb
├── requirements.txt
└── .env                    # API keys and config (gitignored)
```

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/health` | Health check and DB connectivity |
| `GET` | `/api/dataset/info` | Dataset metadata (rows, columns) |
| `POST` | `/api/pipeline/start` | Start pipeline — body: `{"models": ["PD","LGD","EAD","EL"]}` |
| `GET` | `/api/pipeline/stream/{run_id}` | SSE event stream for real-time progress |
| `GET` | `/api/pipeline/status/{run_id}` | Poll pipeline status (fallback) |
| `GET` | `/api/reports/{run_id}` | List generated reports |
| `GET` | `/api/reports/{run_id}/download/{filename}` | Download a .docx report |
| `GET` | `/api/models/list` | List all trained model versions |
| `GET` | `/api/models/{model_id}/metrics` | Metrics for a specific model |

### SSE Event Types

| Event | Description |
|-------|-------------|
| `agent_start` | Agent begins execution |
| `agent_log` | Agent streaming output (text, tool calls) |
| `agent_metric` | Metric reported (AUC, Gini, RMSE, etc.) |
| `agent_complete` | Agent finished |
| `tournament_start` | Model tournament begins |
| `model_trained` | Individual model trained with rank and score |
| `champion_declared` | Tournament winner selected |
| `pipeline_complete` | All agents finished, reports ready |

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Agent Framework | [AWS Strands](https://github.com/strands-agents/strands-agents) with Anthropic Claude |
| ML Libraries | scikit-learn, XGBoost, LightGBM, statsmodels, optbinning |
| Backend API | FastAPI, uvicorn, sse-starlette |
| Report Generation | python-docx, matplotlib, seaborn |
| Frontend | React 18, TypeScript, Vite, Tailwind CSS |
| Data | SQLite, pandas, SQLAlchemy |

## Regulatory Framework

The platform follows the **Basel II/III IRB Advanced Approach** and generates documentation conforming to the **C1 Standalone Use Case Template**:

- **PD Evaluation**: AUC-ROC, Gini, KS, Brier Score, Hosmer-Lemeshow, PSI
- **LGD/EAD Evaluation**: RMSE, MAE, R-squared, decile alignment, PSI
- **Stress Testing**: Base, Adverse (PD x1.5, LGD floor 0.45), Severe (PD x2.0, LGD floor 0.60)
- **Regulatory Output**: statsmodels Logit/OLS coefficient tables with p-values, confidence intervals, and odds ratios are always produced regardless of champion model

## License

Private repository. All rights reserved.
