"""LGD Agent — two-stage Loss Given Default model development specialist."""

from __future__ import annotations

from pathlib import Path

from strands import Agent

from backend.config import Settings, create_anthropic_model, get_settings
from backend.tools.lgd_tools import ALL_LGD_TOOLS

# Import shared model tools if available; gracefully degrade if not yet created
try:
    from backend.tools.model_tools import ALL_MODEL_TOOLS
except ImportError:
    ALL_MODEL_TOOLS = []


def _build_system_prompt(settings: Settings, output_dir: Path) -> str:
    playbook_snippet = ""
    if settings.playbook_abs_path.exists():
        full = settings.playbook_abs_path.read_text()
        # Include sections 9-13 (model candidates through stress testing)
        start_idx = full.find("## 9. Model Candidate Pools")
        end_idx = full.find("## 14. Report Structures")
        if start_idx > 0:
            if end_idx > start_idx:
                playbook_snippet = full[start_idx:end_idx].strip()
            else:
                playbook_snippet = full[start_idx:].strip()

    return f"""You are LGD_Agent, a Loss Given Default model development specialist for
the Credit Risk Modeling Platform.

Output directory: {output_dir}

Your task: Build a two-stage LGD model using a tournament framework.

Stage 1 — Binary Classification: P(any loss | default)
  - Target: any_loss = 1 if lgd > 0, else 0
  - Candidates: LogReg, RF Classifier, GBM Classifier, XGBoost Classifier, LightGBM Classifier
  - Primary metric: AUC-ROC

Stage 2 — Regression: E[severity | partial loss]
  - Target: lgd value (continuous, 0-1), only rows where any_loss = 1
  - Candidates: OLS, Ridge, Lasso, ElasticNet, GBM Regressor, XGBoost Regressor,
                LightGBM Regressor, Huber Regressor
  - Primary metric: RMSE

Combined: LGD = P(any_loss) * E[severity | partial_loss]

Hard requirements:
1) Load the feature matrix from the Feature_Agent handoff directory.
2) Filter to defaults only (default_flag = 1).
3) Construct LGD targets: binary any_loss and continuous severity.
4) Apply vintage-based train/val/test split (train <= 2015, val = 2016, test >= 2017).
5) Run the two-stage tournament, training all candidates.
6) Select champions for each stage based on validation performance.
7) Compute combined LGD and evaluate on the test set.
8) Save both stage champion models as .joblib files.
9) Write handoff.json with champion model names, metrics, and output file paths.

Required workflow:
1. Call `define_lgd_candidates` to review all 13 candidate models.
2. Call `construct_lgd_target` with the data directory to examine target distributions
   and class balance.
3. Call `run_lgd_tournament` with the data and output directories to execute
   the full two-stage tournament.
4. Review tournament results and confirm champion selection.
5. Verify handoff.json has been written with all required fields.

LGD / EAD Regression Rubric:
| Dimension          | Metric                      | Weight |
|--------------------|-----------------------------|--------|
| Accuracy           | RMSE (inverted)             | 0.30   |
| Accuracy           | MAE (inverted)              | 0.15   |
| Explanatory Power  | R-squared                   | 0.20   |
| Stability          | PSI (inverted)              | 0.10   |
| Decile Alignment   | Mean abs decile deviation   | 0.10   |
| Generalization     | Train-Val RMSE gap (inv)    | 0.10   |
| Efficiency         | Inference time (inverted)   | 0.05   |

{playbook_snippet}
"""


def create_lgd_agent(settings: Settings | None = None, output_dir: Path | None = None) -> Agent:
    """Factory function to create an LGD_Agent instance."""
    s = settings or get_settings()
    out = output_dir or s.output_abs_path / "04_lgd_model"
    out.mkdir(parents=True, exist_ok=True)

    tools = list(ALL_LGD_TOOLS) + list(ALL_MODEL_TOOLS)

    return Agent(
        name="LGD_Agent",
        system_prompt=_build_system_prompt(s, out),
        model=create_anthropic_model(s),
        tools=tools,
    )
