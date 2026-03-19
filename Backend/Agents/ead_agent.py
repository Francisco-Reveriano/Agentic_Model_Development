"""EAD Agent — Exposure at Default model development specialist."""

from __future__ import annotations

from pathlib import Path

from strands import Agent

from backend.config import Settings, create_anthropic_model, get_settings
from backend.tools.ead_tools import ALL_EAD_TOOLS

# Import shared model tools if available; gracefully degrade if not yet created
try:
    from backend.tools.model_tools import ALL_MODEL_TOOLS
except ImportError:
    ALL_MODEL_TOOLS = []


def _build_system_prompt(settings: Settings, output_dir: Path) -> str:
    playbook_snippet = ""
    if settings.playbook_abs_path.exists():
        full = settings.playbook_abs_path.read_text()
        start_idx = full.find("## 9. Model Candidate Pools")
        end_idx = full.find("## 14. Report Structures")
        if start_idx > 0:
            if end_idx > start_idx:
                playbook_snippet = full[start_idx:end_idx].strip()
            else:
                playbook_snippet = full[start_idx:].strip()

    return f"""You are EAD_Agent, an Exposure at Default model development specialist for
the Credit Risk Modeling Platform.

Output directory: {output_dir}

Your task: Build an EAD regression model using a tournament framework.

Target: ead (remaining outstanding principal at default, from out_prncp).
CCF: Credit Conversion Factor = ead / funded_amnt (used for validation).

Candidates (9 regression models):
  OLS, Ridge, Lasso, ElasticNet, Huber Regressor, RF Regressor,
  GBM Regressor, XGBoost Regressor, LightGBM Regressor

Primary metric: RMSE (lower is better)

Hard requirements:
1) Load the feature matrix from the Feature_Agent handoff directory.
2) Extract the EAD target and CCF from targets.parquet.
3) Apply vintage-based train/val/test split (train <= 2015, val = 2016, test >= 2017).
4) Train all 9 regression candidates with baseline configurations.
5) Evaluate each model: RMSE, MAE, R-squared on validation set.
6) Perform CCF validation: predicted CCF should be in [0, 1.2] range.
7) Select champion based on lowest validation RMSE.
8) Save champion model as .joblib file.
9) Write handoff.json with champion model name, metrics, and output file paths.

Required workflow:
1. Call `define_ead_candidates` to review all 9 candidate models.
2. Call `construct_ead_target` with the data directory to examine EAD and CCF
   distributions.
3. Optionally call `compute_amortization_schedule` with representative loan
   parameters to validate theoretical exposure curves.
4. Call `run_ead_tournament` with the data and output directories to execute
   the full tournament.
5. Review tournament results, confirm champion selection, and verify CCF
   reasonableness.
6. Verify handoff.json has been written with all required fields.

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


def create_ead_agent(settings: Settings | None = None, output_dir: Path | None = None) -> Agent:
    """Factory function to create an EAD_Agent instance."""
    s = settings or get_settings()
    out = output_dir or s.output_abs_path / "05_ead_model"
    out.mkdir(parents=True, exist_ok=True)

    tools = list(ALL_EAD_TOOLS) + list(ALL_MODEL_TOOLS)

    return Agent(
        name="EAD_Agent",
        system_prompt=_build_system_prompt(s, out),
        model=create_anthropic_model(s),
        tools=tools,
    )
