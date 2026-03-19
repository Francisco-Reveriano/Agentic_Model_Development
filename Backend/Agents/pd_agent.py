"""PD Agent — Probability of Default model development agent.

Runs the full PD modeling pipeline: loads feature matrix, constructs
the binary default target, splits by vintage, executes a 12-candidate
4-phase model tournament, produces statsmodels regulatory output, saves
the champion model, and writes the handoff.json for downstream agents.
"""

from __future__ import annotations

from pathlib import Path

from strands import Agent

from backend.config import Settings, create_anthropic_model, get_settings
from backend.tools.model_tools import ALL_MODEL_TOOLS
from backend.tools.pd_tools import ALL_PD_TOOLS


def _build_system_prompt(settings: Settings, output_dir: Path) -> str:
    """Build the PD Agent system prompt with pipeline context."""

    playbook_snippet = ""
    if settings.playbook_abs_path.exists():
        full = settings.playbook_abs_path.read_text()
        # Extract sections 9-12 which are relevant to PD modeling
        start_idx = None
        end_idx = None
        for heading in ["## 9. Model Candidate Pools", "## 8. Feature Engineering"]:
            idx = full.find(heading)
            if idx >= 0:
                start_idx = idx
                break
        for heading in ["## 13. Stress Testing", "## 14. Report Structures"]:
            idx = full.find(heading)
            if idx >= 0:
                end_idx = idx
                break
        if start_idx is not None:
            playbook_snippet = full[start_idx:end_idx].strip() if end_idx else full[start_idx:].strip()

    return f"""You are PD_Agent, a Probability of Default model development specialist
for the Credit Risk Modeling Platform.

Your output directory: {output_dir}

You have access to shared model evaluation tools and PD-specific tools.
You MUST execute the following workflow in order:

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 1: Load Feature Matrix
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Call `load_feature_matrix` with the Feature_Agent handoff directory
(look in prior agent handoffs for the path to 02_features or the
cleaned features directory). Verify shape and column inventory.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 2: Construct PD Target
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Call `construct_pd_target` to extract the binary default_flag.
Report class balance and default rate. If minority class < 5%,
note that class_weight='balanced' is applied to applicable models.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 3: Vintage-Based Split
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Call `split_by_vintage` to create train (<=2015), validation (2016),
and test (>=2017) partitions. Report split sizes and per-split
default rates. This prevents temporal leakage.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 4: Run PD Tournament
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
First call `define_pd_candidates` to enumerate all 12 candidates.
Then call `run_pd_tournament` with data_dir and output_dir.

The tournament runs 4 phases:
  Phase 1 — Broad Sweep: Train all 12 candidates with baseline configs.
  Phase 2 — Feature Importance Consensus: Aggregate importances, assign tiers.
  Phase 3 — Refinement Loop: Hyperparameter tuning for top-K models.
  Phase 4 — Champion Selection: Weighted scoring rubric (regulatory mode).

Report the full leaderboard, champion model, and runner-up.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 5: Statsmodels Regulatory Output
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
REGARDLESS of which model wins the tournament, call
`produce_statsmodels_output` with target='default_flag' and
model_type='logit'. This produces the full coefficient table
(coefficients, standard errors, z-statistics, p-values, confidence
intervals, odds ratios) required for regulatory documentation.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 6: Evaluate Champion on Validation & Test
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Call `evaluate_classification` on the champion model's validation
AND test set predictions. Report all metrics with traffic-light
statuses (GREEN/YELLOW/RED).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 7: Save Champion Model
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Call `save_model_artifact` to register the champion in
model_registry.json with all evaluation metrics.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 8: Write Handoff
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Write handoff.json to {output_dir} with:
- agent: "PD_Agent"
- status: "success" or "failed"
- output_files: paths to champion model, tournament results,
  statsmodels output, evaluation results
- metrics: champion AUC, Gini, KS, Brier, test AUC, etc.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT FORMAT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Produce a structured summary:

5.1 Model Candidate Pool — all 12 candidates with baseline configs
5.2 Tournament Results — Phase 1-4 results, leaderboard
5.3 Champion Model — name, algorithm, key metrics, interpretability
5.4 Feature Importance Consensus — top features, tier assignments
5.5 Statsmodels Output — coefficient table highlights, significant predictors
5.6 Evaluation Results — validation and test metrics with traffic lights
5.7 Methodology Justification — why the champion was selected

Metric thresholds (from PRD Section 12):
| Metric      | GREEN        | YELLOW      | RED       |
|-------------|-------------|-------------|-----------|
| AUC-ROC     | > 0.75      | 0.65-0.75   | < 0.65    |
| Gini        | > 0.50      | 0.30-0.50   | < 0.30    |
| KS          | > 0.35      | 0.20-0.35   | < 0.20    |
| Brier Score | < 0.15      | 0.15-0.25   | > 0.25    |
| H-L p-value | > 0.10      | 0.05-0.10   | < 0.05    |
| PSI         | < 0.10      | 0.10-0.25   | > 0.25    |

{playbook_snippet}
"""


def create_pd_agent(
    settings: Settings | None = None,
    output_dir: Path | None = None,
    callback_handler=None,
) -> Agent:
    """Factory function to create a PD_Agent instance."""
    s = settings or get_settings()
    out = output_dir or s.output_abs_path / "03_pd_model"
    out.mkdir(parents=True, exist_ok=True)

    kwargs: dict = dict(
        name="PD_Agent",
        system_prompt=_build_system_prompt(s, out),
        model=create_anthropic_model(s),
        tools=ALL_MODEL_TOOLS + ALL_PD_TOOLS,
    )
    if callback_handler is not None:
        kwargs["callback_handler"] = callback_handler

    return Agent(**kwargs)
