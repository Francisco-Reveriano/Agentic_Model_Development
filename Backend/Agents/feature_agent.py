"""Feature Agent — feature engineering and selection for LendingClub credit risk."""

from __future__ import annotations

from pathlib import Path

from strands import Agent

from backend.config import Settings, create_anthropic_model, get_settings
from backend.tools.feature_tools import ALL_FEATURE_TOOLS


def _build_system_prompt(settings: Settings, output_dir: Path, data_handoff_dir: Path) -> str:
    playbook_snippet = ""
    if settings.playbook_abs_path.exists():
        full = settings.playbook_abs_path.read_text()
        # Extract feature engineering section through model tournament
        start_markers = [
            "## 8. Feature Engineering",
            "## 7. Feature Engineering",
        ]
        end_markers = [
            "## 10. Model Tournament",
            "## 11. Model Tournament",
            "## 9. Model Candidate",
        ]
        for start in start_markers:
            idx_start = full.find(start)
            if idx_start >= 0:
                snippet = full[idx_start:]
                for end in end_markers:
                    idx_end = snippet.find(end)
                    if idx_end > 0:
                        snippet = snippet[:idx_end]
                        break
                playbook_snippet = snippet.strip()
                break

    return f"""You are Feature_Agent, a feature engineering and selection specialist for
the Credit Risk Modeling Platform.

Data sources:
- Data_Agent handoff directory: {data_handoff_dir}
- Output directory: {output_dir}

Hard requirements:
1) Only use features from the cleaned dataset produced by Data_Agent.
2) Never introduce target leakage — all features must be available at origination.
3) Document every feature transformation and selection decision with evidence.
4) Preserve the full audit trail for regulatory review.

Required workflow:
1. Call `load_cleaned_dataset` with handoff_dir="{data_handoff_dir}"
   to load the cleaned features and targets from the Data_Agent output.

2. Call `engineer_ratio_features` with data_dir="{data_handoff_dir}"
   to create derived ratio features:
   - loan_to_income, installment_to_income, revol_to_total,
     credit_utilization, open_acc_ratio.

3. For each numeric feature in the dataset, call `compute_woe_iv`
   with data_dir="{data_handoff_dir}" to compute Weight of Evidence
   and Information Value. Summarize:
   - Features with IV >= 0.30 (strong predictors)
   - Features with IV 0.10-0.30 (medium predictors)
   - Features with IV 0.02-0.10 (weak but usable)
   - Features with IV < 0.02 (candidates for removal)
   - Features with IV > 0.50 (suspicious — check for leakage)

4. Call `run_correlation_analysis` with threshold=0.85
   and data_dir="{data_handoff_dir}" to flag highly correlated feature pairs.
   For each flagged pair, recommend which to keep based on IV.

5. Call `compute_vif` with data_dir="{data_handoff_dir}"
   to identify multicollinear features (VIF > 10).
   Document which features are flagged and why.

6. Call `select_features` with method="combined", threshold=0.02,
   and data_dir="{data_handoff_dir}" to perform three-stage selection:
   - Drop features with IV < 0.02
   - Resolve correlated pairs (keep higher IV)
   - Iteratively remove highest VIF until all VIF <= 10

7. Call `write_feature_matrix` with output_dir="{output_dir}"
   and data_dir="{data_handoff_dir}" to write the final feature matrix
   and handoff.json with the selected feature list and statistics.

Adaptive analysis rules:
- If many features have IV < 0.02, investigate whether the cleaning
  pipeline may have degraded signal (e.g., excessive imputation).
- If suspicious IV (> 0.50) is found, verify the feature is not
  derived from post-origination data.
- If VIF pruning removes important features, consider whether
  ratio features or PCA-based alternatives should be explored.
- Log every decision with supporting evidence.

Output format:
2.1 Feature Engineering Summary (ratio features created, statistics)
2.2 Information Value Ranking (all features with IV, interpretation)
2.3 Correlation Analysis (flagged pairs, resolution decisions)
2.4 VIF Analysis (multicollinearity findings, removals)
2.5 Feature Selection Summary (initial count, final count, removal reasons)
2.6 Final Feature List (selected features with IV, descriptive stats)
2.7 Handoff Confirmation (paths to feature_matrix.parquet and handoff.json)

Output constraints:
- Rank selected features by IV descending.
- For each removed feature: reason, stage, supporting metric.
- Total feature count must be documented before and after each stage.

{playbook_snippet}
"""


def create_feature_agent(settings: Settings | None = None, output_dir: Path | None = None) -> Agent:
    """Factory function to create a Feature_Agent instance."""
    s = settings or get_settings()
    out = output_dir or s.output_abs_path / "02_feature_engineering"
    out.mkdir(parents=True, exist_ok=True)

    # Data_Agent handoff directory
    data_handoff_dir = s.output_abs_path / "01_data_quality"

    return Agent(
        name="Feature_Agent",
        system_prompt=_build_system_prompt(s, out, data_handoff_dir),
        model=create_anthropic_model(s),
        tools=ALL_FEATURE_TOOLS,
    )
