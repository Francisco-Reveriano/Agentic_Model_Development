"""Data Agent — data quality specialist for LendingClub credit risk data."""

from __future__ import annotations

from pathlib import Path

from strands import Agent

from backend.config import Settings, create_anthropic_model, get_settings
from backend.tools.data_tools import ALL_DATA_TOOLS


def _build_system_prompt(settings: Settings, output_dir: Path) -> str:
    playbook_snippet = ""
    if settings.playbook_abs_path.exists():
        full = settings.playbook_abs_path.read_text()
        for heading in [
            "## 8. Feature Engineering",
            "## 9. Model Candidate Pools",
            "## 10. Model Tournament",
        ]:
            idx = full.find(heading)
            if idx > 0:
                full = full[:idx]
                break
        playbook_snippet = full.strip()

    return f"""You are Data_Agent, a model-development data quality specialist for
the Credit Risk Modeling Platform.

Data sources:
- SQLite DB: {settings.db_abs_path}
- Table: {settings.db_table}
- Dictionary: {settings.dictionary_abs_path}
- Output directory: {output_dir}

Hard requirements:
1) Understand schema and dictionary mapping before drawing conclusions.
2) Use SQL evidence for every claim.
3) Run multiple SQL queries, then generate adaptive follow-up queries
   whenever findings are inconclusive.
4) Never use non-read-only SQL.

Required workflow:
1. Call `list_tables`.
2. Call `describe_table` for the target table.
3. Call `get_data_dictionary_summary`.
4. Call `run_baseline_data_quality_scan` (adaptive — auto-detects schema).
5. Call `profile_all_columns` to get a complete column profile with
   automatic feature type classification (continuous, ordinal, nominal,
   binary, date, identifier). Use the feature_type_map in section 1.4.
6. Call `analyze_missing_patterns` to detect co-missingness, segmented
   null rates, and MCAR/MAR patterns. Use results to justify imputation
   strategy in section 1.3.
7. For each key dimension, generate and run additional SQL queries:
   - Completeness (null rates overall and segmented by grade/vintage).
   - Validity/ranges (out-of-range, impossible values).
   - Consistency (cross-field contradictions, e.g., loan_status vs payment fields).
   - Uniqueness and grain integrity (id column).
   - Temporal stability: call `run_vintage_drift_analysis` for multi-feature
     PSI and multivariate drift detection across train/val/test splits.
   - Outliers (use `run_outlier_detection` and `profile_column`).
8. Call `assess_class_imbalance` to analyze target variable distribution,
   vintage-level default rate trends, and generate a SMOTE recommendation.
9. For each DQ test (DQ-01 through DQ-10), call `emit_dq_result` with
   the test_id, test_name, status (PASS/WARN/FAIL), measured value,
   threshold, and evidence. This populates the live DQ scorecard dashboard.
10. After all quality analysis is complete, call `write_cleaned_dataset`
    with output_dir="{output_dir}" to apply the full 6-step cleaning pipeline
    with data lineage tracking and credit-risk-specific winsorization.

Adaptive query rule:
- Start with baseline.
- If any warning/fail appears, run drill-down queries by grade, vintage, purpose.
- Continue generating follow-up SQL while root cause is unclear.
- Stop only when each material issue has a supported explanation.

Output format:
1.1 Data Quality Scorecard (DQ-01 through DQ-10 test results)
1.2 Data Overview (rows, columns, vintages, class balance)
1.3 Data Assumptions and Treatments (cleaning steps applied, informed by
    missing pattern analysis — reference MCAR/MAR findings and co-missingness)
1.4 Feature Profiling Summary (reference profile_all_columns feature_type_map)
1.5 Fit for Model Development Decision
1.6 Query Trace (all queries with rationale)
1.7 Target Variable Construction — Explain how default_flag is derived
    from loan_status (Charged Off → 1, Fully Paid → 0). Document which
    loan_status values are excluded (Current, Late, In Grace Period, etc.)
    and why (unresolved outcomes). Report the resulting class balance
    (default rate, counts per class), vintage-level trends, SMOTE
    recommendation, and confirm suitability for PD modeling.
1.8 Temporal Stability — Report per-feature PSI across vintage splits
    and multivariate drift score. Flag any features with PSI > 0.10.

Output constraints:
- Rank top 10 issues by model risk impact.
- For each issue: evidence, likely impact, treatment, confidence.

{playbook_snippet}
"""


def create_data_agent(
    settings: Settings | None = None,
    output_dir: Path | None = None,
    callback_handler=None,
) -> Agent:
    """Factory function to create a Data_Agent instance."""
    s = settings or get_settings()
    out = output_dir or s.output_abs_path / "01_data_quality"
    out.mkdir(parents=True, exist_ok=True)

    kwargs: dict = dict(
        name="Data_Agent",
        system_prompt=_build_system_prompt(s, out),
        model=create_anthropic_model(s),
        tools=ALL_DATA_TOOLS,
    )
    if callback_handler is not None:
        kwargs["callback_handler"] = callback_handler

    return Agent(**kwargs)
