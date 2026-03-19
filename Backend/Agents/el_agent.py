"""EL Agent — Expected Loss calculation combining PD, LGD, and EAD models."""

from __future__ import annotations

from pathlib import Path

from strands import Agent

from backend.config import Settings, create_anthropic_model, get_settings
from backend.tools.el_tools import ALL_EL_TOOLS
from backend.tools.model_tools import ALL_MODEL_TOOLS


def _build_system_prompt(settings: Settings, output_dir: Path) -> str:
    playbook_snippet = ""
    if settings.playbook_abs_path.exists():
        full = settings.playbook_abs_path.read_text()
        for section in ["## 13. Stress Testing", "## 14. Report Structures"]:
            idx = full.find(section)
            if idx > 0:
                end_idx = full.find("\n## ", idx + len(section))
                if end_idx < 0:
                    end_idx = len(full)
                playbook_snippet += full[idx:end_idx] + "\n\n"

    return f"""You are EL_Agent, the Expected Loss specialist for the Credit Risk
Modeling Platform.

Output directory: {output_dir}

Your role:
Combine the three champion models (PD, LGD, EAD) to compute Expected Loss
at the loan level and portfolio level, then run stress testing scenarios.

Required workflow:
1. Load champion models from the PD, LGD, and EAD handoff directories.
2. Load the test dataset (issue_year >= 2017) from the feature matrix.
3. Generate predictions from all three champion models on the test set.
4. Compute loan-level EL = PD * LGD * EAD for each loan.
5. Generate portfolio roll-up by grade, vintage (issue_year), and purpose.
6. Run all three stress test scenarios:
   - Base: no adjustment
   - Adverse: PD * 1.5, LGD floor = 0.45
   - Severe: PD * 2.0, LGD floor = 0.60
7. Compute summary statistics: mean EL, median EL, total portfolio EL,
   EL by risk band, EL distribution percentiles.
8. Write all results and handoff.json.

Output format:
- el_results.parquet: loan-level PD, LGD, EAD, EL predictions
- stress_test_results.json: EL under all 3 scenarios
- portfolio_rollup.json: aggregated EL by dimensions
- handoff.json: status, file paths, key metrics

{playbook_snippet}
"""


def create_el_agent(
    settings: Settings | None = None, output_dir: Path | None = None
) -> Agent:
    """Factory function to create an EL_Agent instance."""
    s = settings or get_settings()
    out = output_dir or s.output_abs_path / "06_expected_loss"
    out.mkdir(parents=True, exist_ok=True)

    return Agent(
        name="EL_Agent",
        system_prompt=_build_system_prompt(s, out),
        model=create_anthropic_model(s),
        tools=ALL_EL_TOOLS + ALL_MODEL_TOOLS,
    )
