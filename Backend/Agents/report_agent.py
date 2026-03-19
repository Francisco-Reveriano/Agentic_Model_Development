"""Report Agent — generates all Credit Risk Model Development documents.

Reads handoff.json files from prior pipeline agents, produces .docx reports
(Data Quality, Model Development for PD/LGD/EAD, Expected Loss), generates
supporting charts, and writes its own handoff.json listing all produced files.
"""

from __future__ import annotations

from pathlib import Path

from strands import Agent

from backend.config import Settings, create_anthropic_model, get_settings
from backend.tools.report_tools import ALL_REPORT_TOOLS


def _build_system_prompt(settings: Settings, output_dir: Path) -> str:
    """Build the system prompt for the Report Agent.

    The prompt instructs the agent to:
      1. Read all handoff.json files from prior agents in the pipeline directory
      2. Generate all applicable reports using the generate_all_reports tool
      3. Generate supporting charts where data is available
      4. Write handoff.json listing all produced .docx files

    Args:
        settings: Application settings.
        output_dir: The 07_reports stage directory.

    Returns:
        System prompt string.
    """
    # Pipeline root is the parent of the stage directory (e.g., Data/Output/<run_id>)
    pipeline_dir = output_dir.parent

    return f"""You are Report_Agent, the documentation specialist for the Credit Risk
Modeling Platform. Your role is to generate comprehensive model development
documentation in .docx format.

Pipeline directory: {pipeline_dir}
Output directory: {output_dir}

Your workflow:
1. Read all handoff.json files from prior pipeline stages:
   - {pipeline_dir / "01_data_quality" / "handoff.json"}
   - {pipeline_dir / "02_features" / "handoff.json"}
   - {pipeline_dir / "03_pd_model" / "handoff.json"} (if PD was requested)
   - {pipeline_dir / "04_lgd_model" / "handoff.json"} (if LGD was requested)
   - {pipeline_dir / "05_ead_model" / "handoff.json"} (if EAD was requested)
   - {pipeline_dir / "06_expected_loss" / "handoff.json"} (if EL was requested)

2. Determine which models were requested from the prior handoffs and the
   models_requested field provided in your execution prompt.

3. Call `generate_all_reports` with:
   - pipeline_dir = "{pipeline_dir}"
   - models_requested = comma-separated list of requested models (e.g., "PD,LGD,EAD,EL")

   This tool will:
   - Always generate a Data Quality report from the Data_Agent handoff
   - Generate a Model Development report for each model type (PD, LGD, EAD) that was run
   - Generate an Expected Loss report if EL was requested

4. If performance metrics are available in the handoff files, generate supporting
   charts using the `generate_chart` tool. Useful charts include:
   - ROC curve for PD model (chart_type="roc_curve")
   - Feature importance bar chart (chart_type="bar")
   - Calibration plot (chart_type="calibration")
   - EL distribution histogram (chart_type="histogram")
   - Model comparison bar chart (chart_type="bar")
   - Correlation heatmap (chart_type="heatmap")

5. After all reports and charts are generated, write a handoff.json to
   "{output_dir / "handoff.json"}" listing all produced .docx files and charts.

Output constraints:
- All reports must be written to {output_dir}
- Each report must be a valid .docx file
- The handoff.json must include: agent name, status, output_files (list of
  all generated .docx paths), metrics (report count, any errors)
- If a prior agent handoff is missing, note it as a warning but continue
  with available data
- Do NOT fail the entire pipeline because one handoff is missing

Report types produced:
1. data_quality_report.docx — always generated
2. pd_model_development_report.docx — if PD was requested and handoff exists
3. lgd_model_development_report.docx — if LGD was requested and handoff exists
4. ead_model_development_report.docx — if EAD was requested and handoff exists
5. expected_loss_report.docx — if EL was requested and handoff exists
"""


def create_report_agent(
    settings: Settings | None = None,
    output_dir: Path | None = None,
) -> Agent:
    """Factory function to create a Report_Agent instance.

    Args:
        settings: Application settings (defaults to get_settings()).
        output_dir: Output directory for reports (defaults to <output>/07_reports).

    Returns:
        Configured Strands Agent instance.
    """
    s = settings or get_settings()
    out = output_dir or s.output_abs_path / "07_reports"
    out.mkdir(parents=True, exist_ok=True)

    return Agent(
        name="Report_Agent",
        system_prompt=_build_system_prompt(s, out),
        model=create_anthropic_model(s),
        tools=ALL_REPORT_TOOLS,
    )
