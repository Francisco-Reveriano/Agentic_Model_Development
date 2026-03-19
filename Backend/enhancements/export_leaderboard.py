"""
Leaderboard Export Utilities

Exports model tournament results in CSV and Excel formats for
stakeholder reporting and model comparison analysis.
"""

from pathlib import Path
from typing import Dict, List, Optional, Any
import json


def export_leaderboard_csv(
    tournament_results: Dict[str, Any],
    output_path: str,
    include_all_phases: bool = True,
) -> str:
    """
    Export tournament leaderboard to CSV format.

    Exports all or filtered tournament candidates with their metrics
    to a CSV file suitable for spreadsheet analysis.

    Args:
        tournament_results: Tournament results dictionary
        output_path: Output file path (e.g., 'leaderboard.csv')
        include_all_phases: Include all phases (True) or Phase 1 only (False)

    Returns:
        Path to created CSV file

    Example:
        >>> results = tournament.run()
        >>> csv_path = export_leaderboard_csv(results, "leaderboard.csv")
        >>> print(f"Leaderboard exported to {csv_path}")
    """
    try:
        import csv
    except ImportError:
        raise ImportError("CSV module required")

    # Gather candidates from appropriate phases
    all_candidates = []

    # Phase 1: All initial candidates
    phase1_results = tournament_results.get("phase1_results", [])
    for i, result in enumerate(phase1_results):
        if isinstance(result, dict):
            candidate = {
                "rank": i + 1,
                "phase": 1,
                "model": result.get("model", "Unknown"),
            }

            metrics = result.get("metrics", {})
            candidate.update(metrics)

            all_candidates.append(candidate)

    # Phase 3: Refined candidates (if include_all_phases)
    if include_all_phases:
        phase3_results = tournament_results.get("phase3_results", [])
        for i, result in enumerate(phase3_results):
            if isinstance(result, dict):
                # Check if already in all_candidates
                model_name = result.get("model", "Unknown")
                if not any(c["model"] == model_name for c in all_candidates):
                    candidate = {
                        "rank": len(all_candidates) + 1,
                        "phase": 3,
                        "model": model_name,
                    }

                    metrics = result.get("metrics", {})
                    candidate.update(metrics)

                    all_candidates.append(candidate)

    # Write CSV
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    if not all_candidates:
        # Write empty file
        with open(output_file, "w") as f:
            f.write("No candidates to export\n")
        return str(output_file)

    # Get all unique metric keys
    all_keys = set()
    for candidate in all_candidates:
        all_keys.update(candidate.keys())

    fieldnames = ["rank", "phase", "model"] + sorted(
        [k for k in all_keys if k not in ["rank", "phase", "model"]]
    )

    with open(output_file, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for candidate in all_candidates:
            row = {field: candidate.get(field, "") for field in fieldnames}
            writer.writerow(row)

    return str(output_file)


def export_leaderboard_excel(
    tournament_results: Dict[str, Any],
    output_path: str,
    include_all_phases: bool = True,
) -> str:
    """
    Export tournament leaderboard to Excel format.

    Exports candidates with metrics to Excel with multiple sheets for
    different phases and formatted headers/metrics.

    Args:
        tournament_results: Tournament results dictionary
        output_path: Output file path (e.g., 'leaderboard.xlsx')
        include_all_phases: Include all phases (True) or Phase 1 only (False)

    Returns:
        Path to created Excel file

    Raises:
        ImportError: If openpyxl or pandas not available

    Example:
        >>> results = tournament.run()
        >>> excel_path = export_leaderboard_excel(results, "leaderboard.xlsx")
    """
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("pandas required for Excel export")

    try:
        from openpyxl.styles import Font, PatternFill, Alignment
        from openpyxl.utils.dataframe import dataframe_to_rows
    except ImportError:
        raise ImportError("openpyxl required for Excel formatting")

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Prepare Phase 1 data
    phase1_data = []
    phase1_results = tournament_results.get("phase1_results", [])

    for i, result in enumerate(phase1_results):
        if isinstance(result, dict):
            row = {
                "Rank": i + 1,
                "Model": result.get("model", "Unknown"),
            }

            metrics = result.get("metrics", {})
            row.update(metrics)

            phase1_data.append(row)

    # Prepare Phase 3 data (if available)
    phase3_data = []
    if include_all_phases:
        phase3_results = tournament_results.get("phase3_results", [])

        for i, result in enumerate(phase3_results):
            if isinstance(result, dict):
                row = {
                    "Rank": i + 1,
                    "Model": result.get("model", "Unknown"),
                }

                metrics = result.get("metrics", {})
                row.update(metrics)

                phase3_data.append(row)

    # Create Excel workbook with sheets
    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        # Phase 1 sheet
        if phase1_data:
            df_phase1 = pd.DataFrame(phase1_data)
            df_phase1.to_excel(writer, sheet_name="Phase 1", index=False)

        # Phase 3 sheet
        if phase3_data:
            df_phase3 = pd.DataFrame(phase3_data)
            df_phase3.to_excel(writer, sheet_name="Phase 3", index=False)

        # Summary sheet
        summary_data = {
            "Metric": ["Phase 1 Candidates", "Phase 3 Candidates", "Champion"],
            "Value": [
                len(phase1_results),
                len(phase3_results),
                tournament_results.get("champion", "Unknown"),
            ],
        }
        df_summary = pd.DataFrame(summary_data)
        df_summary.to_excel(writer, sheet_name="Summary", index=False)

        # Get workbook and format
        workbook = writer.book

        # Format headers
        header_fill = PatternFill(start_color="366092", end_color="366092", fill_type="solid")
        header_font = Font(bold=True, color="FFFFFF")

        for sheet_name in writer.sheets:
            worksheet = writer.sheets[sheet_name]

            for cell in worksheet[1]:
                cell.fill = header_fill
                cell.font = header_font
                cell.alignment = Alignment(horizontal="center", vertical="center")

            # Auto-adjust column widths
            for column in worksheet.columns:
                max_length = 0
                column_letter = column[0].column_letter

                for cell in column:
                    try:
                        if len(str(cell.value)) > max_length:
                            max_length = len(str(cell.value))
                    except:
                        pass

                adjusted_width = min(max_length + 2, 50)
                worksheet.column_dimensions[column_letter].width = adjusted_width

    return str(output_file)


def export_leaderboard_json(
    tournament_results: Dict[str, Any],
    output_path: str,
) -> str:
    """
    Export tournament leaderboard to JSON format.

    Args:
        tournament_results: Tournament results dictionary
        output_path: Output file path (e.g., 'leaderboard.json')

    Returns:
        Path to created JSON file
    """
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    leaderboard_data = {
        "tournament_metadata": {
            "champion": tournament_results.get("champion"),
            "champion_metrics": tournament_results.get("champion_metrics"),
            "phase1_count": len(tournament_results.get("phase1_results", [])),
            "phase3_count": len(tournament_results.get("phase3_results", [])),
        },
        "phase1_results": tournament_results.get("phase1_results", []),
        "phase3_results": tournament_results.get("phase3_results", []),
    }

    with open(output_file, "w") as f:
        json.dump(leaderboard_data, f, indent=2, default=str)

    return str(output_file)


def format_metrics_for_export(metrics: Dict[str, Any]) -> Dict[str, str]:
    """
    Format metrics for clean export.

    Rounds numerical values and formats for readability.

    Args:
        metrics: Raw metrics dictionary

    Returns:
        Formatted metrics dictionary
    """
    formatted = {}

    for key, value in metrics.items():
        if isinstance(value, float):
            formatted[key] = f"{value:.4f}"
        elif isinstance(value, int):
            formatted[key] = str(value)
        else:
            formatted[key] = str(value)

    return formatted


def generate_leaderboard_summary(
    tournament_results: Dict[str, Any],
) -> str:
    """
    Generate human-readable leaderboard summary.

    Args:
        tournament_results: Tournament results

    Returns:
        Formatted summary text
    """
    phase1_results = tournament_results.get("phase1_results", [])
    champion = tournament_results.get("champion", "Unknown")
    champion_metrics = tournament_results.get("champion_metrics", {})

    summary = f"""
Tournament Leaderboard Summary
==============================

Champion: {champion}
Champion Metrics:
{chr(10).join(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}" for k, v in champion_metrics.items())}

Top 5 Candidates:
"""

    for i, result in enumerate(phase1_results[:5]):
        if isinstance(result, dict):
            model = result.get("model", "Unknown")
            metrics = result.get("metrics", {})
            auc = metrics.get("auc", "N/A")

            if isinstance(auc, float):
                auc = f"{auc:.4f}"

            summary += f"\n  {i+1}. {model:25} (AUC: {auc})"

    return summary.strip()


def export_comparison_table(
    tournament_results: Dict[str, Any],
    output_path: str,
    metrics_to_compare: Optional[List[str]] = None,
) -> str:
    """
    Export focused comparison table with selected metrics.

    Args:
        tournament_results: Tournament results
        output_path: Output file path
        metrics_to_compare: List of metrics to include (default: common metrics)

    Returns:
        Path to created file
    """
    if metrics_to_compare is None:
        metrics_to_compare = ["auc", "accuracy", "precision", "recall", "f1_score"]

    phase1_results = tournament_results.get("phase1_results", [])

    try:
        import csv
    except ImportError:
        raise ImportError("CSV module required")

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["Rank", "Model"] + metrics_to_compare

    with open(output_file, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for i, result in enumerate(phase1_results):
            if isinstance(result, dict):
                row = {
                    "Rank": i + 1,
                    "Model": result.get("model", "Unknown"),
                }

                metrics = result.get("metrics", {})
                for metric in metrics_to_compare:
                    value = metrics.get(metric, "")
                    if isinstance(value, float):
                        value = f"{value:.4f}"
                    row[metric] = value

                writer.writerow(row)

    return str(output_file)
