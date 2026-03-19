"""
Pipeline Run History Tracking

Scans and catalogs completed pipeline runs, enables comparison between runs,
and tracks performance trends over time.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime


def scan_pipeline_runs(
    output_dir: str = "Data/Output",
    pattern: str = "pipeline_run_*",
) -> List[Dict[str, Any]]:
    """
    Scan output directory for completed pipeline runs.

    Returns metadata for all discovered runs in chronological order.

    Args:
        output_dir: Root output directory (default: 'Data/Output')
        pattern: Directory name pattern (default: 'pipeline_run_*')

    Returns:
        List of run metadata dictionaries, each containing:
        - run_id: Directory name (pipeline_run_YYYYMMDD_HHMMSS)
        - timestamp: ISO 8601 timestamp
        - status: 'completed', 'failed', 'partial', 'running'
        - stages_completed: List of completed stage names
        - champion_model: Champion model from final tournament
        - metrics: Champion metrics (AUC, accuracy, etc.)
        - artifacts_count: Number of artifacts produced
        - duration_seconds: Wall-clock execution time

    Example:
        >>> runs = scan_pipeline_runs()
        >>> for run in runs:
        ...     print(f"{run['run_id']}: {run['champion_model']} (AUC={run['metrics']['auc']:.3f})")
    """
    output_path = Path(output_dir)
    if not output_path.exists():
        return []

    runs = []

    # Find all pipeline_run_* directories
    for run_dir in sorted(output_path.glob(pattern)):
        if not run_dir.is_dir():
            continue

        run_meta = _extract_run_metadata(run_dir)
        if run_meta:
            runs.append(run_meta)

    # Sort by timestamp (newest first)
    runs.sort(key=lambda x: x.get("timestamp", ""), reverse=True)

    return runs


def _extract_run_metadata(run_dir: Path) -> Optional[Dict[str, Any]]:
    """
    Extract metadata from a single run directory.

    Args:
        run_dir: Path to pipeline_run_* directory

    Returns:
        Dictionary of run metadata, or None if invalid
    """
    run_id = run_dir.name

    # Parse timestamp from directory name
    try:
        # Format: pipeline_run_YYYYMMDD_HHMMSS
        time_str = run_id.replace("pipeline_run_", "")
        timestamp = datetime.strptime(time_str, "%Y%m%d_%H%M%S").isoformat()
    except (ValueError, AttributeError):
        timestamp = None

    # Check for handoff.json in final stage to determine status
    report_stage = run_dir / "07_reports"
    status = "running"
    stages_completed = []

    for i in range(1, 8):
        stage_dir = run_dir / f"{i:02d}_*"
        # Glob for stage directory
        stage_dirs = list(run_dir.glob(f"{i:02d}_*"))

        if stage_dirs:
            stage_name = stage_dirs[0].name
            stages_completed.append(stage_name)

            if (stage_dirs[0] / "handoff.json").exists():
                with open(stage_dirs[0] / "handoff.json") as f:
                    handoff = json.load(f)
                    if handoff.get("status") == "failed":
                        status = "failed"

    if report_stage.exists() and any(report_stage.glob("*.docx")):
        status = "completed"
    elif len(stages_completed) < 7:
        status = "partial"

    # Extract champion and metrics
    champion_model = None
    metrics = {}

    # Look for tournament results in PD stage
    pd_stage_dirs = list(run_dir.glob("03_*"))
    if pd_stage_dirs:
        tournament_file = pd_stage_dirs[0] / "pd_tournament_results.json"
        if tournament_file.exists():
            try:
                with open(tournament_file) as f:
                    tournament = json.load(f)
                    champion_model = tournament.get("champion")
                    metrics = tournament.get("champion_metrics", {})
            except (json.JSONDecodeError, IOError):
                pass

    # Count artifacts
    artifact_count = len(list(run_dir.rglob("*.parquet"))) + len(
        list(run_dir.rglob("*.joblib"))
    )

    # Estimate duration
    duration = _estimate_duration(run_dir)

    return {
        "run_id": run_id,
        "timestamp": timestamp,
        "status": status,
        "stages_completed": stages_completed,
        "champion_model": champion_model,
        "metrics": metrics,
        "artifacts_count": artifact_count,
        "duration_seconds": duration,
    }


def _estimate_duration(run_dir: Path) -> Optional[int]:
    """
    Estimate pipeline execution duration.

    Args:
        run_dir: Path to pipeline_run_* directory

    Returns:
        Estimated duration in seconds, or None if cannot estimate
    """
    # Look for timestamps in handoff.json files
    timestamps = []

    for handoff_file in run_dir.rglob("handoff.json"):
        try:
            with open(handoff_file) as f:
                handoff = json.load(f)
                if "timestamp" in handoff:
                    timestamps.append(handoff["timestamp"])
        except (json.JSONDecodeError, IOError):
            pass

    if len(timestamps) < 2:
        return None

    # Parse ISO format timestamps
    try:
        times = [datetime.fromisoformat(ts.replace("Z", "+00:00")) for ts in timestamps]
        duration = (times[-1] - times[0]).total_seconds()
        return max(0, int(duration))
    except (ValueError, AttributeError):
        return None


def compare_runs(
    run1_id: str,
    run2_id: str,
    output_dir: str = "Data/Output",
) -> Dict[str, Any]:
    """
    Compare metrics between two pipeline runs.

    Args:
        run1_id: First run ID (e.g., 'pipeline_run_20260319_140000')
        run2_id: Second run ID
        output_dir: Output directory root

    Returns:
        Dictionary with comparison:
        - 'run1': run1 metadata
        - 'run2': run2 metadata
        - 'improvements': Dict of metric changes (run2 - run1)
        - 'run2_better': Bool whether run2 is better overall

    Example:
        >>> comparison = compare_runs(
        ...     'pipeline_run_20260319_140000',
        ...     'pipeline_run_20260319_150000'
        ... )
        >>> print(f"AUC improvement: {comparison['improvements']['auc']:+.3f}")
    """
    runs = scan_pipeline_runs(output_dir)

    run1_meta = next((r for r in runs if r["run_id"] == run1_id), None)
    run2_meta = next((r for r in runs if r["run_id"] == run2_id), None)

    if not run1_meta or not run2_meta:
        return {
            "error": "One or both runs not found",
            "run1": run1_meta,
            "run2": run2_meta,
        }

    # Compare metrics
    improvements = {}
    metrics1 = run1_meta.get("metrics", {})
    metrics2 = run2_meta.get("metrics", {})

    for metric in metrics1.keys():
        if metric in metrics2:
            val1 = metrics1[metric]
            val2 = metrics2[metric]
            if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                improvements[metric] = val2 - val1

    # Determine overall winner
    auc_improvement = improvements.get("auc", 0)
    accuracy_improvement = improvements.get("accuracy", 0)
    overall_improvement = (auc_improvement + accuracy_improvement) / 2

    return {
        "run1": run1_meta,
        "run2": run2_meta,
        "improvements": improvements,
        "overall_improvement": overall_improvement,
        "run2_better": overall_improvement > 0,
    }


def get_run_artifacts(
    run_id: str,
    output_dir: str = "Data/Output",
) -> Dict[str, List[Path]]:
    """
    Get all artifacts from a completed run.

    Args:
        run_id: Run ID
        output_dir: Output directory root

    Returns:
        Dictionary mapping artifact type -> list of file paths:
        - parquet: Feature matrices, results
        - joblib: Trained models
        - json: Metrics, configurations
        - docx: Reports
    """
    run_dir = Path(output_dir) / run_id

    artifacts = {
        "parquet": list(run_dir.rglob("*.parquet")),
        "joblib": list(run_dir.rglob("*.joblib")),
        "json": list(run_dir.rglob("*.json")),
        "docx": list(run_dir.rglob("*.docx")),
        "csv": list(run_dir.rglob("*.csv")),
    }

    return artifacts


def find_best_run(
    output_dir: str = "Data/Output",
    metric: str = "auc",
) -> Optional[Dict[str, Any]]:
    """
    Find the best pipeline run by a specific metric.

    Args:
        output_dir: Output directory root
        metric: Metric to optimize (default: 'auc')

    Returns:
        Best run metadata, or None if no runs exist
    """
    runs = scan_pipeline_runs(output_dir)

    best_run = None
    best_value = -float("inf")

    for run in runs:
        if run["status"] != "completed":
            continue

        value = run.get("metrics", {}).get(metric)
        if value is not None and isinstance(value, (int, float)):
            if value > best_value:
                best_value = value
                best_run = run

    return best_run


def generate_run_report(
    run_id: str,
    output_dir: str = "Data/Output",
) -> str:
    """
    Generate human-readable report for a single run.

    Args:
        run_id: Run ID
        output_dir: Output directory root

    Returns:
        Formatted text report
    """
    runs = scan_pipeline_runs(output_dir)
    run = next((r for r in runs if r["run_id"] == run_id), None)

    if not run:
        return f"Run {run_id} not found"

    report = f"""
Pipeline Run Report
===================

Run ID: {run["run_id"]}
Status: {run["status"]}
Timestamp: {run["timestamp"]}

Stages Completed:
{chr(10).join(f"  - {stage}" for stage in run["stages_completed"])}

Champion Model: {run["champion_model"]}

Metrics:
{chr(10).join(f"  {k}: {v:.4f}" for k, v in run["metrics"].items() if isinstance(v, (int, float)))}

Duration: {run["duration_seconds"]} seconds ({run["duration_seconds"] // 60} minutes)
Artifacts: {run["artifacts_count"]} files generated
"""

    return report.strip()
