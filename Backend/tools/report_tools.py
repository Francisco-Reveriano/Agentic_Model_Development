"""Report Agent tools — generate .docx reports and supporting charts.

Tools:
  1. generate_all_reports  — reads handoff.json files, produces all applicable .docx
  2. generate_chart        — creates matplotlib chart images for report embedding
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — must be before pyplot import
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from strands import tool

from backend.report_generator import ReportGenerator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ok(payload: dict[str, Any]) -> dict:
    return {"status": "success", "content": [{"text": json.dumps(payload, default=str, indent=2)}]}


def _error(message: str) -> dict:
    return {"status": "error", "content": [{"text": message}]}


def _read_handoff(directory: Path) -> dict | None:
    """Read handoff.json from a directory, returning None if missing."""
    path = directory / "handoff.json"
    if path.exists():
        try:
            return json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            return None
    return None


def _collect_charts(directory: Path) -> list[str]:
    """Collect all .png chart paths from a directory."""
    charts: list[str] = []
    if directory.exists():
        for ext in ("*.png", "*.jpg", "*.jpeg"):
            charts.extend(str(p) for p in directory.glob(ext))
    return sorted(charts)


# ---------------------------------------------------------------------------
# Tool 1: generate_all_reports
# ---------------------------------------------------------------------------

@tool
def generate_all_reports(pipeline_dir: str, models_requested: str) -> dict:
    """Read all handoff.json files from the pipeline and generate applicable .docx reports.

    Always produces a Data Quality report.
    Produces a Model Development report for each model (PD, LGD, EAD) that was run.
    Produces an Expected Loss report if EL was requested.

    Args:
        pipeline_dir: Root pipeline output directory (contains 01_data_quality/, etc.).
        models_requested: Comma-separated model types that were requested (e.g. "PD,LGD,EAD,EL").
    """
    try:
        root = Path(pipeline_dir)
        if not root.exists():
            return _error(f"Pipeline directory not found: {pipeline_dir}")

        models = [m.strip().upper() for m in models_requested.split(",") if m.strip()]
        generator = ReportGenerator()

        # Determine output directory for reports
        report_dir = root / "07_reports"
        report_dir.mkdir(parents=True, exist_ok=True)

        generated_files: list[str] = []
        errors: list[str] = []

        # --- Stage directories ---
        dq_dir = root / "01_data_quality"
        feature_dir = root / "02_features"
        pd_dir = root / "03_pd_model"
        lgd_dir = root / "04_lgd_model"
        ead_dir = root / "05_ead_model"
        el_dir = root / "06_expected_loss"

        # --- Read all available handoffs ---
        dq_handoff = _read_handoff(dq_dir)
        feature_handoff = _read_handoff(feature_dir)
        pd_handoff = _read_handoff(pd_dir)
        lgd_handoff = _read_handoff(lgd_dir)
        ead_handoff = _read_handoff(ead_dir)
        el_handoff = _read_handoff(el_dir)

        # --- 1. Always produce DQ report ---
        if dq_handoff:
            try:
                dq_data = dict(dq_handoff)
                dq_data["charts"] = _collect_charts(dq_dir)
                path = generator.generate_dq_report(dq_data, report_dir)
                generated_files.append(str(path))
            except Exception as exc:
                errors.append(f"DQ report generation failed: {exc}")
        else:
            errors.append("No Data_Agent handoff.json found — skipping DQ report.")

        # --- 2. Model Development reports for each requested model ---
        model_handoff_map = {
            "PD": (pd_handoff, pd_dir),
            "LGD": (lgd_handoff, lgd_dir),
            "EAD": (ead_handoff, ead_dir),
        }

        for model_type in ["PD", "LGD", "EAD"]:
            if model_type not in models:
                continue

            handoff, stage_dir = model_handoff_map[model_type]
            if not handoff:
                errors.append(f"No {model_type}_Agent handoff.json found — skipping {model_type} report.")
                continue

            try:
                # Build combined artifacts dict for the model report
                artifacts = dict(handoff)
                artifacts["charts"] = _collect_charts(stage_dir)

                # Enrich with feature info if available
                if feature_handoff:
                    artifacts.setdefault("features", {})
                    if "output_files" in feature_handoff:
                        artifacts["features"]["source_files"] = feature_handoff["output_files"]
                    if "metrics" in feature_handoff:
                        feat_metrics = feature_handoff["metrics"]
                        artifacts["features"].setdefault(
                            "selected_features",
                            feat_metrics.get("selected_features", []),
                        )
                        artifacts["features"].setdefault(
                            "feature_importance",
                            feat_metrics.get("feature_importance", {}),
                        )
                        artifacts.setdefault("metrics", {})["feature_count"] = feat_metrics.get(
                            "feature_count", feat_metrics.get("final_feature_count", "N/A")
                        )

                # Enrich with data info from DQ handoff
                if dq_handoff:
                    artifacts.setdefault("data", {})
                    artifacts["data"]["sources"] = {
                        "Database": str(dq_handoff.get("output_files", {}).get("cleaned_features", "LendingClub SQLite")),
                    }

                path = generator.generate_model_report(model_type, artifacts, report_dir)
                generated_files.append(str(path))
            except Exception as exc:
                errors.append(f"{model_type} model report generation failed: {exc}")

        # --- 3. EL report if requested ---
        if "EL" in models:
            if el_handoff:
                try:
                    el_data = dict(el_handoff)
                    el_data["charts"] = _collect_charts(el_dir)

                    # Enrich with model summaries
                    model_summaries = {}
                    for mt, (ho, _) in model_handoff_map.items():
                        if ho and "metrics" in ho:
                            model_summaries[mt] = ho["metrics"]
                    el_data["model_summaries"] = model_summaries

                    path = generator.generate_el_report(el_data, report_dir)
                    generated_files.append(str(path))
                except Exception as exc:
                    errors.append(f"EL report generation failed: {exc}")
            else:
                errors.append("No EL_Agent handoff.json found — skipping EL report.")

        # --- Write report handoff ---
        handoff_data = {
            "agent": "Report_Agent",
            "status": "success" if generated_files else "failed",
            "output_files": {
                "reports": generated_files,
            },
            "metrics": {
                "reports_generated": len(generated_files),
                "models_requested": models,
                "errors": errors,
            },
        }
        (report_dir / "handoff.json").write_text(
            json.dumps(handoff_data, indent=2, default=str)
        )

        return _ok({
            "reports_generated": generated_files,
            "report_count": len(generated_files),
            "errors": errors if errors else None,
            "output_dir": str(report_dir),
        })

    except Exception as exc:
        return _error(f"Failed to generate reports: {exc}")


# ---------------------------------------------------------------------------
# Tool 2: generate_chart
# ---------------------------------------------------------------------------

@tool
def generate_chart(chart_type: str, data_json: str, output_path: str) -> dict:
    """Generate a chart image using matplotlib.

    Supported chart types: histogram, bar, line, heatmap, roc_curve, calibration.

    Args:
        chart_type: One of "histogram", "bar", "line", "heatmap", "roc_curve", "calibration".
        data_json: JSON string containing the chart data. Structure depends on chart_type:
            - histogram: {"values": [...], "bins": 30, "title": "...", "xlabel": "...", "ylabel": "..."}
            - bar: {"labels": [...], "values": [...], "title": "...", "xlabel": "...", "ylabel": "..."}
            - line: {"x": [...], "y": [...], "title": "...", "xlabel": "...", "ylabel": "...", "series": [{"x":..., "y":..., "label":...}]}
            - heatmap: {"matrix": [[...]], "xlabels": [...], "ylabels": [...], "title": "..."}
            - roc_curve: {"fpr": [...], "tpr": [...], "auc": 0.85, "title": "..."}
            - calibration: {"predicted": [...], "observed": [...], "title": "..."}
        output_path: File path for the output image (e.g., "/path/to/chart.png").
    """
    try:
        data = json.loads(data_json)
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        supported_types = {"histogram", "bar", "line", "heatmap", "roc_curve", "calibration"}
        if chart_type not in supported_types:
            return _error(
                f"Unsupported chart type '{chart_type}'. "
                f"Supported: {', '.join(sorted(supported_types))}"
            )

        # Common styling
        plt.style.use("seaborn-v0_8-whitegrid")
        fig, ax = plt.subplots(figsize=(10, 6))

        title = data.get("title", chart_type.replace("_", " ").title())
        xlabel = data.get("xlabel", "")
        ylabel = data.get("ylabel", "")

        if chart_type == "histogram":
            values = np.array(data["values"], dtype=float)
            bins = data.get("bins", 30)
            ax.hist(values, bins=bins, color="#2E86AB", edgecolor="white", alpha=0.85)
            ax.set_xlabel(xlabel or "Value")
            ax.set_ylabel(ylabel or "Frequency")

        elif chart_type == "bar":
            labels = data["labels"]
            values = [float(v) for v in data["values"]]
            colors = data.get("colors", None)
            if colors is None:
                cmap = plt.cm.get_cmap("Blues")
                max_val = max(values) if values else 1
                colors = [cmap(0.3 + 0.6 * v / max_val) for v in values]
            bars = ax.bar(labels, values, color=colors, edgecolor="white")
            ax.set_xlabel(xlabel or "Category")
            ax.set_ylabel(ylabel or "Value")
            # Add value labels on bars
            for bar_item, val in zip(bars, values):
                ax.text(
                    bar_item.get_x() + bar_item.get_width() / 2,
                    bar_item.get_height() + max(values) * 0.01,
                    f"{val:.4f}" if isinstance(val, float) and val < 10 else f"{val:,.0f}",
                    ha="center", va="bottom", fontsize=8,
                )
            plt.xticks(rotation=45, ha="right")

        elif chart_type == "line":
            series_list = data.get("series", [])
            if series_list:
                for s in series_list:
                    ax.plot(s["x"], s["y"], label=s.get("label", ""), linewidth=2)
                ax.legend()
            else:
                x = data.get("x", list(range(len(data.get("y", [])))))
                y = data["y"]
                ax.plot(x, y, color="#2E86AB", linewidth=2)
            ax.set_xlabel(xlabel or "X")
            ax.set_ylabel(ylabel or "Y")

        elif chart_type == "heatmap":
            matrix = np.array(data["matrix"], dtype=float)
            xlabels = data.get("xlabels", [str(i) for i in range(matrix.shape[1])])
            ylabels = data.get("ylabels", [str(i) for i in range(matrix.shape[0])])
            im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto")
            ax.set_xticks(range(len(xlabels)))
            ax.set_xticklabels(xlabels, rotation=45, ha="right")
            ax.set_yticks(range(len(ylabels)))
            ax.set_yticklabels(ylabels)
            fig.colorbar(im, ax=ax)
            # Add text annotations
            for i in range(matrix.shape[0]):
                for j in range(matrix.shape[1]):
                    ax.text(
                        j, i, f"{matrix[i, j]:.2f}",
                        ha="center", va="center",
                        color="white" if matrix[i, j] > matrix.max() * 0.6 else "black",
                        fontsize=8,
                    )

        elif chart_type == "roc_curve":
            fpr = np.array(data["fpr"], dtype=float)
            tpr = np.array(data["tpr"], dtype=float)
            auc_val = data.get("auc", None)
            label = f"ROC (AUC = {auc_val:.4f})" if auc_val is not None else "ROC Curve"
            ax.plot(fpr, tpr, color="#2E86AB", linewidth=2, label=label)
            ax.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1, label="Random")
            ax.set_xlabel(xlabel or "False Positive Rate")
            ax.set_ylabel(ylabel or "True Positive Rate")
            ax.legend(loc="lower right")
            ax.set_xlim([0, 1])
            ax.set_ylim([0, 1.02])

        elif chart_type == "calibration":
            predicted = np.array(data["predicted"], dtype=float)
            observed = np.array(data["observed"], dtype=float)
            ax.plot(predicted, observed, "o-", color="#2E86AB", linewidth=2, markersize=6, label="Model")
            ax.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1, label="Perfect Calibration")
            ax.set_xlabel(xlabel or "Mean Predicted Probability")
            ax.set_ylabel(ylabel or "Observed Frequency")
            ax.legend(loc="lower right")
            ax.set_xlim([0, 1])
            ax.set_ylim([0, 1.02])

        ax.set_title(title, fontsize=14, fontweight="bold", color="#1A3C6D")
        fig.tight_layout()
        fig.savefig(str(out), dpi=150, bbox_inches="tight")
        plt.close(fig)

        return _ok({
            "chart_type": chart_type,
            "output_path": str(out),
            "title": title,
        })

    except json.JSONDecodeError as exc:
        return _error(f"Invalid JSON in data_json: {exc}")
    except KeyError as exc:
        return _error(f"Missing required key in data_json: {exc}")
    except Exception as exc:
        return _error(f"Failed to generate {chart_type} chart: {exc}")


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

ALL_REPORT_TOOLS = [
    generate_all_reports,
    generate_chart,
]
