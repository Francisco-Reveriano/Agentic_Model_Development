"""EL Agent tools — Expected Loss computation, portfolio rollup, and stress testing.

Combines PD, LGD, and EAD champion models to compute loan-level Expected Loss:
  EL = PD * LGD * EAD
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from strands import tool

from backend.config import get_settings

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ok(payload: Dict[str, Any]) -> dict:
    return {"status": "success", "content": [{"text": json.dumps(payload, default=str, indent=2)}]}


def _error(msg: str) -> dict:
    return {"status": "error", "content": [{"text": msg}]}


def _load_handoff(directory: Path) -> Dict[str, Any]:
    """Load handoff.json from a directory."""
    handoff_path = directory / "handoff.json"
    if not handoff_path.exists():
        return {}
    return json.loads(handoff_path.read_text())


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@tool
def load_champion_models(pd_dir: str, lgd_dir: str, ead_dir: str) -> dict:
    """Load all three champion models (PD, LGD, EAD) from their handoff directories.

    Loads the joblib-serialized champion models and scalers, plus the feature
    lists for each component model. Returns a summary of what was loaded.

    Args:
        pd_dir: Directory containing PD champion model and handoff.json.
        lgd_dir: Directory containing LGD champion models (stage1 + stage2) and handoff.json.
        ead_dir: Directory containing EAD champion model and handoff.json.
    """
    try:
        pd_path = Path(pd_dir)
        lgd_path = Path(lgd_dir)
        ead_path = Path(ead_dir)

        loaded = {}
        errors = []

        # --- PD Model ---
        pd_handoff = _load_handoff(pd_path)
        pd_model_file = pd_path / "pd_champion.joblib"
        pd_scaler_file = pd_path / "pd_scaler.joblib"
        pd_features_file = pd_path / "feature_list.json"

        if pd_model_file.exists():
            pd_model = joblib.load(pd_model_file)
            pd_scaler = joblib.load(pd_scaler_file) if pd_scaler_file.exists() else None
            pd_features = json.loads(pd_features_file.read_text()) if pd_features_file.exists() else []
            loaded["pd"] = {
                "model_type": type(pd_model).__name__,
                "scaler_loaded": pd_scaler is not None,
                "feature_count": len(pd_features),
                "champion": pd_handoff.get("metrics", {}).get("champion", "unknown"),
                "val_auc": pd_handoff.get("metrics", {}).get("val_auc"),
            }
        else:
            errors.append(f"PD champion model not found at {pd_model_file}")

        # --- LGD Model (two-stage) ---
        lgd_handoff = _load_handoff(lgd_path)
        lgd_s1_file = lgd_path / "lgd_stage1_champion.joblib"
        lgd_s2_file = lgd_path / "lgd_stage2_champion.joblib"
        lgd_scaler_file = lgd_path / "lgd_scaler.joblib"
        lgd_features_file = lgd_path / "feature_list.json"

        if lgd_s1_file.exists() and lgd_s2_file.exists():
            lgd_s1_model = joblib.load(lgd_s1_file)
            lgd_s2_model = joblib.load(lgd_s2_file)
            lgd_scaler = joblib.load(lgd_scaler_file) if lgd_scaler_file.exists() else None
            lgd_features = json.loads(lgd_features_file.read_text()) if lgd_features_file.exists() else []
            loaded["lgd"] = {
                "stage1_model_type": type(lgd_s1_model).__name__,
                "stage2_model_type": type(lgd_s2_model).__name__,
                "scaler_loaded": lgd_scaler is not None,
                "feature_count": len(lgd_features),
                "stage1_champion": lgd_handoff.get("metrics", {}).get("stage1_champion", "unknown"),
                "stage2_champion": lgd_handoff.get("metrics", {}).get("stage2_champion", "unknown"),
                "combined_test_rmse": lgd_handoff.get("metrics", {}).get("combined_test_rmse"),
            }
        else:
            missing = []
            if not lgd_s1_file.exists():
                missing.append(str(lgd_s1_file))
            if not lgd_s2_file.exists():
                missing.append(str(lgd_s2_file))
            errors.append(f"LGD champion models not found: {missing}")

        # --- EAD Model ---
        ead_handoff = _load_handoff(ead_path)
        ead_model_file = ead_path / "ead_champion.joblib"
        ead_scaler_file = ead_path / "ead_scaler.joblib"
        ead_features_file = ead_path / "feature_list.json"

        if ead_model_file.exists():
            ead_model = joblib.load(ead_model_file)
            ead_scaler = joblib.load(ead_scaler_file) if ead_scaler_file.exists() else None
            ead_features = json.loads(ead_features_file.read_text()) if ead_features_file.exists() else []
            loaded["ead"] = {
                "model_type": type(ead_model).__name__,
                "scaler_loaded": ead_scaler is not None,
                "feature_count": len(ead_features),
                "champion": ead_handoff.get("metrics", {}).get("champion", "unknown"),
                "val_rmse": ead_handoff.get("metrics", {}).get("val_rmse"),
            }
        else:
            errors.append(f"EAD champion model not found at {ead_model_file}")

        if errors:
            return _ok({
                "status": "partial",
                "loaded_models": loaded,
                "errors": errors,
                "all_loaded": False,
            })

        return _ok({
            "status": "complete",
            "loaded_models": loaded,
            "all_loaded": True,
            "model_count": 3,
        })
    except Exception as exc:
        return _error(f"Failed to load champion models: {exc}")


@tool
def compute_expected_loss(data_dir: str, pd_dir: str, lgd_dir: str, ead_dir: str, output_dir: str) -> dict:
    """Compute loan-level Expected Loss: EL = PD * LGD * EAD.

    Loads test data and all three champion models, generates predictions,
    computes EL at the loan level, and writes el_results.parquet.

    Args:
        data_dir: Directory containing feature_matrix.parquet and targets.parquet.
        pd_dir: Directory containing PD champion model artifacts.
        lgd_dir: Directory containing LGD champion model artifacts.
        ead_dir: Directory containing EAD champion model artifacts.
        output_dir: Directory to write EL results.
    """
    try:
        d = Path(data_dir)
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        # --- Load features and targets ---
        features_path = d / "feature_matrix.parquet"
        if not features_path.exists():
            features_path = d / "cleaned_features.parquet"
        if not features_path.exists():
            return _error(f"No feature matrix found in {d}")

        targets_path = d / "targets.parquet"
        if not targets_path.exists():
            return _error(f"targets.parquet not found in {d}")

        features = pd.read_parquet(features_path)
        targets = pd.read_parquet(targets_path)

        # Align indices
        common_idx = features.index.intersection(targets.index)
        features = features.loc[common_idx]
        targets = targets.loc[common_idx]

        # Use test set (issue_year >= 2017) for EL computation
        if "issue_year" in targets.columns:
            test_mask = targets["issue_year"] >= 2017
        else:
            # Fallback: use all data
            test_mask = pd.Series(True, index=targets.index)

        feat_test = features[test_mask].copy()
        tgt_test = targets[test_mask].copy()

        numeric_cols = feat_test.select_dtypes(include=[np.number]).columns.tolist()
        feat_test = feat_test[numeric_cols]
        feat_test = feat_test.replace([np.inf, -np.inf], np.nan)
        feat_test = feat_test.fillna(feat_test.median())

        X_test = feat_test.values

        # --- Load and apply PD model ---
        pd_path = Path(pd_dir)
        pd_model = joblib.load(pd_path / "pd_champion.joblib")
        pd_scaler_file = pd_path / "pd_scaler.joblib"
        pd_scaler = joblib.load(pd_scaler_file) if pd_scaler_file.exists() else None

        pd_features_file = pd_path / "feature_list.json"
        if pd_features_file.exists():
            pd_feature_list = json.loads(pd_features_file.read_text())
            pd_cols = [c for c in pd_feature_list if c in feat_test.columns]
            X_pd = feat_test[pd_cols].values if pd_cols else X_test
        else:
            X_pd = X_test

        if pd_scaler is not None:
            X_pd = pd_scaler.transform(X_pd)

        if hasattr(pd_model, "predict_proba"):
            pd_predictions = pd_model.predict_proba(X_pd)[:, 1]
        elif hasattr(pd_model, "predict"):
            pd_predictions = pd_model.predict(X_pd)
        else:
            return _error("PD model does not have predict_proba or predict method.")

        pd_predictions = np.clip(pd_predictions, 0, 1)

        # --- Load and apply LGD model (two-stage) ---
        lgd_path = Path(lgd_dir)
        lgd_s1 = joblib.load(lgd_path / "lgd_stage1_champion.joblib")
        lgd_s2 = joblib.load(lgd_path / "lgd_stage2_champion.joblib")
        lgd_scaler_file = lgd_path / "lgd_scaler.joblib"
        lgd_scaler = joblib.load(lgd_scaler_file) if lgd_scaler_file.exists() else None

        lgd_features_file = lgd_path / "feature_list.json"
        if lgd_features_file.exists():
            lgd_feature_list = json.loads(lgd_features_file.read_text())
            lgd_cols = [c for c in lgd_feature_list if c in feat_test.columns]
            X_lgd = feat_test[lgd_cols].values if lgd_cols else X_test
        else:
            X_lgd = X_test

        if lgd_scaler is not None:
            X_lgd = lgd_scaler.transform(X_lgd)

        # Stage 1: P(any_loss)
        if hasattr(lgd_s1, "predict_proba"):
            p_any_loss = lgd_s1.predict_proba(X_lgd)[:, 1]
        else:
            p_any_loss = lgd_s1.predict(X_lgd)

        # Stage 2: E[severity]
        # Check if OLS (statsmodels) model — it needs a constant
        s2_model_type = type(lgd_s2).__name__
        if s2_model_type in ("OLSResults", "RegressionResultsWrapper"):
            import statsmodels.api as sm
            X_lgd_c = sm.add_constant(X_lgd, has_constant="add")
            e_severity = lgd_s2.predict(X_lgd_c)
        else:
            e_severity = lgd_s2.predict(X_lgd)

        e_severity = np.clip(e_severity, 0, 1)

        # Combined LGD = P(any_loss) * E[severity | partial_loss]
        lgd_predictions = np.clip(p_any_loss * e_severity, 0, 1)

        # --- Load and apply EAD model ---
        ead_path = Path(ead_dir)
        ead_model = joblib.load(ead_path / "ead_champion.joblib")
        ead_scaler_file = ead_path / "ead_scaler.joblib"
        ead_scaler = joblib.load(ead_scaler_file) if ead_scaler_file.exists() else None

        ead_features_file = ead_path / "feature_list.json"
        if ead_features_file.exists():
            ead_feature_list = json.loads(ead_features_file.read_text())
            ead_cols = [c for c in ead_feature_list if c in feat_test.columns]
            X_ead = feat_test[ead_cols].values if ead_cols else X_test
        else:
            X_ead = X_test

        if ead_scaler is not None:
            X_ead = ead_scaler.transform(X_ead)

        ead_model_type = type(ead_model).__name__
        if ead_model_type in ("OLSResults", "RegressionResultsWrapper"):
            import statsmodels.api as sm
            X_ead_c = sm.add_constant(X_ead, has_constant="add")
            ead_predictions = ead_model.predict(X_ead_c)
        else:
            ead_predictions = ead_model.predict(X_ead)

        ead_predictions = np.clip(ead_predictions, 0, None)

        # --- Compute Expected Loss ---
        el = pd_predictions * lgd_predictions * ead_predictions

        # Build results DataFrame
        el_df = pd.DataFrame({
            "pd": pd_predictions,
            "lgd": lgd_predictions,
            "ead": ead_predictions,
            "expected_loss": el,
        }, index=feat_test.index)

        # Add target columns for comparison
        if "default_flag" in tgt_test.columns:
            el_df["actual_default"] = tgt_test["default_flag"].values
        if "lgd" in tgt_test.columns:
            el_df["actual_lgd"] = tgt_test["lgd"].values
        if "ead" in tgt_test.columns:
            el_df["actual_ead"] = tgt_test["ead"].values

        # Add grouping columns if available in features
        if "grade_ord" in feat_test.columns:
            grade_map_inv = {1: "A", 2: "B", 3: "C", 4: "D", 5: "E", 6: "F", 7: "G"}
            el_df["grade"] = feat_test["grade_ord"].map(grade_map_inv).values
        if "issue_year" in tgt_test.columns:
            el_df["issue_year"] = tgt_test["issue_year"].values

        # Write EL results
        el_df.to_parquet(out / "el_results.parquet", index=False)

        # Summary statistics
        summary = {
            "total_loans": len(el_df),
            "portfolio_el": round(float(el.sum()), 2),
            "mean_el": round(float(el.mean()), 2),
            "median_el": round(float(np.median(el)), 2),
            "total_ead": round(float(ead_predictions.sum()), 2),
            "el_rate": round(float(el.sum() / max(ead_predictions.sum(), 1)), 6),
            "mean_pd": round(float(pd_predictions.mean()), 4),
            "mean_lgd": round(float(lgd_predictions.mean()), 4),
            "mean_ead": round(float(ead_predictions.mean()), 2),
            "pd_distribution": {
                "p05": round(float(np.percentile(pd_predictions, 5)), 4),
                "p25": round(float(np.percentile(pd_predictions, 25)), 4),
                "p50": round(float(np.percentile(pd_predictions, 50)), 4),
                "p75": round(float(np.percentile(pd_predictions, 75)), 4),
                "p95": round(float(np.percentile(pd_predictions, 95)), 4),
            },
            "lgd_distribution": {
                "p05": round(float(np.percentile(lgd_predictions, 5)), 4),
                "p25": round(float(np.percentile(lgd_predictions, 25)), 4),
                "p50": round(float(np.percentile(lgd_predictions, 50)), 4),
                "p75": round(float(np.percentile(lgd_predictions, 75)), 4),
                "p95": round(float(np.percentile(lgd_predictions, 95)), 4),
            },
            "output_file": str(out / "el_results.parquet"),
        }

        return _ok(summary)
    except Exception as exc:
        return _error(f"Failed to compute expected loss: {exc}")


@tool
def portfolio_rollup(el_path: str, group_by: str = "grade") -> dict:
    """Aggregate Expected Loss by a grouping column.

    Reads el_results.parquet and computes portfolio-level summary statistics
    grouped by the specified column (e.g., grade, issue_year).

    Args:
        el_path: Path to el_results.parquet.
        group_by: Column to group by (default: 'grade'). Common options:
                  'grade', 'issue_year'.
    """
    try:
        el_file = Path(el_path)
        if not el_file.exists():
            return _error(f"EL results file not found: {el_file}")

        el_df = pd.read_parquet(el_file)

        if group_by not in el_df.columns:
            available = el_df.columns.tolist()
            return _error(f"Column '{group_by}' not found in el_results.parquet. Available: {available}")

        # Compute aggregations
        grouped = el_df.groupby(group_by).agg(
            loan_count=("expected_loss", "count"),
            total_el=("expected_loss", "sum"),
            mean_el=("expected_loss", "mean"),
            median_el=("expected_loss", "median"),
            total_ead=("ead", "sum"),
            mean_pd=("pd", "mean"),
            mean_lgd=("lgd", "mean"),
            mean_ead=("ead", "mean"),
        ).reset_index()

        # Compute EL rate = total_el / total_ead
        grouped["el_rate"] = (grouped["total_el"] / grouped["total_ead"].replace(0, np.nan)).fillna(0)

        # Compute share of portfolio EL
        total_portfolio_el = grouped["total_el"].sum()
        grouped["el_share_pct"] = (grouped["total_el"] / max(total_portfolio_el, 1) * 100)

        # Round for readability
        for col in ["total_el", "mean_el", "median_el", "total_ead", "mean_ead"]:
            grouped[col] = grouped[col].round(2)
        for col in ["mean_pd", "mean_lgd", "el_rate", "el_share_pct"]:
            grouped[col] = grouped[col].round(4)

        # Convert to records
        rollup_table = grouped.to_dict(orient="records")

        # Portfolio totals
        portfolio_summary = {
            "group_by": group_by,
            "group_count": len(rollup_table),
            "total_loans": int(grouped["loan_count"].sum()),
            "total_portfolio_el": round(float(total_portfolio_el), 2),
            "total_portfolio_ead": round(float(grouped["total_ead"].sum()), 2),
            "portfolio_el_rate": round(float(total_portfolio_el / max(grouped["total_ead"].sum(), 1)), 6),
        }

        return _ok({
            "portfolio_summary": portfolio_summary,
            "rollup_table": rollup_table,
        })
    except Exception as exc:
        return _error(f"Failed to compute portfolio rollup: {exc}")


@tool
def run_stress_test(el_path: str, scenario: str = "base") -> dict:
    """Apply stress test multipliers to Expected Loss predictions.

    Stress scenarios adjust PD and LGD components:
    - Base: No change (current economic conditions).
    - Adverse: PD * 1.5, LGD floor 0.45 (moderate recession).
    - Severe: PD * 2.0, LGD floor 0.60 (severe economic downturn).

    EL is recomputed as stressed_PD * stressed_LGD * EAD.

    Args:
        el_path: Path to el_results.parquet.
        scenario: Stress scenario name: 'base', 'adverse', or 'severe'.
    """
    try:
        el_file = Path(el_path)
        if not el_file.exists():
            return _error(f"EL results file not found: {el_file}")

        el_df = pd.read_parquet(el_file)

        required_cols = ["pd", "lgd", "ead", "expected_loss"]
        missing = [c for c in required_cols if c not in el_df.columns]
        if missing:
            return _error(f"Missing required columns in el_results: {missing}")

        scenario_lower = scenario.lower().strip()

        # Define stress parameters
        stress_params = {
            "base": {"pd_multiplier": 1.0, "lgd_floor": None, "description": "No adjustment — current economic conditions"},
            "adverse": {"pd_multiplier": 1.5, "lgd_floor": 0.45, "description": "Moderate recession — PD x 1.5, LGD floor 0.45"},
            "severe": {"pd_multiplier": 2.0, "lgd_floor": 0.60, "description": "Severe economic downturn — PD x 2.0, LGD floor 0.60"},
        }

        if scenario_lower not in stress_params:
            return _error(f"Unknown scenario '{scenario}'. Valid: {list(stress_params.keys())}")

        params = stress_params[scenario_lower]

        # Apply stress adjustments
        stressed_pd = np.clip(el_df["pd"].values * params["pd_multiplier"], 0, 1)

        stressed_lgd = el_df["lgd"].values.copy()
        if params["lgd_floor"] is not None:
            stressed_lgd = np.maximum(stressed_lgd, params["lgd_floor"])
        stressed_lgd = np.clip(stressed_lgd, 0, 1)

        ead = el_df["ead"].values
        stressed_el = stressed_pd * stressed_lgd * ead

        # Baseline for comparison
        baseline_el = el_df["expected_loss"].values

        # Summary
        result = {
            "scenario": scenario_lower,
            "description": params["description"],
            "stress_parameters": {
                "pd_multiplier": params["pd_multiplier"],
                "lgd_floor": params["lgd_floor"],
            },
            "baseline": {
                "total_el": round(float(baseline_el.sum()), 2),
                "mean_el": round(float(baseline_el.mean()), 2),
                "mean_pd": round(float(el_df["pd"].mean()), 4),
                "mean_lgd": round(float(el_df["lgd"].mean()), 4),
            },
            "stressed": {
                "total_el": round(float(stressed_el.sum()), 2),
                "mean_el": round(float(stressed_el.mean()), 2),
                "mean_pd": round(float(stressed_pd.mean()), 4),
                "mean_lgd": round(float(stressed_lgd.mean()), 4),
            },
            "impact": {
                "el_change_pct": round(float((stressed_el.sum() - baseline_el.sum()) / max(baseline_el.sum(), 1) * 100), 2),
                "el_change_absolute": round(float(stressed_el.sum() - baseline_el.sum()), 2),
                "pd_change_pct": round(float((stressed_pd.mean() - el_df["pd"].mean()) / max(el_df["pd"].mean(), 1e-10) * 100), 2),
                "lgd_change_pct": round(float((stressed_lgd.mean() - el_df["lgd"].mean()) / max(el_df["lgd"].mean(), 1e-10) * 100), 2),
            },
            "distribution": {
                "stressed_el_p05": round(float(np.percentile(stressed_el, 5)), 2),
                "stressed_el_p25": round(float(np.percentile(stressed_el, 25)), 2),
                "stressed_el_p50": round(float(np.percentile(stressed_el, 50)), 2),
                "stressed_el_p75": round(float(np.percentile(stressed_el, 75)), 2),
                "stressed_el_p95": round(float(np.percentile(stressed_el, 95)), 2),
                "stressed_el_max": round(float(stressed_el.max()), 2),
            },
            "total_loans": len(el_df),
        }

        # Save stressed results alongside the scenario name
        output_dir = el_file.parent
        stressed_df = el_df.copy()
        stressed_df["stressed_pd"] = stressed_pd
        stressed_df["stressed_lgd"] = stressed_lgd
        stressed_df["stressed_el"] = stressed_el
        stressed_df.to_parquet(output_dir / f"el_stressed_{scenario_lower}.parquet", index=False)
        result["output_file"] = str(output_dir / f"el_stressed_{scenario_lower}.parquet")

        return _ok(result)
    except Exception as exc:
        return _error(f"Failed to run stress test: {exc}")


# --- Collect all tools for agent registration ---
ALL_EL_TOOLS = [
    load_champion_models,
    compute_expected_loss,
    portfolio_rollup,
    run_stress_test,
]
