"""Feature Agent tools — feature engineering and selection for LendingClub credit risk.

Provides tools for WoE/IV computation, correlation analysis, VIF calculation,
ratio feature engineering, and feature selection pipeline.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from strands import tool

from backend.config import get_settings

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ok(payload: Dict[str, Any]) -> dict:
    return {"status": "success", "content": [{"text": json.dumps(payload, default=str, indent=2)}]}


def _error(message: str) -> dict:
    return {"status": "error", "content": [{"text": message}]}


def _interpret_iv(iv: float) -> str:
    """Return human-readable interpretation of Information Value."""
    if iv < 0.02:
        return "not useful"
    elif iv < 0.10:
        return "weak"
    elif iv < 0.30:
        return "medium"
    elif iv < 0.50:
        return "strong"
    else:
        return "suspicious"


def _resolve_data_dir(data_dir: str) -> Path:
    """Resolve the data directory, falling back to settings default."""
    if data_dir:
        return Path(data_dir)
    settings = get_settings()
    return settings.output_abs_path / "01_data_quality"


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@tool
def load_cleaned_dataset(handoff_dir: str) -> dict:
    """Load the cleaned dataset produced by Data_Agent.

    Reads cleaned_features.parquet and targets.parquet from the specified
    handoff directory and returns shape information.

    Args:
        handoff_dir: Directory containing cleaned_features.parquet and targets.parquet.
    """
    try:
        d = Path(handoff_dir)
        features_path = d / "cleaned_features.parquet"
        targets_path = d / "targets.parquet"

        if not features_path.exists():
            return _error(f"cleaned_features.parquet not found in {d}")
        if not targets_path.exists():
            return _error(f"targets.parquet not found in {d}")

        features = pd.read_parquet(features_path)
        targets = pd.read_parquet(targets_path)

        numeric_cols = features.select_dtypes(include=[np.number]).columns.tolist()
        non_numeric_cols = [c for c in features.columns if c not in numeric_cols]

        null_counts = features.isnull().sum()
        cols_with_nulls = {col: int(cnt) for col, cnt in null_counts.items() if cnt > 0}

        return _ok({
            "features_shape": list(features.shape),
            "targets_shape": list(targets.shape),
            "feature_columns": features.columns.tolist(),
            "target_columns": targets.columns.tolist(),
            "numeric_feature_count": len(numeric_cols),
            "non_numeric_feature_count": len(non_numeric_cols),
            "non_numeric_columns": non_numeric_cols,
            "columns_with_nulls": cols_with_nulls,
            "default_rate": round(float(targets["default_flag"].mean()), 4) if "default_flag" in targets.columns else None,
            "handoff_dir": str(d),
        })
    except Exception as exc:
        return _error(f"Failed to load cleaned dataset: {exc}")


@tool
def compute_woe_iv(feature: str, target_col: str = "default_flag", data_dir: str = "") -> dict:
    """Compute Weight of Evidence and Information Value for a single feature.

    Uses optbinning.OptimalBinning to compute optimal bins, WoE values,
    and the overall Information Value for the feature against the target.

    Args:
        feature: Name of the feature column to analyze.
        target_col: Name of the binary target column (default: 'default_flag').
        data_dir: Directory containing cleaned_features.parquet and targets.parquet.
    """
    try:
        d = _resolve_data_dir(data_dir)
        features = pd.read_parquet(d / "cleaned_features.parquet")
        targets = pd.read_parquet(d / "targets.parquet")

        if feature not in features.columns:
            return _error(f"Feature '{feature}' not found in cleaned_features.parquet. Available: {features.columns.tolist()[:20]}")
        if target_col not in targets.columns:
            return _error(f"Target '{target_col}' not found in targets.parquet. Available: {targets.columns.tolist()}")

        x = features[feature].values.astype(float)
        y = targets[target_col].values.astype(float)

        # Drop rows where feature or target is NaN
        mask = ~(np.isnan(x) | np.isnan(y))
        x = x[mask]
        y = y[mask]

        if len(x) == 0:
            return _error(f"No valid (non-NaN) rows for feature '{feature}'.")

        from optbinning import OptimalBinning

        optb = OptimalBinning(name=feature, dtype="numerical", solver="cp")
        optb.fit(x, y)

        binning_table = optb.binning_table
        table_df = binning_table.build()

        # Extract IV from the binning table
        # The last row of the table contains the totals including overall IV
        iv_value = float(binning_table.iv)

        # Build WoE table as list of dicts for JSON serialization
        woe_records = []
        for idx, row in table_df.iterrows():
            if str(idx) in ("Totals",):
                continue
            record = {
                "bin": str(row.get("Bin", idx)),
                "count": int(row["Count"]) if pd.notna(row.get("Count")) else 0,
                "count_pct": round(float(row["Count (%)"]), 4) if pd.notna(row.get("Count (%)")) else 0.0,
                "event_rate": round(float(row["Event rate"]), 4) if pd.notna(row.get("Event rate")) else 0.0,
                "woe": round(float(row["WoE"]), 4) if pd.notna(row.get("WoE")) else 0.0,
                "iv": round(float(row["IV"]), 6) if pd.notna(row.get("IV")) else 0.0,
            }
            woe_records.append(record)

        interpretation = _interpret_iv(iv_value)

        return _ok({
            "feature": feature,
            "iv": round(iv_value, 6),
            "iv_interpretation": interpretation,
            "n_bins": len(woe_records),
            "n_records_used": int(len(x)),
            "woe_table": woe_records,
        })
    except Exception as exc:
        return _error(f"Failed to compute WoE/IV for '{feature}': {exc}")


@tool
def run_correlation_analysis(threshold: float = 0.85, data_dir: str = "") -> dict:
    """Compute Pearson correlation matrix and flag highly correlated pairs.

    Identifies all feature pairs with absolute correlation exceeding the
    threshold, flagging them for potential removal.

    Args:
        threshold: Absolute correlation threshold for flagging (default: 0.85).
        data_dir: Directory containing cleaned_features.parquet.
    """
    try:
        d = _resolve_data_dir(data_dir)
        features = pd.read_parquet(d / "cleaned_features.parquet")

        numeric_df = features.select_dtypes(include=[np.number])
        if numeric_df.shape[1] < 2:
            return _error("Need at least 2 numeric features for correlation analysis.")

        corr_matrix = numeric_df.corr(method="pearson")

        # Find pairs above threshold
        flagged_pairs: List[Dict[str, Any]] = []
        seen = set()
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                col_i = corr_matrix.columns[i]
                col_j = corr_matrix.columns[j]
                r = corr_matrix.iloc[i, j]
                if abs(r) > threshold:
                    pair_key = tuple(sorted([col_i, col_j]))
                    if pair_key not in seen:
                        seen.add(pair_key)
                        flagged_pairs.append({
                            "feature_1": col_i,
                            "feature_2": col_j,
                            "correlation": round(float(r), 4),
                            "abs_correlation": round(abs(float(r)), 4),
                            "recommendation": "remove_one",
                        })

        # Sort by absolute correlation descending
        flagged_pairs.sort(key=lambda x: x["abs_correlation"], reverse=True)

        # Compute summary statistics on the correlation matrix
        upper_triangle = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape, dtype=bool), k=1)
        )
        all_corrs = upper_triangle.stack().values

        return _ok({
            "total_features_analyzed": int(numeric_df.shape[1]),
            "threshold": threshold,
            "flagged_pair_count": len(flagged_pairs),
            "flagged_pairs": flagged_pairs,
            "correlation_stats": {
                "mean_abs_correlation": round(float(np.mean(np.abs(all_corrs))), 4),
                "median_abs_correlation": round(float(np.median(np.abs(all_corrs))), 4),
                "max_abs_correlation": round(float(np.max(np.abs(all_corrs))), 4),
                "pct_above_050": round(float(np.mean(np.abs(all_corrs) > 0.50) * 100), 2),
                "pct_above_threshold": round(float(np.mean(np.abs(all_corrs) > threshold) * 100), 2),
            },
        })
    except Exception as exc:
        return _error(f"Failed to run correlation analysis: {exc}")


@tool
def compute_vif(data_dir: str = "") -> dict:
    """Compute Variance Inflation Factor for all numeric features.

    Uses statsmodels variance_inflation_factor to identify multicollinearity.
    Features with VIF > 10 are flagged for potential removal.

    Args:
        data_dir: Directory containing cleaned_features.parquet.
    """
    try:
        d = _resolve_data_dir(data_dir)
        features = pd.read_parquet(d / "cleaned_features.parquet")

        numeric_df = features.select_dtypes(include=[np.number]).dropna()

        if numeric_df.shape[1] < 2:
            return _error("Need at least 2 numeric features for VIF computation.")

        # Remove constant columns (zero variance) to avoid infinite VIF
        non_const = numeric_df.loc[:, numeric_df.std() > 0]
        dropped_const = [c for c in numeric_df.columns if c not in non_const.columns]

        from statsmodels.stats.outliers_influence import variance_inflation_factor

        # Add intercept for VIF computation
        X = non_const.values.astype(float)

        vif_data: List[Dict[str, Any]] = []
        for i in range(X.shape[1]):
            try:
                vif_val = variance_inflation_factor(X, i)
            except Exception:
                vif_val = float("inf")
            col_name = non_const.columns[i]
            vif_data.append({
                "feature": col_name,
                "vif": round(float(vif_val), 2) if np.isfinite(vif_val) else 999999.99,
                "flagged": bool(vif_val > 10),
            })

        # Sort by VIF descending
        vif_data.sort(key=lambda x: x["vif"], reverse=True)

        flagged_count = sum(1 for v in vif_data if v["flagged"])
        flagged_features = [v["feature"] for v in vif_data if v["flagged"]]

        return _ok({
            "total_features_analyzed": len(vif_data),
            "flagged_count": flagged_count,
            "flagged_features": flagged_features,
            "dropped_constant_columns": dropped_const,
            "vif_table": vif_data,
            "interpretation": "VIF > 10 indicates high multicollinearity; consider removing or combining.",
        })
    except Exception as exc:
        return _error(f"Failed to compute VIF: {exc}")


@tool
def engineer_ratio_features(data_dir: str = "") -> dict:
    """Create derived ratio features from the cleaned dataset.

    Computes the following ratios:
    - loan_to_income = loan_amnt / annual_inc
    - installment_to_income = installment / (annual_inc / 12)
    - revol_to_total = revol_bal / (tot_cur_bal + 1)
    - credit_utilization = revol_bal / (total_rev_hi_lim + 1)
    - open_acc_ratio = open_acc / (total_acc + 1)

    Writes updated features back to cleaned_features.parquet.

    Args:
        data_dir: Directory containing cleaned_features.parquet.
    """
    try:
        d = _resolve_data_dir(data_dir)
        features_path = d / "cleaned_features.parquet"
        features = pd.read_parquet(features_path)

        created: List[str] = []
        skipped: List[Dict[str, str]] = []

        # loan_to_income
        if "loan_amnt" in features.columns and "annual_inc" in features.columns:
            features["loan_to_income"] = features["loan_amnt"] / features["annual_inc"].replace(0, np.nan)
            features["loan_to_income"] = features["loan_to_income"].fillna(0)
            created.append("loan_to_income")
        else:
            missing = [c for c in ["loan_amnt", "annual_inc"] if c not in features.columns]
            skipped.append({"feature": "loan_to_income", "reason": f"Missing columns: {missing}"})

        # installment_to_income
        if "installment" in features.columns and "annual_inc" in features.columns:
            monthly_inc = features["annual_inc"] / 12
            features["installment_to_income"] = features["installment"] / monthly_inc.replace(0, np.nan)
            features["installment_to_income"] = features["installment_to_income"].fillna(0)
            created.append("installment_to_income")
        else:
            missing = [c for c in ["installment", "annual_inc"] if c not in features.columns]
            skipped.append({"feature": "installment_to_income", "reason": f"Missing columns: {missing}"})

        # revol_to_total
        if "revol_bal" in features.columns and "tot_cur_bal" in features.columns:
            features["revol_to_total"] = features["revol_bal"] / (features["tot_cur_bal"] + 1)
            created.append("revol_to_total")
        else:
            missing = [c for c in ["revol_bal", "tot_cur_bal"] if c not in features.columns]
            skipped.append({"feature": "revol_to_total", "reason": f"Missing columns: {missing}"})

        # credit_utilization
        if "revol_bal" in features.columns and "total_rev_hi_lim" in features.columns:
            features["credit_utilization"] = features["revol_bal"] / (features["total_rev_hi_lim"] + 1)
            created.append("credit_utilization")
        else:
            missing = [c for c in ["revol_bal", "total_rev_hi_lim"] if c not in features.columns]
            skipped.append({"feature": "credit_utilization", "reason": f"Missing columns: {missing}"})

        # open_acc_ratio
        if "open_acc" in features.columns and "total_acc" in features.columns:
            features["open_acc_ratio"] = features["open_acc"] / (features["total_acc"] + 1)
            created.append("open_acc_ratio")
        else:
            missing = [c for c in ["open_acc", "total_acc"] if c not in features.columns]
            skipped.append({"feature": "open_acc_ratio", "reason": f"Missing columns: {missing}"})

        # Replace infinities with NaN, then fill with 0
        features = features.replace([np.inf, -np.inf], np.nan)
        for col in created:
            features[col] = features[col].fillna(0)

        # Write updated features back
        features.to_parquet(features_path, index=False)

        # Compute summary statistics for new features
        ratio_stats = {}
        for col in created:
            s = features[col]
            ratio_stats[col] = {
                "mean": round(float(s.mean()), 6),
                "median": round(float(s.median()), 6),
                "std": round(float(s.std()), 6),
                "min": round(float(s.min()), 6),
                "max": round(float(s.max()), 6),
                "p01": round(float(s.quantile(0.01)), 6),
                "p99": round(float(s.quantile(0.99)), 6),
            }

        return _ok({
            "features_created": created,
            "features_skipped": skipped,
            "total_features_after": int(features.shape[1]),
            "total_rows": int(features.shape[0]),
            "ratio_statistics": ratio_stats,
            "output_path": str(features_path),
        })
    except Exception as exc:
        return _error(f"Failed to engineer ratio features: {exc}")


@tool
def select_features(method: str = "iv", threshold: float = 0.02, data_dir: str = "") -> dict:
    """Select features based on IV threshold, correlation, and VIF analysis.

    Three-stage feature selection:
    1. Remove features with IV below the threshold.
    2. For highly correlated pairs (|r| > 0.85), keep the one with higher IV.
    3. Remove features with VIF > 10 (iteratively, removing highest VIF first).

    Writes the final feature_matrix.parquet and selection_report.json.

    Args:
        method: Selection method — 'iv', 'combined' (iv + correlation + vif), or 'all'.
                Defaults to 'iv'.
        threshold: IV threshold for feature inclusion (default: 0.02).
        data_dir: Directory containing cleaned_features.parquet and targets.parquet.
    """
    try:
        d = _resolve_data_dir(data_dir)
        features = pd.read_parquet(d / "cleaned_features.parquet")
        targets = pd.read_parquet(d / "targets.parquet")

        if "default_flag" not in targets.columns:
            return _error("Target 'default_flag' not found in targets.parquet.")

        numeric_cols = features.select_dtypes(include=[np.number]).columns.tolist()
        y = targets["default_flag"].values.astype(float)

        # --- Stage 1: IV-based selection ---
        from optbinning import OptimalBinning

        iv_scores: Dict[str, float] = {}
        iv_details: List[Dict[str, Any]] = []
        for col in numeric_cols:
            try:
                x = features[col].values.astype(float)
                mask = ~(np.isnan(x) | np.isnan(y))
                x_clean = x[mask]
                y_clean = y[mask]
                if len(x_clean) < 100:
                    iv_scores[col] = 0.0
                    iv_details.append({"feature": col, "iv": 0.0, "interpretation": "not useful", "reason": "insufficient data"})
                    continue

                optb = OptimalBinning(name=col, dtype="numerical", solver="cp")
                optb.fit(x_clean, y_clean)
                iv_val = float(optb.binning_table.iv)
                iv_scores[col] = iv_val
                iv_details.append({
                    "feature": col,
                    "iv": round(iv_val, 6),
                    "interpretation": _interpret_iv(iv_val),
                })
            except Exception as e:
                iv_scores[col] = 0.0
                iv_details.append({"feature": col, "iv": 0.0, "interpretation": "not useful", "reason": str(e)})

        # Sort IV details by IV descending
        iv_details.sort(key=lambda x: x["iv"], reverse=True)

        # Features passing IV threshold
        iv_passed = [col for col in numeric_cols if iv_scores.get(col, 0) >= threshold]
        iv_dropped = [col for col in numeric_cols if iv_scores.get(col, 0) < threshold]

        selected = list(iv_passed)
        removal_log: List[Dict[str, str]] = []
        for col in iv_dropped:
            removal_log.append({
                "feature": col,
                "stage": "iv_filter",
                "reason": f"IV={iv_scores.get(col, 0):.6f} < threshold={threshold}",
            })

        # --- Stage 2: Correlation-based pruning ---
        if method in ("combined", "all") and len(selected) >= 2:
            sel_df = features[selected].select_dtypes(include=[np.number])
            corr_matrix = sel_df.corr(method="pearson")

            to_remove_corr = set()
            for i in range(len(corr_matrix.columns)):
                for j in range(i + 1, len(corr_matrix.columns)):
                    col_i = corr_matrix.columns[i]
                    col_j = corr_matrix.columns[j]
                    r = abs(corr_matrix.iloc[i, j])
                    if r > 0.85:
                        # Remove the one with lower IV
                        iv_i = iv_scores.get(col_i, 0)
                        iv_j = iv_scores.get(col_j, 0)
                        if iv_i >= iv_j:
                            drop_col = col_j
                            keep_col = col_i
                        else:
                            drop_col = col_i
                            keep_col = col_j
                        if drop_col not in to_remove_corr:
                            to_remove_corr.add(drop_col)
                            removal_log.append({
                                "feature": drop_col,
                                "stage": "correlation_filter",
                                "reason": f"|r|={r:.4f} with '{keep_col}'; IV({drop_col})={iv_scores.get(drop_col, 0):.6f} < IV({keep_col})={iv_scores.get(keep_col, 0):.6f}",
                            })

            selected = [c for c in selected if c not in to_remove_corr]

        # --- Stage 3: VIF-based pruning (iterative) ---
        if method in ("combined", "all") and len(selected) >= 2:
            from statsmodels.stats.outliers_influence import variance_inflation_factor

            max_vif_iterations = 20
            iteration = 0
            while iteration < max_vif_iterations and len(selected) >= 2:
                sel_numeric = features[selected].dropna()
                # Remove constant columns
                sel_numeric = sel_numeric.loc[:, sel_numeric.std() > 0]
                if sel_numeric.shape[1] < 2:
                    break

                X = sel_numeric.values.astype(float)
                vifs = {}
                for i in range(X.shape[1]):
                    try:
                        vifs[sel_numeric.columns[i]] = variance_inflation_factor(X, i)
                    except Exception:
                        vifs[sel_numeric.columns[i]] = float("inf")

                max_vif_col = max(vifs, key=vifs.get)
                max_vif_val = vifs[max_vif_col]

                if max_vif_val <= 10:
                    break

                selected.remove(max_vif_col)
                removal_log.append({
                    "feature": max_vif_col,
                    "stage": "vif_filter",
                    "reason": f"VIF={max_vif_val:.2f} > 10 (iteration {iteration + 1})",
                })
                iteration += 1

        # --- Write outputs ---
        feature_matrix = features[selected].copy()
        feature_matrix.to_parquet(d / "feature_matrix.parquet", index=False)

        selection_report = {
            "method": method,
            "iv_threshold": threshold,
            "initial_numeric_features": len(numeric_cols),
            "features_after_iv_filter": len(iv_passed),
            "features_after_all_filters": len(selected),
            "total_removed": len(numeric_cols) - len(selected),
            "selected_features": selected,
            "removal_log": removal_log,
            "iv_details": iv_details,
        }

        (d / "selection_report.json").write_text(
            json.dumps(selection_report, indent=2, default=str)
        )

        return _ok({
            "method": method,
            "iv_threshold": threshold,
            "initial_features": len(numeric_cols),
            "selected_features_count": len(selected),
            "removed_features_count": len(numeric_cols) - len(selected),
            "selected_features": selected,
            "removal_summary": {
                "iv_filter": len(iv_dropped),
                "correlation_filter": sum(1 for r in removal_log if r["stage"] == "correlation_filter"),
                "vif_filter": sum(1 for r in removal_log if r["stage"] == "vif_filter"),
            },
            "output_files": {
                "feature_matrix": str(d / "feature_matrix.parquet"),
                "selection_report": str(d / "selection_report.json"),
            },
        })
    except Exception as exc:
        return _error(f"Failed to select features: {exc}")


@tool
def write_feature_matrix(output_dir: str, data_dir: str = "") -> dict:
    """Write final feature matrix and handoff.json with feature list and statistics.

    Reads the selected feature_matrix.parquet (or cleaned_features.parquet
    if feature_matrix does not exist) and targets, then writes:
    - feature_matrix.parquet (copied to output_dir if different from data_dir)
    - handoff.json with feature list, selection statistics, and summary

    Args:
        output_dir: Directory to write the final feature matrix and handoff.
        data_dir: Directory containing the feature selection outputs.
    """
    try:
        d = _resolve_data_dir(data_dir)
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        # Load feature matrix (prefer selected, fall back to cleaned)
        feature_matrix_path = d / "feature_matrix.parquet"
        if feature_matrix_path.exists():
            feature_matrix = pd.read_parquet(feature_matrix_path)
            source = "feature_matrix.parquet"
        else:
            feature_matrix_path = d / "cleaned_features.parquet"
            if not feature_matrix_path.exists():
                return _error(f"No feature matrix or cleaned features found in {d}")
            feature_matrix = pd.read_parquet(feature_matrix_path)
            source = "cleaned_features.parquet (no selection applied)"

        targets_path = d / "targets.parquet"
        if not targets_path.exists():
            return _error(f"targets.parquet not found in {d}")
        targets = pd.read_parquet(targets_path)

        # Load selection report if available
        selection_report_path = d / "selection_report.json"
        selection_report = {}
        if selection_report_path.exists():
            selection_report = json.loads(selection_report_path.read_text())

        # Compute feature statistics
        feature_stats: List[Dict[str, Any]] = []
        for col in feature_matrix.columns:
            s = feature_matrix[col]
            stat = {
                "feature": col,
                "dtype": str(s.dtype),
                "null_count": int(s.isnull().sum()),
                "null_rate": round(float(s.isnull().mean()), 4),
                "unique_count": int(s.nunique()),
            }
            if pd.api.types.is_numeric_dtype(s):
                stat.update({
                    "mean": round(float(s.mean()), 4),
                    "std": round(float(s.std()), 4),
                    "min": round(float(s.min()), 4),
                    "max": round(float(s.max()), 4),
                })
            feature_stats.append(stat)

        # Write feature matrix to output directory
        output_matrix_path = out / "feature_matrix.parquet"
        feature_matrix.to_parquet(output_matrix_path, index=False)

        # Also copy targets for downstream convenience
        output_targets_path = out / "targets.parquet"
        targets.to_parquet(output_targets_path, index=False)

        # Build handoff
        handoff = {
            "agent": "Feature_Agent",
            "status": "success",
            "source": source,
            "output_files": {
                "feature_matrix": str(output_matrix_path),
                "targets": str(output_targets_path),
            },
            "feature_list": feature_matrix.columns.tolist(),
            "metrics": {
                "total_rows": int(feature_matrix.shape[0]),
                "total_features": int(feature_matrix.shape[1]),
                "default_rate": round(float(targets["default_flag"].mean()), 4) if "default_flag" in targets.columns else None,
                "default_count": int(targets["default_flag"].sum()) if "default_flag" in targets.columns else None,
            },
            "selection_summary": {
                "method": selection_report.get("method", "none"),
                "iv_threshold": selection_report.get("iv_threshold"),
                "initial_features": selection_report.get("initial_numeric_features"),
                "selected_features": selection_report.get("features_after_all_filters"),
                "removed_features": selection_report.get("total_removed"),
            } if selection_report else {},
        }

        handoff_path = out / "handoff.json"
        handoff_path.write_text(json.dumps(handoff, indent=2, default=str))

        return _ok({
            "handoff_path": str(handoff_path),
            "feature_matrix_path": str(output_matrix_path),
            "targets_path": str(output_targets_path),
            "feature_count": int(feature_matrix.shape[1]),
            "row_count": int(feature_matrix.shape[0]),
            "feature_list": feature_matrix.columns.tolist(),
            "feature_stats": feature_stats,
        })
    except Exception as exc:
        return _error(f"Failed to write feature matrix: {exc}")


# --- Collect all tools for agent registration ---
ALL_FEATURE_TOOLS = [
    load_cleaned_dataset,
    compute_woe_iv,
    run_correlation_analysis,
    compute_vif,
    engineer_ratio_features,
    select_features,
    write_feature_matrix,
]
