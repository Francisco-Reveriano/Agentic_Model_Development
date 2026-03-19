"""Shared model evaluation tools used by PD, LGD, and EAD agents.

Provides tools for loading data, splitting by vintage, evaluating
classification and regression models, saving model artifacts, and
producing statsmodels regulatory output.
"""

from __future__ import annotations

import json
import shutil
import time
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


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@tool
def load_feature_matrix(handoff_dir: str) -> dict:
    """Load the feature matrix and targets from a Feature_Agent handoff directory.

    Reads feature_matrix.parquet (or cleaned_features.parquet) and targets.parquet
    produced by the Feature_Agent or Data_Agent.

    Args:
        handoff_dir: Path to the directory containing feature_matrix.parquet and targets.parquet.
    """
    try:
        hdir = Path(handoff_dir)
        if not hdir.exists():
            return _error(f"Handoff directory not found: {hdir}")

        # Try feature_matrix.parquet first, fall back to cleaned_features.parquet
        feature_path = hdir / "feature_matrix.parquet"
        if not feature_path.exists():
            feature_path = hdir / "cleaned_features.parquet"
        if not feature_path.exists():
            return _error(
                f"Neither feature_matrix.parquet nor cleaned_features.parquet "
                f"found in {hdir}"
            )

        targets_path = hdir / "targets.parquet"
        if not targets_path.exists():
            return _error(f"targets.parquet not found in {hdir}")

        X = pd.read_parquet(feature_path)
        y = pd.read_parquet(targets_path)

        # Basic validation
        if len(X) != len(y):
            return _error(
                f"Row count mismatch: features={len(X)}, targets={len(y)}. "
                f"Data may be corrupted."
            )

        # Gather column-level info
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        non_numeric_cols = [c for c in X.columns if c not in numeric_cols]
        null_counts = int(X.isnull().sum().sum())

        return _ok({
            "feature_matrix_path": str(feature_path),
            "targets_path": str(targets_path),
            "feature_shape": list(X.shape),
            "target_shape": list(y.shape),
            "feature_columns": list(X.columns),
            "target_columns": list(y.columns),
            "numeric_feature_count": len(numeric_cols),
            "non_numeric_feature_count": len(non_numeric_cols),
            "non_numeric_columns": non_numeric_cols,
            "total_null_values": null_counts,
            "feature_dtypes_summary": X.dtypes.value_counts().to_dict(),
        })
    except Exception as exc:
        return _error(f"Failed to load feature matrix: {exc}")


@tool
def split_by_vintage(data_dir: str) -> dict:
    """Split dataset by vintage year into train/validation/test partitions.

    Uses issue_year from targets.parquet:
    - Train: issue_year <= 2015
    - Validation: issue_year == 2016
    - Test: issue_year >= 2017

    Writes train_idx.npy, val_idx.npy, test_idx.npy to data_dir.

    Args:
        data_dir: Directory containing targets.parquet; split indices are written here.
    """
    try:
        ddir = Path(data_dir)
        targets_path = ddir / "targets.parquet"
        if not targets_path.exists():
            return _error(f"targets.parquet not found in {ddir}")

        targets = pd.read_parquet(targets_path)
        if "issue_year" not in targets.columns:
            return _error("targets.parquet does not contain 'issue_year' column.")

        issue_year = targets["issue_year"]

        train_mask = issue_year <= 2015
        val_mask = issue_year == 2016
        test_mask = issue_year >= 2017

        train_idx = np.where(train_mask)[0]
        val_idx = np.where(val_mask)[0]
        test_idx = np.where(test_mask)[0]

        # Write index arrays
        np.save(ddir / "train_idx.npy", train_idx)
        np.save(ddir / "val_idx.npy", val_idx)
        np.save(ddir / "test_idx.npy", test_idx)

        # Per-split default rates if default_flag is available
        split_info: Dict[str, Any] = {
            "train_size": int(len(train_idx)),
            "val_size": int(len(val_idx)),
            "test_size": int(len(test_idx)),
            "total_rows": int(len(targets)),
            "train_pct": round(len(train_idx) / len(targets) * 100, 1),
            "val_pct": round(len(val_idx) / len(targets) * 100, 1),
            "test_pct": round(len(test_idx) / len(targets) * 100, 1),
            "train_year_range": f"<= 2015",
            "val_year_range": "2016",
            "test_year_range": ">= 2017",
            "index_files": {
                "train": str(ddir / "train_idx.npy"),
                "val": str(ddir / "val_idx.npy"),
                "test": str(ddir / "test_idx.npy"),
            },
        }

        if "default_flag" in targets.columns:
            for name, idx in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
                subset = targets.iloc[idx]
                default_rate = float(subset["default_flag"].mean())
                default_count = int(subset["default_flag"].sum())
                split_info[f"{name}_default_rate"] = round(default_rate, 4)
                split_info[f"{name}_default_count"] = default_count

        # Vintage distribution
        vintage_dist = issue_year.value_counts().sort_index().to_dict()
        split_info["vintage_distribution"] = {int(k): int(v) for k, v in vintage_dist.items()}

        return _ok(split_info)
    except Exception as exc:
        return _error(f"Failed to split by vintage: {exc}")


@tool
def evaluate_classification(y_true_path: str, y_pred_path: str, y_prob_path: str) -> dict:
    """Evaluate a binary classification model with regulatory-grade metrics.

    Computes AUC-ROC, Gini coefficient, KS statistic, Brier score, and
    Hosmer-Lemeshow test.  Returns all metrics with traffic-light status
    (GREEN / YELLOW / RED) based on regulatory thresholds from the PRD.

    Args:
        y_true_path: Path to .npz or .npy file with true binary labels.
        y_pred_path: Path to .npz or .npy file with predicted binary labels.
        y_prob_path: Path to .npz or .npy file with predicted probabilities.
    """
    try:
        from scipy.stats import ks_2samp
        from sklearn.metrics import (
            accuracy_score,
            brier_score_loss,
            classification_report,
            confusion_matrix,
            f1_score,
            precision_score,
            recall_score,
            roc_auc_score,
        )

        # Load arrays — support both .npy and .npz formats
        def _load_array(path: str) -> np.ndarray:
            p = Path(path)
            if p.suffix == ".npz":
                data = np.load(p)
                # Return the first array in the archive
                return data[list(data.keys())[0]]
            return np.load(p)

        y_true = _load_array(y_true_path).ravel()
        y_pred = _load_array(y_pred_path).ravel()
        y_prob = _load_array(y_prob_path).ravel()

        # Core metrics
        auc = float(roc_auc_score(y_true, y_prob))
        gini = 2 * auc - 1

        # KS statistic: max separation between cumulative distributions
        pos_probs = y_prob[y_true == 1]
        neg_probs = y_prob[y_true == 0]
        ks_stat, ks_pvalue = ks_2samp(pos_probs, neg_probs)
        ks_stat = float(ks_stat)

        brier = float(brier_score_loss(y_true, y_prob))
        accuracy = float(accuracy_score(y_true, y_pred))
        precision = float(precision_score(y_true, y_pred, zero_division=0))
        recall = float(recall_score(y_true, y_pred, zero_division=0))
        f1 = float(f1_score(y_true, y_pred, zero_division=0))

        cm = confusion_matrix(y_true, y_pred).tolist()

        # Traffic-light thresholds (from PRD Section 12)
        def _traffic_light_auc(v: float) -> str:
            if v > 0.75:
                return "GREEN"
            elif v >= 0.65:
                return "YELLOW"
            return "RED"

        def _traffic_light_gini(v: float) -> str:
            if v > 0.50:
                return "GREEN"
            elif v >= 0.30:
                return "YELLOW"
            return "RED"

        def _traffic_light_ks(v: float) -> str:
            if v > 0.35:
                return "GREEN"
            elif v >= 0.20:
                return "YELLOW"
            return "RED"

        def _traffic_light_brier(v: float) -> str:
            if v < 0.15:
                return "GREEN"
            elif v <= 0.25:
                return "YELLOW"
            return "RED"

        metrics: Dict[str, Any] = {
            "auc": round(auc, 6),
            "auc_status": _traffic_light_auc(auc),
            "gini": round(gini, 6),
            "gini_status": _traffic_light_gini(gini),
            "ks_statistic": round(ks_stat, 6),
            "ks_pvalue": round(float(ks_pvalue), 6),
            "ks_status": _traffic_light_ks(ks_stat),
            "brier_score": round(brier, 6),
            "brier_status": _traffic_light_brier(brier),
            "accuracy": round(accuracy, 6),
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1_score": round(f1, 6),
            "confusion_matrix": cm,
        }

        # Hosmer-Lemeshow test (decile-based)
        try:
            n_groups = 10
            sorted_idx = np.argsort(y_prob)
            groups = np.array_split(sorted_idx, n_groups)

            hl_stat = 0.0
            hl_table = []
            for g_idx, group in enumerate(groups):
                obs = y_true[group]
                pred = y_prob[group]
                n_g = len(group)
                obs_events = float(obs.sum())
                exp_events = float(pred.sum())
                obs_non = n_g - obs_events
                exp_non = n_g - exp_events

                if exp_events > 0:
                    hl_stat += (obs_events - exp_events) ** 2 / exp_events
                if exp_non > 0:
                    hl_stat += (obs_non - exp_non) ** 2 / exp_non

                hl_table.append({
                    "decile": g_idx + 1,
                    "n": n_g,
                    "observed_events": int(obs_events),
                    "expected_events": round(exp_events, 2),
                    "observed_non_events": int(obs_non),
                    "expected_non_events": round(exp_non, 2),
                })

            from scipy.stats import chi2
            hl_df = n_groups - 2
            hl_pvalue = float(1 - chi2.cdf(hl_stat, hl_df))

            def _traffic_light_hl(p: float) -> str:
                if p > 0.10:
                    return "GREEN"
                elif p >= 0.05:
                    return "YELLOW"
                return "RED"

            metrics["hosmer_lemeshow_statistic"] = round(hl_stat, 4)
            metrics["hosmer_lemeshow_pvalue"] = round(hl_pvalue, 6)
            metrics["hosmer_lemeshow_status"] = _traffic_light_hl(hl_pvalue)
            metrics["hosmer_lemeshow_table"] = hl_table
        except Exception as hl_exc:
            metrics["hosmer_lemeshow_error"] = str(hl_exc)

        # Overall pass/fail summary
        statuses = [v for k, v in metrics.items() if k.endswith("_status")]
        red_count = statuses.count("RED")
        yellow_count = statuses.count("YELLOW")
        if red_count > 0:
            metrics["overall_status"] = "RED"
        elif yellow_count > 0:
            metrics["overall_status"] = "YELLOW"
        else:
            metrics["overall_status"] = "GREEN"

        metrics["red_count"] = red_count
        metrics["yellow_count"] = yellow_count
        metrics["green_count"] = statuses.count("GREEN")

        return _ok(metrics)
    except Exception as exc:
        return _error(f"Failed to evaluate classification model: {exc}")


@tool
def evaluate_regression(y_true_path: str, y_pred_path: str) -> dict:
    """Evaluate a regression model with RMSE, MAE, and R-squared.

    Args:
        y_true_path: Path to .npy or .npz file with true continuous values.
        y_pred_path: Path to .npy or .npz file with predicted continuous values.
    """
    try:
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

        def _load_array(path: str) -> np.ndarray:
            p = Path(path)
            if p.suffix == ".npz":
                data = np.load(p)
                return data[list(data.keys())[0]]
            return np.load(p)

        y_true = _load_array(y_true_path).ravel()
        y_pred = _load_array(y_pred_path).ravel()

        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        mae = float(mean_absolute_error(y_true, y_pred))
        r2 = float(r2_score(y_true, y_pred))

        # Residual analysis
        residuals = y_true - y_pred
        residual_mean = float(np.mean(residuals))
        residual_std = float(np.std(residuals))
        residual_skew = float(pd.Series(residuals).skew())
        residual_kurtosis = float(pd.Series(residuals).kurtosis())

        # Decile analysis
        sorted_idx = np.argsort(y_pred)
        n_groups = 10
        groups = np.array_split(sorted_idx, n_groups)
        decile_table = []
        for g_idx, group in enumerate(groups):
            decile_table.append({
                "decile": g_idx + 1,
                "n": len(group),
                "mean_actual": round(float(y_true[group].mean()), 6),
                "mean_predicted": round(float(y_pred[group].mean()), 6),
                "abs_deviation": round(
                    float(abs(y_true[group].mean() - y_pred[group].mean())), 6
                ),
            })

        mean_abs_decile_deviation = float(
            np.mean([d["abs_deviation"] for d in decile_table])
        )

        metrics = {
            "rmse": round(rmse, 6),
            "mae": round(mae, 6),
            "r2": round(r2, 6),
            "residual_mean": round(residual_mean, 6),
            "residual_std": round(residual_std, 6),
            "residual_skewness": round(residual_skew, 4),
            "residual_kurtosis": round(residual_kurtosis, 4),
            "mean_abs_decile_deviation": round(mean_abs_decile_deviation, 6),
            "decile_table": decile_table,
            "n_samples": int(len(y_true)),
        }

        return _ok(metrics)
    except Exception as exc:
        return _error(f"Failed to evaluate regression model: {exc}")


@tool
def save_model_artifact(
    model_path: str,
    model_type: str,
    algorithm: str,
    metrics: str,
    output_dir: str,
) -> dict:
    """Save a trained model artifact and register it in model_registry.json.

    Copies the joblib model file to the output directory and appends
    an entry to model_registry.json with metadata and metrics.

    Args:
        model_path: Path to the joblib-serialized model file.
        model_type: Model category — 'PD', 'LGD', or 'EAD'.
        algorithm: Algorithm name (e.g., 'XGBoost', 'Logistic Regression L2').
        metrics: JSON string with evaluation metrics.
        output_dir: Directory to copy the model into and write the registry.
    """
    try:
        src = Path(model_path)
        if not src.exists():
            return _error(f"Model file not found: {src}")

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        # Copy model file
        dest = out / src.name
        shutil.copy2(src, dest)

        # Parse metrics
        try:
            metrics_dict = json.loads(metrics)
        except json.JSONDecodeError:
            metrics_dict = {"raw": metrics}

        # Build registry entry
        entry = {
            "model_type": model_type,
            "algorithm": algorithm,
            "model_file": str(dest),
            "source_file": str(src),
            "registered_at": pd.Timestamp.now(tz="UTC").isoformat(),
            "metrics": metrics_dict,
        }

        # Read or create registry
        registry_path = out / "model_registry.json"
        if registry_path.exists():
            registry = json.loads(registry_path.read_text())
        else:
            registry = {"models": []}

        registry["models"].append(entry)
        registry["last_updated"] = pd.Timestamp.now(tz="UTC").isoformat()

        registry_path.write_text(json.dumps(registry, indent=2, default=str))

        return _ok({
            "model_file": str(dest),
            "registry_path": str(registry_path),
            "model_type": model_type,
            "algorithm": algorithm,
            "total_registered_models": len(registry["models"]),
        })
    except Exception as exc:
        return _error(f"Failed to save model artifact: {exc}")


@tool
def produce_statsmodels_output(
    data_dir: str,
    target: str = "default_flag",
    model_type: str = "logit",
) -> dict:
    """Run a statsmodels Logit or OLS regression and return the full coefficient table.

    This is always produced for regulatory documentation, regardless of which
    model wins the tournament.  Returns coefficients, standard errors,
    z/t-statistics, p-values, confidence intervals, and odds ratios (for Logit).

    Args:
        data_dir: Directory containing feature_matrix.parquet (or cleaned_features.parquet)
                  and targets.parquet, plus train_idx.npy.
        target: Target column name in targets.parquet (default: 'default_flag').
        model_type: 'logit' for Logistic regression or 'ols' for OLS linear regression.
    """
    try:
        import statsmodels.api as sm

        ddir = Path(data_dir)

        # Load features
        feature_path = ddir / "feature_matrix.parquet"
        if not feature_path.exists():
            feature_path = ddir / "cleaned_features.parquet"
        if not feature_path.exists():
            return _error(f"No feature matrix found in {ddir}")

        X = pd.read_parquet(feature_path)
        targets = pd.read_parquet(ddir / "targets.parquet")

        if target not in targets.columns:
            return _error(f"Target column '{target}' not found. Available: {list(targets.columns)}")

        y = targets[target]

        # Use training split if available
        train_idx_path = ddir / "train_idx.npy"
        if train_idx_path.exists():
            train_idx = np.load(train_idx_path)
            X_train = X.iloc[train_idx].copy()
            y_train = y.iloc[train_idx].copy()
        else:
            X_train = X.copy()
            y_train = y.copy()

        # Drop rows with any remaining NaN
        mask = X_train.notna().all(axis=1) & y_train.notna()
        X_train = X_train.loc[mask]
        y_train = y_train.loc[mask]

        # Add constant for intercept
        X_train_const = sm.add_constant(X_train, has_constant="add")

        # Fit model
        model_type_lower = model_type.lower()
        if model_type_lower == "logit":
            model = sm.Logit(y_train, X_train_const)
            result = model.fit(method="bfgs", maxiter=1000, disp=False)
        elif model_type_lower in ("ols", "linear"):
            model = sm.OLS(y_train, X_train_const)
            result = model.fit()
        else:
            return _error(f"Unsupported model_type: {model_type}. Use 'logit' or 'ols'.")

        # Build coefficient table
        coef_table = []
        params = result.params
        std_errs = result.bse
        conf_int = result.conf_int()

        # z-values for logit, t-values for OLS
        if model_type_lower == "logit":
            test_stats = result.zvalues
            stat_name = "z_value"
        else:
            test_stats = result.tvalues
            stat_name = "t_value"

        pvalues = result.pvalues

        for i, col_name in enumerate(X_train_const.columns):
            entry: Dict[str, Any] = {
                "variable": str(col_name),
                "coefficient": round(float(params.iloc[i]), 6),
                "std_error": round(float(std_errs.iloc[i]), 6),
                stat_name: round(float(test_stats.iloc[i]), 4),
                "p_value": round(float(pvalues.iloc[i]), 6),
                "ci_lower": round(float(conf_int.iloc[i, 0]), 6),
                "ci_upper": round(float(conf_int.iloc[i, 1]), 6),
                "significant_5pct": bool(pvalues.iloc[i] < 0.05),
                "significant_1pct": bool(pvalues.iloc[i] < 0.01),
            }
            if model_type_lower == "logit":
                entry["odds_ratio"] = round(float(np.exp(params.iloc[i])), 6)
            coef_table.append(entry)

        # Model summary statistics
        summary_stats: Dict[str, Any] = {
            "model_type": model_type_lower,
            "n_observations": int(result.nobs),
            "n_features": len(X_train.columns),
            "converged": bool(getattr(result, "mle_retvals", {}).get("converged", True)),
        }

        if model_type_lower == "logit":
            summary_stats["pseudo_r2_mcfadden"] = round(float(result.prsquared), 6)
            summary_stats["log_likelihood"] = round(float(result.llf), 4)
            summary_stats["aic"] = round(float(result.aic), 4)
            summary_stats["bic"] = round(float(result.bic), 4)
            summary_stats["llr_pvalue"] = round(float(result.llr_pvalue), 6)
        else:
            summary_stats["r_squared"] = round(float(result.rsquared), 6)
            summary_stats["adj_r_squared"] = round(float(result.rsquared_adj), 6)
            summary_stats["f_statistic"] = round(float(result.fvalue), 4)
            summary_stats["f_pvalue"] = round(float(result.f_pvalue), 6)
            summary_stats["aic"] = round(float(result.aic), 4)
            summary_stats["bic"] = round(float(result.bic), 4)

        # Count significant predictors
        sig_5pct = sum(1 for e in coef_table if e["significant_5pct"] and e["variable"] != "const")
        sig_1pct = sum(1 for e in coef_table if e["significant_1pct"] and e["variable"] != "const")
        total_predictors = len(coef_table) - 1  # Exclude intercept

        summary_stats["significant_at_5pct"] = sig_5pct
        summary_stats["significant_at_1pct"] = sig_1pct
        summary_stats["total_predictors"] = total_predictors

        # Save the full summary to a text file
        summary_text = result.summary().as_text()
        summary_path = ddir / f"statsmodels_{model_type_lower}_summary.txt"
        summary_path.write_text(summary_text)

        # Save coefficient table as CSV
        coef_df = pd.DataFrame(coef_table)
        coef_csv_path = ddir / f"statsmodels_{model_type_lower}_coefficients.csv"
        coef_df.to_csv(coef_csv_path, index=False)

        return _ok({
            "summary_stats": summary_stats,
            "coefficient_table": coef_table,
            "output_files": {
                "summary_text": str(summary_path),
                "coefficient_csv": str(coef_csv_path),
            },
        })
    except Exception as exc:
        return _error(f"Failed to produce statsmodels output: {exc}")


# ---------------------------------------------------------------------------
# Export all tools
# ---------------------------------------------------------------------------

ALL_MODEL_TOOLS = [
    load_feature_matrix,
    split_by_vintage,
    evaluate_classification,
    evaluate_regression,
    save_model_artifact,
    produce_statsmodels_output,
]
