"""EAD Agent tools — Exposure at Default model tournament and amortization utilities.

Regression tournament on EAD target with CCF (Credit Conversion Factor) validation.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import (
    ElasticNet,
    HuberRegressor,
    Lasso,
    Ridge,
)
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.preprocessing import StandardScaler
from strands import tool

from backend.config import get_settings

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ok(payload: Dict[str, Any]) -> dict:
    return {"status": "success", "content": [{"text": json.dumps(payload, default=str, indent=2)}]}


def _error(msg: str) -> dict:
    return {"status": "error", "content": [{"text": msg}]}


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@tool
def define_ead_candidates() -> dict:
    """Return the catalogue of 9 EAD regression model candidates.

    Candidates: OLS, Ridge, Lasso, ElasticNet, Huber, RF Regressor,
    GBM Regressor, XGBoost Regressor, LightGBM Regressor.

    Returns candidate definitions with baseline configs and hyperparam
    search distributions.
    """
    try:
        candidates = [
            {
                "name": "OLS",
                "library": "statsmodels",
                "baseline_config": {},
                "hyperparam_distributions": {},
            },
            {
                "name": "Ridge",
                "library": "sklearn",
                "baseline_config": {"alpha": 1.0},
                "hyperparam_distributions": {"alpha": "loguniform(0.01, 100.0)"},
            },
            {
                "name": "Lasso",
                "library": "sklearn",
                "baseline_config": {"alpha": 0.1, "max_iter": 5000},
                "hyperparam_distributions": {"alpha": "loguniform(0.01, 100.0)"},
            },
            {
                "name": "ElasticNet",
                "library": "sklearn",
                "baseline_config": {"alpha": 0.5, "l1_ratio": 0.5, "max_iter": 5000},
                "hyperparam_distributions": {
                    "alpha": "loguniform(0.01, 100.0)",
                    "l1_ratio": "uniform(0.1, 0.9)",
                },
            },
            {
                "name": "HuberRegressor",
                "library": "sklearn",
                "baseline_config": {"epsilon": 1.35, "max_iter": 500},
                "hyperparam_distributions": {"epsilon": "uniform(1.1, 0.9)"},
            },
            {
                "name": "RandomForestRegressor",
                "library": "sklearn",
                "baseline_config": {"n_estimators": 200, "max_depth": 6, "min_samples_leaf": 100, "n_jobs": -1},
                "hyperparam_distributions": {
                    "n_estimators": "randint(100, 800)",
                    "max_depth": "randint(3, 10)",
                    "min_samples_leaf": "randint(50, 500)",
                },
            },
            {
                "name": "GBMRegressor",
                "library": "sklearn",
                "baseline_config": {"n_estimators": 200, "max_depth": 3, "loss": "huber", "learning_rate": 0.05},
                "hyperparam_distributions": {
                    "n_estimators": "randint(100, 600)",
                    "max_depth": "randint(2, 6)",
                    "learning_rate": "uniform(0.01, 0.15)",
                    "subsample": "uniform(0.6, 0.4)",
                },
            },
            {
                "name": "XGBoostRegressor",
                "library": "xgboost",
                "baseline_config": {
                    "n_estimators": 300, "max_depth": 3, "learning_rate": 0.03, "reg_alpha": 0.1,
                },
                "hyperparam_distributions": {
                    "n_estimators": "randint(200, 800)",
                    "max_depth": "randint(3, 7)",
                    "learning_rate": "uniform(0.01, 0.09)",
                    "reg_alpha": "uniform(0, 1.0)",
                    "reg_lambda": "uniform(0.5, 2.0)",
                },
            },
            {
                "name": "LightGBMRegressor",
                "library": "lightgbm",
                "baseline_config": {
                    "n_estimators": 300, "num_leaves": 31, "learning_rate": 0.03,
                    "min_child_samples": 100, "verbosity": -1,
                },
                "hyperparam_distributions": {
                    "n_estimators": "randint(200, 800)",
                    "num_leaves": "randint(15, 63)",
                    "learning_rate": "uniform(0.01, 0.09)",
                    "min_child_samples": "randint(50, 300)",
                    "colsample_bytree": "uniform(0.5, 0.5)",
                },
            },
        ]

        return _ok({
            "total_candidates": len(candidates),
            "task_type": "regression",
            "target": "ead (exposure at default)",
            "candidates": candidates,
        })
    except Exception as exc:
        return _error(f"Failed to define EAD candidates: {exc}")


@tool
def construct_ead_target(data_dir: str) -> dict:
    """Extract EAD target and CCF from targets.parquet.

    Reads targets.parquet, extracts the ead column and computes the
    Credit Conversion Factor (CCF = ead / funded_amnt).

    Args:
        data_dir: Directory containing targets.parquet from Data_Agent.
    """
    try:
        d = Path(data_dir)
        targets_path = d / "targets.parquet"
        if not targets_path.exists():
            return _error(f"targets.parquet not found in {d}")

        targets = pd.read_parquet(targets_path)

        if "ead" not in targets.columns:
            return _error("Column 'ead' not found in targets.parquet.")

        ead = targets["ead"]
        total_rows = len(ead)
        non_null_count = int(ead.notna().sum())

        # EAD statistics
        ead_clean = ead.dropna()
        ead_stats = {
            "count": non_null_count,
            "null_count": total_rows - non_null_count,
            "mean": round(float(ead_clean.mean()), 2),
            "median": round(float(ead_clean.median()), 2),
            "std": round(float(ead_clean.std()), 2),
            "min": round(float(ead_clean.min()), 2),
            "max": round(float(ead_clean.max()), 2),
            "p05": round(float(ead_clean.quantile(0.05)), 2),
            "p25": round(float(ead_clean.quantile(0.25)), 2),
            "p75": round(float(ead_clean.quantile(0.75)), 2),
            "p95": round(float(ead_clean.quantile(0.95)), 2),
        }

        # CCF statistics
        ccf_result = {}
        if "ccf" in targets.columns:
            ccf = targets["ccf"].dropna()
            ccf_result = {
                "count": int(len(ccf)),
                "mean": round(float(ccf.mean()), 4),
                "median": round(float(ccf.median()), 4),
                "std": round(float(ccf.std()), 4),
                "min": round(float(ccf.min()), 4),
                "max": round(float(ccf.max()), 4),
                "pct_ccf_gt_1": round(float((ccf > 1.0).mean() * 100), 2),
                "pct_ccf_eq_1": round(float((ccf == 1.0).mean() * 100), 2),
                "pct_ccf_lt_0_5": round(float((ccf < 0.5).mean() * 100), 2),
            }

        # Default vs non-default EAD comparison
        ead_by_default = {}
        if "default_flag" in targets.columns:
            for flag_val, label in [(0, "non_default"), (1, "default")]:
                subset = targets[targets["default_flag"] == flag_val]["ead"].dropna()
                if len(subset) > 0:
                    ead_by_default[label] = {
                        "count": int(len(subset)),
                        "mean_ead": round(float(subset.mean()), 2),
                        "median_ead": round(float(subset.median()), 2),
                    }

        return _ok({
            "total_rows": total_rows,
            "ead_stats": ead_stats,
            "ccf_stats": ccf_result,
            "ead_by_default_status": ead_by_default,
        })
    except Exception as exc:
        return _error(f"Failed to construct EAD target: {exc}")


@tool
def compute_amortization_schedule(funded_amnt: float, int_rate: float, term: int) -> dict:
    """Compute a theoretical amortization schedule for a single loan.

    Calculates the month-by-month principal, interest, and remaining balance
    for a fully amortizing loan. Useful for validating EAD predictions against
    theoretical exposure curves.

    Args:
        funded_amnt: Original funded amount (principal).
        int_rate: Annual interest rate as a percentage (e.g., 12.5 for 12.5%).
        term: Loan term in months (e.g., 36 or 60).
    """
    try:
        if funded_amnt <= 0:
            return _error("funded_amnt must be positive.")
        if int_rate < 0:
            return _error("int_rate must be non-negative.")
        if term <= 0:
            return _error("term must be a positive integer.")

        monthly_rate = int_rate / 100.0 / 12.0

        # Monthly payment calculation (standard amortization formula)
        if monthly_rate > 0:
            payment = funded_amnt * (monthly_rate * (1 + monthly_rate) ** term) / ((1 + monthly_rate) ** term - 1)
        else:
            payment = funded_amnt / term

        schedule: List[Dict[str, Any]] = []
        balance = funded_amnt
        total_interest = 0.0
        total_principal = 0.0

        for month in range(1, term + 1):
            interest_payment = balance * monthly_rate
            principal_payment = payment - interest_payment
            balance = max(balance - principal_payment, 0.0)

            total_interest += interest_payment
            total_principal += principal_payment

            schedule.append({
                "month": month,
                "payment": round(payment, 2),
                "principal": round(principal_payment, 2),
                "interest": round(interest_payment, 2),
                "remaining_balance": round(balance, 2),
                "cumulative_principal_pct": round(total_principal / funded_amnt * 100, 2),
            })

        # Summary at key milestones
        milestones = {}
        for pct in [0.25, 0.50, 0.75]:
            target_month = int(term * pct)
            if 0 < target_month <= len(schedule):
                entry = schedule[target_month - 1]
                milestones[f"{int(pct * 100)}pct_term"] = {
                    "month": target_month,
                    "remaining_balance": entry["remaining_balance"],
                    "remaining_pct": round(entry["remaining_balance"] / funded_amnt * 100, 2),
                }

        return _ok({
            "funded_amnt": funded_amnt,
            "int_rate": int_rate,
            "term": term,
            "monthly_payment": round(payment, 2),
            "total_interest": round(total_interest, 2),
            "total_payments": round(total_interest + total_principal, 2),
            "milestones": milestones,
            "schedule": schedule,
        })
    except Exception as exc:
        return _error(f"Failed to compute amortization schedule: {exc}")


@tool
def run_ead_tournament(data_dir: str, output_dir: str) -> dict:
    """Run EAD regression tournament with CCF validation.

    Trains all 9 regression candidates on the EAD target, evaluates on
    validation/test sets, and validates predictions via CCF reasonableness.

    Args:
        data_dir: Directory containing feature_matrix.parquet and targets.parquet.
        output_dir: Directory to write model artifacts and results.
    """
    try:
        d = Path(data_dir)
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        settings = get_settings()

        # --- Load data ---
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

        if "ead" not in targets.columns:
            return _error("Column 'ead' not found in targets.")
        if "issue_year" not in targets.columns:
            return _error("Column 'issue_year' not found in targets — needed for vintage split.")

        # Use only numeric columns
        numeric_cols = features.select_dtypes(include=[np.number]).columns.tolist()
        features = features[numeric_cols]

        # Handle NaN/inf
        features = features.replace([np.inf, -np.inf], np.nan)
        features = features.fillna(features.median())

        # EAD target
        y = targets["ead"].values
        # Keep track of funded_amnt for CCF validation (from features if available)
        funded_amnt_col = None
        if "funded_amnt" in features.columns:
            funded_amnt_col = features["funded_amnt"].values

        # Vintage-based split
        train_mask = targets["issue_year"] <= 2015
        val_mask = targets["issue_year"] == 2016
        test_mask = targets["issue_year"] >= 2017

        X_train = features[train_mask].values
        X_val = features[val_mask].values
        X_test = features[test_mask].values

        y_train = y[train_mask.values]
        y_val = y[val_mask.values]
        y_test = y[test_mask.values]

        # Scale features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)

        # --- Define models ---
        models = {
            "Ridge": Ridge(alpha=1.0),
            "Lasso": Lasso(alpha=0.1, max_iter=5000),
            "ElasticNet": ElasticNet(alpha=0.5, l1_ratio=0.5, max_iter=5000),
            "HuberRegressor": HuberRegressor(epsilon=1.35, max_iter=500),
            "RandomForestRegressor": RandomForestRegressor(
                n_estimators=200, max_depth=6, min_samples_leaf=100, n_jobs=-1,
            ),
            "GBMRegressor": GradientBoostingRegressor(
                n_estimators=200, max_depth=3, loss="huber", learning_rate=0.05,
            ),
        }

        # OLS via statsmodels
        try:
            import statsmodels.api as sm
            models["OLS"] = "statsmodels"
        except ImportError:
            pass

        # XGBoost regressor
        try:
            from xgboost import XGBRegressor
            models["XGBoostRegressor"] = XGBRegressor(
                n_estimators=300, max_depth=3, learning_rate=0.03, reg_alpha=0.1,
                verbosity=0,
            )
        except ImportError:
            pass

        # LightGBM regressor
        try:
            from lightgbm import LGBMRegressor
            models["LightGBMRegressor"] = LGBMRegressor(
                n_estimators=300, num_leaves=31, learning_rate=0.03,
                min_child_samples=100, verbosity=-1,
            )
        except ImportError:
            pass

        # --- Train and evaluate ---
        results: List[Dict[str, Any]] = []
        trained_models = {}

        for name, model in models.items():
            t0 = time.time()
            try:
                if model == "statsmodels":
                    import statsmodels.api as sm
                    X_train_c = sm.add_constant(X_train, has_constant="add")
                    X_val_c = sm.add_constant(X_val, has_constant="add")
                    X_test_c = sm.add_constant(X_test, has_constant="add")
                    ols_model = sm.OLS(y_train, X_train_c).fit()
                    val_preds = ols_model.predict(X_val_c)
                    train_preds = ols_model.predict(X_train_c)
                    test_preds = ols_model.predict(X_test_c)
                    trained_model = ols_model
                else:
                    model.fit(X_train, y_train)
                    val_preds = model.predict(X_val)
                    train_preds = model.predict(X_train)
                    test_preds = model.predict(X_test)
                    trained_model = model

                train_time = time.time() - t0

                # Ensure non-negative predictions
                val_preds = np.clip(val_preds, 0, None)
                train_preds = np.clip(train_preds, 0, None)
                test_preds = np.clip(test_preds, 0, None)

                val_rmse = float(np.sqrt(mean_squared_error(y_val, val_preds)))
                val_mae = float(mean_absolute_error(y_val, val_preds))
                val_r2 = float(r2_score(y_val, val_preds))
                train_rmse = float(np.sqrt(mean_squared_error(y_train, train_preds)))
                test_rmse = float(np.sqrt(mean_squared_error(y_test, test_preds)))

                # CCF validation (if funded_amnt available)
                ccf_validation = {}
                if funded_amnt_col is not None:
                    val_funded = funded_amnt_col[val_mask.values]
                    val_funded_safe = np.where(val_funded == 0, np.nan, val_funded)
                    predicted_ccf = val_preds / val_funded_safe
                    predicted_ccf = predicted_ccf[~np.isnan(predicted_ccf)]
                    if len(predicted_ccf) > 0:
                        ccf_validation = {
                            "mean_predicted_ccf": round(float(np.nanmean(predicted_ccf)), 4),
                            "median_predicted_ccf": round(float(np.nanmedian(predicted_ccf)), 4),
                            "pct_ccf_gt_1": round(float(np.mean(predicted_ccf > 1.0) * 100), 2),
                            "pct_ccf_negative": round(float(np.mean(predicted_ccf < 0) * 100), 2),
                            "ccf_reasonable": bool(np.nanmean(predicted_ccf) <= 1.2 and np.nanmean(predicted_ccf) >= 0),
                        }

                result = {
                    "model": name,
                    "train_rmse": round(train_rmse, 4),
                    "val_rmse": round(val_rmse, 4),
                    "val_mae": round(val_mae, 4),
                    "val_r2": round(val_r2, 4),
                    "test_rmse": round(test_rmse, 4),
                    "overfit_gap_rmse": round(train_rmse - val_rmse, 4),
                    "train_time_s": round(train_time, 2),
                    "ccf_validation": ccf_validation,
                    "status": "success",
                }
                results.append(result)
                trained_models[name] = trained_model

            except Exception as e:
                results.append({
                    "model": name, "status": "failed", "error": str(e),
                    "train_time_s": round(time.time() - t0, 2),
                })

        # --- Select champion ---
        successful = [r for r in results if r["status"] == "success"]
        if not successful:
            return _error("All EAD models failed.")

        champion_result = min(successful, key=lambda r: r["val_rmse"])
        champion_name = champion_result["model"]
        champion_model = trained_models[champion_name]

        # Save champion model and artifacts
        joblib.dump(champion_model, out / "ead_champion.joblib")
        joblib.dump(scaler, out / "ead_scaler.joblib")
        (out / "feature_list.json").write_text(json.dumps(numeric_cols, indent=2))

        # Save tournament results
        tournament_output = {
            "results": results,
            "champion": champion_name,
            "champion_val_rmse": champion_result["val_rmse"],
            "champion_val_r2": champion_result["val_r2"],
            "data_summary": {
                "total_rows": len(features) + 0,  # prevent ref before assign
                "train_rows": int(train_mask.sum()),
                "val_rows": int(val_mask.sum()),
                "test_rows": int(test_mask.sum()),
                "feature_count": len(numeric_cols),
                "ead_mean": round(float(np.mean(y)), 2),
                "ead_std": round(float(np.std(y)), 2),
            },
        }
        (out / "tournament_results.json").write_text(
            json.dumps(tournament_output, indent=2, default=str)
        )

        # Write handoff.json
        handoff = {
            "agent": "EAD_Agent",
            "status": "success",
            "output_files": {
                "champion_model": str(out / "ead_champion.joblib"),
                "scaler": str(out / "ead_scaler.joblib"),
                "feature_list": str(out / "feature_list.json"),
                "tournament_results": str(out / "tournament_results.json"),
            },
            "metrics": {
                "champion": champion_name,
                "val_rmse": champion_result["val_rmse"],
                "val_mae": champion_result["val_mae"],
                "val_r2": champion_result["val_r2"],
                "test_rmse": champion_result["test_rmse"],
                "ccf_validation": champion_result.get("ccf_validation", {}),
            },
        }
        (out / "handoff.json").write_text(json.dumps(handoff, indent=2, default=str))

        return _ok(tournament_output)
    except Exception as exc:
        return _error(f"Failed to run EAD tournament: {exc}")


# --- Collect all tools for agent registration ---
ALL_EAD_TOOLS = [
    define_ead_candidates,
    construct_ead_target,
    compute_amortization_schedule,
    run_ead_tournament,
]
