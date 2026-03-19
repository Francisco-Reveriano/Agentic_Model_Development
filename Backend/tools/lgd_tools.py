"""LGD Agent tools — two-stage Loss Given Default model tournament.

Stage 1: Binary classification (any loss? yes/no)
Stage 2: Regression on LGD severity (for loans with partial loss)
Combined: LGD = P(any_loss) * E[severity | partial_loss]
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List

import joblib
import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint, uniform
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
)
from sklearn.linear_model import (
    ElasticNet,
    HuberRegressor,
    Lasso,
    LogisticRegression,
    Ridge,
)
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
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
def define_lgd_candidates() -> dict:
    """Return the full catalogue of 13 LGD model candidates across two stages.

    Stage 1 (classification — binary any_loss target):
      LogReg, RF Classifier, GBM Classifier, XGBoost Classifier, LightGBM Classifier

    Stage 2 (regression — severity lgd target, rows where lgd > 0):
      OLS, Ridge, Lasso, ElasticNet, GBM Regressor, XGBoost Regressor,
      LightGBM Regressor, Huber Regressor

    Returns candidate definitions with baseline configs and hyperparam search
    distributions.
    """
    try:
        candidates = {
            "stage_1_classification": [
                {
                    "name": "LogisticRegression",
                    "library": "sklearn",
                    "baseline_config": {"C": 0.1, "class_weight": "balanced", "solver": "lbfgs", "max_iter": 1000},
                    "hyperparam_distributions": {
                        "C": "loguniform(0.001, 10.0)",
                        "penalty": "['l1', 'l2']",
                        "solver": "['lbfgs', 'liblinear']",
                    },
                },
                {
                    "name": "RandomForestClassifier",
                    "library": "sklearn",
                    "baseline_config": {"n_estimators": 200, "max_depth": 5, "min_samples_leaf": 100, "n_jobs": -1},
                    "hyperparam_distributions": {
                        "n_estimators": "randint(100, 800)",
                        "max_depth": "randint(3, 10)",
                        "min_samples_leaf": "randint(50, 500)",
                    },
                },
                {
                    "name": "GBMClassifier",
                    "library": "sklearn",
                    "baseline_config": {"n_estimators": 200, "max_depth": 3, "learning_rate": 0.05, "subsample": 0.8},
                    "hyperparam_distributions": {
                        "n_estimators": "randint(100, 600)",
                        "max_depth": "randint(2, 6)",
                        "learning_rate": "uniform(0.01, 0.15)",
                        "subsample": "uniform(0.6, 0.4)",
                    },
                },
                {
                    "name": "XGBoostClassifier",
                    "library": "xgboost",
                    "baseline_config": {
                        "n_estimators": 300, "max_depth": 4, "learning_rate": 0.03,
                        "reg_alpha": 0.1, "eval_metric": "logloss", "use_label_encoder": False,
                    },
                    "hyperparam_distributions": {
                        "n_estimators": "randint(200, 800)",
                        "max_depth": "randint(3, 7)",
                        "learning_rate": "uniform(0.01, 0.09)",
                        "reg_alpha": "uniform(0, 1.0)",
                        "reg_lambda": "uniform(0.5, 2.0)",
                        "colsample_bytree": "uniform(0.5, 0.5)",
                    },
                },
                {
                    "name": "LightGBMClassifier",
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
            ],
            "stage_2_regression": [
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
                    },
                },
                {
                    "name": "HuberRegressor",
                    "library": "sklearn",
                    "baseline_config": {"epsilon": 1.35, "max_iter": 500},
                    "hyperparam_distributions": {"epsilon": "uniform(1.1, 0.9)"},
                },
            ],
        }

        return _ok({
            "total_candidates": 13,
            "stage_1_count": len(candidates["stage_1_classification"]),
            "stage_2_count": len(candidates["stage_2_regression"]),
            "candidates": candidates,
        })
    except Exception as exc:
        return _error(f"Failed to define LGD candidates: {exc}")


@tool
def construct_lgd_target(data_dir: str) -> dict:
    """Construct LGD targets from targets.parquet — binary any_loss and continuous severity.

    Reads targets.parquet, filters to defaults (default_flag=1), computes:
    - any_loss: 1 if lgd > 0, else 0 (binary classification target)
    - lgd: continuous severity (regression target, used where any_loss = 1)

    Args:
        data_dir: Directory containing targets.parquet from Data_Agent.
    """
    try:
        d = Path(data_dir)
        targets_path = d / "targets.parquet"
        if not targets_path.exists():
            return _error(f"targets.parquet not found in {d}")

        targets = pd.read_parquet(targets_path)

        if "default_flag" not in targets.columns:
            return _error("Column 'default_flag' not found in targets.parquet.")
        if "lgd" not in targets.columns:
            return _error("Column 'lgd' not found in targets.parquet.")

        # Filter to defaults only
        defaults = targets[targets["default_flag"] == 1].copy()
        total_defaults = len(defaults)

        if total_defaults == 0:
            return _error("No default records found (default_flag=1).")

        # Binary target: any loss at all?
        defaults["any_loss"] = (defaults["lgd"] > 0).astype(int)

        # Severity subset (where there was some loss)
        severity_subset = defaults[defaults["any_loss"] == 1]

        # Class balance for the binary target
        any_loss_count = int(defaults["any_loss"].sum())
        no_loss_count = total_defaults - any_loss_count

        # LGD severity statistics (for loans with loss)
        lgd_severity = severity_subset["lgd"]

        result = {
            "total_defaults": total_defaults,
            "binary_target": {
                "any_loss_count": any_loss_count,
                "no_loss_count": no_loss_count,
                "any_loss_rate": round(any_loss_count / total_defaults, 4),
                "class_balance_ratio": round(any_loss_count / max(no_loss_count, 1), 4),
            },
            "severity_target": {
                "n_rows": len(severity_subset),
                "lgd_mean": round(float(lgd_severity.mean()), 4) if len(lgd_severity) > 0 else None,
                "lgd_median": round(float(lgd_severity.median()), 4) if len(lgd_severity) > 0 else None,
                "lgd_std": round(float(lgd_severity.std()), 4) if len(lgd_severity) > 0 else None,
                "lgd_min": round(float(lgd_severity.min()), 4) if len(lgd_severity) > 0 else None,
                "lgd_max": round(float(lgd_severity.max()), 4) if len(lgd_severity) > 0 else None,
                "lgd_p25": round(float(lgd_severity.quantile(0.25)), 4) if len(lgd_severity) > 0 else None,
                "lgd_p75": round(float(lgd_severity.quantile(0.75)), 4) if len(lgd_severity) > 0 else None,
            },
        }

        return _ok(result)
    except Exception as exc:
        return _error(f"Failed to construct LGD target: {exc}")


@tool
def run_lgd_tournament(data_dir: str, output_dir: str) -> dict:
    """Run two-stage LGD model tournament: classification + regression.

    Stage 1: Trains classification models on binary any_loss target.
    Stage 2: Trains regression models on lgd severity (rows where lgd > 0).
    Combines: LGD = P(any_loss) * E[severity | partial_loss].

    Saves both stage champion models and combined results.

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

        # Filter to defaults only
        default_mask = targets["default_flag"] == 1
        feat_defaults = features[default_mask].copy()
        tgt_defaults = targets[default_mask].copy()

        if len(feat_defaults) == 0:
            return _error("No default records found for LGD modeling.")

        # Construct targets
        tgt_defaults["any_loss"] = (tgt_defaults["lgd"] > 0).astype(int)

        # Vintage-based split
        if "issue_year" not in tgt_defaults.columns:
            return _error("Column 'issue_year' not found in targets — needed for vintage split.")

        train_mask = tgt_defaults["issue_year"] <= 2015
        val_mask = tgt_defaults["issue_year"] == 2016
        test_mask = tgt_defaults["issue_year"] >= 2017

        # Use only numeric feature columns
        numeric_cols = feat_defaults.select_dtypes(include=[np.number]).columns.tolist()
        feat_defaults = feat_defaults[numeric_cols]

        # Handle NaN/inf in features
        feat_defaults = feat_defaults.replace([np.inf, -np.inf], np.nan)
        feat_defaults = feat_defaults.fillna(feat_defaults.median())

        X_train = feat_defaults[train_mask].values
        X_val = feat_defaults[val_mask].values
        X_test = feat_defaults[test_mask].values

        # Scale features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)

        # =====================================================================
        # STAGE 1: Classification — any_loss binary target
        # =====================================================================
        y_train_cls = tgt_defaults.loc[train_mask, "any_loss"].values
        y_val_cls = tgt_defaults.loc[val_mask, "any_loss"].values
        y_test_cls = tgt_defaults.loc[test_mask, "any_loss"].values

        stage1_models = {
            "LogisticRegression": LogisticRegression(
                C=0.1, class_weight="balanced", solver="lbfgs", max_iter=1000,
            ),
            "RandomForestClassifier": RandomForestClassifier(
                n_estimators=200, max_depth=5, min_samples_leaf=100, n_jobs=-1,
            ),
            "GBMClassifier": GradientBoostingClassifier(
                n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.8,
            ),
        }

        # Conditionally add XGBoost and LightGBM
        try:
            from xgboost import XGBClassifier
            stage1_models["XGBoostClassifier"] = XGBClassifier(
                n_estimators=300, max_depth=4, learning_rate=0.03,
                reg_alpha=0.1, eval_metric="logloss", use_label_encoder=False,
                verbosity=0,
            )
        except ImportError:
            pass

        try:
            from lightgbm import LGBMClassifier
            stage1_models["LightGBMClassifier"] = LGBMClassifier(
                n_estimators=300, num_leaves=31, learning_rate=0.03,
                min_child_samples=100, verbosity=-1,
            )
        except ImportError:
            pass

        stage1_results: List[Dict[str, Any]] = []
        stage1_trained = {}

        for name, model in stage1_models.items():
            t0 = time.time()
            try:
                model.fit(X_train, y_train_cls)
                train_time = time.time() - t0

                # Predict probabilities
                if hasattr(model, "predict_proba"):
                    val_proba = model.predict_proba(X_val)[:, 1]
                    test_proba = model.predict_proba(X_test)[:, 1]
                    train_proba = model.predict_proba(X_train)[:, 1]
                else:
                    val_proba = model.decision_function(X_val)
                    test_proba = model.decision_function(X_test)
                    train_proba = model.decision_function(X_train)

                val_preds = (val_proba >= 0.5).astype(int)

                val_auc = roc_auc_score(y_val_cls, val_proba) if len(np.unique(y_val_cls)) > 1 else 0.0
                train_auc = roc_auc_score(y_train_cls, train_proba) if len(np.unique(y_train_cls)) > 1 else 0.0
                val_acc = accuracy_score(y_val_cls, val_preds)
                val_brier = brier_score_loss(y_val_cls, val_proba)

                result = {
                    "model": name,
                    "stage": 1,
                    "train_auc": round(train_auc, 4),
                    "val_auc": round(val_auc, 4),
                    "val_accuracy": round(val_acc, 4),
                    "val_brier": round(val_brier, 4),
                    "overfit_gap": round(train_auc - val_auc, 4),
                    "train_time_s": round(train_time, 2),
                    "status": "success",
                }
                stage1_results.append(result)
                stage1_trained[name] = model

            except Exception as e:
                stage1_results.append({
                    "model": name, "stage": 1, "status": "failed", "error": str(e),
                    "train_time_s": round(time.time() - t0, 2),
                })

        # Select Stage 1 champion
        successful_s1 = [r for r in stage1_results if r["status"] == "success"]
        if not successful_s1:
            return _error("All Stage 1 classification models failed.")

        stage1_champion_result = max(successful_s1, key=lambda r: r["val_auc"])
        stage1_champion_name = stage1_champion_result["model"]
        stage1_champion_model = stage1_trained[stage1_champion_name]

        # Save Stage 1 champion
        joblib.dump(stage1_champion_model, out / "lgd_stage1_champion.joblib")
        joblib.dump(scaler, out / "lgd_scaler.joblib")

        # =====================================================================
        # STAGE 2: Regression — severity lgd target (only rows where lgd > 0)
        # =====================================================================
        severity_mask = tgt_defaults["lgd"] > 0
        feat_severity = feat_defaults[severity_mask]
        tgt_severity = tgt_defaults[severity_mask]

        if len(feat_severity) == 0:
            return _error("No rows with lgd > 0 for Stage 2 regression.")

        s2_train_mask = tgt_severity["issue_year"] <= 2015
        s2_val_mask = tgt_severity["issue_year"] == 2016
        s2_test_mask = tgt_severity["issue_year"] >= 2017

        X2_train = scaler.transform(feat_severity[s2_train_mask].values)
        X2_val = scaler.transform(feat_severity[s2_val_mask].values)
        X2_test = scaler.transform(feat_severity[s2_test_mask].values)

        y2_train = tgt_severity.loc[s2_train_mask, "lgd"].values
        y2_val = tgt_severity.loc[s2_val_mask, "lgd"].values
        y2_test = tgt_severity.loc[s2_test_mask, "lgd"].values

        stage2_models = {
            "Ridge": Ridge(alpha=1.0),
            "Lasso": Lasso(alpha=0.1, max_iter=5000),
            "ElasticNet": ElasticNet(alpha=0.5, l1_ratio=0.5, max_iter=5000),
            "GBMRegressor": GradientBoostingRegressor(
                n_estimators=200, max_depth=3, loss="huber", learning_rate=0.05,
            ),
            "HuberRegressor": HuberRegressor(epsilon=1.35, max_iter=500),
        }

        # OLS via statsmodels
        try:
            import statsmodels.api as sm
            stage2_models["OLS"] = "statsmodels"
        except ImportError:
            pass

        # XGBoost regressor
        try:
            from xgboost import XGBRegressor
            stage2_models["XGBoostRegressor"] = XGBRegressor(
                n_estimators=300, max_depth=3, learning_rate=0.03, reg_alpha=0.1,
                verbosity=0,
            )
        except ImportError:
            pass

        # LightGBM regressor
        try:
            from lightgbm import LGBMRegressor
            stage2_models["LightGBMRegressor"] = LGBMRegressor(
                n_estimators=300, num_leaves=31, learning_rate=0.03,
                min_child_samples=100, verbosity=-1,
            )
        except ImportError:
            pass

        stage2_results: List[Dict[str, Any]] = []
        stage2_trained = {}

        for name, model in stage2_models.items():
            t0 = time.time()
            try:
                if model == "statsmodels":
                    import statsmodels.api as sm
                    X2_train_c = sm.add_constant(X2_train, has_constant="add")
                    X2_val_c = sm.add_constant(X2_val, has_constant="add")
                    ols_model = sm.OLS(y2_train, X2_train_c).fit()
                    val_preds = np.clip(ols_model.predict(X2_val_c), 0, 1)
                    train_preds = np.clip(ols_model.predict(X2_train_c), 0, 1)
                    trained_model = ols_model
                else:
                    model.fit(X2_train, y2_train)
                    val_preds = np.clip(model.predict(X2_val), 0, 1)
                    train_preds = np.clip(model.predict(X2_train), 0, 1)
                    trained_model = model

                train_time = time.time() - t0

                val_rmse = float(np.sqrt(mean_squared_error(y2_val, val_preds)))
                val_mae = float(mean_absolute_error(y2_val, val_preds))
                val_r2 = float(r2_score(y2_val, val_preds))
                train_rmse = float(np.sqrt(mean_squared_error(y2_train, train_preds)))

                result = {
                    "model": name,
                    "stage": 2,
                    "train_rmse": round(train_rmse, 4),
                    "val_rmse": round(val_rmse, 4),
                    "val_mae": round(val_mae, 4),
                    "val_r2": round(val_r2, 4),
                    "overfit_gap_rmse": round(train_rmse - val_rmse, 4),
                    "train_time_s": round(train_time, 2),
                    "status": "success",
                }
                stage2_results.append(result)
                stage2_trained[name] = trained_model

            except Exception as e:
                stage2_results.append({
                    "model": name, "stage": 2, "status": "failed", "error": str(e),
                    "train_time_s": round(time.time() - t0, 2),
                })

        # Select Stage 2 champion (lowest val RMSE)
        successful_s2 = [r for r in stage2_results if r["status"] == "success"]
        if not successful_s2:
            return _error("All Stage 2 regression models failed.")

        stage2_champion_result = min(successful_s2, key=lambda r: r["val_rmse"])
        stage2_champion_name = stage2_champion_result["model"]
        stage2_champion_model = stage2_trained[stage2_champion_name]

        # Save Stage 2 champion
        joblib.dump(stage2_champion_model, out / "lgd_stage2_champion.joblib")

        # =====================================================================
        # Combined LGD prediction on test set
        # =====================================================================
        # Stage 1: P(any_loss) on full test set
        if hasattr(stage1_champion_model, "predict_proba"):
            p_any_loss = stage1_champion_model.predict_proba(X_test)[:, 1]
        else:
            p_any_loss = stage1_champion_model.decision_function(X_test)

        # Stage 2: E[severity] on full test set (apply model to all, weight by p_any_loss)
        if stage2_champion_name == "OLS":
            import statsmodels.api as sm
            X_test_c = sm.add_constant(X_test, has_constant="add")
            e_severity = np.clip(stage2_champion_model.predict(X_test_c), 0, 1)
        else:
            e_severity = np.clip(stage2_champion_model.predict(X_test), 0, 1)

        # Combined LGD = P(any_loss) * E[severity | partial_loss]
        combined_lgd = p_any_loss * e_severity
        combined_lgd = np.clip(combined_lgd, 0, 1)

        # Evaluate combined LGD against actual
        y_test_lgd = tgt_defaults.loc[test_mask, "lgd"].values
        combined_rmse = float(np.sqrt(mean_squared_error(y_test_lgd, combined_lgd)))
        combined_mae = float(mean_absolute_error(y_test_lgd, combined_lgd))
        combined_r2 = float(r2_score(y_test_lgd, combined_lgd))

        # Save feature list
        feature_list = numeric_cols
        (out / "feature_list.json").write_text(json.dumps(feature_list, indent=2))

        # Save tournament results
        tournament_output = {
            "stage_1": {
                "results": stage1_results,
                "champion": stage1_champion_name,
                "champion_val_auc": stage1_champion_result["val_auc"],
            },
            "stage_2": {
                "results": stage2_results,
                "champion": stage2_champion_name,
                "champion_val_rmse": stage2_champion_result["val_rmse"],
            },
            "combined": {
                "test_rmse": round(combined_rmse, 4),
                "test_mae": round(combined_mae, 4),
                "test_r2": round(combined_r2, 4),
                "mean_predicted_lgd": round(float(combined_lgd.mean()), 4),
                "mean_actual_lgd": round(float(y_test_lgd.mean()), 4),
            },
            "data_summary": {
                "total_defaults": len(feat_defaults),
                "train_defaults": int(train_mask.sum()),
                "val_defaults": int(val_mask.sum()),
                "test_defaults": int(test_mask.sum()),
                "severity_rows": len(feat_severity),
                "feature_count": len(numeric_cols),
            },
        }
        (out / "tournament_results.json").write_text(
            json.dumps(tournament_output, indent=2, default=str)
        )

        # Write handoff.json
        handoff = {
            "agent": "LGD_Agent",
            "status": "success",
            "output_files": {
                "stage1_champion": str(out / "lgd_stage1_champion.joblib"),
                "stage2_champion": str(out / "lgd_stage2_champion.joblib"),
                "scaler": str(out / "lgd_scaler.joblib"),
                "feature_list": str(out / "feature_list.json"),
                "tournament_results": str(out / "tournament_results.json"),
            },
            "metrics": {
                "stage1_champion": stage1_champion_name,
                "stage1_val_auc": stage1_champion_result["val_auc"],
                "stage2_champion": stage2_champion_name,
                "stage2_val_rmse": stage2_champion_result["val_rmse"],
                "combined_test_rmse": round(combined_rmse, 4),
                "combined_test_mae": round(combined_mae, 4),
                "combined_test_r2": round(combined_r2, 4),
            },
        }
        (out / "handoff.json").write_text(json.dumps(handoff, indent=2, default=str))

        return _ok(tournament_output)
    except Exception as exc:
        return _error(f"Failed to run LGD tournament: {exc}")


# --- Collect all tools for agent registration ---
ALL_LGD_TOOLS = [
    define_lgd_candidates,
    construct_lgd_target,
    run_lgd_tournament,
]
