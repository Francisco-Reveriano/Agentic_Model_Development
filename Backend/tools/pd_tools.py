"""PD-specific tools for the Probability of Default agent.

Provides tools for defining PD candidate models (12 candidates from the PRD),
constructing the PD binary target, and running the full PD model tournament.
"""

from __future__ import annotations

import json
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
# PD Candidate Model Definitions (from PRD Section 9.1 & 11)
# ---------------------------------------------------------------------------

PD_CANDIDATE_DEFINITIONS: List[Dict[str, Any]] = [
    {
        "id": 1,
        "name": "Logistic Regression L2",
        "library": "sklearn",
        "class": "sklearn.linear_model.LogisticRegression",
        "baseline_params": {
            "C": 0.1,
            "penalty": "l2",
            "class_weight": "balanced",
            "solver": "lbfgs",
            "max_iter": 1000,
            "random_state": 42,
        },
        "search_distributions": {
            "C": {"type": "loguniform", "low": 0.001, "high": 10.0},
            "penalty": {"type": "choice", "values": ["l1", "l2", "elasticnet"]},
        },
        "interpretable": True,
        "category": "linear",
    },
    {
        "id": 2,
        "name": "Logistic Regression L1",
        "library": "sklearn",
        "class": "sklearn.linear_model.LogisticRegression",
        "baseline_params": {
            "C": 0.05,
            "penalty": "l1",
            "class_weight": "balanced",
            "solver": "liblinear",
            "max_iter": 1000,
            "random_state": 42,
        },
        "search_distributions": {
            "C": {"type": "loguniform", "low": 0.001, "high": 10.0},
        },
        "interpretable": True,
        "category": "linear",
    },
    {
        "id": 3,
        "name": "Elastic Net Logistic",
        "library": "sklearn",
        "class": "sklearn.linear_model.LogisticRegression",
        "baseline_params": {
            "penalty": "elasticnet",
            "l1_ratio": 0.5,
            "C": 0.1,
            "solver": "saga",
            "class_weight": "balanced",
            "max_iter": 2000,
            "random_state": 42,
        },
        "search_distributions": {
            "C": {"type": "loguniform", "low": 0.001, "high": 10.0},
            "l1_ratio": {"type": "uniform", "low": 0.1, "high": 0.9},
        },
        "interpretable": True,
        "category": "linear",
    },
    {
        "id": 4,
        "name": "Decision Tree",
        "library": "sklearn",
        "class": "sklearn.tree.DecisionTreeClassifier",
        "baseline_params": {
            "max_depth": 5,
            "min_samples_leaf": 200,
            "class_weight": "balanced",
            "random_state": 42,
        },
        "search_distributions": {
            "max_depth": {"type": "randint", "low": 3, "high": 8},
            "min_samples_leaf": {"type": "randint", "low": 50, "high": 500},
        },
        "interpretable": True,
        "category": "tree",
    },
    {
        "id": 5,
        "name": "Random Forest",
        "library": "sklearn",
        "class": "sklearn.ensemble.RandomForestClassifier",
        "baseline_params": {
            "n_estimators": 200,
            "max_depth": 6,
            "class_weight": "balanced",
            "n_jobs": -1,
            "random_state": 42,
        },
        "search_distributions": {
            "n_estimators": {"type": "randint", "low": 100, "high": 800},
            "max_depth": {"type": "randint", "low": 3, "high": 10},
            "min_samples_leaf": {"type": "randint", "low": 50, "high": 500},
        },
        "interpretable": False,
        "category": "ensemble",
    },
    {
        "id": 6,
        "name": "Gradient Boosting (GBM)",
        "library": "sklearn",
        "class": "sklearn.ensemble.GradientBoostingClassifier",
        "baseline_params": {
            "n_estimators": 200,
            "max_depth": 3,
            "learning_rate": 0.05,
            "random_state": 42,
        },
        "search_distributions": {
            "n_estimators": {"type": "randint", "low": 100, "high": 600},
            "learning_rate": {"type": "uniform", "low": 0.01, "high": 0.15},
            "max_depth": {"type": "randint", "low": 2, "high": 6},
            "subsample": {"type": "uniform", "low": 0.6, "high": 1.0},
        },
        "interpretable": False,
        "category": "ensemble",
    },
    {
        "id": 7,
        "name": "AdaBoost",
        "library": "sklearn",
        "class": "sklearn.ensemble.AdaBoostClassifier",
        "baseline_params": {
            "n_estimators": 200,
            "learning_rate": 0.05,
            "random_state": 42,
        },
        "search_distributions": {
            "n_estimators": {"type": "randint", "low": 100, "high": 600},
            "learning_rate": {"type": "uniform", "low": 0.01, "high": 0.15},
        },
        "interpretable": False,
        "category": "ensemble",
    },
    {
        "id": 8,
        "name": "Extra Trees",
        "library": "sklearn",
        "class": "sklearn.ensemble.ExtraTreesClassifier",
        "baseline_params": {
            "n_estimators": 200,
            "max_depth": 6,
            "class_weight": "balanced",
            "n_jobs": -1,
            "random_state": 42,
        },
        "search_distributions": {
            "n_estimators": {"type": "randint", "low": 100, "high": 800},
            "max_depth": {"type": "randint", "low": 3, "high": 10},
            "min_samples_leaf": {"type": "randint", "low": 50, "high": 500},
        },
        "interpretable": False,
        "category": "ensemble",
    },
    {
        "id": 9,
        "name": "XGBoost",
        "library": "xgboost",
        "class": "xgboost.XGBClassifier",
        "baseline_params": {
            "n_estimators": 500,
            "max_depth": 4,
            "learning_rate": 0.03,
            "reg_alpha": 0.1,
            "scale_pos_weight": "auto",
            "use_label_encoder": False,
            "eval_metric": "logloss",
            "random_state": 42,
            "n_jobs": -1,
        },
        "search_distributions": {
            "n_estimators": {"type": "randint", "low": 200, "high": 800},
            "max_depth": {"type": "randint", "low": 3, "high": 7},
            "learning_rate": {"type": "uniform", "low": 0.01, "high": 0.10},
            "reg_alpha": {"type": "uniform", "low": 0.0, "high": 1.0},
            "reg_lambda": {"type": "uniform", "low": 0.5, "high": 2.5},
            "colsample_bytree": {"type": "uniform", "low": 0.5, "high": 1.0},
            "min_child_weight": {"type": "randint", "low": 50, "high": 300},
        },
        "interpretable": False,
        "category": "boosting",
    },
    {
        "id": 10,
        "name": "LightGBM",
        "library": "lightgbm",
        "class": "lightgbm.LGBMClassifier",
        "baseline_params": {
            "n_estimators": 500,
            "num_leaves": 31,
            "learning_rate": 0.03,
            "class_weight": "balanced",
            "random_state": 42,
            "n_jobs": -1,
            "verbose": -1,
        },
        "search_distributions": {
            "n_estimators": {"type": "randint", "low": 200, "high": 800},
            "num_leaves": {"type": "randint", "low": 15, "high": 63},
            "learning_rate": {"type": "uniform", "low": 0.01, "high": 0.10},
            "min_child_samples": {"type": "randint", "low": 50, "high": 300},
            "colsample_bytree": {"type": "uniform", "low": 0.5, "high": 1.0},
        },
        "interpretable": False,
        "category": "boosting",
    },
    {
        "id": 11,
        "name": "Logit statsmodels",
        "library": "statsmodels",
        "class": "statsmodels.api.Logit",
        "baseline_params": {
            "method": "bfgs",
            "maxiter": 1000,
        },
        "search_distributions": {},
        "interpretable": True,
        "category": "statistical",
    },
    {
        "id": 12,
        "name": "Probit statsmodels",
        "library": "statsmodels",
        "class": "statsmodels.api.Probit",
        "baseline_params": {
            "method": "bfgs",
        },
        "search_distributions": {},
        "interpretable": True,
        "category": "statistical",
    },
]

# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@tool
def define_pd_candidates() -> dict:
    """Return the full list of 12 PD candidate models with their configurations.

    Returns each model's name, library, class path, baseline hyperparameters,
    Phase 3 hyperparameter search distributions, interpretability flag, and
    model category.  This is the canonical candidate pool from the PRD
    (Section 9.1).
    """
    try:
        summary = {
            "total_candidates": len(PD_CANDIDATE_DEFINITIONS),
            "by_library": {},
            "by_category": {},
            "interpretable_count": sum(
                1 for c in PD_CANDIDATE_DEFINITIONS if c["interpretable"]
            ),
            "candidates": PD_CANDIDATE_DEFINITIONS,
        }

        # Count by library
        for c in PD_CANDIDATE_DEFINITIONS:
            lib = c["library"]
            summary["by_library"][lib] = summary["by_library"].get(lib, 0) + 1
        # Count by category
        for c in PD_CANDIDATE_DEFINITIONS:
            cat = c["category"]
            summary["by_category"][cat] = summary["by_category"].get(cat, 0) + 1

        return _ok(summary)
    except Exception as exc:
        return _error(f"Failed to define PD candidates: {exc}")


@tool
def construct_pd_target(data_dir: str) -> dict:
    """Construct the PD binary target from targets.parquet.

    Reads the default_flag column and returns class balance information
    including default rate, class counts, and suitability for modeling.

    Args:
        data_dir: Directory containing targets.parquet.
    """
    try:
        ddir = Path(data_dir)
        targets_path = ddir / "targets.parquet"
        if not targets_path.exists():
            return _error(f"targets.parquet not found in {ddir}")

        targets = pd.read_parquet(targets_path)
        if "default_flag" not in targets.columns:
            return _error(
                f"'default_flag' not found in targets.parquet. "
                f"Available columns: {list(targets.columns)}"
            )

        y = targets["default_flag"]
        total = len(y)
        n_default = int(y.sum())
        n_non_default = total - n_default
        default_rate = float(y.mean())

        # Check class balance (PRD DQ-02: minority < 5% triggers treatment)
        minority_pct = min(default_rate, 1 - default_rate) * 100
        class_balance_ok = minority_pct >= 5.0

        # Vintage-level default rates if issue_year available
        vintage_default_rates = {}
        if "issue_year" in targets.columns:
            for year, group in targets.groupby("issue_year"):
                vintage_default_rates[int(year)] = {
                    "count": int(len(group)),
                    "default_rate": round(float(group["default_flag"].mean()), 4),
                    "default_count": int(group["default_flag"].sum()),
                }

        # Save target array as npy for downstream use
        y_path = ddir / "y_pd.npy"
        np.save(y_path, y.values)

        result = {
            "target_column": "default_flag",
            "total_observations": total,
            "default_count": n_default,
            "non_default_count": n_non_default,
            "default_rate": round(default_rate, 6),
            "minority_class_pct": round(minority_pct, 2),
            "class_balance_adequate": class_balance_ok,
            "class_weight_recommended": not class_balance_ok,
            "target_saved_to": str(y_path),
        }

        if vintage_default_rates:
            result["vintage_default_rates"] = vintage_default_rates

        # Assessment
        if default_rate < 0.01:
            result["assessment"] = (
                "WARN: Very low default rate (<1%). Consider SMOTE or "
                "aggressive class_weight='balanced'."
            )
        elif default_rate > 0.50:
            result["assessment"] = (
                "WARN: Default rate > 50%. Verify target construction — "
                "this is unusual for consumer lending."
            )
        else:
            result["assessment"] = (
                f"Default rate of {default_rate:.1%} is within expected range "
                f"for consumer lending. Class weighting via class_weight='balanced' "
                f"is applied to applicable models."
            )

        return _ok(result)
    except Exception as exc:
        return _error(f"Failed to construct PD target: {exc}")


@tool
def run_pd_tournament(data_dir: str, output_dir: str) -> dict:
    """Run the full 4-phase PD model tournament across all 12 candidates.

    Phase 1: Broad sweep — train all candidates with baseline configs.
    Phase 2: Feature importance consensus — aggregate importances across models.
    Phase 3: Refinement loop — hyperparameter tuning for top-K models.
    Phase 4: Champion selection — weighted scoring rubric.

    Saves the champion model as a joblib artifact, along with tournament
    results summary.

    Args:
        data_dir: Directory containing feature_matrix.parquet (or cleaned_features.parquet),
                  targets.parquet, and split index files (train_idx.npy, val_idx.npy, test_idx.npy).
        output_dir: Directory to write tournament results and champion model.
    """
    try:
        import joblib
        import statsmodels.api as sm
        from sklearn.metrics import roc_auc_score
        from sklearn.preprocessing import StandardScaler

        settings = get_settings()
        ddir = Path(data_dir)
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        # --- Load data ---
        feature_path = ddir / "feature_matrix.parquet"
        if not feature_path.exists():
            feature_path = ddir / "cleaned_features.parquet"
        if not feature_path.exists():
            return _error(f"No feature matrix found in {ddir}")

        X = pd.read_parquet(feature_path)
        targets = pd.read_parquet(ddir / "targets.parquet")
        y = targets["default_flag"].values

        # Load split indices
        train_idx = np.load(ddir / "train_idx.npy")
        val_idx = np.load(ddir / "val_idx.npy")
        test_idx = np.load(ddir / "test_idx.npy")

        X_train, y_train = X.iloc[train_idx].values, y[train_idx]
        X_val, y_val = X.iloc[val_idx].values, y[val_idx]
        X_test, y_test = X.iloc[test_idx].values, y[test_idx]

        feature_names = list(X.columns)

        # Scale for linear models
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)

        # Compute scale_pos_weight for XGBoost
        n_neg = int((y_train == 0).sum())
        n_pos = int((y_train == 1).sum())
        scale_pos_weight = n_neg / max(n_pos, 1)

        # ================================================================
        # PHASE 1: Broad Sweep
        # ================================================================
        phase1_results: List[Dict[str, Any]] = []
        trained_models: Dict[str, Any] = {}
        feature_importances: Dict[str, np.ndarray] = {}

        for candidate in PD_CANDIDATE_DEFINITIONS:
            name = candidate["name"]
            t0 = time.time()

            try:
                if candidate["library"] == "sklearn":
                    # Dynamically import and instantiate
                    module_path, class_name = candidate["class"].rsplit(".", 1)
                    import importlib
                    mod = importlib.import_module(module_path)
                    cls = getattr(mod, class_name)

                    params = dict(candidate["baseline_params"])
                    model = cls(**params)

                    # Use scaled data for linear models, raw for tree-based
                    if candidate["category"] == "linear":
                        model.fit(X_train_scaled, y_train)
                        val_proba = model.predict_proba(X_val_scaled)[:, 1]
                        train_proba = model.predict_proba(X_train_scaled)[:, 1]
                    else:
                        model.fit(X_train, y_train)
                        val_proba = model.predict_proba(X_val)[:, 1]
                        train_proba = model.predict_proba(X_train)[:, 1]

                    # Extract feature importances
                    if hasattr(model, "coef_"):
                        fi = np.abs(model.coef_).ravel()
                    elif hasattr(model, "feature_importances_"):
                        fi = model.feature_importances_
                    else:
                        fi = np.ones(len(feature_names))
                    feature_importances[name] = fi

                elif candidate["library"] == "xgboost":
                    from xgboost import XGBClassifier

                    params = dict(candidate["baseline_params"])
                    # Replace 'auto' scale_pos_weight with computed value
                    if params.get("scale_pos_weight") == "auto":
                        params["scale_pos_weight"] = scale_pos_weight

                    model = XGBClassifier(**params)
                    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
                    val_proba = model.predict_proba(X_val)[:, 1]
                    train_proba = model.predict_proba(X_train)[:, 1]
                    feature_importances[name] = model.feature_importances_

                elif candidate["library"] == "lightgbm":
                    from lightgbm import LGBMClassifier

                    params = dict(candidate["baseline_params"])
                    model = LGBMClassifier(**params)
                    model.fit(
                        X_train, y_train,
                        eval_set=[(X_val, y_val)],
                    )
                    val_proba = model.predict_proba(X_val)[:, 1]
                    train_proba = model.predict_proba(X_train)[:, 1]
                    feature_importances[name] = model.feature_importances_.astype(float)

                elif candidate["library"] == "statsmodels":
                    X_train_sm = sm.add_constant(
                        pd.DataFrame(X_train_scaled, columns=feature_names),
                        has_constant="add",
                    )
                    X_val_sm = sm.add_constant(
                        pd.DataFrame(X_val_scaled, columns=feature_names),
                        has_constant="add",
                    )

                    sm_params = dict(candidate["baseline_params"])
                    method = sm_params.pop("method", "bfgs")
                    maxiter = sm_params.pop("maxiter", 1000)

                    if "Logit" in candidate["name"] or "Logit" in candidate["class"]:
                        sm_model = sm.Logit(y_train, X_train_sm)
                    else:
                        sm_model = sm.Probit(y_train, X_train_sm)

                    result = sm_model.fit(method=method, maxiter=maxiter, disp=False)
                    val_proba = result.predict(X_val_sm)
                    train_proba = result.predict(X_train_sm)

                    # Use absolute z-values as importance proxy
                    fi = np.abs(result.params.values[1:])  # Exclude constant
                    feature_importances[name] = fi

                    model = result  # Store the result object
                else:
                    continue

                train_time = time.time() - t0
                val_auc = float(roc_auc_score(y_val, val_proba))
                train_auc = float(roc_auc_score(y_train, train_proba))
                auc_gap = abs(train_auc - val_auc)

                trained_models[name] = {
                    "model": model,
                    "scaler_needed": candidate["category"] == "linear"
                        or candidate["library"] == "statsmodels",
                    "val_proba": val_proba,
                    "train_proba": train_proba,
                }

                phase1_results.append({
                    "name": name,
                    "library": candidate["library"],
                    "category": candidate["category"],
                    "val_auc": round(val_auc, 6),
                    "train_auc": round(train_auc, 6),
                    "auc_gap": round(auc_gap, 6),
                    "train_time_s": round(train_time, 2),
                    "interpretable": candidate["interpretable"],
                    "status": "trained",
                })

            except Exception as model_exc:
                train_time = time.time() - t0
                phase1_results.append({
                    "name": name,
                    "library": candidate["library"],
                    "category": candidate["category"],
                    "val_auc": 0.0,
                    "train_auc": 0.0,
                    "auc_gap": 0.0,
                    "train_time_s": round(train_time, 2),
                    "interpretable": candidate["interpretable"],
                    "status": f"failed: {str(model_exc)[:200]}",
                })

        # Sort by validation AUC descending
        phase1_results.sort(key=lambda r: r["val_auc"], reverse=True)

        # ================================================================
        # PHASE 2: Feature Importance Consensus
        # ================================================================
        n_features = len(feature_names)
        consensus_scores = np.zeros(n_features)
        total_weight = 0.0

        for name, fi in feature_importances.items():
            # Get this model's validation AUC as weight
            match = [r for r in phase1_results if r["name"] == name and r["status"] == "trained"]
            if not match:
                continue
            weight = match[0]["val_auc"]
            # Normalize importances to sum to 1
            fi_arr = np.array(fi[:n_features])  # Ensure correct length
            if len(fi_arr) < n_features:
                fi_arr = np.pad(fi_arr, (0, n_features - len(fi_arr)))
            fi_sum = fi_arr.sum()
            if fi_sum > 0:
                fi_norm = fi_arr / fi_sum
            else:
                fi_norm = np.ones(n_features) / n_features
            consensus_scores += weight * fi_norm
            total_weight += weight

        if total_weight > 0:
            consensus_scores /= total_weight

        # Assign tiers based on percentiles
        p80 = np.percentile(consensus_scores, 80)
        p50 = np.percentile(consensus_scores, 50)
        p20 = np.percentile(consensus_scores, 20)

        feature_tiers: Dict[str, int] = {}
        tier_counts = {1: 0, 2: 0, 3: 0, 4: 0}
        for i, fname in enumerate(feature_names):
            score = consensus_scores[i]
            if score >= p80:
                tier = 1
            elif score >= p50:
                tier = 2
            elif score >= p20:
                tier = 3
            else:
                tier = 4
            feature_tiers[fname] = tier
            tier_counts[tier] += 1

        # Top features by consensus
        top_feature_idx = np.argsort(consensus_scores)[::-1]
        top_features = [
            {"feature": feature_names[i], "consensus_score": round(float(consensus_scores[i]), 6), "tier": feature_tiers[feature_names[i]]}
            for i in top_feature_idx[:20]
        ]

        # ================================================================
        # PHASE 3: Refinement Loop (top-K models)
        # ================================================================
        top_k = settings.tournament_top_k
        max_iters = settings.tournament_max_iterations
        convergence_threshold = settings.tournament_convergence_threshold
        prune_threshold = settings.tournament_prune_threshold

        # Select top-K trained models
        trained_names = [r["name"] for r in phase1_results if r["status"] == "trained"]
        top_k_names = trained_names[:top_k]

        refinement_log: List[Dict[str, Any]] = []
        best_score_ever = max(
            (r["val_auc"] for r in phase1_results if r["status"] == "trained"),
            default=0.0,
        )

        # Build feature tier subsets
        tier12_mask = np.array([feature_tiers.get(f, 4) <= 2 for f in feature_names])
        tier123_mask = np.array([feature_tiers.get(f, 4) <= 3 for f in feature_names])

        for iteration in range(1, max_iters + 1):
            iter_best = 0.0
            iter_results: List[Dict[str, Any]] = []

            for name in list(top_k_names):
                candidate = next(
                    (c for c in PD_CANDIDATE_DEFINITIONS if c["name"] == name), None
                )
                if candidate is None or candidate["library"] == "statsmodels":
                    # Statsmodels models are not tuned in Phase 3
                    continue
                if not candidate["search_distributions"]:
                    continue

                try:
                    from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
                    from scipy.stats import loguniform, randint, uniform
                    import importlib

                    module_path, class_name = candidate["class"].rsplit(".", 1)
                    mod = importlib.import_module(module_path)
                    cls = getattr(mod, class_name)

                    # Build param distributions
                    param_dist = {}
                    for param_name, dist_spec in candidate["search_distributions"].items():
                        if dist_spec["type"] == "loguniform":
                            param_dist[param_name] = loguniform(dist_spec["low"], dist_spec["high"])
                        elif dist_spec["type"] == "uniform":
                            param_dist[param_name] = uniform(dist_spec["low"], dist_spec["high"] - dist_spec["low"])
                        elif dist_spec["type"] == "randint":
                            param_dist[param_name] = randint(dist_spec["low"], dist_spec["high"])
                        elif dist_spec["type"] == "choice":
                            param_dist[param_name] = dist_spec["values"]

                    # Base params (exclude search params)
                    base_params = {
                        k: v for k, v in candidate["baseline_params"].items()
                        if k not in candidate["search_distributions"]
                    }

                    # Handle XGBoost scale_pos_weight
                    if candidate["library"] == "xgboost" and base_params.get("scale_pos_weight") == "auto":
                        base_params["scale_pos_weight"] = scale_pos_weight

                    base_model = cls(**base_params)

                    # Use appropriate data based on model category
                    if candidate["category"] == "linear":
                        X_search = X_train_scaled
                    else:
                        X_search = X_train

                    cv = TimeSeriesSplit(n_splits=settings.tournament_cv_splits)
                    search = RandomizedSearchCV(
                        base_model,
                        param_dist,
                        n_iter=min(settings.tournament_random_search_iter, 20),
                        cv=cv,
                        scoring="roc_auc",
                        random_state=42 + iteration,
                        n_jobs=-1,
                        error_score=0.0,
                    )
                    search.fit(X_search, y_train)

                    # Evaluate on validation
                    if candidate["category"] == "linear":
                        val_proba = search.predict_proba(X_val_scaled)[:, 1]
                    else:
                        val_proba = search.predict_proba(X_val)[:, 1]

                    refined_auc = float(roc_auc_score(y_val, val_proba))

                    # Update the trained model if improved
                    original_auc = next(
                        (r["val_auc"] for r in phase1_results if r["name"] == name),
                        0.0,
                    )
                    if refined_auc > original_auc:
                        trained_models[name]["model"] = search.best_estimator_
                        trained_models[name]["val_proba"] = val_proba
                        # Update phase1_results
                        for r in phase1_results:
                            if r["name"] == name:
                                r["val_auc"] = round(refined_auc, 6)
                                break

                    iter_results.append({
                        "name": name,
                        "original_auc": round(original_auc, 6),
                        "refined_auc": round(refined_auc, 6),
                        "improvement": round(refined_auc - original_auc, 6),
                        "best_params": {k: round(v, 6) if isinstance(v, float) else v
                                        for k, v in search.best_params_.items()},
                    })

                    iter_best = max(iter_best, refined_auc)

                except Exception as refine_exc:
                    iter_results.append({
                        "name": name,
                        "status": f"refinement_failed: {str(refine_exc)[:200]}",
                    })

            # Check convergence
            improvement = iter_best - best_score_ever
            best_score_ever = max(best_score_ever, iter_best)

            refinement_log.append({
                "iteration": iteration,
                "best_score": round(best_score_ever, 6),
                "improvement": round(improvement, 6),
                "models_refined": len(iter_results),
                "details": iter_results,
            })

            # Prune models below threshold
            leader_score = best_score_ever
            top_k_names = [
                name for name in top_k_names
                if any(
                    r["val_auc"] >= leader_score - prune_threshold
                    for r in phase1_results
                    if r["name"] == name
                )
            ]

            if improvement < convergence_threshold:
                break

        # Re-sort phase1_results after refinement
        phase1_results.sort(key=lambda r: r["val_auc"], reverse=True)

        # ================================================================
        # PHASE 4: Champion Selection (Weighted Scoring Rubric)
        # ================================================================
        # PRD Section 10, Phase 4 — regulatory scoring mode
        scoring_mode = settings.tournament_scoring_mode

        if scoring_mode == "regulatory":
            weights = {
                "auc": 0.20,
                "gini": 0.15,
                "ks": 0.15,
                "brier_inv": 0.10,
                "hl_pvalue": 0.10,
                "psi_inv": 0.10,
                "interpretability": 0.10,
                "auc_gap_inv": 0.05,
                "speed_inv": 0.05,
            }
        else:
            weights = {
                "auc": 0.35,
                "gini": 0.20,
                "ks": 0.15,
                "brier_inv": 0.05,
                "hl_pvalue": 0.00,
                "psi_inv": 0.05,
                "interpretability": 0.00,
                "auc_gap_inv": 0.10,
                "speed_inv": 0.10,
            }

        from scipy.stats import ks_2samp
        from sklearn.metrics import brier_score_loss

        champion_scores: List[Dict[str, Any]] = []

        for r in phase1_results:
            if r["status"] != "trained" or r["name"] not in trained_models:
                continue

            name = r["name"]
            val_proba = trained_models[name]["val_proba"]
            val_auc = r["val_auc"]
            gini = 2 * val_auc - 1

            # KS
            pos_probs = val_proba[y_val == 1]
            neg_probs = val_proba[y_val == 0]
            ks_stat = float(ks_2samp(pos_probs, neg_probs).statistic)

            # Brier
            brier = float(brier_score_loss(y_val, val_proba))

            # Interpretability
            candidate = next(
                (c for c in PD_CANDIDATE_DEFINITIONS if c["name"] == name), {}
            )
            is_interpretable = candidate.get("interpretable", False)

            # Composite score (normalize to [0, 1] range where possible)
            score = (
                weights["auc"] * val_auc
                + weights["gini"] * (gini + 1) / 2  # Normalize Gini from [-1,1] to [0,1]
                + weights["ks"] * min(ks_stat, 1.0)
                + weights["brier_inv"] * (1 - brier)
                + weights["interpretability"] * (1.0 if is_interpretable else 0.0)
                + weights["auc_gap_inv"] * max(0, 1 - r["auc_gap"] * 10)
                + weights["speed_inv"] * max(0, 1 - r["train_time_s"] / 60)
            )

            champion_scores.append({
                "name": name,
                "composite_score": round(score, 6),
                "val_auc": round(val_auc, 6),
                "gini": round(gini, 6),
                "ks_statistic": round(ks_stat, 6),
                "brier_score": round(brier, 6),
                "auc_gap": round(r["auc_gap"], 6),
                "train_time_s": round(r["train_time_s"], 2),
                "interpretable": is_interpretable,
                "category": r["category"],
            })

        champion_scores.sort(key=lambda s: s["composite_score"], reverse=True)

        # Declare champion
        champion = champion_scores[0] if champion_scores else None
        runner_up = champion_scores[1] if len(champion_scores) > 1 else None

        # ================================================================
        # Save champion model
        # ================================================================
        if champion:
            champion_name = champion["name"]
            champion_model_data = trained_models[champion_name]

            # Save model artifact
            model_path = out / "champion_pd_model.joblib"
            joblib.dump(champion_model_data["model"], model_path)

            # Save scaler if needed
            if champion_model_data["scaler_needed"]:
                scaler_path = out / "champion_pd_scaler.joblib"
                joblib.dump(scaler, scaler_path)

            # Save validation predictions
            np.save(out / "val_y_true.npy", y_val)
            np.save(out / "val_y_prob.npy", champion_model_data["val_proba"])
            val_y_pred = (champion_model_data["val_proba"] >= 0.5).astype(int)
            np.save(out / "val_y_pred.npy", val_y_pred)

            # Save test predictions
            if champion_name in trained_models:
                model_obj = trained_models[champion_name]["model"]
                needs_scaling = trained_models[champion_name]["scaler_needed"]

                if hasattr(model_obj, "predict_proba"):
                    test_proba = model_obj.predict_proba(
                        X_test_scaled if needs_scaling else X_test
                    )[:, 1]
                elif hasattr(model_obj, "predict"):
                    # statsmodels
                    X_test_sm = sm.add_constant(
                        pd.DataFrame(X_test_scaled, columns=feature_names),
                        has_constant="add",
                    )
                    test_proba = model_obj.predict(X_test_sm)
                else:
                    test_proba = np.zeros(len(y_test))

                np.save(out / "test_y_true.npy", y_test)
                np.save(out / "test_y_prob.npy", test_proba)
                test_y_pred = (test_proba >= 0.5).astype(int)
                np.save(out / "test_y_pred.npy", test_y_pred)

                test_auc = float(roc_auc_score(y_test, test_proba))
            else:
                test_auc = 0.0

        # ================================================================
        # Build results summary
        # ================================================================
        tournament_results = {
            "phase_1_broad_sweep": {
                "models_attempted": len(PD_CANDIDATE_DEFINITIONS),
                "models_trained": sum(1 for r in phase1_results if r["status"] == "trained"),
                "models_failed": sum(1 for r in phase1_results if r["status"] != "trained"),
                "results": phase1_results,
            },
            "phase_2_feature_consensus": {
                "total_features": len(feature_names),
                "tier_counts": tier_counts,
                "top_20_features": top_features,
            },
            "phase_3_refinement": {
                "iterations_run": len(refinement_log),
                "top_k": top_k,
                "convergence_threshold": convergence_threshold,
                "log": refinement_log,
            },
            "phase_4_champion_selection": {
                "scoring_mode": scoring_mode,
                "weights": weights,
                "rankings": champion_scores,
                "champion": champion,
                "runner_up": runner_up,
            },
        }

        if champion:
            tournament_results["champion_summary"] = {
                "name": champion["name"],
                "val_auc": champion["val_auc"],
                "test_auc": round(test_auc, 6),
                "gini": champion["gini"],
                "composite_score": champion["composite_score"],
                "model_file": str(model_path),
                "interpretable": champion["interpretable"],
            }

        # Save tournament results
        results_path = out / "tournament_results.json"
        results_path.write_text(json.dumps(tournament_results, indent=2, default=str))

        return _ok(tournament_results)
    except Exception as exc:
        return _error(f"Failed to run PD tournament: {exc}")


# ---------------------------------------------------------------------------
# Export all tools
# ---------------------------------------------------------------------------

ALL_PD_TOOLS = [
    define_pd_candidates,
    construct_pd_target,
    run_pd_tournament,
]
