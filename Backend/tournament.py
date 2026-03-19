"""
Core 4-phase Model Tournament Engine
=====================================
Shared by PD (Probability of Default), LGD (Loss Given Default),
and EAD (Exposure at Default) agents.

Phases:
  1. Broad Sweep        -- fit every candidate with baseline params
  2. Feature Consensus   -- weighted-average importance across models, tiered
  3. Refinement Loop     -- hyperparameter search x feature-set search, prune
  4. Champion Selection  -- weighted rubric scoring
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Literal

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    brier_score_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class CandidateModel:
    """Blueprint for one candidate model to enter the tournament."""
    name: str
    library: str  # "sklearn", "xgboost", "lightgbm", "statsmodels"
    estimator_class: type
    baseline_params: dict
    hyperparam_distributions: dict  # for RandomizedSearchCV


@dataclass
class LeaderboardEntry:
    """Single row on the tournament leaderboard."""
    name: str
    library: str
    estimator: Any  # fitted model
    metrics: dict[str, float]  # auc, gini, ks, brier, rmse, mae, r2 etc.
    feature_importances: np.ndarray | None
    training_time_s: float
    feature_count: int
    status: str  # "trained" | "failed"
    error: str | None = None


@dataclass
class TournamentSettings:
    """Knobs that govern every phase of the tournament."""
    top_k: int = 5
    max_iterations: int = 5
    convergence_threshold: float = 0.002
    prune_threshold: float = 0.03
    cv_splits: int = 5
    random_search_iter: int = 50
    use_shap: bool = True
    feature_tiers: bool = True
    scoring_mode: str = "regulatory"  # "regulatory" | "performance"


@dataclass
class FeatureConsensus:
    """Cross-model agreement on feature importance."""
    feature_names: list[str]
    consensus_scores: np.ndarray
    tiers: dict[str, int]  # feature_name -> tier (1-4)
    tier_counts: dict[int, int]


@dataclass
class TournamentResult:
    """Final output of the full tournament pipeline."""
    champion: LeaderboardEntry
    runner_up: LeaderboardEntry | None
    leaderboard: list[LeaderboardEntry]
    feature_consensus: FeatureConsensus | None
    iterations_completed: int
    converged: bool


# ---------------------------------------------------------------------------
# Helper: Hosmer-Lemeshow statistic
# ---------------------------------------------------------------------------

def _hosmer_lemeshow(y_true: np.ndarray, y_prob: np.ndarray, g: int = 10) -> float:
    """Return the Hosmer-Lemeshow chi-squared p-value.

    A higher p-value indicates better calibration.  Returns 1.0 on any
    failure so that the metric degrades gracefully.
    """
    try:
        order = np.argsort(y_prob)
        y_true_sorted = np.asarray(y_true, dtype=float)[order]
        y_prob_sorted = np.asarray(y_prob, dtype=float)[order]
        groups = np.array_split(np.arange(len(y_true_sorted)), g)

        hl_stat = 0.0
        for idx in groups:
            obs_1 = y_true_sorted[idx].sum()
            n_g = len(idx)
            exp_1 = y_prob_sorted[idx].sum()
            exp_0 = n_g - exp_1
            if exp_1 > 0:
                hl_stat += (obs_1 - exp_1) ** 2 / exp_1
            if exp_0 > 0:
                obs_0 = n_g - obs_1
                hl_stat += (obs_0 - exp_0) ** 2 / exp_0

        p_value = 1.0 - scipy_stats.chi2.cdf(hl_stat, g - 2)
        return float(p_value)
    except Exception:
        return 1.0


# ---------------------------------------------------------------------------
# ModelTournament
# ---------------------------------------------------------------------------

class ModelTournament:
    """Four-phase tournament that pits candidate models against each other."""

    def __init__(
        self,
        model_type: Literal["classification", "regression"],
        candidate_pool: list[CandidateModel],
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        settings: TournamentSettings | None = None,
        event_callback: Callable | None = None,
    ):
        self.model_type = model_type
        self.candidate_pool = candidate_pool
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.X_test = X_test
        self.y_test = y_test
        self.settings = settings or TournamentSettings()
        self.event_callback = event_callback

        # Internal bookkeeping
        self._primary_metric = "auc" if model_type == "classification" else "rmse"
        self._primary_ascending = model_type == "regression"  # lower RMSE is better

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _emit(self, method_name: str, **kwargs: Any) -> None:
        """Safely invoke *event_callback.<method_name>* if the callback exists."""
        if self.event_callback is not None:
            fn = getattr(self.event_callback, method_name, None)
            if fn is not None:
                try:
                    fn(**kwargs)
                except Exception:
                    pass  # never let a callback crash the tournament

    def _evaluate_model(self, model: Any, X: pd.DataFrame, y: pd.Series) -> dict[str, float]:
        """Compute all relevant metrics for *model* on the given data."""
        metrics: dict[str, float] = {}

        if self.model_type == "classification":
            # Predicted probabilities for the positive class
            if hasattr(model, "predict_proba"):
                y_prob = model.predict_proba(X)[:, 1]
            elif hasattr(model, "decision_function"):
                raw = model.decision_function(X)
                # Sigmoid to get probabilities
                y_prob = 1.0 / (1.0 + np.exp(-raw))
            else:
                y_prob = model.predict(X).astype(float)

            y_true = np.asarray(y, dtype=float)

            # AUC
            try:
                auc = roc_auc_score(y_true, y_prob)
            except ValueError:
                auc = 0.5
            metrics["auc"] = auc

            # Gini coefficient
            metrics["gini"] = 2.0 * auc - 1.0

            # KS statistic (Kolmogorov-Smirnov)
            try:
                pos_probs = y_prob[y_true == 1]
                neg_probs = y_prob[y_true == 0]
                if len(pos_probs) > 0 and len(neg_probs) > 0:
                    ks_stat, _ = scipy_stats.ks_2samp(pos_probs, neg_probs)
                else:
                    ks_stat = 0.0
            except Exception:
                ks_stat = 0.0
            metrics["ks"] = ks_stat

            # Brier score (lower is better)
            try:
                metrics["brier"] = brier_score_loss(y_true, y_prob)
            except Exception:
                metrics["brier"] = 1.0

            # Hosmer-Lemeshow p-value (higher is better calibration)
            metrics["hl_pvalue"] = _hosmer_lemeshow(y_true, y_prob)

        else:
            # Regression metrics
            y_pred = model.predict(X)
            y_true = np.asarray(y, dtype=float)

            metrics["rmse"] = float(np.sqrt(mean_squared_error(y_true, y_pred)))
            metrics["mae"] = float(mean_absolute_error(y_true, y_pred))
            metrics["r2"] = float(r2_score(y_true, y_pred))

        return metrics

    def _extract_importances(self, model: Any, X: pd.DataFrame) -> np.ndarray:
        """Extract normalised feature importances from *model*.

        Strategy:
          1. Linear models expose ``.coef_`` -- use ``abs(coef_)`` normalised.
          2. Tree models expose ``.feature_importances_`` -- use directly.
          3. Fallback: ``sklearn.inspection.permutation_importance``.
        """
        n_features = X.shape[1]

        # Strategy 1: linear coefficients
        if hasattr(model, "coef_"):
            coef = np.asarray(model.coef_).flatten()
            if coef.shape[0] == n_features:
                abs_coef = np.abs(coef)
                total = abs_coef.sum()
                if total > 0:
                    return abs_coef / total
                return np.ones(n_features) / n_features

        # Strategy 2: tree-based importances
        if hasattr(model, "feature_importances_"):
            imp = np.asarray(model.feature_importances_).flatten()
            if imp.shape[0] == n_features:
                total = imp.sum()
                if total > 0:
                    return imp / total
                return np.ones(n_features) / n_features

        # Strategy 3: permutation importance (slower, model-agnostic)
        try:
            scoring = "roc_auc" if self.model_type == "classification" else "neg_root_mean_squared_error"
            result = permutation_importance(
                model, X, self.y_val, n_repeats=5, random_state=42, scoring=scoring, n_jobs=-1,
            )
            imp = np.maximum(result.importances_mean, 0.0)
            total = imp.sum()
            if total > 0:
                return imp / total
        except Exception:
            pass

        return np.ones(n_features) / n_features

    def _primary_score(self, entry: LeaderboardEntry) -> float:
        """Return a *higher-is-better* scalar used for sorting."""
        val = entry.metrics.get(self._primary_metric, 0.0)
        if self._primary_ascending:
            return -val  # negate RMSE so that lower RMSE -> higher score
        return val

    def _sort_leaderboard(self, leaderboard: list[LeaderboardEntry]) -> list[LeaderboardEntry]:
        """Return leaderboard sorted best-first (highest primary score first)."""
        return sorted(leaderboard, key=self._primary_score, reverse=True)

    # ------------------------------------------------------------------
    # Phase 1 -- Broad Sweep
    # ------------------------------------------------------------------

    def phase1_broad_sweep(self) -> list[LeaderboardEntry]:
        """Fit every candidate with its baseline params and rank on validation."""
        leaderboard: list[LeaderboardEntry] = []

        for candidate in self.candidate_pool:
            t0 = time.perf_counter()
            try:
                estimator = candidate.estimator_class(**candidate.baseline_params)

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    estimator.fit(self.X_train, self.y_train)

                elapsed = time.perf_counter() - t0
                metrics = self._evaluate_model(estimator, self.X_val, self.y_val)
                importances = self._extract_importances(estimator, self.X_val)

                entry = LeaderboardEntry(
                    name=candidate.name,
                    library=candidate.library,
                    estimator=estimator,
                    metrics=metrics,
                    feature_importances=importances,
                    training_time_s=elapsed,
                    feature_count=self.X_train.shape[1],
                    status="trained",
                )
            except Exception as exc:
                elapsed = time.perf_counter() - t0
                entry = LeaderboardEntry(
                    name=candidate.name,
                    library=candidate.library,
                    estimator=None,
                    metrics={},
                    feature_importances=None,
                    training_time_s=elapsed,
                    feature_count=self.X_train.shape[1],
                    status="failed",
                    error=str(exc),
                )

            leaderboard.append(entry)
            self._emit(
                "on_model_trained",
                model_name=entry.name,
                metrics=entry.metrics,
                status=entry.status,
                training_time_s=entry.training_time_s,
            )

        leaderboard = self._sort_leaderboard(leaderboard)
        return leaderboard

    # ------------------------------------------------------------------
    # Phase 2 -- Feature Consensus
    # ------------------------------------------------------------------

    def phase2_feature_consensus(self, leaderboard: list[LeaderboardEntry]) -> FeatureConsensus:
        """Build a cross-model weighted feature importance consensus."""
        feature_names = list(self.X_train.columns)
        n_features = len(feature_names)

        # Collect (importance_vector, weight) pairs for every trained model
        weighted_importances = np.zeros(n_features, dtype=float)
        total_weight = 0.0

        for entry in leaderboard:
            if entry.status != "trained" or entry.feature_importances is None:
                continue

            imp = entry.feature_importances
            if imp.shape[0] != n_features:
                continue

            # Weight = model's primary metric (higher-is-better sense)
            weight = self._primary_score(entry)
            # Shift so weights are positive even for regression
            weight = max(weight, 1e-8)

            weighted_importances += imp * weight
            total_weight += weight

        if total_weight > 0:
            consensus_scores = weighted_importances / total_weight
        else:
            consensus_scores = np.ones(n_features) / n_features

        # Normalise to sum to 1
        cs_total = consensus_scores.sum()
        if cs_total > 0:
            consensus_scores = consensus_scores / cs_total

        # Assign tiers based on cumulative importance (sorted descending)
        sorted_indices = np.argsort(-consensus_scores)
        n = len(sorted_indices)
        tier_boundaries = [
            int(np.ceil(n * 0.20)),   # Tier 1: top 20%
            int(np.ceil(n * 0.50)),   # Tier 2: next 30% (cumul 50%)
            int(np.ceil(n * 0.80)),   # Tier 3: next 30% (cumul 80%)
        ]

        tiers: dict[str, int] = {}
        for rank, idx in enumerate(sorted_indices):
            fname = feature_names[idx]
            if rank < tier_boundaries[0]:
                tiers[fname] = 1
            elif rank < tier_boundaries[1]:
                tiers[fname] = 2
            elif rank < tier_boundaries[2]:
                tiers[fname] = 3
            else:
                tiers[fname] = 4

        tier_counts: dict[int, int] = {t: 0 for t in (1, 2, 3, 4)}
        for t in tiers.values():
            tier_counts[t] += 1

        consensus = FeatureConsensus(
            feature_names=feature_names,
            consensus_scores=consensus_scores,
            tiers=tiers,
            tier_counts=tier_counts,
        )

        self._emit(
            "on_feature_consensus",
            tier_counts=tier_counts,
            top_features=[feature_names[i] for i in sorted_indices[:10]],
        )

        return consensus

    # ------------------------------------------------------------------
    # Phase 3 -- Refinement Loop
    # ------------------------------------------------------------------

    def phase3_refinement_loop(
        self,
        leaderboard: list[LeaderboardEntry],
        consensus: FeatureConsensus,
    ) -> list[LeaderboardEntry]:
        """Hyperparameter + feature-set search for the top-K models."""
        settings = self.settings

        # Only keep successfully trained models
        trained = [e for e in leaderboard if e.status == "trained"]
        top_k = trained[: settings.top_k]
        if not top_k:
            return leaderboard

        # Build feature-set variants from consensus tiers
        tier12_cols = [f for f, t in consensus.tiers.items() if t in (1, 2)]
        tier123_cols = [f for f, t in consensus.tiers.items() if t in (1, 2, 3)]
        full_cols = list(self.X_train.columns)

        feature_sets: list[tuple[str, list[str]]] = [
            ("tier12", tier12_cols),
            ("tier123", tier123_cols),
            ("full", full_cols),
        ]
        # Drop empty sets and ensure columns exist in data
        feature_sets = [
            (label, [c for c in cols if c in self.X_train.columns])
            for label, cols in feature_sets
            if cols
        ]
        # Deduplicate identical column lists
        seen_col_keys: set[str] = set()
        unique_feature_sets: list[tuple[str, list[str]]] = []
        for label, cols in feature_sets:
            key = ",".join(sorted(cols))
            if key not in seen_col_keys:
                seen_col_keys.add(key)
                unique_feature_sets.append((label, cols))
        feature_sets = unique_feature_sets if unique_feature_sets else [("full", full_cols)]

        # Map candidate name -> CandidateModel for hyperparam distributions
        candidate_map: dict[str, CandidateModel] = {c.name: c for c in self.candidate_pool}

        previous_best_score: float | None = None
        iterations_completed = 0

        for iteration in range(1, settings.max_iterations + 1):
            iterations_completed = iteration
            new_entries: list[LeaderboardEntry] = []

            for entry in top_k:
                candidate = candidate_map.get(entry.name)
                if candidate is None:
                    new_entries.append(entry)
                    continue

                best_entry_for_model = entry  # keep current as fallback

                for fs_label, fs_cols in feature_sets:
                    if not fs_cols:
                        continue

                    X_tr = self.X_train[fs_cols]
                    X_va = self.X_val[fs_cols]

                    # Run RandomizedSearchCV if distributions are provided
                    if candidate.hyperparam_distributions:
                        t0 = time.perf_counter()
                        try:
                            base_estimator = candidate.estimator_class(**candidate.baseline_params)
                            scoring = (
                                "roc_auc"
                                if self.model_type == "classification"
                                else "neg_root_mean_squared_error"
                            )
                            tscv = TimeSeriesSplit(n_splits=settings.cv_splits)

                            search = RandomizedSearchCV(
                                estimator=base_estimator,
                                param_distributions=candidate.hyperparam_distributions,
                                n_iter=settings.random_search_iter,
                                scoring=scoring,
                                cv=tscv,
                                random_state=42 + iteration,
                                n_jobs=-1,
                                refit=True,
                                error_score="raise",
                            )

                            with warnings.catch_warnings():
                                warnings.simplefilter("ignore")
                                search.fit(X_tr, self.y_train)

                            elapsed = time.perf_counter() - t0
                            best_model = search.best_estimator_
                            metrics = self._evaluate_model(best_model, X_va, self.y_val)
                            importances = self._extract_importances(best_model, X_va)

                            refined_entry = LeaderboardEntry(
                                name=f"{entry.name}|{fs_label}|iter{iteration}",
                                library=entry.library,
                                estimator=best_model,
                                metrics=metrics,
                                feature_importances=importances,
                                training_time_s=elapsed,
                                feature_count=len(fs_cols),
                                status="trained",
                            )

                            # Keep the best variant for this model
                            if self._primary_score(refined_entry) > self._primary_score(best_entry_for_model):
                                best_entry_for_model = refined_entry

                        except Exception as exc:
                            elapsed = time.perf_counter() - t0
                            # If search fails, keep previous best
                            pass
                    else:
                        # No hyperparam distributions -- just refit on the feature subset
                        t0 = time.perf_counter()
                        try:
                            estimator = candidate.estimator_class(**candidate.baseline_params)
                            with warnings.catch_warnings():
                                warnings.simplefilter("ignore")
                                estimator.fit(X_tr, self.y_train)

                            elapsed = time.perf_counter() - t0
                            metrics = self._evaluate_model(estimator, X_va, self.y_val)
                            importances = self._extract_importances(estimator, X_va)

                            refined_entry = LeaderboardEntry(
                                name=f"{entry.name}|{fs_label}|iter{iteration}",
                                library=entry.library,
                                estimator=estimator,
                                metrics=metrics,
                                feature_importances=importances,
                                training_time_s=elapsed,
                                feature_count=len(fs_cols),
                                status="trained",
                            )

                            if self._primary_score(refined_entry) > self._primary_score(best_entry_for_model):
                                best_entry_for_model = refined_entry

                        except Exception:
                            pass

                new_entries.append(best_entry_for_model)

            # Sort and determine current best
            new_entries = self._sort_leaderboard(new_entries)
            current_best_score = self._primary_score(new_entries[0]) if new_entries else 0.0

            # Convergence check
            converged = False
            if previous_best_score is not None:
                improvement = abs(current_best_score - previous_best_score)
                if improvement < settings.convergence_threshold:
                    converged = True

            previous_best_score = current_best_score

            # Prune models too far behind the leader
            if len(new_entries) > 1:
                leader_score = self._primary_score(new_entries[0])
                pruned: list[LeaderboardEntry] = [new_entries[0]]
                for e in new_entries[1:]:
                    gap = leader_score - self._primary_score(e)
                    if gap <= settings.prune_threshold:
                        pruned.append(e)
                new_entries = pruned

            top_k = new_entries

            self._emit(
                "on_iteration_update",
                iteration=iteration,
                best_score=current_best_score,
                models_remaining=len(top_k),
                converged=converged,
            )

            if converged:
                break

        # Merge refined top-K back into the full leaderboard
        refined_names_base = {e.name.split("|")[0] for e in top_k}
        remaining = [e for e in leaderboard if e.name not in refined_names_base and e.status == "trained"]
        full_leaderboard = self._sort_leaderboard(list(top_k) + remaining)

        return full_leaderboard

    # ------------------------------------------------------------------
    # Phase 4 -- Champion Selection
    # ------------------------------------------------------------------

    def phase4_champion_selection(self, leaderboard: list[LeaderboardEntry]) -> TournamentResult:
        """Apply a weighted rubric to pick the champion."""
        trained = [e for e in leaderboard if e.status == "trained"]
        if not trained:
            raise ValueError("No successfully trained models in the leaderboard.")

        # ---- define rubrics ----
        if self.model_type == "classification":
            if self.settings.scoring_mode == "regulatory":
                rubric: dict[str, float] = {
                    "auc": 0.20,
                    "gini": 0.15,
                    "ks": 0.15,
                    "brier": 0.10,
                    "hl_pvalue": 0.10,
                    "psi": 0.10,
                    "interpretability": 0.10,
                    "generalization": 0.05,
                    "efficiency": 0.05,
                }
            else:
                rubric = {
                    "auc": 0.35,
                    "gini": 0.20,
                    "ks": 0.15,
                    "brier": 0.10,
                    "generalization": 0.10,
                    "efficiency": 0.10,
                }
        else:
            rubric = {
                "rmse": 0.30,
                "mae": 0.15,
                "r2": 0.20,
                "psi": 0.10,
                "decile": 0.10,
                "generalization": 0.10,
                "efficiency": 0.05,
            }

        # ---- compute per-model rubric scores ----
        # First collect raw metric arrays for min-max normalisation
        metric_keys_in_rubric = [k for k in rubric if k in ("auc", "gini", "ks", "brier", "hl_pvalue", "rmse", "mae", "r2")]
        raw_values: dict[str, list[float]] = {k: [] for k in metric_keys_in_rubric}
        for entry in trained:
            for k in metric_keys_in_rubric:
                raw_values[k].append(entry.metrics.get(k, 0.0))

        # Min-max ranges
        metric_min: dict[str, float] = {}
        metric_max: dict[str, float] = {}
        for k, vals in raw_values.items():
            metric_min[k] = min(vals) if vals else 0.0
            metric_max[k] = max(vals) if vals else 1.0

        # Higher-is-better for these metrics; lower-is-better for brier, rmse, mae
        lower_is_better = {"brier", "rmse", "mae"}

        def _normalise(key: str, value: float) -> float:
            lo = metric_min.get(key, 0.0)
            hi = metric_max.get(key, 1.0)
            rng = hi - lo
            if rng < 1e-12:
                return 1.0
            normed = (value - lo) / rng
            if key in lower_is_better:
                normed = 1.0 - normed
            return float(np.clip(normed, 0.0, 1.0))

        # Compute test-set metrics for generalization gap
        test_metrics_cache: dict[str, dict[str, float]] = {}
        for entry in trained:
            if entry.estimator is not None:
                try:
                    # Use the correct feature set for the model
                    X_test_subset = self.X_test
                    if entry.feature_count < self.X_test.shape[1]:
                        # The model was trained on a feature subset -- infer columns
                        # from the estimator if possible; otherwise use full set
                        if hasattr(entry.estimator, "feature_names_in_"):
                            cols = [c for c in entry.estimator.feature_names_in_ if c in self.X_test.columns]
                            if cols:
                                X_test_subset = self.X_test[cols]
                    test_metrics_cache[entry.name] = self._evaluate_model(entry.estimator, X_test_subset, self.y_test)
                except Exception:
                    test_metrics_cache[entry.name] = {}

        # Interpretability proxy: prefer fewer features and linear models
        max_features = max(e.feature_count for e in trained) if trained else 1
        linear_libs = {"sklearn", "statsmodels"}

        # Training-time efficiency: normalise by longest time
        max_time = max(e.training_time_s for e in trained) if trained else 1.0

        scored: list[tuple[float, LeaderboardEntry]] = []

        for entry in trained:
            total = 0.0

            # Metric-based components
            for k in metric_keys_in_rubric:
                weight = rubric.get(k, 0.0)
                val = entry.metrics.get(k, 0.0)
                total += weight * _normalise(k, val)

            # PSI proxy: stability between val and test predictions
            psi_weight = rubric.get("psi", 0.0)
            if psi_weight > 0 and entry.estimator is not None:
                try:
                    X_test_subset = self.X_test
                    if hasattr(entry.estimator, "feature_names_in_"):
                        cols = [c for c in entry.estimator.feature_names_in_ if c in self.X_test.columns]
                        if cols:
                            X_test_subset = self.X_test[cols]

                    if self.model_type == "classification":
                        if hasattr(entry.estimator, "predict_proba"):
                            p_val = entry.estimator.predict_proba(self.X_val)[:, 1]
                            p_test = entry.estimator.predict_proba(X_test_subset)[:, 1]
                        else:
                            p_val = entry.estimator.predict(self.X_val).astype(float)
                            p_test = entry.estimator.predict(X_test_subset).astype(float)
                    else:
                        p_val = entry.estimator.predict(self.X_val).astype(float)
                        p_test = entry.estimator.predict(X_test_subset).astype(float)

                    psi_score = self._compute_psi(p_val, p_test)
                    # PSI < 0.10 is stable => high score; PSI > 0.25 is unstable
                    psi_normed = float(np.clip(1.0 - psi_score / 0.25, 0.0, 1.0))
                    total += psi_weight * psi_normed
                except Exception:
                    total += psi_weight * 0.5  # neutral if computation fails

            # Decile lift (regression): correlation of predicted vs actual in deciles
            decile_weight = rubric.get("decile", 0.0)
            if decile_weight > 0 and entry.estimator is not None:
                try:
                    X_val_subset = self.X_val
                    if hasattr(entry.estimator, "feature_names_in_"):
                        cols = [c for c in entry.estimator.feature_names_in_ if c in self.X_val.columns]
                        if cols:
                            X_val_subset = self.X_val[cols]
                    y_pred = entry.estimator.predict(X_val_subset)
                    decile_corr = self._decile_correlation(np.asarray(self.y_val), y_pred)
                    total += decile_weight * float(np.clip(decile_corr, 0.0, 1.0))
                except Exception:
                    total += decile_weight * 0.5

            # Interpretability score
            interp_weight = rubric.get("interpretability", 0.0)
            if interp_weight > 0:
                feat_ratio = 1.0 - (entry.feature_count / max_features) if max_features > 0 else 0.0
                lib_bonus = 0.3 if entry.library in linear_libs else 0.0
                total += interp_weight * float(np.clip(0.7 * feat_ratio + lib_bonus, 0.0, 1.0))

            # Generalization gap: compare val metric to test metric
            gen_weight = rubric.get("generalization", 0.0)
            if gen_weight > 0:
                test_m = test_metrics_cache.get(entry.name, {})
                if self._primary_metric in entry.metrics and self._primary_metric in test_m:
                    val_score = entry.metrics[self._primary_metric]
                    test_score = test_m[self._primary_metric]
                    gap = abs(val_score - test_score)
                    # Smaller gap -> higher score; gap > 0.10 is bad
                    gen_normed = float(np.clip(1.0 - gap / 0.10, 0.0, 1.0))
                    total += gen_weight * gen_normed
                else:
                    total += gen_weight * 0.5

            # Efficiency: faster is better
            eff_weight = rubric.get("efficiency", 0.0)
            if eff_weight > 0 and max_time > 0:
                time_ratio = 1.0 - (entry.training_time_s / max_time)
                total += eff_weight * float(np.clip(time_ratio, 0.0, 1.0))

            scored.append((total, entry))

        # Sort by total rubric score descending
        scored.sort(key=lambda x: x[0], reverse=True)

        champion = scored[0][1]
        runner_up = scored[1][1] if len(scored) > 1 else None

        sorted_entries = [entry for _, entry in scored]

        self._emit(
            "on_champion_declared",
            champion_name=champion.name,
            champion_metrics=champion.metrics,
            runner_up_name=runner_up.name if runner_up else None,
        )

        # We don't have full iteration info at this level; caller fills it in
        return TournamentResult(
            champion=champion,
            runner_up=runner_up,
            leaderboard=sorted_entries,
            feature_consensus=None,  # attached by run_full_tournament
            iterations_completed=0,
            converged=False,
        )

    # ------------------------------------------------------------------
    # PSI & decile helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_psi(expected: np.ndarray, actual: np.ndarray, buckets: int = 10) -> float:
        """Population Stability Index between two score distributions."""
        try:
            # Create buckets from the expected distribution
            breakpoints = np.linspace(0, 100, buckets + 1)
            expected_percents = np.percentile(expected, breakpoints)

            # Bin both distributions
            expected_counts = np.histogram(expected, bins=expected_percents)[0].astype(float)
            actual_counts = np.histogram(actual, bins=expected_percents)[0].astype(float)

            # Convert to proportions, avoid zeros
            expected_props = expected_counts / expected_counts.sum()
            actual_props = actual_counts / actual_counts.sum()

            expected_props = np.clip(expected_props, 1e-8, None)
            actual_props = np.clip(actual_props, 1e-8, None)

            psi = np.sum((actual_props - expected_props) * np.log(actual_props / expected_props))
            return float(psi)
        except Exception:
            return 0.0

    @staticmethod
    def _decile_correlation(y_true: np.ndarray, y_pred: np.ndarray, n_deciles: int = 10) -> float:
        """Rank correlation between predicted and actual means within deciles."""
        try:
            order = np.argsort(y_pred)
            groups = np.array_split(order, n_deciles)
            pred_means = [y_pred[g].mean() for g in groups if len(g) > 0]
            true_means = [y_true[g].mean() for g in groups if len(g) > 0]
            if len(pred_means) < 3:
                return 0.5
            corr, _ = scipy_stats.spearmanr(pred_means, true_means)
            if np.isnan(corr):
                return 0.5
            return float((corr + 1.0) / 2.0)  # map [-1,1] -> [0,1]
        except Exception:
            return 0.5

    # ------------------------------------------------------------------
    # Full Tournament
    # ------------------------------------------------------------------

    def run_full_tournament(self) -> TournamentResult:
        """Execute all four phases and return the final result."""

        # Phase 1
        leaderboard = self.phase1_broad_sweep()

        # Phase 2
        consensus = self.phase2_feature_consensus(leaderboard)

        # Phase 3
        refined_leaderboard = self.phase3_refinement_loop(leaderboard, consensus)

        # Phase 4
        result = self.phase4_champion_selection(refined_leaderboard)

        # Attach consensus and iteration metadata
        result.feature_consensus = consensus

        # Determine iterations completed and convergence from phase 3 state
        # We re-derive these from the leaderboard entry names
        iteration_numbers = []
        for entry in refined_leaderboard:
            parts = entry.name.split("|")
            for part in parts:
                if part.startswith("iter"):
                    try:
                        iteration_numbers.append(int(part.replace("iter", "")))
                    except ValueError:
                        pass

        if iteration_numbers:
            result.iterations_completed = max(iteration_numbers)
            result.converged = result.iterations_completed < self.settings.max_iterations
        else:
            result.iterations_completed = 0
            result.converged = False

        return result
