"""
Parallel Training of Model Candidates

Accelerates model tournament by training multiple candidates in parallel
using joblib. Automatically handles resource constraints and provides
progress tracking.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional, Callable

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ParallelTrainingConfig:
    """
    Configuration for parallel candidate training.

    Controls resource allocation and behavior of parallel training.
    """

    n_jobs: int = 4
    """Number of parallel jobs (-1 = use all cores, default: 4)"""

    verbose: int = 1
    """Verbosity level for joblib (0-10, default: 1)"""

    backend: str = "threading"
    """Joblib backend: 'threading' or 'multiprocessing' (default: 'threading')"""

    max_memory: Optional[str] = None
    """Memory limit per job (e.g., '4GB'), None for unlimited"""

    timeout: Optional[int] = None
    """Timeout per job in seconds, None for unlimited"""

    prefer_processes: bool = False
    """Use multiprocessing instead of threading for CPU-bound work (default: False)"""

    def get_joblib_kwargs(self) -> dict:
        """
        Get kwargs for joblib.Parallel initialization.

        Returns:
            Dictionary of kwargs for Parallel()
        """
        kwargs = {
            "n_jobs": self.n_jobs,
            "verbose": self.verbose,
            "backend": self.backend,
        }

        if self.timeout is not None:
            kwargs["timeout"] = self.timeout

        if self.max_memory is not None:
            kwargs["max_nbytes"] = self.max_memory

        return kwargs


def train_candidates_parallel(
    candidates: List[Tuple[str, Any]],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_jobs: int = 4,
    verbose: bool = True,
    config: Optional[ParallelTrainingConfig] = None,
) -> Dict[str, Tuple[Any, dict]]:
    """
    Train multiple model candidates in parallel.

    Distributes candidate training across multiple cores/threads using joblib.
    Each candidate is trained on the same train/validation split.

    Args:
        candidates: List of (name, model_instance) tuples to train
        X_train: Training feature matrix
        y_train: Training target vector
        X_val: Validation feature matrix
        y_val: Validation target vector
        n_jobs: Number of parallel jobs (default: 4)
        verbose: Print progress (default: True)
        config: ParallelTrainingConfig instance (optional)

    Returns:
        Dictionary mapping candidate_name -> (trained_model, metrics_dict)

    Example:
        >>> from sklearn.ensemble import RandomForestClassifier
        >>> from xgboost import XGBClassifier
        >>> candidates = [
        ...     ("rf_50", RandomForestClassifier(n_estimators=50)),
        ...     ("xgb", XGBClassifier()),
        ... ]
        >>> results = train_candidates_parallel(
        ...     candidates, X_train, y_train, X_val, y_val, n_jobs=4
        ... )
        >>> print(results["rf_50"][1]["auc"])  # Get RF AUC
    """
    if config is None:
        config = ParallelTrainingConfig(n_jobs=n_jobs, verbose=1 if verbose else 0)

    try:
        from joblib import Parallel, delayed
    except ImportError:
        # Fallback to sequential training
        return _train_candidates_sequential(
            candidates, X_train, y_train, X_val, y_val, verbose
        )

    def train_single_candidate(
        name: str, model: Any, X_tr: np.ndarray, y_tr: np.ndarray,
        X_v: np.ndarray, y_v: np.ndarray
    ) -> Tuple[str, Any, dict]:
        """Train a single candidate and compute metrics."""
        try:
            model.fit(X_tr, y_tr)
            metrics = _compute_candidate_metrics(model, X_v, y_v)
            return name, model, metrics
        except Exception as e:
            # Return error information
            return name, None, {"error": str(e)}

    # Schedule parallel training jobs
    joblib_kwargs = config.get_joblib_kwargs()
    parallel = Parallel(**joblib_kwargs)

    jobs = [
        delayed(train_single_candidate)(
            name, model, X_train, y_train, X_val, y_val
        )
        for name, model in candidates
    ]

    results_list = parallel(jobs)

    # Reorganize results into dictionary
    results = {}
    for name, model, metrics in results_list:
        if model is not None:
            results[name] = (model, metrics)
        else:
            results[name] = (None, metrics)  # Error case

    return results


def _train_candidates_sequential(
    candidates: List[Tuple[str, Any]],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    verbose: bool = True,
) -> Dict[str, Tuple[Any, dict]]:
    """
    Fallback sequential training when joblib is unavailable.

    Args:
        candidates: List of (name, model) tuples
        X_train: Training features
        y_train: Training targets
        X_val: Validation features
        y_val: Validation targets
        verbose: Print progress

    Returns:
        Dictionary of training results
    """
    results = {}

    for i, (name, model) in enumerate(candidates):
        if verbose:
            logger.info("Training %d/%d: %s", i + 1, len(candidates), name)

        try:
            model.fit(X_train, y_train)
            metrics = _compute_candidate_metrics(model, X_val, y_val)
            results[name] = (model, metrics)
        except Exception as e:
            results[name] = (None, {"error": str(e)})

    return results


def _compute_candidate_metrics(model: Any, X_val: np.ndarray, y_val: np.ndarray) -> dict:
    """
    Compute standard metrics for a candidate model.

    Args:
        model: Trained model
        X_val: Validation features
        y_val: Validation targets

    Returns:
        Dictionary of computed metrics
    """
    metrics = {}

    try:
        # Predictions
        if hasattr(model, "predict_proba"):
            y_pred_proba = model.predict_proba(X_val)[:, 1]
        else:
            y_pred_proba = model.predict(X_val)

        y_pred = model.predict(X_val)

        # AUC
        try:
            from sklearn.metrics import roc_auc_score
            metrics["auc"] = roc_auc_score(y_val, y_pred_proba)
        except Exception:
            pass

        # Accuracy
        try:
            from sklearn.metrics import accuracy_score
            metrics["accuracy"] = accuracy_score(y_val, y_pred)
        except Exception:
            pass

        # Precision, Recall
        try:
            from sklearn.metrics import precision_score, recall_score
            metrics["precision"] = precision_score(y_val, y_pred, zero_division=0)
            metrics["recall"] = recall_score(y_val, y_pred, zero_division=0)
        except Exception:
            pass

        # F1
        try:
            from sklearn.metrics import f1_score
            metrics["f1"] = f1_score(y_val, y_pred, zero_division=0)
        except Exception:
            pass

    except Exception as e:
        metrics["error"] = str(e)

    return metrics


def estimate_parallel_speedup(
    n_candidates: int, n_jobs: int = 4, overhead_percent: float = 10
) -> float:
    """
    Estimate speedup from parallel training.

    Accounts for joblib overhead and assumes linear speedup up to n_jobs.

    Args:
        n_candidates: Number of models to train
        n_jobs: Number of parallel jobs
        overhead_percent: Estimated parallel overhead as percentage (default: 10)

    Returns:
        Estimated speedup factor (e.g., 3.5x faster)
    """
    # Theoretical speedup with perfect parallelization
    theoretical_speedup = min(n_candidates, n_jobs)

    # Account for overhead
    overhead_factor = 1 + (overhead_percent / 100)
    actual_speedup = theoretical_speedup / overhead_factor

    return max(1.0, actual_speedup)


def get_recommended_n_jobs(n_candidates: int) -> int:
    """
    Recommend number of parallel jobs based on candidate count.

    Args:
        n_candidates: Number of candidates to train

    Returns:
        Recommended n_jobs value
    """
    import os

    n_cpu = os.cpu_count() or 4

    # Use at least 2 jobs if possible, up to n_cpu
    if n_candidates == 1:
        return 1
    elif n_candidates <= 4:
        return min(n_candidates, n_cpu)
    else:
        return min(n_cpu, max(4, n_candidates // 3))
