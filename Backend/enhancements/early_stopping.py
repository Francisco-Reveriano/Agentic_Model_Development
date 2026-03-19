"""
Early Stopping for Tree-Based Models

Implements early stopping callbacks for XGBoost and LightGBM to reduce training time
while maintaining model performance. Monitors validation metrics and stops training
when improvement plateaus.
"""

from dataclasses import dataclass
from typing import Any, Tuple, Optional

import numpy as np


@dataclass
class EarlyStoppingConfig:
    """
    Configuration for early stopping behavior.

    Controls when training stops based on validation metric plateaus.
    """

    enabled: bool = True
    """Enable early stopping (default: True)"""

    rounds: int = 50
    """Number of rounds without improvement before stopping (default: 50)"""

    metric: str = "auc"
    """Metric to monitor: 'auc', 'logloss', 'error', 'rmse' (default: 'auc')"""

    minimize: bool = False
    """Whether to minimize (True) or maximize (False) metric (default: False for AUC)"""

    verbose: bool = True
    """Print early stopping notifications (default: True)"""

    min_delta: float = 0.0001
    """Minimum improvement threshold to reset patience (default: 0.0001)"""

    def get_xgboost_callbacks(self) -> Optional[list]:
        """
        Get XGBoost early stopping callbacks.

        Returns:
            List with XGBoost early stopping callback, or None if not enabled
        """
        if not self.enabled:
            return None

        try:
            import xgboost as xgb

            return [
                xgb.callback.EarlyStopping(
                    rounds=self.rounds,
                    metric_name=self.metric,
                    data_name="validation",
                    save_best=True,
                    maximize=not self.minimize,
                )
            ]
        except ImportError:
            return None

    def get_lightgbm_callbacks(self) -> Optional[list]:
        """
        Get LightGBM early stopping callbacks.

        Returns:
            List with LightGBM early stopping callback, or None if not enabled
        """
        if not self.enabled:
            return None

        try:
            import lightgbm as lgb

            return [
                lgb.early_stopping(
                    stopping_rounds=self.rounds,
                    first_metric_only=True,
                    verbose=self.verbose,
                )
            ]
        except ImportError:
            return None


def apply_early_stopping(
    model: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    rounds: int = 50,
    config: Optional[EarlyStoppingConfig] = None,
) -> Tuple[Any, dict]:
    """
    Apply early stopping to a tree-based model during training.

    Monitors validation performance and stops training when no improvement
    is observed for 'rounds' consecutive iterations.

    Args:
        model: XGBoost or LightGBM model to train
        X_train: Training feature matrix
        y_train: Training target vector
        X_val: Validation feature matrix
        y_val: Validation target vector
        rounds: Number of rounds without improvement (default: 50)
        config: EarlyStoppingConfig instance (optional)

    Returns:
        Tuple of (trained_model, history_dict) containing:
        - trained_model: Fitted model
        - history_dict: Training history with keys like 'train_auc', 'val_auc'

    Example:
        >>> import xgboost as xgb
        >>> model = xgb.XGBClassifier(n_estimators=1000)
        >>> fitted_model, history = apply_early_stopping(
        ...     model, X_train, y_train, X_val, y_val, rounds=50
        ... )
        >>> print(f"Stopped at iteration {len(history['train_auc'])}")
    """
    if config is None:
        config = EarlyStoppingConfig(rounds=rounds)

    if not config.enabled:
        # Train normally without early stopping
        model.fit(X_train, y_train)
        return model, {"message": "Early stopping disabled"}

    model_type = type(model).__name__
    history = {"model_type": model_type, "early_stopping_applied": True}

    # XGBoost handling
    if "XGB" in model_type:
        try:
            import xgboost as xgb

            callbacks = config.get_xgboost_callbacks()

            model.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)],
                callbacks=callbacks or [],
                verbose=False,
            )

            if hasattr(model, "evals_result_"):
                history.update(model.evals_result_)

        except ImportError:
            # Fallback to basic training
            model.fit(X_train, y_train)

    # LightGBM handling
    elif "LGB" in model_type or "Lgb" in model_type:
        try:
            import lightgbm as lgb

            callbacks = config.get_lightgbm_callbacks()

            model.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)],
                callbacks=callbacks or [],
                verbose_eval=0,
            )

            if hasattr(model, "evals_result_"):
                history.update(model.evals_result_)

        except ImportError:
            # Fallback to basic training
            model.fit(X_train, y_train)

    # For other sklearn-compatible models, early stopping not applicable
    else:
        model.fit(X_train, y_train)
        history["message"] = f"Early stopping not applicable to {model_type}"

    return model, history


def should_use_early_stopping(model_class: type) -> bool:
    """
    Check if a model class supports early stopping.

    Args:
        model_class: Model class to check

    Returns:
        True if model supports early stopping, False otherwise
    """
    model_name = model_class.__name__

    early_stopping_models = {
        "XGBClassifier",
        "XGBRegressor",
        "LGBMClassifier",
        "LGBMRegressor",
        "lgbm.LGBMClassifier",
        "lgbm.LGBMRegressor",
    }

    return model_name in early_stopping_models


def estimate_early_stopping_improvement(
    val_metric_history: list, rounds: int = 50
) -> float:
    """
    Estimate the improvement from using early stopping.

    Simulates when early stopping would have triggered and calculates
    the cost in terms of validation metric loss.

    Args:
        val_metric_history: List of validation metric values over time
        rounds: Early stopping patience (number of rounds without improvement)

    Returns:
        Estimated percentage improvement in training speed (0.0-1.0)
    """
    if len(val_metric_history) <= rounds:
        return 0.0

    # Find when early stopping would have triggered
    best_idx = 0
    best_metric = val_metric_history[0]

    for i, metric in enumerate(val_metric_history):
        if metric > best_metric:
            best_metric = metric
            best_idx = i

        if i - best_idx >= rounds:
            # Would have stopped at iteration i
            early_stop_idx = i
            total_iterations = len(val_metric_history)
            improvement = 1.0 - (early_stop_idx / total_iterations)
            return max(0.0, improvement)

    return 0.0
