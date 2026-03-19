"""
Scoring Mode Configuration

Manages model selection rubric differences between regulatory and performance modes.
Regulatory mode prioritizes interpretability and regulatory compliance.
Performance mode prioritizes predictive accuracy and discriminatory power.
"""

from dataclasses import dataclass, field
from typing import Dict, Literal


@dataclass
class ScoringModeConfig:
    """
    Configuration for model selection scoring rubric.

    Defines relative weights for different evaluation metrics depending on
    whether the model is being selected for regulatory compliance or performance.
    """

    mode: Literal["regulatory", "performance"] = "regulatory"
    """Scoring mode: 'regulatory' or 'performance' (default: 'regulatory')"""

    regulatory_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "auc": 0.15,
            "accuracy": 0.10,
            "interpretability": 0.25,
            "stability": 0.15,
            "regulatory_compliance": 0.20,
            "gini": 0.10,
            "psi": 0.05,
        }
    )
    """
    Weights for regulatory mode metrics (must sum to 1.0).
    Emphasizes interpretability, stability, and compliance over raw performance.
    """

    performance_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "auc": 0.30,
            "accuracy": 0.20,
            "precision": 0.15,
            "recall": 0.15,
            "f1_score": 0.10,
            "interpretability": 0.05,
            "stability": 0.05,
        }
    )
    """
    Weights for performance mode metrics (must sum to 1.0).
    Emphasizes discriminatory power and accuracy over interpretability.
    """

    def get_weights(self) -> Dict[str, float]:
        """
        Get the active weights for the current mode.

        Returns:
            Dictionary of {metric: weight} for current mode
        """
        if self.mode == "regulatory":
            return self.regulatory_weights.copy()
        else:
            return self.performance_weights.copy()

    def set_mode(self, mode: Literal["regulatory", "performance"]) -> None:
        """
        Set the scoring mode.

        Args:
            mode: 'regulatory' or 'performance'

        Raises:
            ValueError: If mode is not valid
        """
        if mode not in ("regulatory", "performance"):
            raise ValueError(f"Invalid mode: {mode}. Must be 'regulatory' or 'performance'")
        self.mode = mode

    def validate_weights(self) -> bool:
        """
        Validate that weights are properly configured.

        Returns:
            True if weights are valid

        Raises:
            ValueError: If weights don't sum to 1.0 or contain invalid values
        """
        for mode_name, weights in [
            ("regulatory", self.regulatory_weights),
            ("performance", self.performance_weights),
        ]:
            total = sum(weights.values())
            if abs(total - 1.0) > 1e-6:
                raise ValueError(
                    f"{mode_name} weights sum to {total:.6f}, must be 1.0"
                )

            for metric, weight in weights.items():
                if weight < 0 or weight > 1:
                    raise ValueError(
                        f"{mode_name} weight for {metric} is {weight}, must be 0-1"
                    )

        return True


def get_rubric_weights(
    mode: Literal["regulatory", "performance"] = "regulatory"
) -> Dict[str, float]:
    """
    Get model selection rubric weights for a scoring mode.

    Args:
        mode: 'regulatory' or 'performance' (default: 'regulatory')

    Returns:
        Dictionary of {metric_name: weight} with weights summing to 1.0

    Raises:
        ValueError: If mode is invalid

    Example:
        >>> regulatory_weights = get_rubric_weights("regulatory")
        >>> print(regulatory_weights["interpretability"])  # 0.25
        >>> print(sum(regulatory_weights.values()))  # 1.0

        >>> performance_weights = get_rubric_weights("performance")
        >>> print(performance_weights["auc"])  # 0.30
    """
    if mode == "regulatory":
        return {
            "auc": 0.15,
            "accuracy": 0.10,
            "interpretability": 0.25,
            "stability": 0.15,
            "regulatory_compliance": 0.20,
            "gini": 0.10,
            "psi": 0.05,
        }
    elif mode == "performance":
        return {
            "auc": 0.30,
            "accuracy": 0.20,
            "precision": 0.15,
            "recall": 0.15,
            "f1_score": 0.10,
            "interpretability": 0.05,
            "stability": 0.05,
        }
    else:
        raise ValueError(f"Invalid mode: {mode}. Must be 'regulatory' or 'performance'")


def compute_rubric_score(
    metrics: Dict[str, float],
    mode: Literal["regulatory", "performance"] = "regulatory",
) -> float:
    """
    Compute overall rubric score for a model candidate.

    Calculates weighted sum of individual metrics using mode-appropriate weights.

    Args:
        metrics: Dictionary of {metric_name: value} for a model
        mode: 'regulatory' or 'performance' (default: 'regulatory')

    Returns:
        Weighted rubric score (0.0-1.0)

    Example:
        >>> metrics = {
        ...     "auc": 0.85,
        ...     "accuracy": 0.92,
        ...     "interpretability": 0.75,
        ...     "stability": 0.80,
        ...     "regulatory_compliance": 0.90,
        ...     "gini": 0.70,
        ...     "psi": 0.05,
        ... }
        >>> score = compute_rubric_score(metrics, mode="regulatory")
        >>> print(f"Regulatory score: {score:.3f}")
    """
    weights = get_rubric_weights(mode)

    total_score = 0.0
    weighted_count = 0

    for metric_name, weight in weights.items():
        if metric_name in metrics:
            metric_value = metrics[metric_name]
            # Clamp metric values to 0-1 range
            metric_value = max(0.0, min(1.0, metric_value))
            total_score += weight * metric_value
            weighted_count += weight

    # Normalize by actual weights found
    if weighted_count > 0:
        total_score = total_score / weighted_count

    return total_score


def compare_modes_for_candidate(
    metrics: Dict[str, float],
) -> Dict[str, float]:
    """
    Compare how a candidate model scores in both modes.

    Args:
        metrics: Dictionary of candidate's performance metrics

    Returns:
        Dictionary with keys:
        - 'regulatory_score': Score in regulatory mode
        - 'performance_score': Score in performance mode
        - 'regulatory_advantage': (regulatory_score - performance_score)

    Example:
        >>> metrics = {...}
        >>> scores = compare_modes_for_candidate(metrics)
        >>> if scores["regulatory_advantage"] > 0:
        ...     print("Model better for regulatory deployment")
    """
    regulatory_score = compute_rubric_score(metrics, mode="regulatory")
    performance_score = compute_rubric_score(metrics, mode="performance")

    return {
        "regulatory_score": regulatory_score,
        "performance_score": performance_score,
        "regulatory_advantage": regulatory_score - performance_score,
    }


def select_best_candidate(
    candidates: Dict[str, Dict[str, float]],
    mode: Literal["regulatory", "performance"] = "regulatory",
) -> tuple:
    """
    Select best model candidate using rubric scoring.

    Args:
        candidates: Dictionary of {candidate_name: metrics_dict}
        mode: Scoring mode (default: 'regulatory')

    Returns:
        Tuple of (best_candidate_name, best_score)

    Example:
        >>> candidates = {
        ...     "logistic_regression": {...metrics...},
        ...     "xgboost": {...metrics...},
        ...     "lightgbm": {...metrics...},
        ... }
        >>> champion, score = select_best_candidate(candidates, mode="regulatory")
        >>> print(f"Champion in regulatory mode: {champion} (score: {score:.3f})")
    """
    best_name = None
    best_score = -1.0

    for candidate_name, metrics in candidates.items():
        score = compute_rubric_score(metrics, mode=mode)
        if score > best_score:
            best_score = score
            best_name = candidate_name

    return best_name, best_score


# Pre-computed rubric configurations for common use cases
REGULATORY_RUBRIC = {
    "auc": 0.15,
    "accuracy": 0.10,
    "interpretability": 0.25,
    "stability": 0.15,
    "regulatory_compliance": 0.20,
    "gini": 0.10,
    "psi": 0.05,
}

PERFORMANCE_RUBRIC = {
    "auc": 0.30,
    "accuracy": 0.20,
    "precision": 0.15,
    "recall": 0.15,
    "f1_score": 0.10,
    "interpretability": 0.05,
    "stability": 0.05,
}
