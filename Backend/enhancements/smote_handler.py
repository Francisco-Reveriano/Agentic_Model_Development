"""
SMOTE Class Imbalance Handler

Detects and corrects class imbalance using Synthetic Minority Over-sampling Technique (SMOTE).
Automatically applies SMOTE when minority class representation falls below threshold.
"""

from dataclasses import dataclass
from typing import Tuple, Optional

import numpy as np


@dataclass
class SMOTEConfig:
    """
    Configuration for SMOTE class imbalance handling.

    Controls when and how SMOTE is applied to training data.
    """

    enabled: bool = False
    """Enable SMOTE application (default: False - disabled by default)"""

    minority_threshold: float = 0.05
    """
    Minimum minority class ratio to trigger SMOTE.
    If minority_class_ratio < threshold, SMOTE is applied.
    Default: 0.05 (5%)
    """

    minority_threshold_max: float = 0.40
    """
    Maximum minority class ratio. If above this, don't apply SMOTE.
    Default: 0.40 (40%) - only apply if severe imbalance
    """

    k_neighbors: int = 5
    """Number of nearest neighbors for SMOTE (default: 5)"""

    random_state: int = 42
    """Random seed for reproducibility (default: 42)"""

    sampling_strategy: str = "auto"
    """
    Resampling strategy:
    - 'auto': Balance to 1:1 ratio
    - float (0.0-1.0): Ratio of minority to majority samples
    Default: 'auto'
    """

    verbose: bool = True
    """Print SMOTE application details (default: True)"""


def apply_smote_if_needed(
    X_train: np.ndarray,
    y_train: np.ndarray,
    minority_threshold: float = 0.05,
    enabled: bool = False,
    config: Optional[SMOTEConfig] = None,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Apply SMOTE to training data if class imbalance is detected.

    Analyzes target distribution and applies SMOTE only if minority class
    falls below the specified threshold. Safe to call even if no imbalance
    exists.

    Args:
        X_train: Training feature matrix (n_samples, n_features)
        y_train: Training target vector (n_samples,)
        minority_threshold: Min minority class ratio to trigger SMOTE (default: 0.05)
        enabled: Whether SMOTE is enabled (default: False)
        config: SMOTEConfig instance (optional, overrides other params)

    Returns:
        Tuple of (X_resampled, y_resampled, info_dict) where:
        - X_resampled: Features after optional SMOTE
        - y_resampled: Targets after optional SMOTE
        - info_dict: Dictionary with keys:
            - 'smote_applied': bool
            - 'original_class_dist': dict of class -> count
            - 'resampled_class_dist': dict of class -> count
            - 'imbalance_ratio': float (minority/majority)

    Example:
        >>> X_balanced, y_balanced, info = apply_smote_if_needed(
        ...     X_train, y_train, minority_threshold=0.10, enabled=True
        ... )
        >>> print(f"SMOTE applied: {info['smote_applied']}")
        >>> print(f"Imbalance ratio: {info['imbalance_ratio']:.3f}")
    """
    if config is None:
        config = SMOTEConfig(
            enabled=enabled,
            minority_threshold=minority_threshold,
        )

    info = {
        "smote_applied": False,
        "original_class_dist": {},
        "resampled_class_dist": {},
        "imbalance_ratio": None,
    }

    # Analyze original class distribution
    unique_classes, counts = np.unique(y_train, return_counts=True)
    info["original_class_dist"] = dict(zip(unique_classes.tolist(), counts.tolist()))

    if not config.enabled:
        info["reason"] = "SMOTE disabled in config"
        return X_train, y_train, info

    # Check if class imbalance exists
    if len(unique_classes) < 2:
        info["reason"] = "Only one class in training data"
        return X_train, y_train, info

    # Calculate imbalance ratio
    minority_count = counts.min()
    majority_count = counts.max()
    minority_ratio = minority_count / (minority_count + majority_count)
    info["imbalance_ratio"] = minority_ratio

    if config.verbose:
        print(f"Original class distribution: {info['original_class_dist']}")
        print(f"Minority ratio: {minority_ratio:.3f}")

    # Check if SMOTE should be applied
    if minority_ratio >= config.minority_threshold_max:
        if config.verbose:
            print(f"Minority ratio {minority_ratio:.3f} >= threshold {config.minority_threshold_max:.3f}")
            print("Skipping SMOTE (classes not severely imbalanced)")
        return X_train, y_train, info

    if minority_ratio < config.minority_threshold:
        if config.verbose:
            print(f"Minority ratio {minority_ratio:.3f} < threshold {config.minority_threshold:.3f}")
            print("Applying SMOTE to balance classes...")

        try:
            from imblearn.over_sampling import SMOTE

            smote = SMOTE(
                k_neighbors=config.k_neighbors,
                random_state=config.random_state,
                sampling_strategy=config.sampling_strategy,
            )

            X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

            # Calculate resampled distribution
            unique_resampled, counts_resampled = np.unique(
                y_resampled, return_counts=True
            )
            info["resampled_class_dist"] = dict(
                zip(unique_resampled.tolist(), counts_resampled.tolist())
            )

            info["smote_applied"] = True
            info["reason"] = "Class imbalance detected and corrected"

            if config.verbose:
                print(f"After SMOTE: {info['resampled_class_dist']}")
                print(f"Samples added: {len(y_resampled) - len(y_train)}")

            return X_resampled, y_resampled, info

        except ImportError:
            info["reason"] = "SMOTE not available (imbalanced-learn not installed)"
            if config.verbose:
                print("Warning: SMOTE requested but imbalanced-learn not installed")
            return X_train, y_train, info

    else:
        info["reason"] = f"Minority ratio {minority_ratio:.3f} acceptable"
        return X_train, y_train, info


def detect_class_imbalance(
    y_train: np.ndarray,
    threshold: float = 0.05,
) -> bool:
    """
    Detect if training data has significant class imbalance.

    Args:
        y_train: Training target vector
        threshold: Minority class ratio threshold (default: 0.05)

    Returns:
        True if minority class ratio < threshold
    """
    unique_classes, counts = np.unique(y_train, return_counts=True)

    if len(unique_classes) < 2:
        return False

    minority_count = counts.min()
    total_count = counts.sum()
    minority_ratio = minority_count / total_count

    return minority_ratio < threshold


def calculate_smote_samples_needed(
    y_train: np.ndarray,
    target_ratio: float = 0.5,
) -> int:
    """
    Calculate number of synthetic samples SMOTE would generate.

    Args:
        y_train: Training target vector
        target_ratio: Target minority/majority ratio (default: 0.5)

    Returns:
        Number of synthetic samples to generate
    """
    unique_classes, counts = np.unique(y_train, return_counts=True)

    if len(unique_classes) < 2:
        return 0

    minority_count = counts.min()
    majority_count = counts.max()

    target_minority = int(majority_count * target_ratio)
    samples_needed = max(0, target_minority - minority_count)

    return samples_needed


def get_class_weights(y_train: np.ndarray) -> dict:
    """
    Calculate class weights for imbalanced classification.

    Alternative to SMOTE: use class weights in model training.

    Args:
        y_train: Training target vector

    Returns:
        Dictionary mapping class -> weight (higher for minority)

    Example:
        >>> weights = get_class_weights(y_train)
        >>> model = LogisticRegression(class_weight=weights)
    """
    unique_classes, counts = np.unique(y_train, return_counts=True)
    total = counts.sum()

    weights = {}
    for cls, count in zip(unique_classes, counts):
        # Weight inversely proportional to class frequency
        weight = total / (len(unique_classes) * count)
        weights[int(cls)] = weight

    return weights
