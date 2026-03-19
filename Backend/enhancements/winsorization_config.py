"""
Feature Winsorization Configuration

Implements robust outlier handling via winsorization. Limits extreme values
to specified percentiles while preserving distribution shape.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd


@dataclass
class WinsorizeConfig:
    """
    Configuration for per-feature winsorization thresholds.

    Winsorization replaces extreme values with the specified percentile values.
    Conservative defaults appropriate for credit risk modeling per Basel II/III.
    """

    enabled: bool = True
    """Enable winsorization (default: True)"""

    method: str = "percentile"
    """
    Winsorization method:
    - 'percentile': Use [lower_pct, upper_pct] (default)
    - 'std': Use mean +/- std_multiplier * std
    - 'iqr': Use IQR-based bounds
    """

    lower_percentile: float = 1.0
    """Lower percentile threshold (default: 1.0, clip bottom 1%)"""

    upper_percentile: float = 99.0
    """Upper percentile threshold (default: 99.0, clip top 1%)"""

    std_multiplier: float = 3.0
    """
    For 'std' method: multiply standard deviation.
    Default: 3.0 (±3 sigma covers ~99.7% of normal distribution)
    """

    iqr_multiplier: float = 1.5
    """
    For 'iqr' method: multiplier for interquartile range.
    Default: 1.5 (standard boxplot definition)
    """

    per_feature_thresholds: Dict[str, Tuple[float, float]] = field(
        default_factory=dict
    )
    """
    Optional per-feature thresholds override defaults.
    Keys are feature names, values are (lower_bound, upper_bound) tuples.
    """

    verbose: bool = True
    """Print winsorization details (default: True)"""

    def get_threshold(self, feature_name: str) -> Optional[Tuple[float, float]]:
        """
        Get winsorization threshold for a feature.

        Args:
            feature_name: Name of feature

        Returns:
            Tuple of (lower_bound, upper_bound) or None if not configured
        """
        return self.per_feature_thresholds.get(feature_name, None)

    def set_threshold(
        self, feature_name: str, lower: float, upper: float
    ) -> None:
        """
        Set custom winsorization threshold for a feature.

        Args:
            feature_name: Name of feature
            lower: Lower bound
            upper: Upper bound
        """
        self.per_feature_thresholds[feature_name] = (lower, upper)


def apply_winsorization(
    df: pd.DataFrame,
    config: Optional[WinsorizeConfig] = None,
    columns: Optional[list] = None,
) -> Tuple[pd.DataFrame, dict]:
    """
    Apply winsorization to features in a dataframe.

    Clips extreme values to specified percentiles while preserving data shape.
    Particularly useful for financial data with occasional extreme outliers.

    Args:
        df: Input dataframe
        config: WinsorizeConfig instance (optional)
        columns: Specific columns to winsorize (default: all numeric)

    Returns:
        Tuple of (winsorized_df, info_dict) where:
        - winsorized_df: DataFrame with winsorized values
        - info_dict: Dictionary with:
            - 'values_clipped': Total number of clipped values
            - 'per_column': Dict of {col: count} clipped per column
            - 'clipping_rates': Dict of {col: percentage} clipped

    Example:
        >>> df = pd.DataFrame({'amount': [1, 2, 999999], 'rate': [0.5, 0.75, 0.8]})
        >>> config = WinsorizeConfig(lower_percentile=5, upper_percentile=95)
        >>> df_win, info = apply_winsorization(df, config)
        >>> print(f"Values clipped: {info['values_clipped']}")
    """
    if config is None:
        config = WinsorizeConfig()

    if not config.enabled:
        return df.copy(), {"enabled": False}

    df_winsorized = df.copy()

    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()

    info = {
        "values_clipped": 0,
        "per_column": {},
        "clipping_rates": {},
        "thresholds": {},
    }

    for col in columns:
        if col not in df.columns:
            continue

        col_data = df[col].dropna()
        if len(col_data) == 0:
            continue

        # Determine thresholds
        custom_threshold = config.get_threshold(col)

        if custom_threshold is not None:
            lower_bound, upper_bound = custom_threshold
        elif config.method == "percentile":
            lower_bound = col_data.quantile(config.lower_percentile / 100)
            upper_bound = col_data.quantile(config.upper_percentile / 100)
        elif config.method == "std":
            mean = col_data.mean()
            std = col_data.std()
            lower_bound = mean - config.std_multiplier * std
            upper_bound = mean + config.std_multiplier * std
        elif config.method == "iqr":
            q1 = col_data.quantile(0.25)
            q3 = col_data.quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - config.iqr_multiplier * iqr
            upper_bound = q3 + config.iqr_multiplier * iqr
        else:
            lower_bound = col_data.min()
            upper_bound = col_data.max()

        # Count clipped values before clipping
        clipped_count = (
            (df_winsorized[col] < lower_bound).sum()
            + (df_winsorized[col] > upper_bound).sum()
        )

        # Apply winsorization
        df_winsorized[col] = df_winsorized[col].clip(lower=lower_bound, upper=upper_bound)

        info["values_clipped"] += clipped_count
        info["per_column"][col] = clipped_count
        info["clipping_rates"][col] = (clipped_count / len(df[col])) * 100
        info["thresholds"][col] = {"lower": float(lower_bound), "upper": float(upper_bound)}

        if config.verbose and clipped_count > 0:
            clip_rate = info["clipping_rates"][col]
            print(
                f"Column '{col}': clipped {clipped_count} values ({clip_rate:.2f}%) "
                f"to [{lower_bound:.2f}, {upper_bound:.2f}]"
            )

    return df_winsorized, info


def create_credit_risk_winsorize_config() -> WinsorizeConfig:
    """
    Create pre-configured winsorization for typical credit risk features.

    Returns:
        WinsorizeConfig with conservative thresholds suitable for credit modeling
    """
    config = WinsorizeConfig(
        enabled=True,
        method="percentile",
        lower_percentile=1.0,
        upper_percentile=99.0,
    )

    # Typical credit risk feature thresholds
    # Loan amount: 1st to 99th percentile
    config.set_threshold("funded_amnt", 1000, 40000)

    # Annual income: 1st to 99th percentile
    config.set_threshold("annual_inc", 10000, 200000)

    # Debt-to-income: typically 0-50 (allows some extreme cases)
    config.set_threshold("dti", 0.0, 50.0)

    # Interest rate: typically 5-30%
    config.set_threshold("int_rate", 0.05, 0.30)

    # FICO score: typically 300-850
    config.set_threshold("fico_range_low", 300, 850)
    config.set_threshold("fico_range_high", 300, 850)

    # Loan term: 36 or 60 months
    config.set_threshold("term_months", 36, 60)

    # Months since last delinquency: 0 to 600+ months
    config.set_threshold("mths_since_last_delinq", 0, 600)

    return config


def validate_winsorization(
    original_df: pd.DataFrame,
    winsorized_df: pd.DataFrame,
    info: dict,
) -> bool:
    """
    Validate that winsorization was applied correctly.

    Args:
        original_df: Original dataframe before winsorization
        winsorized_df: Winsorized dataframe
        info: Info dict from apply_winsorization

    Returns:
        True if validation passes

    Raises:
        ValueError: If validation fails
    """
    if original_df.shape[0] != winsorized_df.shape[0]:
        raise ValueError("Winsorization changed number of rows")

    if original_df.shape[1] != winsorized_df.shape[1]:
        raise ValueError("Winsorization changed number of columns")

    # Check that non-extreme values were not changed
    for col in original_df.select_dtypes(include=[np.number]).columns:
        if col not in info["thresholds"]:
            continue

        thresholds = info["thresholds"][col]
        lower = thresholds["lower"]
        upper = thresholds["upper"]

        # Get values within bounds
        mask = (original_df[col] >= lower) & (original_df[col] <= upper)

        # These values should be unchanged
        original_within = original_df.loc[mask, col]
        winsorized_within = winsorized_df.loc[mask, col]

        if not original_within.equals(winsorized_within):
            raise ValueError(f"Winsorization modified values within bounds in column '{col}'")

    return True


def estimate_outliers(
    df: pd.DataFrame,
    lower_pct: float = 1.0,
    upper_pct: float = 99.0,
) -> Dict[str, int]:
    """
    Estimate number of outliers using percentile method.

    Args:
        df: Input dataframe
        lower_pct: Lower percentile (default: 1.0)
        upper_pct: Upper percentile (default: 99.0)

    Returns:
        Dictionary mapping column -> outlier count
    """
    outliers = {}

    for col in df.select_dtypes(include=[np.number]).columns:
        col_data = df[col].dropna()

        lower_bound = col_data.quantile(lower_pct / 100)
        upper_bound = col_data.quantile(upper_pct / 100)

        outlier_count = (
            (col_data < lower_bound).sum() + (col_data > upper_bound).sum()
        )

        outliers[col] = outlier_count

    return outliers
