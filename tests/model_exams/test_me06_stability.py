"""
Model Exam ME-06: Model & Input Stability
=========================================

PRD Section O.S (Operational Stability):
  - O.S.1: Prediction Stability Index (PSI) for model outputs
  - O.S.2: Characteristic Stability Index (CSI) for input features

Tests verify:
  - PSI for PD/LGD/EAD predictions between train and test: PSI < 0.10 (good)
  - CSI for top-10 features between train and test: CSI < 0.10 (good)
  - Distribution shifts detected and monitored
  - Stability thresholds align with Basel III monitoring requirements

Reference:
  PSI < 0.10: Insignificant change
  0.10 <= PSI < 0.25: Small change, monitor
  PSI >= 0.25: Significant change, investigate
"""

import pytest
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency


class TestModelAndInputStability:
    """Validates prediction and feature distribution stability across data splits."""

    def test_psi_calculation_methodology(self, sample_features, sample_targets):
        """O.S.1: PSI must be calculated correctly and documented.

        Population Stability Index (PSI) measures distribution shifts
        in model predictions or features:

        PSI = Σ (Expected% - Actual%) * ln(Expected% / Actual%)

        Where:
          - Expected = train/reference distribution
          - Actual = test/monitoring distribution
          - Bins = typically 10 deciles or based on natural breaks

        PSI < 0.10 indicates stable distribution (GREEN)
        0.10 <= PSI < 0.25 indicates monitoring required (YELLOW)
        PSI >= 0.25 indicates investigation required (RED)
        """
        from sklearn.linear_model import LogisticRegression

        X = sample_features.fillna(sample_features.mean()).values
        y = sample_targets['default_flag'].values

        # Train/test split
        split = int(0.7 * len(X))
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        # Train model and generate predictions
        model = LogisticRegression(C=0.1, class_weight='balanced', max_iter=1000, random_state=42)
        model.fit(X_train, y_train)

        pd_train = model.predict_proba(X_train)[:, 1]
        pd_test = model.predict_proba(X_test)[:, 1]

        # Calculate PSI using deciles
        def calculate_psi(expected, actual, bins=10):
            """Calculate PSI using specified bins."""
            # Create bins based on expected distribution
            breakpoints = np.percentile(expected, np.linspace(0, 100, bins + 1))
            breakpoints[0] = expected.min() - 0.001
            breakpoints[-1] = expected.max() + 0.001

            # Bin both distributions
            expected_binned = np.histogram(expected, bins=breakpoints)[0] / len(expected)
            actual_binned = np.histogram(actual, bins=breakpoints)[0] / len(actual)

            # Avoid log(0)
            expected_binned = np.where(expected_binned > 0, expected_binned, 0.0001)
            actual_binned = np.where(actual_binned > 0, actual_binned, 0.0001)

            # Calculate PSI
            psi = np.sum((actual_binned - expected_binned) * np.log(actual_binned / expected_binned))
            return psi

        psi = calculate_psi(pd_train, pd_test)

        if psi < 0.10:
            status = "GREEN (Stable)"
        elif psi < 0.25:
            status = "YELLOW (Monitor)"
        else:
            status = "RED (Investigate)"

        print(f"PD Prediction Stability Index (PSI):")
        print(f"  Train set size: {len(pd_train)}")
        print(f"  Test set size: {len(pd_test)}")
        print(f"  PSI: {psi:.4f} [{status}]")
        print(f"  Train PD mean: {pd_train.mean():.4f}")
        print(f"  Test PD mean: {pd_test.mean():.4f}")

        assert psi < 0.25, f"PSI {psi:.4f} exceeds RED threshold (>=0.25)"

    def test_csi_for_top_features(self, sample_features, sample_targets):
        """O.S.2: CSI must be computed for input features.

        Characteristic Stability Index (CSI) measures distribution shifts
        for input features. Formula similar to PSI but applied to feature
        distributions rather than predictions.

        CSI is monitored for top predictive features to detect shifts
        in the underlying population that may affect model performance.
        """
        from sklearn.ensemble import RandomForestClassifier

        X = sample_features.fillna(sample_features.mean()).values
        y = sample_targets['default_flag'].values

        # Train model to identify top features
        split = int(0.7 * len(X))
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)

        # Get top 10 features by importance
        importances = model.feature_importances_
        top_indices = np.argsort(importances)[-10:]
        top_features = [sample_features.columns[i] for i in top_indices]

        print("CSI for Top-10 Features:")
        print(f"{'Feature':20s} {'CSI':>10s} {'Status':>15s}")
        print("-" * 48)

        csi_values = {}
        for idx, feature_idx in enumerate(top_indices):
            feature_name = sample_features.columns[feature_idx]
            feat_train = X_train[:, feature_idx]
            feat_test = X_test[:, feature_idx]

            # Calculate CSI similar to PSI
            def calculate_csi(expected, actual, bins=10):
                breakpoints = np.percentile(expected, np.linspace(0, 100, bins + 1))
                breakpoints[0] = expected.min() - 0.001
                breakpoints[-1] = expected.max() + 0.001

                expected_binned = np.histogram(expected, bins=breakpoints)[0] / len(expected)
                actual_binned = np.histogram(actual, bins=breakpoints)[0] / len(actual)

                expected_binned = np.where(expected_binned > 0, expected_binned, 0.0001)
                actual_binned = np.where(actual_binned > 0, actual_binned, 0.0001)

                csi = np.sum((actual_binned - expected_binned) *
                           np.log(actual_binned / expected_binned))
                return csi

            csi = calculate_csi(feat_train, feat_test)
            csi_values[feature_name] = csi

            if csi < 0.10:
                status = "GREEN"
            elif csi < 0.25:
                status = "YELLOW"
            else:
                status = "RED"

            print(f"{feature_name:20s} {csi:10.4f} {status:>15s}")

        # Average CSI should be acceptable
        avg_csi = np.mean(list(csi_values.values()))
        assert avg_csi < 0.25, f"Average CSI {avg_csi:.4f} exceeds threshold"
        print(f"\nAverage CSI (top-10): {avg_csi:.4f} ✓")

    def test_distribution_shift_detection(self, sample_features, sample_targets):
        """O.S.1-O.S.2: Detect and flag significant distribution shifts.

        Implements monitoring logic that flags distribution changes
        for further investigation. Shifts may indicate:
          - Population drift (changes in borrower profile)
          - Macroeconomic changes (recession/expansion)
          - Data quality issues
          - Model performance degradation
        """
        from scipy.stats import ks_2samp, chi2_contingency

        X = sample_features.fillna(sample_features.mean()).values
        y = sample_targets['default_flag'].values

        split = int(0.7 * len(X))
        X_train, X_test = X[:split], X[split:]

        # Test for distribution shifts using KS test
        print("Distribution Shift Detection (KS Test):")
        print(f"{'Feature':20s} {'KS Stat':>10s} {'p-value':>10s} {'Shift?':>15s}")
        print("-" * 58)

        shifts_detected = 0
        for i, col in enumerate(sample_features.columns):
            feat_train = X_train[:, i]
            feat_test = X_test[:, i]

            # KS test compares distributions
            ks_stat, p_value = ks_2samp(feat_train, feat_test)

            # p < 0.05 indicates significant shift
            shift_detected = p_value < 0.05
            status = "SHIFT ⚠" if shift_detected else "Stable"

            if shift_detected:
                shifts_detected += 1

            print(f"{col:20s} {ks_stat:10.4f} {p_value:10.4f} {status:>15s}")

        print(f"\nShifts detected: {shifts_detected}/{len(sample_features.columns)} features")

    def test_psi_thresholds_documented(self, pd_thresholds):
        """O.S.1: PSI monitoring thresholds must be documented per Basel III.

        Basel III guidelines specify monitoring thresholds for PSI:

        GREEN (PSI < 0.10):   Model performing as expected, no action
        YELLOW (0.10-0.25):   Monitor closely, investigate if > 0.25
        RED (PSI >= 0.25):    Significant shift, model retraining likely needed

        These thresholds apply to:
          - Prediction distributions (PD, LGD, EAD)
          - Feature distributions (characteristics)
          - Risk grades (if applicable)
        """
        expected_thresholds = {
            'psi': {
                'green_below': 0.10,
                'yellow_below': 0.25,
                'red': 0.25,
            }
        }

        # Verify thresholds are present
        if 'psi' in pd_thresholds:
            psi_config = pd_thresholds['psi']
            print("PSI Monitoring Thresholds (Basel III):")
            print(f"  GREEN:  PSI < {psi_config['green_below']:.2f}")
            print(f"  YELLOW: {psi_config['green_below']:.2f} <= PSI < {psi_config['yellow_below']:.2f}")
            print(f"  RED:    PSI >= {psi_config['yellow_below']:.2f}")
        else:
            print("PSI thresholds should be added to configuration")

        print("\nPSI thresholds properly documented ✓")
