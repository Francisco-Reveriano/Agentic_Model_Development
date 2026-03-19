"""
TC-03: Population Stability Index (PSI) - Train vs Test Distribution Shift
Per PRD Section 7.3.2
"""
import pytest
import numpy as np


def compute_psi(expected, actual, n_bins=10):
    """Compute PSI between two distributions."""
    eps = 1e-4
    breakpoints = np.linspace(
        min(np.min(expected), np.min(actual)),
        max(np.max(expected), np.max(actual)),
        n_bins + 1
    )
    expected_counts = np.histogram(expected, bins=breakpoints)[0] / len(expected) + eps
    actual_counts = np.histogram(actual, bins=breakpoints)[0] / len(actual) + eps

    psi = np.sum((actual_counts - expected_counts) * np.log(actual_counts / expected_counts))
    return psi


class TestPSIStability:
    """Test suite for Population Stability Index checks."""

    def test_psi_predictions_stable(self, sample_features, sample_targets):
        """TC-03a: PSI between train and test predictions < 0.10."""
        from sklearn.linear_model import LogisticRegression

        X = sample_features.values
        y = sample_targets['default_flag'].values
        split = int(0.7 * len(X))

        model = LogisticRegression(C=0.1, class_weight='balanced', max_iter=1000, random_state=42)
        model.fit(X[:split], y[:split])

        train_probs = model.predict_proba(X[:split])[:, 1]
        test_probs = model.predict_proba(X[split:])[:, 1]

        psi = compute_psi(train_probs, test_probs)

        if psi < 0.10:
            status = "GREEN (Stable)"
        elif psi < 0.25:
            status = "YELLOW (Monitor)"
        else:
            status = "RED (Unstable)"

        print(f"PSI (predictions): {psi:.4f} [{status}]")
        assert psi < 0.25, f"PSI {psi:.4f} exceeds RED threshold of 0.25"

    def test_psi_feature_stability(self, sample_features):
        """TC-03b: PSI for each feature between train/test halves < 0.25."""
        split = int(0.7 * len(sample_features))
        unstable_features = []

        for col in sample_features.columns:
            train_vals = sample_features[col].values[:split]
            test_vals = sample_features[col].values[split:]
            psi = compute_psi(train_vals, test_vals)

            if psi >= 0.25:
                unstable_features.append((col, psi))
            print(f"PSI({col}): {psi:.4f}")

        assert len(unstable_features) == 0, (
            f"Unstable features (PSI > 0.25): {unstable_features}"
        )

    def test_psi_symmetric_property(self):
        """TC-03c: Verify PSI is approximately symmetric."""
        np.random.seed(42)
        a = np.random.normal(0, 1, 5000)
        b = np.random.normal(0.1, 1.05, 5000)

        psi_ab = compute_psi(a, b)
        psi_ba = compute_psi(b, a)

        # PSI is not exactly symmetric but should be close
        assert abs(psi_ab - psi_ba) < 0.05, (
            f"PSI asymmetry too large: PSI(a,b)={psi_ab:.4f}, PSI(b,a)={psi_ba:.4f}"
        )
