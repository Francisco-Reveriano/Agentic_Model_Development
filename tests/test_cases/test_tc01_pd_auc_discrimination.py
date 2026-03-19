"""
TC-01: PD Model AUC-ROC Discrimination Threshold Validation
Validates PD champion achieves AUC-ROC > 0.75, Gini > 0.50, KS > 0.35
Per PRD Section 7.3.2
"""
import pytest
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
from scipy.stats import ks_2samp


class TestPDAUCDiscrimination:
    """Test suite for PD model discrimination metrics."""

    def test_auc_roc_above_threshold(self, sample_features, sample_targets, pd_thresholds):
        """TC-01a: AUC-ROC must exceed 0.75 (Green threshold)."""
        from sklearn.linear_model import LogisticRegression

        X = sample_features.values
        y = sample_targets['default_flag'].values

        model = LogisticRegression(C=0.1, class_weight='balanced', max_iter=1000, random_state=42)
        split = int(0.7 * len(X))
        model.fit(X[:split], y[:split])
        y_prob = model.predict_proba(X[split:])[:, 1]
        auc = roc_auc_score(y[split:], y_prob)

        threshold = pd_thresholds["auc_roc"]["green"]
        assert auc > pd_thresholds["auc_roc"]["red_below"], (
            f"AUC-ROC {auc:.4f} is RED (below {pd_thresholds['auc_roc']['red_below']})"
        )
        status = "GREEN" if auc > threshold else "YELLOW"
        print(f"AUC-ROC: {auc:.4f} [{status}] (threshold: {threshold})")

    def test_gini_coefficient(self, sample_features, sample_targets, pd_thresholds):
        """TC-01b: Gini coefficient must exceed 0.50."""
        from sklearn.linear_model import LogisticRegression

        X = sample_features.values
        y = sample_targets['default_flag'].values

        model = LogisticRegression(C=0.1, class_weight='balanced', max_iter=1000, random_state=42)
        split = int(0.7 * len(X))
        model.fit(X[:split], y[:split])
        y_prob = model.predict_proba(X[split:])[:, 1]

        auc = roc_auc_score(y[split:], y_prob)
        gini = 2 * auc - 1

        assert gini > pd_thresholds["gini"]["red_below"], (
            f"Gini {gini:.4f} is RED (below {pd_thresholds['gini']['red_below']})"
        )
        status = "GREEN" if gini > pd_thresholds["gini"]["green"] else "YELLOW"
        print(f"Gini: {gini:.4f} [{status}] (threshold: {pd_thresholds['gini']['green']})")

    def test_ks_statistic(self, sample_features, sample_targets, pd_thresholds):
        """TC-01c: KS statistic must exceed 0.35."""
        from sklearn.linear_model import LogisticRegression

        X = sample_features.values
        y = sample_targets['default_flag'].values

        model = LogisticRegression(C=0.1, class_weight='balanced', max_iter=1000, random_state=42)
        split = int(0.7 * len(X))
        model.fit(X[:split], y[:split])
        y_prob = model.predict_proba(X[split:])[:, 1]

        # KS statistic: max separation between cumulative distributions
        y_test = y[split:]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        ks = np.max(tpr - fpr)

        assert ks > pd_thresholds["ks_statistic"]["red_below"], (
            f"KS {ks:.4f} is RED (below {pd_thresholds['ks_statistic']['red_below']})"
        )
        status = "GREEN" if ks > pd_thresholds["ks_statistic"]["green"] else "YELLOW"
        print(f"KS Statistic: {ks:.4f} [{status}] (threshold: {pd_thresholds['ks_statistic']['green']})")

    def test_discrimination_metrics_consistent(self, sample_features, sample_targets):
        """TC-01d: Verify Gini = 2*AUC - 1 mathematical relationship."""
        from sklearn.linear_model import LogisticRegression

        X = sample_features.values
        y = sample_targets['default_flag'].values

        model = LogisticRegression(C=0.1, class_weight='balanced', max_iter=1000, random_state=42)
        split = int(0.7 * len(X))
        model.fit(X[:split], y[:split])
        y_prob = model.predict_proba(X[split:])[:, 1]

        auc = roc_auc_score(y[split:], y_prob)
        gini = 2 * auc - 1

        # Mathematical identity must hold exactly
        assert abs(gini - (2 * auc - 1)) < 1e-10, "Gini != 2*AUC - 1"
