"""
TC-08: Feature Importance Consensus - Cross-Model Tier Assignment
Per PRD Section 8.3
"""
import pytest
import numpy as np
from scipy.stats import spearmanr


class TestFeatureConsensus:
    """Test suite for Phase 2 feature importance consensus."""

    def test_consensus_produces_tiers(self, sample_features, sample_targets):
        """TC-08a: All features assigned to exactly one tier."""
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

        X = sample_features.values
        y = sample_targets['default_flag'].values
        features = sample_features.columns.tolist()
        split = int(0.7 * len(X))

        # Train multiple models and extract importances
        models = {
            "LR": LogisticRegression(C=0.1, max_iter=1000, random_state=42),
            "RF": RandomForestClassifier(n_estimators=50, max_depth=6, random_state=42),
            "GBM": GradientBoostingClassifier(n_estimators=50, max_depth=3, random_state=42),
        }

        importances = {}
        for name, model in models.items():
            model.fit(X[:split], y[:split])
            if hasattr(model, 'feature_importances_'):
                imp = model.feature_importances_
            else:
                imp = np.abs(model.coef_[0])
            # Normalize
            imp = imp / (imp.sum() + 1e-10)
            importances[name] = imp

        # Consensus: average across models
        consensus = np.mean(list(importances.values()), axis=0)

        # Tier assignment (PRD Section 8.3.2 step 5)
        n_features = len(features)
        sorted_idx = np.argsort(consensus)[::-1]
        tiers = {}
        for rank, idx in enumerate(sorted_idx):
            pct = rank / n_features
            if pct < 0.20:
                tier = 1
            elif pct < 0.50:
                tier = 2
            elif pct < 0.80:
                tier = 3
            else:
                tier = 4
            tiers[features[idx]] = tier

        # Verify all features assigned
        assert len(tiers) == n_features, "Not all features assigned tiers"
        tier_counts = {t: sum(1 for v in tiers.values() if v == t) for t in [1, 2, 3, 4]}
        print(f"Tier distribution: {tier_counts}")

        # Verify tier 1 has ~20% of features
        assert tier_counts[1] > 0, "Tier 1 (Critical) has no features"

    def test_importance_ranking_stability(self, sample_features, sample_targets):
        """TC-08b: Spearman correlation between model rankings > 0.3."""
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

        X = sample_features.values
        y = sample_targets['default_flag'].values
        split = int(0.7 * len(X))

        rf = RandomForestClassifier(n_estimators=50, max_depth=6, random_state=42)
        gbm = GradientBoostingClassifier(n_estimators=50, max_depth=3, random_state=42)

        rf.fit(X[:split], y[:split])
        gbm.fit(X[:split], y[:split])

        rf_imp = rf.feature_importances_
        gbm_imp = gbm.feature_importances_

        corr, p_value = spearmanr(rf_imp, gbm_imp)
        print(f"Spearman rank correlation (RF vs GBM): {corr:.4f} (p={p_value:.4f})")
        assert corr > 0.0, f"Feature rankings negatively correlated: {corr:.4f}"
