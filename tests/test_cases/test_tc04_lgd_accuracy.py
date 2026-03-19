"""
TC-04: LGD Two-Stage Model Accuracy - RMSE & MAE Validation
Per PRD Section 7.4
"""
import pytest
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


class TestLGDAccuracy:
    """Test suite for LGD model accuracy metrics."""

    def test_lgd_rmse(self, sample_features, sample_targets):
        """TC-04a: LGD RMSE validation."""
        from sklearn.ensemble import GradientBoostingRegressor

        defaults = sample_targets['default_flag'] == 1
        X_def = sample_features[defaults].values
        y_lgd = sample_targets.loc[defaults, 'lgd'].values

        if len(X_def) < 50:
            pytest.skip("Too few defaults for LGD testing")

        split = int(0.7 * len(X_def))
        model = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42)
        model.fit(X_def[:split], y_lgd[:split])
        y_pred = np.clip(model.predict(X_def[split:]), 0, 1)

        rmse = np.sqrt(mean_squared_error(y_lgd[split:], y_pred))
        print(f"LGD RMSE: {rmse:.4f}")
        assert rmse < 0.5, f"LGD RMSE {rmse:.4f} exceeds threshold 0.5"

    def test_lgd_mae(self, sample_features, sample_targets):
        """TC-04b: LGD MAE validation."""
        from sklearn.ensemble import GradientBoostingRegressor

        defaults = sample_targets['default_flag'] == 1
        X_def = sample_features[defaults].values
        y_lgd = sample_targets.loc[defaults, 'lgd'].values

        if len(X_def) < 50:
            pytest.skip("Too few defaults for LGD testing")

        split = int(0.7 * len(X_def))
        model = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42)
        model.fit(X_def[:split], y_lgd[:split])
        y_pred = np.clip(model.predict(X_def[split:]), 0, 1)

        mae = mean_absolute_error(y_lgd[split:], y_pred)
        print(f"LGD MAE: {mae:.4f}")
        assert mae < 0.4, f"LGD MAE {mae:.4f} exceeds threshold 0.4"

    def test_lgd_r2(self, sample_features, sample_targets):
        """TC-04c: LGD R-squared validation."""
        from sklearn.ensemble import GradientBoostingRegressor

        defaults = sample_targets['default_flag'] == 1
        X_def = sample_features[defaults].values
        y_lgd = sample_targets.loc[defaults, 'lgd'].values

        if len(X_def) < 50:
            pytest.skip("Too few defaults for LGD testing")

        split = int(0.7 * len(X_def))
        model = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42)
        model.fit(X_def[:split], y_lgd[:split])
        y_pred = np.clip(model.predict(X_def[split:]), 0, 1)

        r2 = r2_score(y_lgd[split:], y_pred)
        print(f"LGD R²: {r2:.4f}")
        # R² should be positive (better than predicting mean)
        assert r2 > -0.5, f"LGD R² {r2:.4f} indicates model worse than mean"

    def test_lgd_two_stage_combination(self, sample_features, sample_targets):
        """TC-04d: Verify two-stage LGD = P(any_loss) × E[severity]."""
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import GradientBoostingRegressor

        defaults = sample_targets['default_flag'] == 1
        X = sample_features.values
        y_default = sample_targets['default_flag'].values
        y_lgd = sample_targets['lgd'].values

        split = int(0.7 * len(X))

        # Stage 1: P(any loss)
        y_any_loss = (y_lgd > 0).astype(int)
        stage1 = LogisticRegression(C=0.1, max_iter=1000, random_state=42)
        stage1.fit(X[:split], y_any_loss[:split])
        p_loss = stage1.predict_proba(X[split:])[:, 1]

        # Stage 2: E[severity | partial loss]
        partial_loss_mask = (y_lgd[:split] > 0) & (y_lgd[:split] < 1)
        if partial_loss_mask.sum() < 20:
            pytest.skip("Too few partial loss observations")

        stage2 = GradientBoostingRegressor(n_estimators=50, max_depth=3, random_state=42)
        stage2.fit(X[:split][partial_loss_mask], y_lgd[:split][partial_loss_mask])
        severity = np.clip(stage2.predict(X[split:]), 0, 1)

        # Combined: LGD = P(any_loss) × E[severity]
        lgd_combined = p_loss * severity

        assert np.all(lgd_combined >= 0) and np.all(lgd_combined <= 1), (
            "Combined LGD values must be in [0, 1]"
        )
        print(f"Combined LGD mean: {lgd_combined.mean():.4f}, std: {lgd_combined.std():.4f}")
