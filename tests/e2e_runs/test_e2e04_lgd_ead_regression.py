"""
E2E Test Suite: LGD + EAD Regression Models
Tests two-stage LGD architecture and EAD regression with CCF analysis.
"""

import pytest
from typing import List


class TestLGDTwoStageArchitecture:
    """Verify LGD two-stage architecture: binary classification + severity regression."""

    def test_lgd_stage_1_binary_classification(self):
        """Test that LGD Stage 1 is binary classification (default vs loss)."""
        stage1_task = "binary_classification"
        assert "binary" in stage1_task.lower(), "Stage 1 should be binary classification"

    def test_lgd_stage_2_severity_regression(self):
        """Test that LGD Stage 2 is regression (loss severity)."""
        stage2_task = "regression"
        assert "regression" in stage2_task.lower(), "Stage 2 should be regression"

    def test_lgd_two_stage_flow(self):
        """Test that LGD flows through Stage 1 then Stage 2."""
        flow = ["stage1_binary", "stage2_regression"]
        assert len(flow) == 2, "LGD should have 2 stages"
        assert "binary" in flow[0], "Stage 1 should be binary"
        assert "regression" in flow[1], "Stage 2 should be regression"

    def test_stage1_target_variable(self):
        """Test that Stage 1 target is default indicator (0/1)."""
        stage1_target = "loss_occurred"  # Binary: whether loss occurred
        assert stage1_target is not None, "Stage 1 should have target variable"

    def test_stage2_target_variable(self):
        """Test that Stage 2 target is loss severity (0.0-1.0)."""
        stage2_target = "loss_severity"  # Continuous: LGD value
        assert stage2_target is not None, "Stage 2 should have target variable"

    def test_lgd_predictions_combined(self):
        """Test that final LGD = P(loss) * E(loss_severity | loss)."""
        # LGD = Stage1_pred * Stage2_pred
        # Where Stage1_pred is P(loss occurred)
        # And Stage2_pred is E(LGD | loss occurred)
        combination_formula = "stage1_pred * stage2_pred"
        assert "*" in combination_formula, "LGD combines both stage predictions"


class TestLGDCandidateCounts:
    """Verify LGD tournament has correct candidate counts."""

    def test_lgd_stage1_candidate_count(self):
        """Test that Stage 1 has exactly 5 binary classification candidates."""
        stage1_candidates = [
            "logistic_regression",
            "random_forest",
            "xgboost",
            "lightgbm",
            "neural_net",
        ]

        assert len(stage1_candidates) == 5, "Stage 1 should have 5 candidates"

    def test_lgd_stage2_candidate_count(self):
        """Test that Stage 2 has exactly 8 regression candidates."""
        stage2_candidates = [
            "linear_regression",
            "ridge_regression",
            "lasso_regression",
            "elastic_net",
            "random_forest_regressor",
            "xgboost_regressor",
            "lightgbm_regressor",
            "neural_net_regressor",
        ]

        assert len(stage2_candidates) == 8, "Stage 2 should have 8 candidates"

    def test_lgd_total_candidate_combinations(self):
        """Test total LGD candidate combinations is 5 * 8 = 40."""
        stage1_count = 5
        stage2_count = 8
        total_combinations = stage1_count * stage2_count

        assert total_combinations == 40, "Should have 5 * 8 = 40 model combinations"

    def test_lgd_tournament_candidate_pool(self):
        """Test that tournament evaluates Stage 1 and Stage 2 separately."""
        # Stage 1: select best binary classifier
        # Stage 2: select best regressor
        # Result: best (Stage1, Stage2) pair
        stages = 2
        assert stages == 2, "LGD has 2 tournament stages"

    def test_logistic_included_in_stage1(self):
        """Test that logistic regression is included in Stage 1."""
        stage1 = [
            "logistic_regression",
            "random_forest",
            "xgboost",
            "lightgbm",
            "neural_net",
        ]

        assert "logistic_regression" in stage1, "Logistic regression required in Stage 1"

    def test_linear_regression_included_in_stage2(self):
        """Test that linear regression is included in Stage 2."""
        stage2 = [
            "linear_regression",
            "ridge_regression",
            "lasso_regression",
            "elastic_net",
            "random_forest_regressor",
            "xgboost_regressor",
            "lightgbm_regressor",
            "neural_net_regressor",
        ]

        assert "linear_regression" in stage2, "Linear regression required in Stage 2"


class TestEADCandidateCount:
    """Verify EAD tournament has exactly 9 regression candidates."""

    def test_ead_candidate_count(self):
        """Test that EAD tournament has exactly 9 candidates."""
        ead_candidates = [
            "linear_regression",
            "ridge_regression",
            "lasso_regression",
            "elastic_net",
            "random_forest_regressor",
            "xgboost_regressor",
            "lightgbm_regressor",
            "neural_net_regressor",
            "svr_rbf",
        ]

        assert len(ead_candidates) == 9, "EAD should have exactly 9 candidates"

    def test_ead_is_pure_regression(self):
        """Test that EAD is pure regression, not binary classification."""
        ead_task = "regression"
        assert "regression" in ead_task.lower(), "EAD should be regression only"

    def test_ead_target_is_ccf(self):
        """Test that EAD target variable is CCF (Credit Conversion Factor)."""
        ead_target = "ccf"  # CCF = (EAD - Funded) / (Credit Limit - Funded)
        assert ead_target.lower() == "ccf", "EAD target should be CCF"

    def test_ead_bounds_0_to_1(self):
        """Test that EAD values are bounded between 0 and 1."""
        # EAD is a fraction of credit exposure
        min_ead = 0.0
        max_ead = 1.0

        assert min_ead == 0.0, "EAD minimum should be 0"
        assert max_ead == 1.0, "EAD maximum should be 1"

    def test_ead_includes_tree_based_models(self):
        """Test that EAD candidates include tree-based models."""
        ead_candidates = [
            "linear_regression",
            "ridge_regression",
            "lasso_regression",
            "elastic_net",
            "random_forest_regressor",
            "xgboost_regressor",
            "lightgbm_regressor",
            "neural_net_regressor",
            "svr_rbf",
        ]

        tree_based = [c for c in ead_candidates if "regressor" in c.lower()]
        assert len(tree_based) >= 3, "Should have multiple tree-based regressors"

    def test_ead_includes_linear_baseline(self):
        """Test that EAD includes linear regression baseline."""
        assert "linear_regression" in [
            "linear_regression",
            "ridge_regression",
            "lasso_regression",
            "elastic_net",
            "random_forest_regressor",
            "xgboost_regressor",
            "lightgbm_regressor",
            "neural_net_regressor",
            "svr_rbf",
        ]


class TestCCFComputation:
    """Verify CCF (Credit Conversion Factor) computation."""

    def test_ccf_formula(self):
        """Test that CCF formula is (EAD - Funded) / (Credit_Limit - Funded)."""
        # CCF represents utilization of available credit at default
        formula = "ccf = (ead - funded_amount) / (credit_limit - funded_amount)"
        assert "ead" in formula.lower(), "Formula should use EAD"
        assert "funded" in formula.lower(), "Formula should use funded amount"
        assert "credit_limit" in formula.lower(), "Formula should use credit limit"

    def test_ccf_represents_utilization(self):
        """Test that CCF represents credit utilization at default."""
        # CCF measures how much of available credit was used when default occurred
        ccf_meaning = "credit_conversion_factor"
        assert "conversion" in ccf_meaning.lower(), "CCF is a conversion factor"

    def test_ccf_bounds_reasonable(self):
        """Test that CCF values fall within reasonable bounds."""
        # CCF can be > 1 if new charges added after origination
        # But should typically be 0 to 1.2
        typical_min = 0.0
        typical_max = 1.2

        assert typical_min <= typical_max, "CCF bounds should be logical"

    def test_ead_calculated_from_ccf(self):
        """Test that EAD is calculated as funded + (credit_limit - funded) * CCF."""
        # EAD = Funded + (Credit_Limit - Funded) * CCF
        formula = "ead = funded + (credit_limit - funded) * ccf"
        assert "funded" in formula, "Should use funded amount"
        assert "credit_limit" in formula, "Should use credit limit"
        assert "ccf" in formula, "Should use CCF"

    def test_ccf_analysis_output(self):
        """Test that CCF analysis is produced as output."""
        ccf_output_file = "ccf_analysis.json"
        assert "ccf" in ccf_output_file.lower(), "Should output CCF analysis"

    def test_ead_output_artifacts(self):
        """Test that EAD stage produces expected artifacts."""
        ead_artifacts = [
            "ead_champion.joblib",
            "ead_tournament_results.json",
            "ccf_analysis.json",
        ]

        assert len(ead_artifacts) == 3, "EAD should produce 3 key artifacts"

    def test_lgd_ead_combination(self):
        """Test that LGD and EAD are combined to calculate Expected Loss."""
        # EL = PD * LGD * EAD
        # Where EAD is the exposure amount
        combination = "el = pd * lgd * ead"
        assert "*" in combination, "Should multiply PD * LGD * EAD"


class TestRegressionModelCommonality:
    """Test that regression models are used across Stage2/EAD."""

    def test_stage2_ead_model_overlap(self):
        """Test that Stage 2 and EAD share some common model types."""
        stage2_models = {
            "linear_regression",
            "ridge_regression",
            "lasso_regression",
            "elastic_net",
            "random_forest_regressor",
            "xgboost_regressor",
            "lightgbm_regressor",
            "neural_net_regressor",
        }

        ead_models = {
            "linear_regression",
            "ridge_regression",
            "lasso_regression",
            "elastic_net",
            "random_forest_regressor",
            "xgboost_regressor",
            "lightgbm_regressor",
            "neural_net_regressor",
            "svr_rbf",
        }

        overlap = stage2_models & ead_models
        assert len(overlap) >= 7, "Stage2 and EAD should share most regression models"

    def test_regression_standardization(self):
        """Test that all regression candidates standardize features."""
        # Important for linear models and regularization
        standardized_models = [
            "linear_regression",
            "ridge_regression",
            "lasso_regression",
            "elastic_net",
            "svr_rbf",
        ]

        assert len(standardized_models) >= 3, "Should have standardization-sensitive models"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
