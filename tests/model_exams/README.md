# Model Exam Validation Scripts

This directory contains 10 comprehensive model exam validation scripts that implement the C1 Standalone Use Case Template test framework from the PRD. These tests validate credit risk modeling platform compliance with Basel III IRB Advanced Approach requirements.

## Overview

- **Total Files**: 10 test modules
- **Total Test Methods**: 58
- **Coverage**: Data appropriateness, quality, features, methodology, parameters, stability, robustness, explainability, business integration, and documentation
- **Framework**: pytest with fixtures from conftest.py

## Test Files

### ME-01: Data Appropriateness & Suitability
**File**: `test_me01_data_appropriateness.py`

PRD Section: I.DA.1-I.DA.5

Tests verify:
- Database exists and is accessible
- Observation period spans ≥5 years
- Default count exceeds 500
- Loan type distribution is diverse
- No survivorship bias in loan status

**Test Methods** (5):
- `test_database_exists_and_accessible()`
- `test_observation_period_span()`
- `test_sufficient_defaults_captured()`
- `test_portfolio_representativeness()`
- `test_no_survivorship_bias()`

### ME-02: Data Quality & Integrity
**File**: `test_me02_data_quality.py`

PRD Section: I.DQ.1-I.DQ.8

Tests verify:
- Missing data rate < 5% in critical columns
- No duplicate loan IDs
- Numeric columns have valid ranges
- Data types are consistent
- Temporal consistency in date fields
- Overall completeness > 95%
- Outlier counts are reasonable

**Test Methods** (8):
- `test_missing_data_rate_acceptable()`
- `test_no_duplicate_loan_ids()`
- `test_numeric_columns_valid_ranges()`
- `test_data_type_consistency()`
- `test_no_negative_categorical_values()`
- `test_temporal_consistency()`
- `test_completeness_threshold()`
- `test_outlier_count_reasonable()`

### ME-03: Feature Engineering & Selection
**File**: `test_me03_feature_engineering.py`

PRD Section: M.FE.1-M.FE.7

Tests verify:
- WoE monotonicity for ordinal features
- No target leakage (single-feature AUC < 0.95)
- Correlation analysis computed
- VIF < 10 for selected features (multicollinearity)
- Information Value calculation and documentation
- Feature stability across data splits

**Test Methods** (6):
- `test_woe_monotonicity_for_ordinal_features()`
- `test_no_target_leakage()`
- `test_correlation_analysis_computed()`
- `test_vif_below_threshold_for_selected_features()`
- `test_information_value_calculation()`
- `test_feature_stability_across_splits()`

### ME-04: Methodology Selection & Validation
**File**: `test_me04_methodology_selection.py`

PRD Section: M.MS.1-M.MS.8

Tests verify:
- Multiple methodologies evaluated (≥12 candidates for PD)
- Clear selection criteria defined
- Champion vs runner-up documented
- Scoring rubric weights sum to 1.0
- statsmodels output required (regulatory compliance)
- Interpretability considerations assessed
- Complexity vs performance trade-offs documented

**Test Methods** (7):
- `test_multiple_methodologies_evaluated()`
- `test_selection_criteria_defined_and_applied()`
- `test_champion_vs_runner_up_documented()`
- `test_scoring_rubric_weights_sum_to_one()`
- `test_statsmodels_output_required()`
- `test_interpretability_considerations()`
- `test_complexity_vs_performance_tradeoff()`

### ME-05: Parameter Estimation & Validation
**File**: `test_me05_parameter_estimation.py`

PRD Section: M.PE.1-M.PE.7

Tests verify:
- Coefficient signs align with economic theory
- Key features statistically significant (p < 0.05)
- Hyperparameter tuning uses systematic approach (RandomizedSearchCV with 5-fold CV)
- Reproducibility with documented seeds
- Convergence criteria specified and logged
- Confidence intervals computed for coefficients
- Regularization parameters justified

**Test Methods** (7):
- `test_coefficient_signs_align_with_economic_theory()`
- `test_key_features_statistically_significant()`
- `test_hyperparameter_tuning_systematic()`
- `test_reproducibility_with_documented_seeds()`
- `test_convergence_criteria_specified()`
- `test_confidence_intervals_computed()`
- `test_regularization_parameters_justified()`

### ME-06: Model & Input Stability
**File**: `test_me06_stability.py`

PRD Section: O.S.1-O.S.2

Tests verify:
- PSI (Prediction Stability Index) for PD/LGD/EAD < 0.10
- CSI (Characteristic Stability Index) for top-10 features < 0.10
- Distribution shift detection using KS test
- PSI monitoring thresholds documented (GREEN < 0.10, YELLOW < 0.25, RED ≥ 0.25)

**Test Methods** (4):
- `test_psi_calculation_methodology()`
- `test_csi_for_top_features()`
- `test_distribution_shift_detection()`
- `test_psi_thresholds_documented()`

### ME-07: Model Robustness & Stress Testing
**File**: `test_me07_robustness.py`

PRD Section: O.R.1

Tests verify:
- EL ordering consistent under stress scenarios (Base ≤ Adverse ≤ Severe)
- Feature sensitivity analysis (AUC degradation < 10% when dropping top feature)
- Noise injection produces reasonable prediction volatility
- Model predictions remain within valid ranges under stress

**Test Methods** (4):
- `test_stress_scenario_analysis()`
- `test_feature_sensitivity_analysis()`
- `test_feature_dropout_auc_degradation()`
- `test_noise_injection_prediction_volatility()`

### ME-08: Model Explainability & Interpretability
**File**: `test_me08_explainability.py`

PRD Section: O.E.1

Tests verify:
- SHAP values satisfy fundamental property: prediction = base_value + Σ SHAP
- Partial Dependency Plots (PDP) monotonic for ordinal features
- Feature importance consistent across methods (Permutation, Coefficient, Tree)
- Coefficient interpretability and documentation

**Test Methods** (4):
- `test_shap_value_sum_property()`
- `test_partial_dependency_monotonicity()`
- `test_feature_importance_consistency()`
- `test_coefficient_interpretability()`

### ME-09: Business Integration & End-to-End Pipeline
**File**: `test_me09_business_integration.py`

PRD Section: B.EE.1-B.EE.2

Tests verify:
- All 7 agents execute in correct sequence
- handoff.json chain documents agent communication
- Portfolio EL within 1-5% range (typical for credit portfolios)
- Grade risk ordering monotonic: A < B < C < D < E < F < G
- Reports generated with correct structure
- Agent error handling and recovery

**Test Methods** (6):
- `test_seven_agent_sequence_execution()`
- `test_handoff_json_chain()`
- `test_portfolio_el_within_acceptable_range()`
- `test_grade_risk_ordering_monotonic()`
- `test_reports_generated_with_correct_structure()`
- `test_agent_error_handling_and_recovery()`

### ME-10: Documentation Completeness & Audit Trail
**File**: `test_me10_documentation.py`

PRD Section: TI.D.1

Tests verify:
- ReportGenerator has all required methods (dq, pd, lgd, ead, el)
- DQ Report has 6 required sections
- C1 Template (PD/LGD/EAD) has 14 required sections
- EL Report has 8 required sections
- All reports reference correct model type
- Docstring completeness (>80% coverage)
- Audit trail metadata complete

**Test Methods** (7):
- `test_report_generator_has_all_methods()`
- `test_dq_report_section_structure()`
- `test_c1_template_section_structure()`
- `test_el_report_section_structure()`
- `test_docstring_completeness()`
- `test_model_type_references_in_reports()`
- `test_audit_trail_metadata()`

## Fixture Usage

All tests use fixtures from `tests/conftest.py`:

- `project_root`: Project root directory
- `output_dir`: Most recent pipeline run output directory
- `settings`: Project settings from config.py
- `db_path`: Path to SQLite database
- `sample_features`: Sample feature data (1000 records, 10 features)
- `sample_targets`: Sample target variables (default_flag, lgd, ead)
- `mock_tournament_results`: Mock tournament results for testing
- `pd_thresholds`: PD evaluation metric thresholds (PRD Section 7.3.2)

## Running the Tests

### Run all model exams:
```bash
pytest tests/model_exams/ -v
```

### Run specific model exam:
```bash
pytest tests/model_exams/test_me01_data_appropriateness.py -v
```

### Run specific test method:
```bash
pytest tests/model_exams/test_me01_data_appropriateness.py::TestDataAppropriatenessAndSuitability::test_database_exists_and_accessible -v
```

### Run with coverage:
```bash
pytest tests/model_exams/ --cov=backend --cov-report=html
```

### Run with detailed output:
```bash
pytest tests/model_exams/ -vv -s
```

## Test Design Principles

1. **Self-Contained**: Each test is independent and can run in isolation
2. **Fixture-Based**: Uses pytest fixtures for test data and configuration
3. **Parameterized**: Tests use sample data from conftest.py fixtures
4. **Clear Assertions**: Each test has explicit assertions with helpful error messages
5. **Documented**: Comprehensive docstrings reference PRD sections and regulations
6. **Graceful Skipping**: Uses `pytest.skip()` for optional dependencies (output_dir, etc.)
7. **Logging**: Prints detailed results for audit trail and debugging

## PRD Section Mapping

| Test | PRD Section | Focus Area |
|------|-------------|-----------|
| ME-01 | I.DA.1-I.DA.5 | Data Appropriateness |
| ME-02 | I.DQ.1-I.DQ.8 | Data Quality |
| ME-03 | M.FE.1-M.FE.7 | Feature Engineering |
| ME-04 | M.MS.1-M.MS.8 | Methodology Selection |
| ME-05 | M.PE.1-M.PE.7 | Parameter Estimation |
| ME-06 | O.S.1-O.S.2 | Stability Analysis |
| ME-07 | O.R.1 | Robustness Testing |
| ME-08 | O.E.1 | Explainability |
| ME-09 | B.EE.1-B.EE.2 | Business Integration |
| ME-10 | TI.D.1 | Documentation |

## Basel III Compliance

These tests implement validation requirements from:
- Basel III IRB Advanced Approach
- Federal Reserve SR 15-18 (Guidance on Model Risk Management)
- OCC Bulletin 2011-12 (Model Validation)

Key validation domains covered:
- **Data Validation**: Appropriateness, quality, integrity
- **Model Development**: Feature selection, methodology, parameter estimation
- **Model Performance**: Discrimination, calibration, stability, robustness
- **Model Use**: Business integration, explainability, documentation

## Notes

- Tests using `output_dir` will skip if no pipeline run output is found
- Tests using database will skip if database is not accessible
- Mock fixtures are used for unit tests to avoid external dependencies
- All tests are designed to pass with sample/mock data for CI/CD integration
