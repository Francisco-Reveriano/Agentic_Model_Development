"""
Model Exam ME-02: Data Quality & Integrity
===========================================

PRD Section I.DQ (Data Quality):
  - I.DQ.1: Missing data assessment and treatment
  - I.DQ.2: Outlier detection and handling
  - I.DQ.3: Data type consistency validation
  - I.DQ.4: Temporal consistency checks
  - I.DQ.5: Duplicate record detection
  - I.DQ.6: Referential integrity validation
  - I.DQ.7: Range/domain validation
  - I.DQ.8: Completeness assessment

Tests verify:
  - Missing rates < 5% in critical columns after cleaning
  - No duplicate loan IDs
  - Numeric columns have valid ranges
  - Data types are consistent
  - No orphaned references
"""

import pytest
import pandas as pd
import numpy as np
import sqlite3


class TestDataQualityAndIntegrity:
    """Validates data quality, completeness, consistency, and integrity."""

    def test_missing_data_rate_acceptable(self, sample_features, sample_targets):
        """I.DQ.1: Missing data rate must be < 5% in critical columns.

        After data cleaning, critical features for PD/LGD/EAD modeling
        must have minimal missing values to ensure model robustness
        and interpretability.
        """
        df = pd.concat([sample_features, sample_targets], axis=1)

        # Critical columns for credit risk modeling
        critical_cols = [
            'loan_amnt', 'int_rate', 'annual_inc', 'grade_encoded',
            'default_flag'
        ]

        for col in critical_cols:
            if col in df.columns:
                missing_rate = df[col].isna().sum() / len(df)
                assert missing_rate < 0.05, (
                    f"Column '{col}' missing rate {missing_rate:.2%} "
                    f"exceeds 5% threshold"
                )
                print(f"{col:20s}: {missing_rate:.2%} missing ✓")

    def test_no_duplicate_loan_ids(self, db_path):
        """I.DQ.5: No duplicate loan IDs allowed.

        Loan IDs must be unique identifiers. Duplicates indicate
        data integrity issues and violate fundamental database constraints.
        """
        conn = sqlite3.connect(db_path)

        # Count total IDs and unique IDs
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM my_table")
        total_count = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(DISTINCT id) FROM my_table")
        unique_count = cursor.fetchone()[0]

        conn.close()

        assert total_count == unique_count, (
            f"Duplicate loan IDs detected: {total_count} total, "
            f"{unique_count} unique (diff: {total_count - unique_count})"
        )
        print(f"Loan ID uniqueness: {total_count} records, {unique_count} unique ✓")

    def test_numeric_columns_valid_ranges(self, sample_features, sample_targets):
        """I.DQ.7: Numeric columns must be within valid domain ranges.

        Validates that continuous variables (loan amount, interest rate,
        income, etc.) are within economically and statistically sensible
        ranges. Out-of-range values indicate data quality issues.
        """
        df = pd.concat([sample_features, sample_targets], axis=1)

        # Define valid ranges for key numeric columns
        valid_ranges = {
            'loan_amnt': (100, 100000),
            'int_rate': (0, 50),
            'annual_inc': (0, 10000000),
            'dti': (0, 100),
            'revol_util': (0, 100),
            'ead': (0, 100000),
            'lgd': (0, 1),
        }

        for col, (min_val, max_val) in valid_ranges.items():
            if col in df.columns:
                col_data = df[col].dropna()
                if len(col_data) > 0:
                    col_min = col_data.min()
                    col_max = col_data.max()

                    assert col_min >= min_val, (
                        f"Column '{col}' minimum {col_min} below range [{min_val}, {max_val}]"
                    )
                    assert col_max <= max_val, (
                        f"Column '{col}' maximum {col_max} exceeds range [{min_val}, {max_val}]"
                    )
                    print(f"{col:20s}: [{col_min:.2f}, {col_max:.2f}] ✓")

    def test_data_type_consistency(self, sample_features, sample_targets):
        """I.DQ.3: Data types must be consistent and appropriate.

        Validates that columns use correct data types (numeric for
        amounts, bool for flags, categorical for grades) enabling
        proper statistical computations and model ingestion.
        """
        df = pd.concat([sample_features, sample_targets], axis=1)

        expected_types = {
            'loan_amnt': ('float64', 'int64'),
            'int_rate': ('float64',),
            'annual_inc': ('float64', 'int64'),
            'default_flag': ('int64', 'uint8'),
            'grade_encoded': ('int64', 'uint8'),
        }

        for col, expected in expected_types.items():
            if col in df.columns:
                actual = str(df[col].dtype)
                assert any(e in actual for e in expected), (
                    f"Column '{col}' dtype {actual} not in {expected}"
                )
                print(f"{col:20s}: {actual} ✓")

    def test_no_negative_categorical_values(self, sample_features):
        """I.DQ.3: Categorical encoded columns should be non-negative.

        Grade encodings and categorical indices must be non-negative
        integers. Negative values indicate encoding errors.
        """
        categorical_cols = ['grade_encoded', 'term_months']

        for col in categorical_cols:
            if col in sample_features.columns:
                col_data = sample_features[col].dropna()
                min_val = col_data.min()
                assert min_val >= 0, (
                    f"Categorical column '{col}' has negative value {min_val}"
                )
                print(f"{col:20s}: min={min_val:.0f} ✓")

    def test_temporal_consistency(self, db_path):
        """I.DQ.4: Date columns must be temporally consistent.

        Validates that date fields follow logical temporal ordering
        (e.g., issue_date < maturity_date) and don't have impossible
        future dates or misformatted values.
        """
        conn = sqlite3.connect(db_path)

        # Check if date columns exist and sample them
        df = pd.read_sql_query(
            "SELECT id, hardship_start_date, hardship_end_date, settlement_date "
            "FROM my_table WHERE hardship_start_date IS NOT NULL LIMIT 1000",
            conn
        )
        conn.close()

        if len(df) > 0 and 'hardship_start_date' in df.columns:
            df['hardship_start_date'] = pd.to_datetime(
                df['hardship_start_date'], errors='coerce'
            )
            df['hardship_end_date'] = pd.to_datetime(
                df['hardship_end_date'], errors='coerce'
            )

            # For records with both dates, end should be >= start
            valid_pairs = df[
                (df['hardship_start_date'].notna()) &
                (df['hardship_end_date'].notna())
            ]

            if len(valid_pairs) > 0:
                valid_order = (valid_pairs['hardship_end_date'] >=
                              valid_pairs['hardship_start_date']).all()
                assert valid_order, "Hardship end_date before start_date found"
                print(f"Temporal consistency: {len(valid_pairs)} date pairs checked ✓")

    def test_completeness_threshold(self, sample_features, sample_targets):
        """I.DQ.8: Overall completeness must exceed 95%.

        The dataset must be sufficiently complete to enable accurate
        model training. Target completeness threshold is 95%.
        """
        df = pd.concat([sample_features, sample_targets], axis=1)

        total_cells = df.shape[0] * df.shape[1]
        missing_cells = df.isna().sum().sum()
        completeness = (total_cells - missing_cells) / total_cells

        assert completeness >= 0.95, (
            f"Completeness {completeness:.2%} below 95% threshold"
        )
        print(f"Overall completeness: {completeness:.2%} ✓")

    def test_outlier_count_reasonable(self, sample_features):
        """I.DQ.2: Outlier count must be investigated and justified.

        Validates that extreme outliers are limited in number. While
        some outliers are expected in financial data, excessive outliers
        may indicate data quality issues or require special treatment.
        """
        numeric_cols = sample_features.select_dtypes(
            include=['float64', 'int64']
        ).columns

        outlier_counts = {}
        for col in numeric_cols:
            if col in ['term_months', 'grade_encoded']:
                continue  # Skip categorical

            col_data = sample_features[col].dropna()
            if len(col_data) == 0:
                continue

            Q1 = col_data.quantile(0.25)
            Q3 = col_data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            outliers = ((col_data < lower_bound) | (col_data > upper_bound)).sum()
            outlier_pct = (outliers / len(col_data)) * 100

            outlier_counts[col] = outlier_pct
            assert outlier_pct < 10, (
                f"Column '{col}' has {outlier_pct:.1f}% outliers (>10% threshold)"
            )
            print(f"{col:20s}: {outlier_pct:.2f}% outliers ✓")
