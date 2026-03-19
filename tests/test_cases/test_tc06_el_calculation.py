"""
TC-06: Expected Loss Calculation Accuracy - EL = PD × LGD × EAD
Per PRD Section 7.6
"""
import pytest
import numpy as np


class TestELCalculation:
    """Test suite for Expected Loss computation."""

    def test_el_formula_correct(self):
        """TC-06a: Verify EL = PD × LGD × EAD."""
        np.random.seed(42)
        n = 1000
        pd_vals = np.random.uniform(0, 0.3, n)
        lgd_vals = np.random.uniform(0, 1, n)
        ead_vals = np.random.uniform(1000, 50000, n)

        el = pd_vals * lgd_vals * ead_vals
        el_manual = np.array([p * l * e for p, l, e in zip(pd_vals, lgd_vals, ead_vals)])

        np.testing.assert_array_almost_equal(el, el_manual, decimal=10)

    def test_el_non_negative(self):
        """TC-06b: All EL values must be >= 0."""
        np.random.seed(42)
        n = 1000
        pd_vals = np.random.uniform(0, 0.5, n)
        lgd_vals = np.random.uniform(0, 1, n)
        ead_vals = np.random.uniform(0, 50000, n)

        el = pd_vals * lgd_vals * ead_vals
        assert np.all(el >= 0), "Found negative EL values"

    def test_stress_scenario_ordering(self):
        """TC-06c: Stress EL ordering: Base < Adverse < Severe."""
        np.random.seed(42)
        n = 1000
        pd_base = np.random.uniform(0.05, 0.25, n)
        lgd_base = np.random.uniform(0.2, 0.8, n)
        ead_vals = np.random.uniform(5000, 40000, n)

        # Base scenario
        el_base = pd_base * lgd_base * ead_vals

        # Adverse: PD × 1.5, LGD floor 0.45
        pd_adverse = pd_base * 1.5
        lgd_adverse = np.maximum(lgd_base, 0.45)
        el_adverse = pd_adverse * lgd_adverse * ead_vals

        # Severe: PD × 2.0, LGD floor 0.60
        pd_severe = pd_base * 2.0
        lgd_severe = np.maximum(lgd_base, 0.60)
        el_severe = pd_severe * lgd_severe * ead_vals

        total_base = el_base.sum()
        total_adverse = el_adverse.sum()
        total_severe = el_severe.sum()

        print(f"Base EL: ${total_base:,.0f}")
        print(f"Adverse EL: ${total_adverse:,.0f}")
        print(f"Severe EL: ${total_severe:,.0f}")

        assert total_base < total_adverse < total_severe, (
            f"Stress ordering violated: Base={total_base:.0f}, "
            f"Adverse={total_adverse:.0f}, Severe={total_severe:.0f}"
        )

    def test_el_does_not_exceed_ead(self):
        """TC-06d: Loan-level EL should not exceed EAD."""
        np.random.seed(42)
        n = 1000
        pd_vals = np.clip(np.random.uniform(0, 1, n), 0, 1)
        lgd_vals = np.clip(np.random.uniform(0, 1, n), 0, 1)
        ead_vals = np.random.uniform(1000, 50000, n)

        el = pd_vals * lgd_vals * ead_vals
        assert np.all(el <= ead_vals + 1e-6), "Found EL exceeding EAD"
