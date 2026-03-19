"""
End-to-End Test Suite for Credit Risk Modeling Platform

Tests cover:
- Full pipeline execution (test_e2e01_full_pipeline_regulatory.py)
- PD-only pipeline mode (test_e2e02_pd_only.py)
- Regulatory vs Performance mode scoring (test_e2e03_performance_mode.py)
- LGD two-stage and EAD regression (test_e2e04_lgd_ead_regression.py)
- Pipeline resilience and edge cases (test_e2e05_resilience.py)
"""

__all__ = [
    "test_e2e01_full_pipeline_regulatory",
    "test_e2e02_pd_only",
    "test_e2e03_performance_mode",
    "test_e2e04_lgd_ead_regression",
    "test_e2e05_resilience",
]
