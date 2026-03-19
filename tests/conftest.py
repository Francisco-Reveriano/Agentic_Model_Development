"""
Shared pytest fixtures for Credit Risk Modeling Platform test suite.
Provides test data, model loading, and common validation utilities.
"""
import pytest
import json
import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import MagicMock

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture(scope="session")
def project_root():
    """Return the project root directory."""
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def output_dir(project_root):
    """Find the most recent pipeline run output directory."""
    output_base = project_root / "Data" / "Output"
    if not output_base.exists():
        pytest.skip("No pipeline output directory found")
    runs = sorted(output_base.glob("pipeline_run_*"), reverse=True)
    if not runs:
        pytest.skip("No pipeline runs found")
    return runs[0]


@pytest.fixture(scope="session")
def settings():
    """Load project settings from config."""
    try:
        from backend.config import Settings
        return Settings()
    except Exception:
        pytest.skip("Cannot load Settings - .env may be missing")


@pytest.fixture(scope="session")
def db_path(settings):
    """Return the database path."""
    path = Path(settings.db_path)
    if not path.exists():
        pytest.skip(f"Database not found: {path}")
    return path


@pytest.fixture
def sample_features():
    """Generate sample feature data for unit testing."""
    np.random.seed(42)
    n = 1000
    return pd.DataFrame({
        'loan_amnt': np.random.uniform(1000, 40000, n),
        'int_rate': np.random.uniform(5.0, 25.0, n),
        'annual_inc': np.random.uniform(20000, 200000, n),
        'dti': np.random.uniform(0, 40, n),
        'revol_bal': np.random.uniform(0, 50000, n),
        'revol_util': np.random.uniform(0, 100, n),
        'open_acc': np.random.randint(1, 30, n),
        'total_acc': np.random.randint(5, 80, n),
        'grade_encoded': np.random.randint(0, 7, n),
        'term_months': np.random.choice([36, 60], n),
    })


@pytest.fixture
def sample_targets():
    """Generate sample target variables."""
    np.random.seed(42)
    n = 1000
    default_flag = np.random.binomial(1, 0.15, n)
    lgd = np.where(default_flag == 1, np.random.beta(2, 5, n), 0.0)
    ead = np.random.uniform(1000, 40000, n)
    return pd.DataFrame({
        'default_flag': default_flag,
        'lgd': lgd,
        'ead': ead
    })


@pytest.fixture
def mock_tournament_results():
    """Return mock tournament results for testing."""
    return {
        "model_type": "PD",
        "phase": "champion_selection",
        "champion": {
            "name": "XGBoost",
            "library": "xgboost",
            "metrics": {
                "auc_roc": 0.782,
                "gini": 0.564,
                "ks_statistic": 0.412,
                "brier_score": 0.118,
                "hosmer_lemeshow_p": 0.23
            }
        },
        "runner_up": {
            "name": "LightGBM",
            "library": "lightgbm",
            "metrics": {
                "auc_roc": 0.778,
                "gini": 0.556,
                "ks_statistic": 0.398,
                "brier_score": 0.121,
                "hosmer_lemeshow_p": 0.18
            }
        },
        "leaderboard": [],
        "iterations_completed": 3,
        "converged": True
    }


@pytest.fixture
def pd_thresholds():
    """PRD Section 7.3.2 - PD evaluation metric thresholds."""
    return {
        "auc_roc": {"green": 0.75, "yellow": 0.65, "red_below": 0.65},
        "gini": {"green": 0.50, "yellow": 0.30, "red_below": 0.30},
        "ks_statistic": {"green": 0.35, "yellow": 0.20, "red_below": 0.20},
        "brier_score": {"green_below": 0.15, "yellow_below": 0.25, "red": 0.25},
        "hosmer_lemeshow_p": {"green": 0.10, "yellow": 0.05, "red_below": 0.05},
        "psi": {"green_below": 0.10, "yellow_below": 0.25, "red": 0.25}
    }


def load_model_artifact(output_dir, stage_dir, filename):
    """Load a model artifact (joblib) from a pipeline run."""
    import joblib
    path = output_dir / stage_dir / filename
    if not path.exists():
        return None
    return joblib.load(path)


def load_json_artifact(output_dir, stage_dir, filename):
    """Load a JSON artifact from a pipeline run."""
    path = output_dir / stage_dir / filename
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def load_parquet_artifact(output_dir, stage_dir, filename):
    """Load a parquet artifact from a pipeline run."""
    path = output_dir / stage_dir / filename
    if not path.exists():
        return None
    return pd.read_parquet(path)
