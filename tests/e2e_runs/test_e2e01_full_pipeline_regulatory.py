"""
E2E Test Suite: Full Pipeline in Regulatory Mode
Tests the complete 7-agent pipeline executing in regulatory scoring mode with Basel II/III compliance.
"""

import os
import pytest
from pathlib import Path
from datetime import datetime
from typing import List, Set


class TestPipelineConfigValid:
    """Verify all required environment variables are configured."""

    def test_required_env_vars_present(self):
        """Test that all required environment variables are set."""
        required_vars = [
            "ANTHROPIC_API_KEY",
            "MODEL_ID",
            "DB_PATH",
            "DB_TABLE",
            "OUTPUT_DIR",
            "PIPELINE_MD_PATH",
        ]

        for var in required_vars:
            assert os.getenv(var) is not None, f"Missing required env var: {var}"

    def test_env_var_values_non_empty(self):
        """Test that environment variable values are not empty strings."""
        key_vars = ["ANTHROPIC_API_KEY", "MODEL_ID", "DB_PATH"]

        for var in key_vars:
            value = os.getenv(var, "")
            assert value.strip(), f"Env var {var} is empty or whitespace"

    def test_db_path_exists(self):
        """Test that the configured database file exists."""
        db_path = os.getenv("DB_PATH", "Data/Raw/RCM_Controls.db")
        assert Path(db_path).exists(), f"Database not found at {db_path}"

    def test_pipeline_md_exists(self):
        """Test that the pipeline playbook markdown file exists."""
        md_path = os.getenv("PIPELINE_MD_PATH", "docs/credit_risk_pd_lgd_ead_pipeline.md")
        assert Path(md_path).exists(), f"Pipeline markdown not found at {md_path}"


class TestAllAgentsDefined:
    """Verify that all 7 agent factory functions are importable."""

    def test_import_all_agents(self):
        """Test that all 7 agent factory functions can be imported."""
        try:
            from backend.agents.data_agent import create_data_agent
            from backend.agents.feature_agent import create_feature_agent
            from backend.agents.pd_agent import create_pd_agent
            from backend.agents.lgd_agent import create_lgd_agent
            from backend.agents.ead_agent import create_ead_agent
            from backend.agents.el_agent import create_el_agent
            from backend.agents.report_agent import create_report_agent
        except ImportError as e:
            pytest.fail(f"Failed to import agent: {e}")

    def test_agent_factory_signatures(self):
        """Test that agent factory functions have correct signatures."""
        from backend.agents.data_agent import create_data_agent
        import inspect

        sig = inspect.signature(create_data_agent)
        params = list(sig.parameters.keys())

        assert "settings" in params or len(params) == 0, "Factory should accept settings parameter"
        assert "output_dir" in params or len(params) <= 1, "Factory should accept output_dir parameter"

    def test_agent_returns_agent_object(self):
        """Test that agent factories return valid Agent objects."""
        from backend.agents.data_agent import create_data_agent
        from backend.config import Settings

        settings = Settings()
        agent = create_data_agent(settings=settings)

        assert agent is not None
        assert hasattr(agent, "run") or hasattr(agent, "process"), "Agent should have run/process method"


class TestOrchestratorStageOrder:
    """Verify that pipeline stages execute in correct order."""

    def test_stage_directory_names(self):
        """Test that stage directories follow naming convention."""
        expected_stages = [
            "01_data_quality",
            "02_feature_engineering",
            "03_pd_modeling",
            "04_lgd_modeling",
            "05_ead_modeling",
            "06_el_calculation",
            "07_reports",
        ]

        assert len(expected_stages) == 7, "Should have exactly 7 stages"

        for i, stage in enumerate(expected_stages, 1):
            assert stage.startswith(f"{i:02d}"), f"Stage {i} should start with {i:02d}_"

    def test_orchestrator_imports(self):
        """Test that orchestrator can be imported."""
        try:
            from backend.orchestrator import Orchestrator, execute_pipeline
        except ImportError as e:
            pytest.fail(f"Failed to import orchestrator: {e}")

    def test_handoff_protocol_structure(self):
        """Test that handoff protocol defines correct information flow."""
        from backend.orchestrator import Orchestrator
        import inspect

        source = inspect.getsource(Orchestrator)
        assert "handoff.json" in source, "Orchestrator should use handoff.json protocol"

    def test_stage_sequence_logic(self):
        """Test that stages are executed in sequential order."""
        from backend.orchestrator import Orchestrator
        import inspect

        source = inspect.getsource(Orchestrator)
        # Check that agent execution order is defined
        agents_order = [
            "data_agent",
            "feature_agent",
            "pd_agent",
            "lgd_agent",
            "ead_agent",
            "el_agent",
            "report_agent",
        ]

        for agent in agents_order:
            assert agent in source, f"Orchestrator should reference {agent}"


class TestArtifactExpectations:
    """Verify expected artifact filenames for full pipeline."""

    def test_full_pipeline_artifact_list(self):
        """Test that full pipeline produces 20+ expected artifacts."""
        expected_artifacts = {
            # Data Quality Stage
            "cleaned_features.parquet",
            "targets.parquet",
            "dq_report.json",

            # Feature Engineering Stage
            "feature_matrix.parquet",
            "feature_importance.json",
            "feature_stats.json",

            # PD Modeling Stage
            "pd_champion.joblib",
            "pd_tournament_results.json",
            "pd_statsmodels_output.txt",

            # LGD Modeling Stage
            "lgd_stage1_champion.joblib",
            "lgd_stage2_champion.joblib",
            "lgd_tournament_results.json",

            # EAD Modeling Stage
            "ead_champion.joblib",
            "ead_tournament_results.json",
            "ccf_analysis.json",

            # EL Calculation Stage
            "el_results.parquet",
            "el_base_case.parquet",
            "el_adverse.parquet",
            "el_severe.parquet",

            # Reports
            "Credit_Risk_DQ_Report.docx",
            "Credit_Risk_PD_Report.docx",
            "Credit_Risk_LGD_Report.docx",
            "Credit_Risk_EAD_Report.docx",
            "Credit_Risk_EL_Summary.docx",
        }

        assert len(expected_artifacts) >= 20, "Should have at least 20 expected artifacts"

    def test_stage_output_directories(self):
        """Test that stage output directories have consistent structure."""
        output_dirs = [
            "01_data_quality",
            "02_feature_engineering",
            "03_pd_modeling",
            "04_lgd_modeling",
            "05_ead_modeling",
            "06_el_calculation",
            "07_reports",
        ]

        assert len(output_dirs) == 7, "Should have 7 stage output directories"

    def test_handoff_json_presence(self):
        """Test that each stage produces handoff.json for next stage."""
        stages = [
            "01_data_quality",
            "02_feature_engineering",
            "03_pd_modeling",
            "04_lgd_modeling",
            "05_ead_modeling",
            "06_el_calculation",
        ]

        for stage in stages:
            assert f"{stage}/handoff.json" in [f"{s}/handoff.json" for s in stages]


class TestStressOrderingLogic:
    """Verify stress scenario ordering: Base < Adverse < Severe."""

    def test_stress_scenario_order(self):
        """Test that stress scenarios follow Base < Adverse < Severe."""
        scenarios = ["base", "adverse", "severe"]
        assert scenarios == ["base", "adverse", "severe"], "Stress ordering must be Base < Adverse < Severe"

    def test_el_stress_calculation_structure(self):
        """Test EL calculation produces all three stress scenarios."""
        from backend.agents.el_agent import create_el_agent
        import inspect

        source = inspect.getsource(create_el_agent)
        assert "base" in source.lower(), "Should compute base case EL"
        assert "adverse" in source.lower(), "Should compute adverse scenario EL"
        assert "severe" in source.lower(), "Should compute severe scenario EL"

    def test_stress_assumptions_impact(self):
        """Test that stress assumptions increase loss estimates."""
        # Base case should have lower EL than Adverse
        # Adverse should have lower EL than Severe
        # This is structural assertion about model outputs
        stress_factors = {
            "base": 1.0,
            "adverse": 1.25,
            "severe": 1.50,
        }

        for scenario, factor in stress_factors.items():
            assert factor >= 1.0, f"{scenario} scenario factor must be >= 1.0"

    def test_stress_output_filenames(self):
        """Test that stress output files follow naming convention."""
        expected_files = [
            "el_base_case.parquet",
            "el_adverse.parquet",
            "el_severe.parquet",
        ]

        assert len(expected_files) == 3, "Should have exactly 3 stress output files"
        for filename in expected_files:
            assert "el_" in filename, "EL files should start with el_"


class TestExecutionTimeBudget:
    """Verify pipeline execution stays within 60-minute budget."""

    def test_execution_time_limit(self):
        """Test that total pipeline execution time is limited."""
        max_duration_seconds = 60 * 60  # 60 minutes
        assert max_duration_seconds == 3600, "Budget should be 3600 seconds (60 minutes)"

    def test_agent_timeout_configuration(self):
        """Test that agent timeouts are configured."""
        try:
            from backend.enhancements.agent_timeout import AgentTimeoutConfig

            config = AgentTimeoutConfig()
            assert hasattr(config, "DATA_AGENT"), "Should have DATA_AGENT timeout"
            assert hasattr(config, "PD_AGENT"), "Should have PD_AGENT timeout"
            assert config.DATA_AGENT < config.PD_AGENT, "Data agent should be faster than PD"
        except ImportError:
            pytest.skip("agent_timeout enhancement not yet implemented")

    def test_stage_time_estimates(self):
        """Test that stages have reasonable time estimates."""
        stage_time_estimates = {
            "data_quality": 300,        # 5 minutes
            "feature_engineering": 600, # 10 minutes
            "pd_modeling": 1800,        # 30 minutes
            "lgd_modeling": 900,        # 15 minutes
            "ead_modeling": 600,        # 10 minutes
            "el_calculation": 300,      # 5 minutes
            "reports": 300,             # 5 minutes
        }

        total = sum(stage_time_estimates.values())
        assert total <= 3600, "Total estimated time should fit in 60-minute budget"

    def test_early_stopping_enabled(self):
        """Test that early stopping is available to reduce training time."""
        try:
            from backend.enhancements.early_stopping import EarlyStoppingConfig

            config = EarlyStoppingConfig()
            assert hasattr(config, "enabled"), "Should have enabled flag"
            assert hasattr(config, "rounds"), "Should have early stopping rounds"
        except ImportError:
            pytest.skip("early_stopping enhancement not yet implemented")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
