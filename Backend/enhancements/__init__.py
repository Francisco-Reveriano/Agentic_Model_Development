"""
Enhancements module for credit risk modeling platform.

Provides optional performance, resilience, and operational improvements:
- Agent timeout management
- Early stopping for tree models
- Parallel training of candidates
- SMOTE handling for class imbalance
- Scoring mode configuration (regulatory vs performance)
- Model comparison utilities
- Feature winsorization
- Pipeline run history tracking
- SSE heartbeat for long-running operations
- Leaderboard export capabilities
"""

__version__ = "1.0.0"

__all__ = [
    "AgentTimeoutConfig",
    "wrap_with_timeout",
    "EarlyStoppingConfig",
    "apply_early_stopping",
    "ParallelTrainingConfig",
    "train_candidates_parallel",
    "SMOTEConfig",
    "apply_smote_if_needed",
    "ScoringModeConfig",
    "get_rubric_weights",
    "generate_comparison_data",
    "WinsorizeConfig",
    "apply_winsorization",
    "scan_pipeline_runs",
    "compare_runs",
    "HeartbeatConfig",
    "emit_heartbeat",
    "create_reconnectable_sse_handler",
    "export_leaderboard_csv",
    "export_leaderboard_excel",
]
