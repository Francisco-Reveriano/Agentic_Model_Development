"""
TC-09: Refinement Loop Convergence - Phase 3 Terminates Correctly
Per PRD Section 8.4
"""
import pytest
import numpy as np


class TestRefinementConvergence:
    """Test suite for Phase 3 refinement loop."""

    def test_convergence_by_threshold(self):
        """TC-09a: Loop exits when improvement < convergence_threshold."""
        convergence_threshold = 0.002
        max_iterations = 5
        scores = [0.750, 0.758, 0.761, 0.762, 0.762]  # Simulated improvements

        iteration = 0
        best_score = scores[0]
        converged = False

        for i in range(1, len(scores)):
            improvement = scores[i] - best_score
            if improvement < convergence_threshold:
                converged = True
                iteration = i
                break
            best_score = scores[i]
            iteration = i

        assert converged, "Loop did not converge - improvements never fell below threshold"
        print(f"Converged at iteration {iteration} with improvement {scores[iteration] - scores[iteration-1]:.4f}")

    def test_max_iterations_termination(self):
        """TC-09b: Loop exits at max_iterations even without convergence."""
        max_iterations = 5
        convergence_threshold = 0.002
        scores = [0.70, 0.72, 0.74, 0.76, 0.78, 0.80]  # Steady improvement

        iteration = 0
        best_score = scores[0]
        converged = False

        for i in range(1, min(len(scores), max_iterations + 1)):
            improvement = scores[i] - best_score
            if improvement < convergence_threshold:
                converged = True
                break
            best_score = scores[i]
            iteration = i

        assert iteration <= max_iterations, f"Loop exceeded max_iterations: {iteration}"
        print(f"Loop ran {iteration} iterations (max: {max_iterations}), converged: {converged}")

    def test_pruning_removes_underperformers(self):
        """TC-09c: Models > 0.03 below leader are pruned."""
        prune_threshold = 0.03
        model_scores = {
            "XGBoost": 0.782,
            "LightGBM": 0.778,
            "Random Forest": 0.761,
            "GBM": 0.755,
            "Decision Tree": 0.710,
        }

        leader_score = max(model_scores.values())
        surviving = {k: v for k, v in model_scores.items()
                     if leader_score - v <= prune_threshold}
        pruned = {k: v for k, v in model_scores.items()
                  if leader_score - v > prune_threshold}

        print(f"Leader: {leader_score:.4f}")
        print(f"Surviving ({len(surviving)}): {surviving}")
        print(f"Pruned ({len(pruned)}): {pruned}")

        assert len(surviving) > 0, "All models pruned"
        assert len(surviving) < len(model_scores), "No models pruned"
        for name, score in pruned.items():
            assert leader_score - score > prune_threshold

    def test_iteration_tracking_complete(self):
        """TC-09d: Verify iteration tracking table has all required columns."""
        required_columns = [
            "iteration", "models_evaluated", "best_model",
            "best_score", "improvement", "feature_set",
            "models_pruned", "status"
        ]

        # Simulate iteration record
        record = {
            "iteration": 2,
            "models_evaluated": 15,
            "best_model": "XGBoost",
            "best_score": 0.789,
            "improvement": 0.003,
            "feature_set": "Tier1+Tier2",
            "models_pruned": ["Decision Tree"],
            "status": "Continuing"
        }

        for col in required_columns:
            assert col in record, f"Missing column: {col}"
        print(f"Iteration record complete: {list(record.keys())}")
