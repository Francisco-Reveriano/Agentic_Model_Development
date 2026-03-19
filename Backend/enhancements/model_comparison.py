"""
Model Comparison Utilities

Generates structured comparison data for visualization of model tournament results.
Supports bar charts, radar charts, and feature importance heatmaps.
"""

from typing import Dict, List, Tuple, Optional, Any
import numpy as np


def generate_comparison_data(
    tournament_results: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Generate structured comparison data from tournament results.

    Transforms raw tournament results into formats suitable for various
    visualizations (bar charts, radar charts, heatmaps).

    Args:
        tournament_results: Tournament results from Phase 1-4
            Expected keys:
            - 'phase1_results': List of {model, metrics} for all candidates
            - 'phase3_results': List of refined candidates after optimization
            - 'champion': Name of selected champion model
            - 'champion_metrics': Metrics of champion

    Returns:
        Dictionary with visualization-ready data:
        - 'bar_chart_data': Dict for bar chart visualization
        - 'radar_chart_data': Dict for radar/spider chart
        - 'heatmap_data': Dict for correlation/feature importance heatmap
        - 'leaderboard': Ranked list of all candidates
        - 'summary': High-level tournament summary

    Example:
        >>> results = tournament.run()
        >>> comp_data = generate_comparison_data(results)
        >>> print(comp_data['leaderboard'][0])  # Top candidate
        {'rank': 1, 'name': 'xgboost', 'score': 0.85, 'auc': 0.87, ...}
    """
    comparison_data = {
        "bar_chart_data": _prepare_bar_chart_data(tournament_results),
        "radar_chart_data": _prepare_radar_chart_data(tournament_results),
        "heatmap_data": _prepare_heatmap_data(tournament_results),
        "leaderboard": _prepare_leaderboard(tournament_results),
        "summary": _prepare_summary(tournament_results),
    }

    return comparison_data


def _prepare_bar_chart_data(tournament_results: Dict[str, Any]) -> Dict[str, List]:
    """
    Prepare data for bar chart showing candidate scores.

    Args:
        tournament_results: Raw tournament results

    Returns:
        Dictionary with:
        - 'models': List of model names
        - 'scores': List of overall rubric scores
        - 'auc': List of AUC values
        - 'accuracy': List of accuracy values
    """
    phase1_results = tournament_results.get("phase1_results", [])

    models = []
    scores = []
    auc_values = []
    accuracy_values = []

    for result in phase1_results:
        if isinstance(result, dict):
            models.append(result.get("model", "Unknown"))
            metrics = result.get("metrics", {})

            scores.append(metrics.get("overall_score", 0.0))
            auc_values.append(metrics.get("auc", 0.0))
            accuracy_values.append(metrics.get("accuracy", 0.0))

    return {
        "models": models,
        "scores": scores,
        "auc": auc_values,
        "accuracy": accuracy_values,
    }


def _prepare_radar_chart_data(tournament_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Prepare data for radar/spider chart comparing top candidates.

    Args:
        tournament_results: Raw tournament results

    Returns:
        Dictionary with:
        - 'metrics': List of metric names
        - 'candidates': Dict mapping candidate_name -> metric_values
    """
    phase1_results = tournament_results.get("phase1_results", [])

    # Get top 5 candidates
    top_n = 5
    candidates_dict = {}

    for i, result in enumerate(phase1_results[:top_n]):
        if isinstance(result, dict):
            model_name = result.get("model", f"Model_{i}")
            metrics = result.get("metrics", {})

            candidate_metrics = [
                metrics.get("auc", 0.0),
                metrics.get("accuracy", 0.0),
                metrics.get("precision", 0.0),
                metrics.get("recall", 0.0),
                metrics.get("f1_score", 0.0),
            ]

            candidates_dict[model_name] = candidate_metrics

    metric_names = ["AUC", "Accuracy", "Precision", "Recall", "F1 Score"]

    return {
        "metrics": metric_names,
        "candidates": candidates_dict,
        "top_n": top_n,
    }


def _prepare_heatmap_data(tournament_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Prepare data for heatmap showing feature importance across models.

    Args:
        tournament_results: Raw tournament results

    Returns:
        Dictionary with:
        - 'models': List of model names
        - 'features': List of feature names
        - 'importance_matrix': 2D array of importance scores
    """
    feature_importance = tournament_results.get("phase2_feature_importance", {})

    if not feature_importance:
        return {
            "models": [],
            "features": [],
            "importance_matrix": [],
        }

    models = list(feature_importance.keys())
    all_features = set()

    # Collect all features mentioned
    for model_features in feature_importance.values():
        if isinstance(model_features, dict):
            all_features.update(model_features.keys())

    features = sorted(list(all_features))

    # Build importance matrix
    importance_matrix = []
    for model in models:
        model_importance = []
        features_dict = feature_importance.get(model, {})

        for feature in features:
            if isinstance(features_dict, dict):
                importance = features_dict.get(feature, 0.0)
            else:
                importance = 0.0

            model_importance.append(importance)

        importance_matrix.append(model_importance)

    return {
        "models": models,
        "features": features,
        "importance_matrix": importance_matrix,
    }


def _prepare_leaderboard(tournament_results: Dict[str, Any]) -> List[Dict]:
    """
    Prepare ranked leaderboard of all candidates.

    Args:
        tournament_results: Raw tournament results

    Returns:
        List of ranked candidates sorted by score, each containing:
        - rank: Integer rank (1=best)
        - name: Model name
        - score: Overall rubric score
        - auc: AUC value
        - accuracy: Accuracy value
        - phase: Which phase the model reached
    """
    phase1_results = tournament_results.get("phase1_results", [])

    leaderboard = []

    for i, result in enumerate(phase1_results):
        if isinstance(result, dict):
            metrics = result.get("metrics", {})

            entry = {
                "rank": i + 1,
                "name": result.get("model", "Unknown"),
                "score": metrics.get("overall_score", 0.0),
                "auc": metrics.get("auc", 0.0),
                "accuracy": metrics.get("accuracy", 0.0),
                "precision": metrics.get("precision", 0.0),
                "recall": metrics.get("recall", 0.0),
                "f1_score": metrics.get("f1_score", 0.0),
                "phase": result.get("phase", 1),
            }

            leaderboard.append(entry)

    # Sort by rank (already in order, but ensure)
    leaderboard.sort(key=lambda x: x["score"], reverse=True)

    # Update ranks
    for i, entry in enumerate(leaderboard):
        entry["rank"] = i + 1

    return leaderboard


def _prepare_summary(tournament_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Prepare high-level tournament summary.

    Args:
        tournament_results: Raw tournament results

    Returns:
        Dictionary with:
        - champion: Name of winning model
        - champion_score: Score of champion
        - runner_up: Name of second-best model
        - runner_up_score: Score of runner-up
        - phase1_count: Number of candidates in Phase 1
        - phase3_count: Number of candidates refined in Phase 3
        - improvement: Improvement from Phase 1 best to Phase 3 best
    """
    champion = tournament_results.get("champion", "Unknown")
    champion_metrics = tournament_results.get("champion_metrics", {})
    champion_score = champion_metrics.get("overall_score", 0.0)

    phase1_results = tournament_results.get("phase1_results", [])
    phase3_results = tournament_results.get("phase3_results", [])

    runner_up = "N/A"
    runner_up_score = 0.0

    if len(phase1_results) > 1:
        second_result = phase1_results[1]
        if isinstance(second_result, dict):
            runner_up = second_result.get("model", "Unknown")
            runner_up_score = second_result.get("metrics", {}).get("overall_score", 0.0)

    # Calculate improvement
    phase1_best = 0.0
    if phase1_results and isinstance(phase1_results[0], dict):
        phase1_best = phase1_results[0].get("metrics", {}).get("overall_score", 0.0)

    phase3_best = 0.0
    if phase3_results and isinstance(phase3_results[0], dict):
        phase3_best = phase3_results[0].get("metrics", {}).get("overall_score", 0.0)

    improvement = phase3_best - phase1_best if phase1_best > 0 else 0.0

    return {
        "champion": champion,
        "champion_score": champion_score,
        "runner_up": runner_up,
        "runner_up_score": runner_up_score,
        "phase1_candidate_count": len(phase1_results),
        "phase3_candidate_count": len(phase3_results),
        "phase1_best_score": phase1_best,
        "phase3_best_score": phase3_best,
        "improvement_from_refinement": improvement,
    }


def calculate_metric_statistics(
    tournament_results: Dict[str, Any],
    metric: str = "auc",
) -> Dict[str, float]:
    """
    Calculate statistics for a specific metric across all candidates.

    Args:
        tournament_results: Raw tournament results
        metric: Metric to analyze (default: 'auc')

    Returns:
        Dictionary with:
        - mean: Mean value
        - std: Standard deviation
        - min: Minimum value
        - max: Maximum value
        - median: Median value
    """
    phase1_results = tournament_results.get("phase1_results", [])

    values = []
    for result in phase1_results:
        if isinstance(result, dict):
            metrics = result.get("metrics", {})
            value = metrics.get(metric, None)
            if value is not None:
                values.append(float(value))

    if not values:
        return {
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
            "median": 0.0,
        }

    values_array = np.array(values)

    return {
        "mean": float(np.mean(values_array)),
        "std": float(np.std(values_array)),
        "min": float(np.min(values_array)),
        "max": float(np.max(values_array)),
        "median": float(np.median(values_array)),
    }
