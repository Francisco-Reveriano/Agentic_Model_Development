"""SSE callback handler for Strands agents.

Bridges synchronous Strands agent execution with the async SSE event queue
so the React frontend receives real-time progress updates.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any


class SSECallbackHandler:
    """Pushes structured events to an asyncio.Queue during agent execution."""

    def __init__(self, agent_name: str, event_queue: asyncio.Queue):
        self.agent_name = agent_name
        self.queue = event_queue
        self._loop = asyncio.get_event_loop()

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _put(self, event_type: str, data: dict[str, Any]) -> None:
        data["agent"] = self.agent_name
        data["timestamp"] = time.time()
        payload = {"event": event_type, "data": data}
        # Schedule the put on the event loop (called from sync agent thread)
        asyncio.run_coroutine_threadsafe(self.queue.put(payload), self._loop)

    # ------------------------------------------------------------------
    # agent lifecycle events
    # ------------------------------------------------------------------

    def on_agent_start(self, stage: int, total_stages: int) -> None:
        self._put("agent_start", {"stage": stage, "total_stages": total_stages})

    def on_agent_complete(self, status: str, duration_s: float) -> None:
        self._put("agent_complete", {"status": status, "duration_s": round(duration_s, 1)})

    def on_agent_error(self, error: str, recoverable: bool = False) -> None:
        self._put("agent_error", {"error": error, "recoverable": recoverable})

    # ------------------------------------------------------------------
    # streaming log events
    # ------------------------------------------------------------------

    def on_log(self, content: str, log_type: str = "text") -> None:
        self._put("agent_log", {"content": content, "type": log_type})

    def on_tool_start(self, tool_name: str, tool_input: dict[str, Any] | None = None) -> None:
        self._put("agent_log", {
            "content": f"Calling tool: {tool_name}",
            "type": "tool_call",
            "tool_name": tool_name,
        })

    def on_tool_end(self, tool_name: str, tool_output: dict[str, Any] | None = None) -> None:
        self._put("agent_log", {
            "content": f"Tool completed: {tool_name}",
            "type": "tool_result",
            "tool_name": tool_name,
        })
        # Try to extract metrics from tool output
        if tool_output and isinstance(tool_output, dict):
            self._extract_metrics(tool_output)

    # ------------------------------------------------------------------
    # metric events
    # ------------------------------------------------------------------

    def on_metric(self, metric: str, value: float) -> None:
        self._put("agent_metric", {"metric": metric, "value": value})

    def _extract_metrics(self, output: dict[str, Any]) -> None:
        """Auto-extract known metric keys from tool outputs."""
        content = output.get("content", [{}])
        if isinstance(content, list) and content:
            try:
                data = json.loads(content[0].get("text", "{}"))
            except (json.JSONDecodeError, AttributeError, IndexError):
                return
        elif isinstance(content, dict):
            data = content
        else:
            return

        metric_keys = {
            "auc", "gini", "ks", "brier", "psi", "rmse", "mae", "r2",
            "row_count", "default_rate", "dq_tests_passed",
        }
        for key in metric_keys:
            if key in data and isinstance(data[key], (int, float)):
                self.on_metric(key, data[key])

    # ------------------------------------------------------------------
    # tournament-specific events
    # ------------------------------------------------------------------

    def on_tournament_start(self, phase: int, model_count: int, model_type: str) -> None:
        self._put("tournament_start", {
            "phase": phase,
            "model_count": model_count,
            "model_type": model_type,
        })

    def on_model_trained(self, model: str, rank: int, primary_metric: float, time_s: float) -> None:
        self._put("model_trained", {
            "model": model,
            "rank": rank,
            "primary_metric": primary_metric,
            "time_s": round(time_s, 1),
        })

    def on_phase_complete(self, phase: int, best_model: str, best_score: float) -> None:
        self._put("phase_complete", {
            "phase": phase,
            "best_model": best_model,
            "best_score": best_score,
        })

    def on_model_pruned(self, model: str, score: float, reason: str) -> None:
        self._put("model_pruned", {
            "model": model,
            "score": score,
            "reason": reason,
        })

    def on_champion_declared(self, champion: str, score: float, runner_up: str) -> None:
        self._put("champion_declared", {
            "champion": champion,
            "score": score,
            "runner_up": runner_up,
        })

    def on_feature_consensus(self, top_features: list[str], tier_counts: dict[int, int]) -> None:
        self._put("feature_consensus", {
            "top_5_features": top_features[:5],
            "tier_counts": tier_counts,
        })

    def on_iteration_update(
        self, iteration: int, best_score: float, improvement: float, models_remaining: int
    ) -> None:
        self._put("iteration_update", {
            "iteration": iteration,
            "best_score": best_score,
            "improvement": improvement,
            "models_remaining": models_remaining,
        })
