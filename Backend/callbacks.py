"""SSE callback handler for Strands agents.

Bridges synchronous Strands agent execution with the async SSE event queue
so the React frontend receives real-time progress updates.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import threading
import time
from typing import Any

logger = logging.getLogger(__name__)

# Sentence boundary pattern — flush buffer when these are found
_SENTENCE_END = re.compile(r'[.!?]\s|[\n]')


    # Map tool names to workflow substeps
TOOL_TO_SUBSTEP = {
    "list_tables": "Schema Discovery",
    "describe_table": "Schema Discovery",
    "get_data_dictionary_summary": "Schema Discovery",
    "run_baseline_data_quality_scan": "Baseline Scan",
    "profile_all_columns": "Column Profiling",
    "profile_column": "Column Profiling",
    "analyze_missing_patterns": "Missing Analysis",
    "run_outlier_detection": "Outlier Detection",
    "assess_class_imbalance": "Class Balance",
    "run_vintage_drift_analysis": "Drift Analysis",
    "emit_dq_result": "DQ Scoring",
    "write_cleaned_dataset": "Cleaning Pipeline",
}


class SSECallbackHandler:
    """Pushes structured events to an asyncio.Queue during agent execution."""

    def __init__(self, agent_name: str, event_queue: asyncio.Queue):
        self.agent_name = agent_name
        self.queue = event_queue
        self._loop = asyncio.get_event_loop()

        # Token buffering for sentence-level streaming
        self._text_buf = ""
        self._reasoning_buf = ""
        self._flush_timer: threading.Timer | None = None
        self._lock = threading.Lock()
        self._active_substeps: set[str] = set()

        # Tool call deduplication: track per-invocation ID (Strands sends
        # current_tool_use on every streaming token during input construction,
        # so we must only emit once per unique tool_use id).
        self._last_tool_use_id: str | None = None
        # Fallback debounce when no id is present
        self._last_emitted_tool: str | None = None
        self._had_non_tool_event: bool = False

    # ------------------------------------------------------------------
    # Strands callback_handler interface
    # ------------------------------------------------------------------

    def __call__(self, **kwargs: Any) -> None:
        """Strands streaming callback — receives text, tool use, and reasoning events."""
        # Text stream events (agent reasoning / output)
        data = kwargs.get("data")
        if isinstance(data, str) and data:
            self._buffer_text(data, "text")
            self._had_non_tool_event = True
            return

        # Tool use stream events — deduplicate: Strands fires current_tool_use
        # on every streaming token while building tool input. We emit once per
        # unique invocation using the tool_use id, or debounce by name.
        current_tool_use = kwargs.get("current_tool_use")
        if current_tool_use and isinstance(current_tool_use, dict):
            tool_name = current_tool_use.get("name", "")
            tool_id = current_tool_use.get("id", "") or current_tool_use.get("toolUseId", "")
            if tool_name:
                is_new_invocation = False
                if tool_id:
                    # Primary: unique id per invocation
                    if tool_id != self._last_tool_use_id:
                        self._last_tool_use_id = tool_id
                        is_new_invocation = True
                else:
                    # Fallback debounce: new name OR non-tool event in between
                    if tool_name != self._last_emitted_tool or self._had_non_tool_event:
                        is_new_invocation = True

                if is_new_invocation:
                    self._last_emitted_tool = tool_name
                    self._had_non_tool_event = False
                    self._flush_all()
                    self._put("agent_log", {
                        "content": f"Calling tool: {tool_name}",
                        "type": "tool_call",
                        "tool_name": tool_name,
                    })
                    # Emit substep event if this tool maps to a workflow substep
                    substep = TOOL_TO_SUBSTEP.get(tool_name)
                    if substep and substep not in self._active_substeps:
                        for prev in list(self._active_substeps):
                            self._put("agent_substep", {"substep": prev, "status": "completed"})
                        self._active_substeps.clear()
                        self._active_substeps.add(substep)
                        self._put("agent_substep", {"substep": substep, "status": "running"})
            return

        # Reasoning text events (thinking)
        reasoning = kwargs.get("reasoningText")
        if isinstance(reasoning, str) and reasoning:
            self._had_non_tool_event = True
            self._buffer_text(reasoning, "reasoning")
            return

    # ------------------------------------------------------------------
    # Token buffering — accumulate tokens, flush on sentence boundaries
    # ------------------------------------------------------------------

    def _buffer_text(self, token: str, log_type: str) -> None:
        with self._lock:
            if log_type == "reasoning":
                self._reasoning_buf += token
                buf = self._reasoning_buf
            else:
                self._text_buf += token
                buf = self._text_buf

            # Check for sentence boundary in the buffer
            if _SENTENCE_END.search(buf):
                self._flush_type(log_type)
            else:
                # Schedule a timeout flush (500ms) so text doesn't get stuck
                self._schedule_flush()

    def _flush_type(self, log_type: str) -> None:
        """Flush a specific buffer type. Must be called with lock held."""
        if log_type == "reasoning" and self._reasoning_buf.strip():
            self._put("agent_log", {"content": self._reasoning_buf, "type": "reasoning"})
            self._reasoning_buf = ""
        elif log_type == "text" and self._text_buf.strip():
            self._put("agent_log", {"content": self._text_buf, "type": "text"})
            self._text_buf = ""

    def _flush_all(self) -> None:
        """Flush both buffers. Thread-safe."""
        with self._lock:
            if self._reasoning_buf.strip():
                self._put("agent_log", {"content": self._reasoning_buf, "type": "reasoning"})
                self._reasoning_buf = ""
            if self._text_buf.strip():
                self._put("agent_log", {"content": self._text_buf, "type": "text"})
                self._text_buf = ""
            if self._flush_timer:
                self._flush_timer.cancel()
                self._flush_timer = None

    def _schedule_flush(self) -> None:
        """Schedule a timeout flush after 500ms of no new tokens."""
        if self._flush_timer:
            self._flush_timer.cancel()
        self._flush_timer = threading.Timer(0.5, self._flush_all)
        self._flush_timer.daemon = True
        self._flush_timer.start()

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _put(self, event_type: str, data: dict[str, Any]) -> None:
        data["agent"] = self.agent_name
        data["timestamp"] = time.time()
        payload = {"event": event_type, "data": data}
        logger.debug("SSE event: %s for agent %s", event_type, self.agent_name)
        # Schedule the put on the event loop (called from sync agent thread)
        asyncio.run_coroutine_threadsafe(self.queue.put(payload), self._loop)

    # ------------------------------------------------------------------
    # agent lifecycle events
    # ------------------------------------------------------------------

    def on_agent_start(self, stage: int, total_stages: int) -> None:
        self._put("agent_start", {"stage": stage, "total_stages": total_stages})

    def on_agent_complete(self, status: str, duration_s: float) -> None:
        self._flush_all()
        self._put("agent_complete", {"status": status, "duration_s": round(duration_s, 1)})

    def on_agent_error(self, error: str, recoverable: bool = False) -> None:
        self._flush_all()
        logger.error("Agent %s error: %s (recoverable=%s)", self.agent_name, error, recoverable)
        self._put("agent_error", {"error": error, "recoverable": recoverable})

    # ------------------------------------------------------------------
    # streaming log events
    # ------------------------------------------------------------------

    def on_log(self, content: str, log_type: str = "text") -> None:
        self._put("agent_log", {"content": content, "type": log_type})

    def on_tool_start(self, tool_name: str, tool_input: dict[str, Any] | None = None) -> None:
        self._flush_all()
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

    # ------------------------------------------------------------------
    # DQ scorecard events (Improvement 7)
    # ------------------------------------------------------------------

    def on_dq_test(
        self, test_id: str, test_name: str, status: str,
        value: str, threshold: str, evidence: str = "",
    ) -> None:
        self._put("dq_scorecard_update", {
            "test_id": test_id,
            "test_name": test_name,
            "status": status,
            "value": value,
            "threshold": threshold,
            "evidence": evidence,
        })

    # ------------------------------------------------------------------
    # Structured table events (Improvement 9)
    # ------------------------------------------------------------------

    def on_table(self, table_name: str, columns: list[str], rows: list[list]) -> None:
        self._put("agent_table", {
            "table_name": table_name,
            "columns": columns,
            "rows": rows[:50],  # Limit to 50 rows
        })

    # ------------------------------------------------------------------
    # Chart data events (Improvement 10)
    # ------------------------------------------------------------------

    def on_chart_data(self, chart_name: str, data: list[dict]) -> None:
        self._put("agent_chart_data", {
            "chart_name": chart_name,
            "data": data[:100],  # Limit data points
        })
