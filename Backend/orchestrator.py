"""Pipeline orchestrator — sequences agents and manages handoff protocol."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from backend.callbacks import SSECallbackHandler
from backend.config import Settings, create_anthropic_model, get_settings

logger = logging.getLogger(__name__)

# Agent stage directory names
STAGE_DIRS = {
    "Data_Agent": "01_data_quality",
    "Feature_Agent": "02_features",
    "PD_Agent": "03_pd_model",
    "LGD_Agent": "04_lgd_model",
    "EAD_Agent": "05_ead_model",
    "EL_Agent": "06_expected_loss",
    "Report_Agent": "07_reports",
}

# Pipeline sequences by model selection
PIPELINE_SEQUENCES = {
    frozenset(["PD"]): [
        "Data_Agent", "Feature_Agent", "PD_Agent", "Report_Agent",
    ],
    frozenset(["LGD"]): [
        "Data_Agent", "Feature_Agent", "LGD_Agent", "Report_Agent",
    ],
    frozenset(["EAD"]): [
        "Data_Agent", "Feature_Agent", "EAD_Agent", "Report_Agent",
    ],
    frozenset(["PD", "LGD", "EAD", "EL"]): [
        "Data_Agent", "Feature_Agent", "PD_Agent", "LGD_Agent",
        "EAD_Agent", "EL_Agent", "Report_Agent",
    ],
}


class PipelineOrchestrator:
    """Runs the agent pipeline, writing handoff.json between stages."""

    def __init__(
        self,
        run_id: str,
        models: list[str],
        event_queue: asyncio.Queue,
        settings: Settings | None = None,
        db_path_override: str | None = None,
        config_overrides: dict[str, Any] | None = None,
    ):
        self.run_id = run_id
        self.models = [m.upper() for m in models]
        self.event_queue = event_queue
        self.settings = settings or get_settings()
        self.db_path_override = db_path_override
        self.config_overrides = config_overrides or {}

        self.output_root = self.settings.output_abs_path / run_id
        self.sequence = self._resolve_sequence()

        from backend.logging_config import add_pipeline_log_handler

        self._run_log_handler = add_pipeline_log_handler(self.output_root)
        logger.info("Pipeline %s initialized -- agents=%s, models=%s", run_id, self.sequence, self.models)

    # ------------------------------------------------------------------

    def _resolve_sequence(self) -> list[str]:
        key = frozenset(self.models)
        if key in PIPELINE_SEQUENCES:
            return PIPELINE_SEQUENCES[key]
        # Fallback: build custom sequence
        seq = ["Data_Agent", "Feature_Agent"]
        for m in ["PD", "LGD", "EAD"]:
            if m in self.models:
                seq.append(f"{m}_Agent")
        if "EL" in self.models:
            seq.append("EL_Agent")
        seq.append("Report_Agent")
        return seq

    def _create_dirs(self) -> None:
        for agent_name in self.sequence:
            stage_dir = self.output_root / STAGE_DIRS.get(agent_name, agent_name)
            stage_dir.mkdir(parents=True, exist_ok=True)

    def _stage_dir(self, agent_name: str) -> Path:
        return self.output_root / STAGE_DIRS.get(agent_name, agent_name)

    def _read_handoff(self, agent_name: str) -> dict[str, Any]:
        handoff_path = self._stage_dir(agent_name) / "handoff.json"
        if handoff_path.exists():
            return json.loads(handoff_path.read_text())
        return {}

    def _write_handoff(
        self,
        agent_name: str,
        status: str,
        started_at: str,
        duration_s: float,
        output_files: dict | None = None,
        metrics: dict | None = None,
        errors: list | None = None,
    ) -> None:
        handoff = {
            "agent": agent_name,
            "status": status,
            "started_at": started_at,
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "duration_s": round(duration_s, 1),
            "output_files": output_files or {},
            "metrics": metrics or {},
            "errors": errors or [],
        }
        path = self._stage_dir(agent_name) / "handoff.json"
        path.write_text(json.dumps(handoff, indent=2, default=str))

    # ------------------------------------------------------------------

    def _get_agent(self, agent_name: str, callback: Any = None) -> Any:
        """Import and return the agent instance by name."""
        if agent_name == "Data_Agent":
            from backend.agents.data_agent import create_data_agent
            return create_data_agent(self.settings, self._stage_dir(agent_name), callback_handler=callback)
        elif agent_name == "Feature_Agent":
            from backend.agents.feature_agent import create_feature_agent
            return create_feature_agent(
                self.settings,
                self._stage_dir(agent_name),
                data_handoff_dir=self._stage_dir("Data_Agent"),
                callback_handler=callback,
            )
        elif agent_name == "PD_Agent":
            from backend.agents.pd_agent import create_pd_agent
            return create_pd_agent(self.settings, self._stage_dir(agent_name), callback_handler=callback)
        elif agent_name == "LGD_Agent":
            from backend.agents.lgd_agent import create_lgd_agent
            return create_lgd_agent(self.settings, self._stage_dir(agent_name))
        elif agent_name == "EAD_Agent":
            from backend.agents.ead_agent import create_ead_agent
            return create_ead_agent(self.settings, self._stage_dir(agent_name))
        elif agent_name == "EL_Agent":
            from backend.agents.el_agent import create_el_agent
            return create_el_agent(self.settings, self._stage_dir(agent_name))
        elif agent_name == "Report_Agent":
            from backend.agents.report_agent import create_report_agent
            return create_report_agent(self.settings, self._stage_dir(agent_name), callback_handler=callback)
        else:
            raise ValueError(f"Unknown agent: {agent_name}")

    def _build_agent_prompt(self, agent_name: str) -> str:
        """Build the execution prompt for each agent, including handoff context."""
        stage_dir = self._stage_dir(agent_name)
        prior_handoffs = {}
        for prev_agent in self.sequence:
            if prev_agent == agent_name:
                break
            ho = self._read_handoff(prev_agent)
            if ho:
                prior_handoffs[prev_agent] = ho

        prompt = (
            f"Execute your full pipeline stage.\n"
            f"Run ID: {self.run_id}\n"
            f"Output directory: {stage_dir}\n"
            f"Models requested: {self.models}\n"
        )
        if prior_handoffs:
            prompt += f"\nPrior agent handoffs:\n{json.dumps(prior_handoffs, indent=2, default=str)}\n"
        prompt += "\nProceed with your complete workflow. Write handoff.json when done."
        return prompt

    # ------------------------------------------------------------------
    # Post-agent fallback: ensure critical output files exist
    # ------------------------------------------------------------------

    @staticmethod
    def _call_tool(tool_func, **kwargs) -> dict:
        """Call a Strands @tool-decorated function directly with kwargs."""
        # Access the raw function if wrapped by @tool decorator
        fn = getattr(tool_func, "_tool_func", tool_func)
        return fn(**kwargs)

    def _ensure_data_agent_outputs(self) -> None:
        """If Data_Agent didn't call write_cleaned_dataset, call it directly."""
        stage_dir = self._stage_dir("Data_Agent")
        if (stage_dir / "cleaned_features.parquet").exists():
            return  # Agent produced outputs correctly

        logger.warning("Data_Agent did not produce parquet files — running write_cleaned_dataset fallback")
        from backend.tools.data_tools import write_cleaned_dataset
        result = self._call_tool(write_cleaned_dataset, output_dir=str(stage_dir), run_id=self.run_id)
        logger.info("write_cleaned_dataset fallback result: %s", result.get("status", "unknown"))

    def _ensure_feature_agent_outputs(self) -> None:
        """If Feature_Agent didn't produce outputs, run feature pipeline directly."""
        feature_dir = self._stage_dir("Feature_Agent")
        data_dir = self._stage_dir("Data_Agent")

        if (feature_dir / "feature_matrix.parquet").exists():
            return  # Agent produced outputs correctly

        logger.warning("Feature_Agent did not produce feature_matrix — running fallback pipeline")
        from backend.tools.feature_tools import (
            engineer_ratio_features,
            select_features,
            write_feature_matrix,
        )

        # Step 1: Engineer ratio features (writes back to data_dir's cleaned_features.parquet)
        result = self._call_tool(engineer_ratio_features, data_dir=str(data_dir))
        logger.info("engineer_ratio_features fallback: %s", result.get("status", "unknown"))

        # Step 2: Select features (writes feature_matrix.parquet to data_dir)
        result = self._call_tool(select_features, method="combined", threshold=0.02, data_dir=str(data_dir))
        logger.info("select_features fallback: %s", result.get("status", "unknown"))

        # Step 3: Write feature matrix to feature_dir
        result = self._call_tool(write_feature_matrix, output_dir=str(feature_dir), data_dir=str(data_dir))
        logger.info("write_feature_matrix fallback: %s", result.get("status", "unknown"))

    # ------------------------------------------------------------------

    async def run(self) -> None:
        """Execute the full pipeline."""
        self._create_dirs()
        total = len(self.sequence)

        await self.event_queue.put({
            "event": "pipeline_start",
            "data": {
                "run_id": self.run_id,
                "agents": self.sequence,
                "models": self.models,
                "total_stages": total,
            },
        })

        completed_agents: list[str] = []
        pipeline_start = time.time()

        try:
            for idx, agent_name in enumerate(self.sequence, 1):
                callback = SSECallbackHandler(agent_name, self.event_queue)
                callback.on_agent_start(stage=idx, total_stages=total)
                logger.info("Starting agent %s (%d/%d)", agent_name, idx, total)

                started_at = datetime.now(timezone.utc).isoformat()
                t0 = time.time()

                try:
                    # Set module-level callback so tools can emit SSE events
                    from backend.tools.data_tools import set_callback_handler
                    set_callback_handler(callback)

                    agent = self._get_agent(agent_name, callback=callback)
                    prompt = self._build_agent_prompt(agent_name)

                    # Run the synchronous Strands agent in a thread
                    result = await asyncio.to_thread(agent, prompt)

                    # Run post-agent fallbacks to ensure critical outputs exist
                    if agent_name == "Data_Agent":
                        await asyncio.to_thread(self._ensure_data_agent_outputs)
                    elif agent_name == "Feature_Agent":
                        await asyncio.to_thread(self._ensure_feature_agent_outputs)

                    duration = time.time() - t0
                    # Only write orchestrator handoff if agent didn't write its own
                    existing = self._read_handoff(agent_name)
                    if not existing or not existing.get("output_files"):
                        self._write_handoff(agent_name, "success", started_at, duration)
                    else:
                        # Merge timing metadata into agent-written handoff
                        existing["started_at"] = started_at
                        existing["completed_at"] = datetime.now(timezone.utc).isoformat()
                        existing["duration_s"] = round(duration, 1)
                        path = self._stage_dir(agent_name) / "handoff.json"
                        path.write_text(json.dumps(existing, indent=2, default=str))
                    callback.on_agent_complete("success", duration)
                    completed_agents.append(agent_name)
                    logger.info("Agent %s completed in %.1fs", agent_name, duration)

                except Exception as exc:
                    duration = time.time() - t0
                    logger.exception("Agent %s failed after %.1fs", agent_name, duration)
                    self._write_handoff(
                        agent_name, "failed", started_at, duration,
                        errors=[str(exc)],
                    )
                    callback.on_agent_error(str(exc), recoverable=False)
                    # Halt pipeline on failure
                    await self.event_queue.put({
                        "event": "pipeline_complete",
                        "data": {
                            "run_id": self.run_id,
                            "status": "failed",
                            "failed_agent": agent_name,
                            "completed_agents": completed_agents,
                            "total_duration_s": round(time.time() - pipeline_start, 1),
                            "error": str(exc),
                        },
                    })
                    return

            # All agents completed successfully
            total_duration = round(time.time() - pipeline_start, 1)
            logger.info("Pipeline %s finished -- %d agents, %.1fs", self.run_id, len(completed_agents), total_duration)
            await self.event_queue.put({
                "event": "pipeline_complete",
                "data": {
                    "run_id": self.run_id,
                    "status": "completed",
                    "completed_agents": completed_agents,
                    "total_duration_s": total_duration,
                    "reports": [
                        str(p.name)
                        for p in (self.output_root / "07_reports").glob("*.docx")
                    ],
                },
            })
        finally:
            logging.getLogger().removeHandler(self._run_log_handler)
            self._run_log_handler.close()
