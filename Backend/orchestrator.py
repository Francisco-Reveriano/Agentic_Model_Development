"""Pipeline orchestrator — sequences agents and manages handoff protocol."""

from __future__ import annotations

import asyncio
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from backend.callbacks import SSECallbackHandler
from backend.config import Settings, create_anthropic_model, get_settings

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

    def _get_agent(self, agent_name: str) -> Any:
        """Import and return the agent instance by name."""
        if agent_name == "Data_Agent":
            from backend.agents.data_agent import create_data_agent
            return create_data_agent(self.settings, self._stage_dir(agent_name))
        elif agent_name == "Feature_Agent":
            from backend.agents.feature_agent import create_feature_agent
            return create_feature_agent(self.settings, self._stage_dir(agent_name))
        elif agent_name == "PD_Agent":
            from backend.agents.pd_agent import create_pd_agent
            return create_pd_agent(self.settings, self._stage_dir(agent_name))
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
            return create_report_agent(self.settings, self._stage_dir(agent_name))
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

        for idx, agent_name in enumerate(self.sequence, 1):
            callback = SSECallbackHandler(agent_name, self.event_queue)
            callback.on_agent_start(stage=idx, total_stages=total)

            started_at = datetime.now(timezone.utc).isoformat()
            t0 = time.time()

            try:
                agent = self._get_agent(agent_name)
                prompt = self._build_agent_prompt(agent_name)

                # Run the synchronous Strands agent in a thread
                result = await asyncio.to_thread(agent, prompt)

                duration = time.time() - t0
                self._write_handoff(agent_name, "success", started_at, duration)
                callback.on_agent_complete("success", duration)
                completed_agents.append(agent_name)

            except Exception as exc:
                duration = time.time() - t0
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
        await self.event_queue.put({
            "event": "pipeline_complete",
            "data": {
                "run_id": self.run_id,
                "status": "completed",
                "completed_agents": completed_agents,
                "total_duration_s": round(time.time() - pipeline_start, 1),
                "reports": [
                    str(p.name)
                    for p in (self.output_root / "07_reports").glob("*.docx")
                ],
            },
        })
