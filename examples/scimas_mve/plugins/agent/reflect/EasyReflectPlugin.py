from typing import Dict, Any
import os

from agentkernel_standalone.mas.agent.base.plugin_base import ReflectPlugin
from agentkernel_standalone.toolkit.logger import get_logger


logger = get_logger(__name__)

PIPELINE_PHASES = {
    "read", "prepare_data", "profile_data", "retrieve_literature",
    "hypothesize", "experiment", "review", "write", "replicate",
}


class EasyReflectPlugin(ReflectPlugin):
    def __init__(self):
        super().__init__()
        self.agent_id = None
        self.state_plug = None
        log_mode = (os.getenv("SCIMAS_LOG_MODE", "compact") or "compact").strip().lower()
        self._verbose_reflect_logs = os.getenv(
            "SCIMAS_VERBOSE_REFLECT_LOGS",
            "1" if log_mode == "verbose" else "0",
        ).lower() in {"1", "true", "yes"}

    async def init(self):
        self.agent_id = self._component.agent.agent_id
        state_comp = self._component.agent.get_component("state")
        self.state_plug = state_comp._plugin

    async def execute(self, current_tick: int) -> Dict[str, Any]:
        last_action = await self.state_plug.get_state("last_action")
        last_effective_action = await self.state_plug.get_state("last_effective_action") or last_action
        last_reward = await self.state_plug.get_state("last_reward") or 0.0
        last_reward_components = await self.state_plug.get_state("last_reward_components") or {}
        last_action_seq = int((await self.state_plug.get_state("last_action_seq")) or 0)
        last_reflected_seq = int((await self.state_plug.get_state("last_reflected_seq")) or 0)
        phase_status: Dict[str, Dict[str, Any]] = await self.state_plug.get_state("phase_status") or {}

        if not last_action or last_action_seq <= last_reflected_seq:
            if self._verbose_reflect_logs:
                logger.info(f"Agent {self.agent_id} has no action to reflect on.")
            return phase_status

        if last_effective_action in PIPELINE_PHASES:
            entry = phase_status.setdefault(last_effective_action, {
                "runs": 0, "successes": 0, "last_result": None, "best_score": None,
            })
            # Patch entries that were created before best_score was added to the schema
            entry.setdefault("best_score", None)
            entry["runs"] += 1

            last_action_ok = await self.state_plug.get_state("last_action_ok")
            if last_action_ok is not None:
                success = bool(last_action_ok)
            else:
                success = last_reward > 0

            if success:
                entry["successes"] += 1
            entry["last_result"] = "success" if success else "failed"

            if last_effective_action == "experiment":
                score = last_reward_components.get("experiment_selection_score")
                if score is None:
                    score = last_reward_components.get("dev_score_norm")
                if score is None:
                    score = last_reward if last_reward > 0 else None
                current_best = entry.get("best_score")
                if score is not None and (current_best is None or score > current_best):
                    entry["best_score"] = score

            await self.state_plug.set_state("phase_status", phase_status)

            if self._verbose_reflect_logs:
                logger.info(
                    f"Agent {self.agent_id} phase_status[{last_effective_action}]: "
                    f"runs={entry['runs']} successes={entry['successes']} last={entry['last_result']}"
                )

        reward_ledger: Dict[str, float] = await self.state_plug.get_state("episode_reward_ledger") or {}
        action_counts: Dict[str, int] = await self.state_plug.get_state("episode_action_counts") or {}

        if last_effective_action:
            reward_ledger[last_effective_action] = reward_ledger.get(last_effective_action, 0.0) + last_reward
            action_counts[last_effective_action] = action_counts.get(last_effective_action, 0) + 1
            await self.state_plug.set_state("episode_reward_ledger", reward_ledger)
            await self.state_plug.set_state("episode_action_counts", action_counts)
        await self.state_plug.set_state("last_reflected_seq", last_action_seq)

        return phase_status
