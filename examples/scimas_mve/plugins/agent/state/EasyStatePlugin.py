from typing import Dict, Any, Optional, List

import os
from agentkernel_standalone.toolkit.logger import get_logger
from agentkernel_standalone.mas.agent.base.plugin_base import StatePlugin

logger = get_logger(__name__)

class EasyStatePlugin(StatePlugin):
    def __init__(
        self,
        state_data: Optional[Dict[str, Any]] = None,
        policy: Optional[Dict[str, float]] = None,
        alpha: float = 0.3,
        beta: float = 2.0,
        action_space: Optional[List[str]] = None,
        budget: int = 10,
    ):
        super().__init__()
        self._state_data: Dict[str, Any] = state_data if state_data is not None else {}
        self.agent_id = None
        log_mode = (os.getenv("SCIMAS_LOG_MODE", "compact") or "compact").strip().lower()
        self._verbose_state_logs = os.getenv(
            "SCIMAS_VERBOSE_STATE_LOGS",
            "1" if log_mode == "verbose" else "0",
        ).lower() in {"1", "true", "yes"}
        self._state_data.setdefault("budget", budget)
        self._state_data.setdefault("phase_status", {})
        self._state_data.setdefault("last_action_ok", False)
        self._state_data.setdefault("exp_count", 0)
        self._state_data.setdefault("observations", [])
        self._state_data.setdefault("replications", [])
        self._state_data.setdefault("shared_observations", [])
        self._state_data.setdefault("hypothesis", [])
        self._state_data.setdefault("notes", [])
        self._state_data.setdefault("shared_notes", [])
        self._state_data.setdefault("data_card", None)
        self._state_data.setdefault("prepare_data_ready", False)
        self._state_data.setdefault("method_card", None)
        self._state_data.setdefault("inbox_evidence", [])
        self._state_data.setdefault("last_action", None)
        self._state_data.setdefault("last_effective_action", None)
        self._state_data.setdefault("last_action_seq", 0)
        self._state_data.setdefault("last_reflected_seq", 0)
        self._state_data.setdefault("last_reward", 0.0)
        self._state_data.setdefault("last_learning_reward", 0.0)
        self._state_data.setdefault("last_reward_components", {})
        self._state_data.setdefault("last_fitness", None)
        self._state_data.setdefault("last_paper_id", None)
        self._state_data.setdefault("current_task_id", None)
        self._state_data.setdefault("claim_backoff_until_tick", 0)
        self._state_data.setdefault("claim_fail_streak", 0)
        self._state_data.setdefault("llm_role_plan", None)
        self._state_data.setdefault("episode_reward_ledger", {})
        self._state_data.setdefault("episode_action_counts", {})
        self._state_data.setdefault("credit_buffer", 0.0)
        self._state_data.setdefault("contribution_credit_total", 0.0)
        self._state_data.setdefault("share_sent_evidence_count", 0)
        self._state_data.setdefault("share_sent_observation_count", 0)
        self._state_data.setdefault("paper_write_count", 0)
        self._state_data.setdefault("review_count", 0)
        self._state_data.setdefault("replication_count", 0)
        self._state_data.setdefault("last_selection_score", None)
        
    async def init(self):
        self.agent_id = self._component.agent.agent_id
        
    async def execute(self, current_tick: int):
        if self._verbose_state_logs:
            logger.info(f"Agent {self.agent_id} state snapshot: {self._state_data}")
    
    async def set_state(self, key: str, value: Any):
        self._state_data[key] = value
        if self._verbose_state_logs:
            logger.info(f"Agent {self.agent_id} state update: {key} -> {value}")

    async def get_state(self, key: str) -> Any:
        return self._state_data.get(key)
