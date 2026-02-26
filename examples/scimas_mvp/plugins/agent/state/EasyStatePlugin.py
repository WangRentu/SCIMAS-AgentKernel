from typing import Dict, Any, Optional, List

import copy
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
        self._default_action_space = action_space or [
            "read",
            "hypothesize",
            "experiment",
            "replicate",
            "write",
            "review",
            "share_evidence",
            "share_observation",
            "claim_task",
            "complete_task",
        ]
        self._default_policy = policy or {
            "read": 0.18,
            "hypothesize": 0.14,
            "experiment": 0.22,
            "replicate": 0.02,
            "write": 0.14,
            "review": 0.08,
            "share_evidence": 0.06,
            "share_observation": 0.06,
            "claim_task": 0.06,
            "complete_task": 0.04,
        }
        self._state_data.setdefault("policy", copy.deepcopy(self._default_policy))
        self._state_data.setdefault("alpha", alpha)
        self._state_data.setdefault("beta", beta)
        self._state_data.setdefault("action_space", list(self._default_action_space))
        self._state_data.setdefault("budget", budget)
        self._state_data.setdefault("exp_count", 0)
        self._state_data.setdefault("observations", [])
        self._state_data.setdefault("replications", [])
        self._state_data.setdefault("shared_observations", [])
        self._state_data.setdefault("hypothesis", [])
        self._state_data.setdefault("notes", [])
        self._state_data.setdefault("shared_notes", [])
        self._state_data.setdefault("inbox_evidence", [])
        self._state_data.setdefault("last_action", None)
        self._state_data.setdefault("last_effective_action", None)
        self._state_data.setdefault("last_reward", 0.0)
        self._state_data.setdefault("last_learning_reward", 0.0)
        self._state_data.setdefault("last_reward_components", {})
        self._state_data.setdefault("last_fitness", None)
        self._state_data.setdefault("last_paper_id", None)
        self._state_data.setdefault("current_task_id", None)
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
        logger.info(f"Agent {self.agent_id} state snapshot: {self._state_data}")
    
    async def set_state(self, key: str, value: Any):
        self._state_data[key] = value
        logger.info(f"Agent {self.agent_id} state update: {key} -> {value}")

    async def get_state(self, key: str) -> Any:
        return self._state_data.get(key)
