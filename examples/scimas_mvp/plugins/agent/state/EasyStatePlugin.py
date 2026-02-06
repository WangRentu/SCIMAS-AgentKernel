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
            "write",
            "review",
        ]
        self._default_policy = policy or {
            "read": 0.25,
            "hypothesize": 0.2,
            "experiment": 0.25,
            "write": 0.2,
            "review": 0.1,
        }
        self._state_data.setdefault("policy", copy.deepcopy(self._default_policy))
        self._state_data.setdefault("alpha", alpha)
        self._state_data.setdefault("beta", beta)
        self._state_data.setdefault("action_space", list(self._default_action_space))
        self._state_data.setdefault("budget", budget)
        self._state_data.setdefault("exp_count", 0)
        self._state_data.setdefault("observations", [])
        self._state_data.setdefault("hypothesis", [])
        self._state_data.setdefault("notes", [])
        self._state_data.setdefault("last_action", None)
        self._state_data.setdefault("last_reward", 0.0)
        self._state_data.setdefault("last_fitness", None)
        
    async def init(self):
        self.agent_id = self._component.agent.agent_id
        
    async def execute(self, current_tick: int):
        logger.info(f"Agent {self.agent_id} state snapshot: {self._state_data}")
    
    async def set_state(self, key: str, value: Any):
        self._state_data[key] = value
        logger.info(f"Agent {self.agent_id} state update: {key} -> {value}")

    async def get_state(self, key: str) -> Any:
        return self._state_data.get(key)
