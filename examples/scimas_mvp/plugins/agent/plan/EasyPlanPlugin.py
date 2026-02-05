import random
from typing import Dict, Any

from agentkernel_standalone.mas.agent.base.plugin_base import PlanPlugin
from agentkernel_standalone.mas.agent.components import *
from agentkernel_standalone.toolkit.logger import get_logger


logger = get_logger(__name__)

class EasyPlanPlugin(PlanPlugin):
    """
    A minimal research planner that samples actions from the policy vector.
    """
    
    def __init__(self):
        super().__init__()
        self.plan = []
        self._rng = random.Random()
        logger.info("EasyPlanPlugin initialized")
        
    async def init(self):
        self.agent_id = self._component.agent.agent_id
        self.state_comp: StateComponent = self._component.agent.get_component("state")
        self.state_plug = self.state_comp._plugin

    async def execute(self, current_tick: int) -> Dict[str, Any]:
        self.plan.clear()
        policy = await self.state_plug.get_state("policy") or {}
        hypothesis = await self.state_plug.get_state("hypothesis") or []
        exp_count = await self.state_plug.get_state("exp_count") or 0
        budget = await self.state_plug.get_state("budget") or 10

        action = self._sample_action(policy)

        if not hypothesis and action in ("write", "review"):
            action = "read"
        if exp_count >= budget and action == "experiment":
            action = "write"

        self.plan.append({"action": action})
        logger.info(f"Agent {self.agent_id} planned action: {action}")

    def _sample_action(self, policy: Dict[str, float]) -> str:
        if not policy:
            return "read"
        total = sum(policy.values())
        if total <= 0:
            return next(iter(policy.keys()))
        pick = self._rng.random() * total
        current = 0.0
        for action, prob in policy.items():
            current += prob
            if pick <= current:
                return action
        return next(iter(policy.keys()))
