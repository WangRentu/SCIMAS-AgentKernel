from typing import Dict, Any
import math

from agentkernel_standalone.mas.agent.base.plugin_base import ReflectPlugin
from agentkernel_standalone.toolkit.logger import get_logger


logger = get_logger(__name__)

class EasyReflectPlugin(ReflectPlugin):
    def __init__(self):
        super().__init__()
        self.agent_id = None
        self.state_plug = None
        
    async def init(self):
        self.agent_id = self._component.agent.agent_id 
        state_comp = self._component.agent.get_component("state")
        self.state_plug = state_comp._plugin
        
    async def execute(self, current_tick: int) -> Dict[str, Any]:
        policy = await self.state_plug.get_state("policy") or {}
        action_space = await self.state_plug.get_state("action_space") or list(policy.keys())
        last_action = await self.state_plug.get_state("last_action")
        last_reward = await self.state_plug.get_state("last_reward") or 0.0
        alpha = await self.state_plug.get_state("alpha") or 0.3
        beta = await self.state_plug.get_state("beta") or 2.0

        if not policy or not last_action:
            logger.info(f"Agent {self.agent_id} has no policy update this tick.")
            return {}

        rewards = {action: (last_reward if action == last_action else 0.0) for action in action_space}
        exp_vals = {action: math.exp(beta * reward) for action, reward in rewards.items()}
        denom = sum(exp_vals.values()) or 1.0
        softmax = {action: (value / denom) for action, value in exp_vals.items()}

        updated = {}
        for action in action_space:
            old_p = policy.get(action, 0.0)
            updated[action] = (1 - alpha) * old_p + alpha * softmax.get(action, 0.0)

        total = sum(updated.values())
        if total > 0:
            updated = {action: value / total for action, value in updated.items()}

        await self.state_plug.set_state("policy", updated)
        logger.info(f"Agent {self.agent_id} updated policy: {updated}")
        return {"policy": updated, "last_action": last_action, "last_reward": last_reward}
