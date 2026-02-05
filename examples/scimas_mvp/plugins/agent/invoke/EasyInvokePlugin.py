from typing import List, Dict, Any
from agentkernel_standalone.mas.agent.base.plugin_base import InvokePlugin
from agentkernel_standalone.toolkit.logger import get_logger
from agentkernel_standalone.types.schemas.action import ActionResult

logger = get_logger(__name__)

class EasyInvokePlugin(InvokePlugin):
    """
    Execute the planned research action via the action component.
    """
    def __init__(self):
        super().__init__()
        self.agent_id = None
        self.plan_comp = None
        self.plans = []
        self.controller = None
        self.state_plug = None
    async def init(self): 
        self.agent_id = self._component.agent.agent_id
        self.plan_comp = self._component.agent.get_component('plan')
        self.plan_plug = self.plan_comp._plugin
        self.controller = self._component.agent.controller
        state_comp = self._component.agent.get_component("state")
        self.state_plug = state_comp._plugin
    async def execute(self, current_tick: int):
        
        self.plans = self.plan_plug.plan
        for plan in self.plans:
            action = plan.get("action")
            if not action:
                continue
            result = await self._run_research_action(action, plan)
            await self._record_result(action, result)

    async def _run_research_action(self, action: str, plan: Dict[str, Any]) -> ActionResult:
        payload = {k: v for k, v in plan.items() if k != "action"}
        payload["agent_id"] = self.agent_id
        return await self.controller.run_action("otheractions", action, **payload)

    async def _record_result(self, action: str, result: ActionResult) -> None:
        if not isinstance(result, ActionResult):
            return
        if isinstance(result.data, dict):
            reward = result.data.get("reward")
            if reward is not None:
                await self.state_plug.set_state("last_action", action)
                await self.state_plug.set_state("last_reward", reward)
        logger.info(f"Agent {self.agent_id} executed {action} with status {result.status}")
