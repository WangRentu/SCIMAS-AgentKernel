import os
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
        self._verbose_invoke_logs = os.getenv("SCIMAS_VERBOSE_INVOKE_LOGS", "0").lower() in {"1", "true", "yes"}
    async def init(self): 
        self.agent_id = self._component.agent.agent_id
        self.plan_comp = self._component.agent.get_component('plan')
        self.plan_plug = self.plan_comp._plugin
        self.controller = self._component.agent.controller
        state_comp = self._component.agent.get_component("state")
        self.state_plug = state_comp._plugin
    async def execute(self, current_tick: int):
        executed = False
        self.plans = self.plan_plug.plan
        for plan in self.plans:
            action = plan.get("action")
            if not action or action == "idle":
                continue
            result = await self._run_research_action(action, plan)
            await self._record_result(action, result)
            executed = True
        if not executed:
            await self._clear_last_result()

    async def _run_research_action(self, action: str, plan: Dict[str, Any]) -> ActionResult:
        payload = {k: v for k, v in plan.items() if k not in ("action", "greedy_improve")}
        payload["agent_id"] = self.agent_id
        if plan.get("greedy_improve"):
            config = payload.get("config") or {}
            if not isinstance(config, dict):
                config = {}
            config["greedy_improve"] = True
            payload["config"] = config
        return await self.controller.run_action("otheractions", action, **payload)

    async def _clear_last_result(self) -> None:
        await self.state_plug.set_state("last_action", None)
        await self.state_plug.set_state("last_effective_action", None)
        await self.state_plug.set_state("last_reward", 0.0)
        await self.state_plug.set_state("last_learning_reward", 0.0)
        await self.state_plug.set_state("last_reward_components", {})
        await self.state_plug.set_state("last_action_ok", None)

    async def _record_result(self, action: str, result: ActionResult) -> None:
        if not isinstance(result, ActionResult):
            await self._clear_last_result()
            return
        payload = result.data if isinstance(result.data, dict) else {}
        effective_action = payload.get("effective_action") or action
        reward_raw = payload.get("reward", 0.0)
        try:
            reward = float(reward_raw or 0.0)
        except (TypeError, ValueError):
            reward = 0.0
        reward_components = payload.get("reward_components") or {}
        learning_reward = self._derive_learning_reward(
            action=action,
            effective_action=effective_action,
            reward=reward,
            reward_components=reward_components,
        )
        action_ok = bool(result.is_successful() and payload.get("ok", True))
        action_seq = int((await self.state_plug.get_state("last_action_seq")) or 0) + 1
        await self.state_plug.set_state("last_action", action)
        await self.state_plug.set_state("last_effective_action", effective_action)
        await self.state_plug.set_state("last_reward", reward)
        await self.state_plug.set_state("last_learning_reward", learning_reward)
        await self.state_plug.set_state("last_reward_components", reward_components)
        await self.state_plug.set_state("last_action_ok", action_ok)
        await self.state_plug.set_state("last_action_seq", action_seq)
        if self._verbose_invoke_logs:
            logger.info(f"Agent {self.agent_id} executed {action} with status {result.status}")

    @staticmethod
    def _derive_learning_reward(
        *,
        action: str,
        effective_action: str,
        reward: float,
        reward_components: Dict[str, Any],
    ) -> float:
        if isinstance(reward_components, dict):
            explicit = reward_components.get("learning_reward")
            if explicit is not None:
                return float(explicit)
            if action == "complete_task":
                return float(reward_components.get("inner_reward", 0.0) or 0.0)
        return reward
