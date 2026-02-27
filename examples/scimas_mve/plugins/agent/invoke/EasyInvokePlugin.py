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
                effective_action = result.data.get("effective_action") or action
                reward_components = result.data.get("reward_components") or {}
                learning_reward = self._derive_learning_reward(
                    action=action,
                    effective_action=effective_action,
                    reward=float(reward),
                    reward_components=reward_components,
                )
                await self.state_plug.set_state("last_action", action)
                await self.state_plug.set_state("last_effective_action", effective_action)
                await self.state_plug.set_state("last_reward", reward)
                await self.state_plug.set_state("last_learning_reward", learning_reward)
                await self.state_plug.set_state("last_reward_components", reward_components)
                await self._update_episode_ledgers(effective_action, float(reward), reward_components)
        if self._verbose_invoke_logs:
            logger.info(f"Agent {self.agent_id} executed {action} with status {result.status}")

    def _derive_learning_reward(
        self,
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
        # Claiming a task is a workflow step; avoid overfitting policy to claim bonuses.
        if action == "claim_task" and effective_action == "claim_task":
            return 0.0
        return reward

    async def _update_episode_ledgers(
        self,
        effective_action: str,
        reward: float,
        reward_components: Dict[str, Any],
    ) -> None:
        ledger = await self.state_plug.get_state("episode_reward_ledger") or {}
        counts = await self.state_plug.get_state("episode_action_counts") or {}
        counts[effective_action] = int(counts.get(effective_action, 0) or 0) + 1
        ledger["total_reward"] = float(ledger.get("total_reward", 0.0) or 0.0) + reward
        if isinstance(reward_components, dict):
            for key, value in reward_components.items():
                if not isinstance(value, (int, float)):
                    continue
                ledger[key] = float(ledger.get(key, 0.0) or 0.0) + float(value)
        await self.state_plug.set_state("episode_reward_ledger", ledger)
        await self.state_plug.set_state("episode_action_counts", counts)
