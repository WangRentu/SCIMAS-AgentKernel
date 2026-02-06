import random
from typing import Dict, Any

from agentkernel_standalone.mas.agent.base.plugin_base import PlanPlugin
from agentkernel_standalone.mas.agent.components import *
from agentkernel_standalone.toolkit.logger import get_logger


logger = get_logger(__name__)

# 基于概率的plan plugin，根据概率化的policy选择action
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
        observations = await self.state_plug.get_state("observations") or []
        notes = await self.state_plug.get_state("notes") or []

        action = self._sample_action(policy)

        if not hypothesis and action in ("write", "review"):
            action = "read"
        if exp_count >= budget and action == "experiment":
            action = "write"

        plan_item: Dict[str, Any] = {"action": action}

        if action == "experiment":
            intervention = await self._propose_intervention(hypothesis, observations, notes)
            if intervention:
                plan_item["intervention"] = intervention

        self.plan.append(plan_item)
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

    async def _propose_intervention(self, hypothesis, observations, notes) -> Dict[str, float]:
        """
        Simple heuristic:
        - Prefer variables not yet in hypothesis.
        - Use literature hints to pick a likely parent if available.
        - Otherwise pick the most unexplored variable.
        """
        spec = await self._component.agent.controller.run_environment("science", "get_world_spec")
        target = spec.get("target")
        vars_all = [v for v in spec.get("variables", []) if v != target]

        # Count how many times each var was intervened
        int_counts = {v: 0 for v in vars_all}
        for obs in observations:
            intervention = (obs or {}).get("intervention") or {}
            for v, delta in intervention.items():
                if delta:
                    int_counts[v] = int_counts.get(v, 0) + 1

        # Literature priors
        lit_score = {v: 0.0 for v in vars_all}
        for note in notes:
            for hint in (note or {}).get("hints", []) or []:
                for v in vars_all:
                    if hint.startswith(f"{v} likely causes {target}."):
                        lit_score[v] += 1.0
                    elif hint.startswith(f"No evidence that {v} causes {target}."):
                        lit_score[v] -= 1.0

        candidates = vars_all
        # Prefer not-yet-in-hypothesis
        prefer = [v for v in candidates if v not in hypothesis] or candidates

        # Score = literature bonus - explored count
        scored = sorted(prefer, key=lambda v: (lit_score.get(v, 0.0), -int_counts.get(v, 0)), reverse=True)
        if not scored:
            return {}

        chosen = scored[0]
        delta = 1.0 if self._rng.random() < 0.5 else -1.0
        return {chosen: delta}
