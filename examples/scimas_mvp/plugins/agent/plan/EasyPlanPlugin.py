import random
import json
from typing import Dict, Any, List, Optional

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
        self.perceive_comp: PerceiveComponent = self._component.agent.get_component("perceive")
        self.perceive_plug = self.perceive_comp._plugin

    async def execute(self, current_tick: int) -> Dict[str, Any]:
        self.plan.clear()
        await self._ingest_shared_messages()
        policy = await self.state_plug.get_state("policy") or {}
        hypothesis = await self.state_plug.get_state("hypothesis") or []
        exp_count = await self.state_plug.get_state("exp_count") or 0
        budget = await self.state_plug.get_state("budget") or 10
        observations = await self.state_plug.get_state("observations") or []
        notes = await self.state_plug.get_state("notes") or []
        shared_notes = await self.state_plug.get_state("shared_notes") or []
        shared_observations = await self.state_plug.get_state("shared_observations") or []

        task_plan = await self._plan_from_taskboard(
            hypothesis=hypothesis,
            observations=observations,
            notes=notes,
        )
        if task_plan:
            self.plan.append(task_plan)
            logger.info(f"Agent {self.agent_id} planned taskboard action: {task_plan.get('action')}")
            return {"plan": self.plan}

        action = self._sample_action(policy)

        if not hypothesis and action in ("write", "review"):
            action = "read"
        if exp_count >= budget and action in ("experiment", "replicate"):
            action = "write"
        if action == "share_evidence" and not notes:
            action = "read"
        if action == "share_observation" and not observations:
            action = "experiment"
        if action == "read" and (len(notes) + len(shared_notes)) > 2 and (len(observations) + len(shared_observations)) < budget:
            action = "experiment"

        plan_item: Dict[str, Any] = {"action": action}

        if action in ("experiment", "replicate"):
            intervention = await self._propose_intervention(hypothesis, observations, notes)
            if intervention:
                plan_item["intervention"] = intervention

        self.plan.append(plan_item)
        logger.info(f"Agent {self.agent_id} planned action: {action}")
        return {"plan": self.plan}

    def _preferred_task_types(self) -> List[str]:
        ordered = ["read", "experiment", "write", "replicate", "review", "hypothesize"]
        if not ordered:
            return []
        offset = abs(hash(self.agent_id)) % len(ordered)
        return ordered[offset:] + ordered[:offset]

    def _choose_open_task(self, tasks: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if not tasks:
            return None
        preferred = self._preferred_task_types()
        for task_type in preferred:
            for task in tasks:
                if task.get("task_type") == task_type:
                    return task
        return tasks[0]

    async def _plan_from_taskboard(
        self,
        hypothesis: List[str],
        observations: List[Dict[str, Any]],
        notes: List[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        listed_claimed = await self._component.agent.controller.run_environment(
            "science",
            "task_list",
            status="claimed",
            agent_id=self.agent_id,
        )
        claimed_tasks = (listed_claimed or {}).get("tasks", []) if isinstance(listed_claimed, dict) else []
        if claimed_tasks:
            task = claimed_tasks[0]
            task_action = task.get("task_type", "read")
            plan_item: Dict[str, Any] = {
                "action": "complete_task",
                "task_id": task.get("task_id"),
                "task_action": task_action,
            }
            task_payload = dict(task.get("payload") or {})
            if task_action in ("experiment", "replicate"):
                intervention = await self._propose_intervention(hypothesis, observations, notes)
                if intervention:
                    task_payload["intervention"] = intervention
            if task_payload:
                plan_item["task_payload"] = task_payload
            return plan_item

        listed_open = await self._component.agent.controller.run_environment(
            "science",
            "task_list",
            status="open",
        )
        open_tasks = (listed_open or {}).get("tasks", []) if isinstance(listed_open, dict) else []
        chosen = self._choose_open_task(open_tasks)
        if not chosen:
            return None
        return {
            "action": "claim_task",
            "task_id": chosen.get("task_id"),
            "task_type": chosen.get("task_type"),
        }

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

    async def _ingest_shared_messages(self) -> None:
        messages = self.perceive_plug.last_tick_messages or []
        if not messages:
            return

        inbox = await self.state_plug.get_state("inbox_evidence") or []
        shared_notes = await self.state_plug.get_state("shared_notes") or []
        shared_observations = await self.state_plug.get_state("shared_observations") or []

        for msg in messages:
            raw = msg.get("content")
            if not isinstance(raw, str):
                continue
            try:
                payload = json.loads(raw)
            except Exception:
                continue

            msg_type = payload.get("type")
            if msg_type == "evidence_share":
                evidence = payload.get("evidence") or {}
                evidence_record = dict(evidence)
                evidence_record["source_agent"] = payload.get("from_agent")
                shared_notes.append(evidence_record)
                inbox.append({"type": msg_type, "from_agent": payload.get("from_agent"), "evidence": evidence_record})
            elif msg_type == "observation_share":
                observation = payload.get("observation") or {}
                observation_record = dict(observation)
                observation_record["source_agent"] = payload.get("from_agent")
                shared_observations.append(observation_record)
                inbox.append(
                    {
                        "type": msg_type,
                        "from_agent": payload.get("from_agent"),
                        "observation": observation_record,
                    }
                )

        await self.state_plug.set_state("shared_notes", shared_notes)
        await self.state_plug.set_state("shared_observations", shared_observations)
        await self.state_plug.set_state("inbox_evidence", inbox)
