import random
import json
import os
from typing import Dict, Any, List, Optional

from agentkernel_standalone.mas.agent.base.plugin_base import PlanPlugin
from agentkernel_standalone.mas.agent.components import *
from agentkernel_standalone.toolkit.logger import get_logger


logger = get_logger(__name__)

class EasyPlanPlugin(PlanPlugin):
    """A minimal AIRS planner that is taskboard-first with policy fallback."""
    
    def __init__(self):
        super().__init__()
        self.plan = []
        self._rng = random.Random()
        log_mode = (os.getenv("SCIMAS_LOG_MODE", "compact") or "compact").strip().lower()
        self._verbose_plan_logs = os.getenv(
            "SCIMAS_VERBOSE_PLAN_LOGS",
            "1" if log_mode == "verbose" else "0",
        ).lower() in {"1", "true", "yes"}
        if self._verbose_plan_logs:
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
        data_card = await self.state_plug.get_state("data_card")
        method_card = await self.state_plug.get_state("method_card")
        prepare_data_ready = bool(await self.state_plug.get_state("prepare_data_ready"))

        task_plan = await self._plan_from_taskboard(
            current_tick=current_tick,
            hypothesis=hypothesis,
            observations=observations,
            notes=notes,
            data_card=data_card if isinstance(data_card, dict) else None,
            method_card=method_card if isinstance(method_card, dict) else None,
        )
        if task_plan:
            self.plan.append(task_plan)
            if self._verbose_plan_logs:
                logger.info(f"Agent {self.agent_id} planned taskboard action: {task_plan.get('action')}")
            return {"plan": self.plan}

        action = self._sample_action(policy)

        if not hypothesis and action in ("write", "review", "experiment"):
            action = "read"
        if exp_count >= budget and action in ("experiment",):
            action = "write"
        if action in ("profile_data", "experiment") and not prepare_data_ready:
            action = "prepare_data"
        if action == "experiment" and not isinstance(data_card, dict):
            action = "profile_data"
        if action == "experiment" and not isinstance(method_card, dict):
            action = "retrieve_literature"
        if action == "share_evidence" and not notes:
            action = "read"
        if action == "share_observation" and not observations:
            action = "experiment"
        if action == "read" and (len(notes) + len(shared_notes)) > 2:
            action = "hypothesize"

        plan_item: Dict[str, Any] = {"action": action}

        self.plan.append(plan_item)
        if self._verbose_plan_logs:
            logger.info(f"Agent {self.agent_id} planned action: {action}")
        return {"plan": self.plan}

    def _preferred_task_types(self) -> List[str]:
        ordered = ["read", "prepare_data", "profile_data", "retrieve_literature", "hypothesize", "experiment", "write", "replicate", "review"]
        if not ordered:
            return []
        offset = abs(hash(self.agent_id)) % len(ordered)
        return ordered[offset:] + ordered[:offset]

    def _choose_open_task(self, tasks: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if not tasks:
            return None
        ready_tasks = [t for t in tasks if t.get("ready", True)]
        tasks = ready_tasks or tasks
        preferred = self._preferred_task_types()
        for task_type in preferred:
            for task in tasks:
                if task.get("task_type") == task_type:
                    return task
        return tasks[0]

    async def _plan_from_taskboard(
        self,
        current_tick: int,
        hypothesis: List[str],
        observations: List[Dict[str, Any]],
        notes: List[Dict[str, Any]],
        data_card: Optional[Dict[str, Any]],
        method_card: Optional[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        remembered_task_id = await self.state_plug.get_state("current_task_id")
        if remembered_task_id:
            try:
                task_get = await self._component.agent.controller.run_environment(
                    "science",
                    "task_get",
                    task_id=str(remembered_task_id),
                    current_tick=current_tick,
                )
            except Exception:
                task_get = None
            remembered_task = (task_get or {}).get("task") if isinstance(task_get, dict) else None
            if isinstance(remembered_task, dict):
                status = str(remembered_task.get("status") or "")
                if status in {"claimed", "running"} and str(remembered_task.get("claimed_by") or "") == str(self.agent_id):
                    plan_item: Dict[str, Any] = {
                        "action": "complete_task",
                        "task_id": remembered_task.get("task_id"),
                        "task_action": remembered_task.get("task_type", "read"),
                    }
                    task_payload = dict(remembered_task.get("payload") or {})
                    if task_payload:
                        plan_item["task_payload"] = task_payload
                    return plan_item

        listed_claimed = await self._component.agent.controller.run_environment(
            "science",
            "task_list",
            status="claimed",
            agent_id=self.agent_id,
            current_tick=current_tick,
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
            if task_payload:
                plan_item["task_payload"] = task_payload
            return plan_item

        listed_running = await self._component.agent.controller.run_environment(
            "science",
            "task_list",
            status="running",
            agent_id=self.agent_id,
            current_tick=current_tick,
        )
        running_tasks = (listed_running or {}).get("tasks", []) if isinstance(listed_running, dict) else []
        if running_tasks:
            task = running_tasks[0]
            task_action = task.get("task_type", "read")
            plan_item = {
                "action": "complete_task",
                "task_id": task.get("task_id"),
                "task_action": task_action,
            }
            task_payload = dict(task.get("payload") or {})
            if task_payload:
                plan_item["task_payload"] = task_payload
            return plan_item

        active_worker = True
        try:
            worker_check = await self._component.agent.controller.run_environment(
                "science",
                "is_active_worker",
                agent_id=self.agent_id,
            )
            if isinstance(worker_check, dict):
                active_worker = bool(worker_check.get("is_active", True))
        except Exception:
            active_worker = True

        backoff_until = int((await self.state_plug.get_state("claim_backoff_until_tick")) or 0)
        in_backoff = current_tick < backoff_until
        if (not active_worker) or in_backoff:
            return {
                "action": self._choose_non_claim_action(
                    hypothesis=hypothesis,
                    observations=observations,
                    notes=notes,
                    data_card=data_card,
                    method_card=method_card,
                ),
                "non_claim_reason": "not_active_worker" if not active_worker else "claim_backoff",
            }

        listed_open = await self._component.agent.controller.run_environment(
            "science",
            "task_list",
            status="open",
            current_tick=current_tick,
        )
        open_tasks = (listed_open or {}).get("tasks", []) if isinstance(listed_open, dict) else []
        chosen = self._choose_open_task(open_tasks)
        if not chosen:
            return {
                "action": self._choose_non_claim_action(
                    hypothesis=hypothesis,
                    observations=observations,
                    notes=notes,
                    data_card=data_card,
                    method_card=method_card,
                ),
                "non_claim_reason": "no_open_task",
            }
        return {
            "action": "claim_task",
            "task_id": chosen.get("task_id"),
            "task_type": chosen.get("task_type"),
        }

    def _choose_non_claim_action(
        self,
        *,
        hypothesis: List[str],
        observations: List[Dict[str, Any]],
        notes: List[Dict[str, Any]],
        data_card: Optional[Dict[str, Any]],
        method_card: Optional[Dict[str, Any]],
    ) -> str:
        # Keep non-active/backoff agents productive with low-cost supporting work.
        prepare_data_ready = bool(data_card) and not bool(data_card.get("degraded", False)) if isinstance(data_card, dict) else False
        if not prepare_data_ready:
            return "prepare_data"
        if not isinstance(data_card, dict):
            return "profile_data"
        if not isinstance(method_card, dict):
            return "retrieve_literature"
        if len(notes) < 2:
            return "read"
        if not hypothesis:
            return "hypothesize"
        if not observations:
            return "review"
        return "share_evidence"

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
