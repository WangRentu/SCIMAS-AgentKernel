import json
import os
from typing import Dict, Any, List

from agentkernel_standalone.mas.agent.base.plugin_base import PlanPlugin
from agentkernel_standalone.mas.agent.components import *
from agentkernel_standalone.toolkit.logger import get_logger


logger = get_logger(__name__)

PIPELINE_DAG = {
    "read":                {"depends_on": [],                                      "min_successes": 1, "max_runs": 3},
    "prepare_data":        {"depends_on": ["read"],                                "min_successes": 1, "max_runs": 2},
    "profile_data":        {"depends_on": ["prepare_data"],                        "min_successes": 1, "max_runs": 2},
    "retrieve_literature": {"depends_on": ["read"],                                "min_successes": 1, "max_runs": 2},
    "hypothesize":         {"depends_on": ["profile_data", "retrieve_literature"], "min_successes": 1, "max_runs": 3, "gate": "vdh_passed"},
    "experiment":          {"depends_on": ["hypothesize"],                         "min_successes": 1, "max_runs": 8, "gate": "vdh_passed", "greedy_search": True},
    "review":              {"depends_on": ["experiment"],                          "min_successes": 1, "max_runs": 3, "min_upstream_successes": {"experiment": 1}},
    "write":               {"depends_on": ["experiment"],                          "min_successes": 1, "max_runs": 2, "min_upstream_successes": {"experiment": 1}},
    "replicate":           {"depends_on": ["write"],                               "min_successes": 1, "max_runs": 1},
}

TOPO_ORDER = [
    "read", "prepare_data", "profile_data", "retrieve_literature",
    "hypothesize", "experiment", "review", "write", "replicate",
]


def _empty_phase_entry() -> Dict[str, Any]:
    return {"runs": 0, "successes": 0, "last_result": "", "best_score": None}


class EasyPlanPlugin(PlanPlugin):

    def __init__(self):
        super().__init__()
        self.plan: List[Dict[str, Any]] = []
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

        phase_status: Dict[str, Dict[str, Any]] = await self.state_plug.get_state("phase_status") or {}
        for phase in PIPELINE_DAG:
            if phase not in phase_status:
                phase_status[phase] = _empty_phase_entry()
        await self.state_plug.set_state("phase_status", phase_status)

        ready_phases: List[str] = []
        blocked_phases: List[str] = []

        for phase in TOPO_ORDER:
            spec = PIPELINE_DAG[phase]
            ps = phase_status[phase]

            if ps["successes"] >= spec["min_successes"]:
                continue
            if ps["runs"] >= spec["max_runs"]:
                continue

            deps_met = all(
                phase_status[dep]["successes"] >= PIPELINE_DAG[dep]["min_successes"]
                for dep in spec["depends_on"]
            )
            if not deps_met:
                blocked_phases.append(phase)
                continue

            upstream_req = spec.get("min_upstream_successes", {})
            upstream_met = all(
                phase_status[up]["successes"] >= count
                for up, count in upstream_req.items()
            )
            if not upstream_met:
                blocked_phases.append(phase)
                continue

            gate = spec.get("gate")
            if gate:
                gate_ok = await self._check_gate(gate, phase)
                if not gate_ok:
                    if phase != "hypothesize":
                        blocked_phases.append(phase)
                        continue

            ready_phases.append(phase)

        if not ready_phases:
            all_complete = all(
                phase_status[p]["successes"] >= PIPELINE_DAG[p]["min_successes"]
                for p in PIPELINE_DAG
            )
            if all_complete:
                self.plan.append({"action": "idle", "reason": "pipeline_complete"})
            else:
                self.plan.append({"action": "idle", "reason": "pipeline_blocked", "blocked_phases": blocked_phases})
            if self._verbose_plan_logs:
                logger.info(f"Agent {self.agent_id} pipeline idle: {self.plan[0].get('reason')}")
            return {"plan": self.plan}

        chosen = ready_phases[0]
        plan_item: Dict[str, Any] = {"action": chosen}

        if PIPELINE_DAG[chosen].get("greedy_search") and phase_status[chosen]["successes"] > 0:
            plan_item["greedy_improve"] = True

        self.plan.append(plan_item)
        if self._verbose_plan_logs:
            logger.info(f"Agent {self.agent_id} planned action: {chosen}")
        return {"plan": self.plan}

    async def _check_gate(self, gate: str, phase: str) -> bool:
        if gate == "vdh_passed":
            last_vdh_report = await self.state_plug.get_state("last_vdh_report")
            return isinstance(last_vdh_report, dict) and bool(last_vdh_report.get("final_ok", False))
        return True

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
