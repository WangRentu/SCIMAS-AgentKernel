import json
from typing import Any, Dict, List, Optional

from agentkernel_standalone.types.schemas.action import ActionResult


class HypothesizeOperator:
    """Encapsulates hypothesize execution path with backward-compatible outputs."""

    def __init__(self, plugin: Any):
        self.plugin = plugin

    @staticmethod
    def _list_of_text(values: Any, *, limit: int = 8) -> List[str]:
        if not isinstance(values, list):
            return []
        out: List[str] = []
        for item in values[: max(1, limit)]:
            text = str(item or "").strip()
            if text:
                out.append(text)
        return out

    def _normalize_hypothesis_output(
        self,
        *,
        data: Dict[str, Any],
        world_spec: Dict[str, Any],
        base_plan: Dict[str, Any],
    ) -> Dict[str, Any]:
        raw = data if isinstance(data, dict) else {}
        wrapped = raw.get("plan_spec") if isinstance(raw.get("plan_spec"), dict) else None
        candidate = wrapped if wrapped is not None else raw
        plan_spec = self.plugin._merge_solver_plan(base_plan, candidate if isinstance(candidate, dict) else {})

        # Adapter for older schema: top-level solver fields should still be visible to gates.
        if isinstance(raw.get("schema_assumptions"), list) and "schema_assumptions" not in plan_spec:
            plan_spec["schema_assumptions"] = self._list_of_text(raw.get("schema_assumptions"), limit=8)
        if isinstance(raw.get("memory_safety"), list) and "memory_safety" not in plan_spec:
            plan_spec["memory_safety"] = self._list_of_text(raw.get("memory_safety"), limit=8)
        if isinstance(raw.get("evidence_refs"), list) and "evidence_refs" not in plan_spec:
            plan_spec["evidence_refs"] = self._list_of_text(raw.get("evidence_refs"), limit=16)

        hypothesis = self._list_of_text(raw.get("hypothesis"), limit=8)
        if not hypothesis:
            hypothesis = self._list_of_text(raw.get("hypothesis_tags"), limit=8)

        scoring = world_spec.get("scoring_column")
        if isinstance(scoring, list):
            selected = str(scoring[0]).strip() if scoring else ""
            if selected:
                if not isinstance(plan_spec.get("target_cols"), list) or not plan_spec.get("target_cols"):
                    plan_spec["target_cols"] = [selected]
                solver_spec = plan_spec.get("solver_spec") if isinstance(plan_spec.get("solver_spec"), dict) else {}
                if not isinstance(solver_spec.get("target_cols"), list) or not solver_spec.get("target_cols"):
                    solver_spec = dict(solver_spec)
                    solver_spec["target_cols"] = [selected]
                    plan_spec["solver_spec"] = solver_spec
                schema_notes = self._list_of_text(plan_spec.get("schema_assumptions"), limit=12)
                marker = f"use manifest['scoring_column'][0] -> {selected}"
                if marker not in schema_notes:
                    schema_notes.append(marker)
                plan_spec["schema_assumptions"] = schema_notes

        return {"plan_spec": plan_spec, "hypothesis": hypothesis}

    async def execute(self, agent_id: str, hypothesis: Optional[List[str]] = None) -> ActionResult:
        if self.plugin._orchestrator_v2:
            ctx = await self.plugin._load_research_context(agent_id=agent_id, include_shared=True)
            if isinstance(ctx, dict):
                notes = list(ctx.get("notes") or [])
                observations = list(ctx.get("observations") or [])
                data_card = ctx.get("data_card")
                method_card = ctx.get("method_card")
                world_spec = dict(ctx.get("world_spec") or {})
            else:
                notes = list(getattr(ctx, "notes", []) or [])
                observations = list(getattr(ctx, "observations", []) or [])
                data_card = getattr(ctx, "data_card", None)
                method_card = getattr(ctx, "method_card", None)
                world_spec = dict(getattr(ctx, "world_spec", {}) or {})
        else:
            notes = (await self.plugin._get_state(agent_id, "notes") or []) + (
                await self.plugin._get_state(agent_id, "shared_notes") or []
            )
            observations = (await self.plugin._get_state(agent_id, "observations") or []) + (
                await self.plugin._get_state(agent_id, "shared_observations") or []
            )
            data_card = await self.plugin._get_state(agent_id, "data_card")
            method_card = await self.plugin._get_state(agent_id, "method_card")
            world_spec = await self.plugin.controller.run_environment("science", "get_world_spec")
        await self.plugin._rag_bootstrap_episode_knowledge(
            agent_id=agent_id,
            world_spec=world_spec,
            data_card=data_card if isinstance(data_card, dict) else None,
            method_card=method_card if isinstance(method_card, dict) else None,
        )
        existing_plan = await self.plugin._get_state(agent_id, "plan_spec") or {}
        plan_spec = self.plugin._merge_solver_plan(
            self.plugin._default_solver_plan(world_spec),
            existing_plan if isinstance(existing_plan, dict) else {},
        )
        inferred = ["metric_alignment", "format_safety"]
        rag_block = {"context": "", "refs": [], "status": "disabled"}

        if hypothesis is None and self.plugin._llm_ready("hypothesize"):
            recent_failure_modes = [
                self.plugin._truncate((o or {}).get("error"), 180)
                for o in observations[-5:]
                if isinstance(o, dict) and not bool((o or {}).get("ok", False)) and str((o or {}).get("error") or "").strip()
            ]
            rag_query = " | ".join(
                [
                    f"task={world_spec.get('task_name')}",
                    f"metric={world_spec.get('metric')}",
                    f"current_strategy={plan_spec.get('strategy')}",
                    f"existing_hypothesis={json.dumps(hypothesis or [], ensure_ascii=False)}",
                    f"failure_modes={json.dumps(recent_failure_modes[:4], ensure_ascii=False)}",
                ]
            )
            rag_result = await self.plugin._rag_retrieve_context(
                agent_id=agent_id,
                action="hypothesize",
                run_id=None,
                paper_id=None,
                query_text=rag_query,
                quotas={"data_card": 2, "method_card": 3, "observation": 2, "diagnosis": 2, "note": 1},
                notes=notes,
                observations=observations,
                data_card=data_card if isinstance(data_card, dict) else None,
                method_card=method_card if isinstance(method_card, dict) else None,
            )
            rag_block = self.plugin._format_rag_prompt_block(result=rag_result)
            cards = []
            for n in notes[-4:]:
                cards.extend((n or {}).get("cards", [])[:3])
            prompt = self.plugin._build_plan_prompt(
                world_spec=world_spec,
                cards=cards,
                recent_runs=observations,
                data_card=data_card,
                method_card=method_card,
                rag_context=rag_block.get("context", ""),
                rag_refs=rag_block.get("refs", []),
                rag_status=rag_block.get("status", ""),
            )
            llm_result = await self.plugin._call_llm_json(agent_id=agent_id, action_name="hypothesize", prompt=prompt)
            if llm_result.get("ok") and isinstance(llm_result.get("data"), dict):
                data = llm_result.get("data") or {}
                if isinstance(data.get("config"), dict) and not isinstance(data.get("solver_spec"), dict):
                    data["solver_spec"] = data.get("config")
                normalized = self._normalize_hypothesis_output(
                    data=data,
                    world_spec=world_spec,
                    base_plan=plan_spec,
                )
                plan_spec = normalized.get("plan_spec") if isinstance(normalized.get("plan_spec"), dict) else plan_spec
                hy = normalized.get("hypothesis")
                if isinstance(hy, list) and hy:
                    inferred = [str(t) for t in hy[:5]]

        if hypothesis is None:
            hypothesis = inferred

        await self.plugin._set_state(agent_id, "hypothesis", hypothesis)
        await self.plugin._set_state(agent_id, "plan_spec", plan_spec)
        feasibility = self.plugin._hypothesis_feasibility(world_spec=world_spec, plan_spec=plan_spec)
        vdh_report: Dict[str, Any] = {
            "final_ok": True,
            "policy": self.plugin._vdh_gate_policy,
            "gate_a": {"ok": True, "source": "disabled", "constraints": {}, "warnings": []},
            "gate_b": {"ok": True, "errors": [], "resource_estimate": {}},
            "gate_c": {
                "ok": True,
                "coverage_score": 1.0,
                "threshold": self.plugin._vdh_evidence_threshold,
                "source": "disabled",
                "errors": [],
            },
            "failures": [],
        }
        if self.plugin._vdh_enable:
            vdh_report = await self.plugin._evaluate_vdh_gates(
                world_spec=world_spec,
                hypothesis=hypothesis,
                plan_spec=plan_spec,
                notes=notes,
                observations=observations,
                data_card=data_card if isinstance(data_card, dict) else None,
            )
        await self.plugin._set_state(agent_id, "last_vdh_report", vdh_report)

        gate_a_ok = bool(((vdh_report.get("gate_a") or {}).get("ok", False)))
        coverage_score = float(((vdh_report.get("gate_c") or {}).get("coverage_score", 0.0) or 0.0))
        gate_b_errors = list(((vdh_report.get("gate_b") or {}).get("errors") or []))
        potential_oom = any(str(e) == "potential_oom" for e in gate_b_errors)
        final_ok = bool(vdh_report.get("final_ok", True))

        vdh_schema_pass_reward = self.plugin._vdh_schema_pass_reward if gate_a_ok else 0.0
        if coverage_score > 0.8:
            vdh_evidence_coverage_reward = self.plugin._vdh_evidence_high_reward * min(1.0, coverage_score)
        else:
            vdh_evidence_coverage_reward = 0.0
        vdh_oom_penalty = -self.plugin._vdh_oom_penalty if potential_oom else 0.0
        vdh_gate_penalty = -self.plugin._vdh_gate_penalty if (self.plugin._vdh_enable and not final_ok) else 0.0

        if self.plugin._vdh_enable:
            hypo_reward = vdh_schema_pass_reward + vdh_evidence_coverage_reward + vdh_oom_penalty + vdh_gate_penalty
        else:
            hypo_reward = float(feasibility.get("reward", 0.0) or 0.0)
        hypo_reward = float(max(-1.2, min(1.2, hypo_reward)))

        reward_components = {
            "learning_reward": float(hypo_reward),
            "hypothesize_reward": float(hypo_reward),
            "hypothesis_feasibility_score": float(feasibility.get("feasibility_score", 0.0) or 0.0),
            "hypothesis_schema_bonus": float(feasibility.get("schema_bonus", 0.0) or 0.0),
            "hypothesis_resource_bonus": float(feasibility.get("resource_bonus", 0.0) or 0.0),
            "vdh_schema_pass_reward": float(vdh_schema_pass_reward),
            "vdh_evidence_coverage_reward": float(vdh_evidence_coverage_reward),
            "vdh_oom_penalty": float(vdh_oom_penalty),
            "vdh_gate_penalty": float(vdh_gate_penalty),
        }
        await self.plugin._log_vdh_gate(agent_id=agent_id, vdh_report=vdh_report, reward_components=reward_components)

        if self.plugin._vdh_enable and self.plugin._vdh_gate_policy == "hard_fail" and not final_ok:
            recovery = await self.plugin._enqueue_vdh_recovery_tasks(vdh_report=vdh_report)
            ar = ActionResult.error(
                method_name="hypothesize",
                message="VDH gate failed; hypothesis rejected and recovery tasks queued.",
                data={
                    "ok": False,
                    "precondition_failed": True,
                    "reason": "vdh_gate_failed",
                    "hypothesis": hypothesis,
                    "plan_spec": plan_spec,
                    "vdh": vdh_report,
                    "recovery_tasks": recovery,
                    "reward": hypo_reward,
                    "effective_action": "hypothesize",
                    "reward_components": reward_components,
                    "feasibility": feasibility,
                },
            )
            await self.plugin._append_trace(agent_id, "hypothesize", hypo_reward, ar.data or {})
            return ar

        experiment_spawn_result: Dict[str, Any] = {"spawned": False, "reason": "not_requested"}
        recovery_service = getattr(self.plugin, "_recovery_service", None)
        if recovery_service is not None and hasattr(recovery_service, "ensure_experiment_spawned"):
            try:
                experiment_spawn_result = await recovery_service.ensure_experiment_spawned(
                    reason="hypothesize_gate_pass",
                    priority=4,
                )
            except Exception as e:
                experiment_spawn_result = {
                    "spawned": False,
                    "reason": f"spawn_failed:{self.plugin._truncate(str(e), 120)}",
                }

        ar = ActionResult.success(
            method_name="hypothesize",
            message="AIRS strategy plan updated.",
            data={
                "hypothesis": hypothesis,
                "plan_spec": plan_spec,
                "vdh": vdh_report,
                "experiment_spawn": experiment_spawn_result,
                "reward": hypo_reward,
                "effective_action": "hypothesize",
                "reward_components": reward_components,
                "feasibility": feasibility,
                "rag_status": rag_block.get("status"),
                "rag_refs": rag_block.get("refs", []),
            },
        )
        await self.plugin._append_trace(agent_id, "hypothesize", hypo_reward, ar.data or {})
        return ar
