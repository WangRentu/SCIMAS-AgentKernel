from typing import Any, Dict, Optional

from agentkernel_standalone.types.schemas.action import ActionResult


class ReplicationOperator:
    """Encapsulates replicate execution path with backward-compatible outputs."""

    def __init__(self, plugin: Any):
        self.plugin = plugin

    async def execute(
        self,
        agent_id: str,
        paper_id: Optional[str] = None,
        intervention: Optional[Dict[str, float]] = None,
        n_samples: int = 50,
    ) -> ActionResult:
        del intervention, n_samples

        target_paper_id = paper_id or await self.plugin._get_state(agent_id, "last_paper_id")
        if self.plugin._strict_task_dependencies and not target_paper_id:
            ar = self.plugin._action_error(
                "replicate",
                "Replication requires a target paper_id under strict task dependencies.",
                effective_action="replicate",
                detail={"precondition_failed": True, "reason": "paper_id_required"},
            )
            await self.plugin._append_trace(agent_id, "replicate", 0.0, ar.data or {})
            return ar

        if self.plugin._orchestrator_v2:
            ctx = await self.plugin._load_research_context(agent_id=agent_id, include_shared=True)
            if isinstance(ctx, dict):
                world_spec = dict(ctx.get("world_spec") or {})
            else:
                world_spec = dict(getattr(ctx, "world_spec", {}) or {})
            if not world_spec:
                world_spec = await self.plugin.controller.run_environment("science", "get_world_spec")
        else:
            world_spec = await self.plugin.controller.run_environment("science", "get_world_spec")

        target_paper = await self.plugin.controller.run_environment("science", "get_paper", paper_id=target_paper_id)
        claimed_metrics = (target_paper or {}).get("claimed_results") if isinstance(target_paper, dict) else {}
        llm_replication_plan: Optional[Dict[str, Any]] = None
        replication_payload: Dict[str, Any] = {"mode": "score_consistency"}
        if self.plugin._llm_ready("replicate") and isinstance(target_paper, dict):
            prompt = self.plugin._build_replication_prompt(
                world_spec=world_spec,
                paper_id=str(target_paper_id),
                paper=target_paper,
                claimed_metrics=claimed_metrics or {},
            )
            llm_result = await self.plugin._call_llm_json(agent_id=agent_id, action_name="replicate", prompt=prompt)
            if llm_result.get("ok") and isinstance(llm_result.get("data"), dict):
                llm_replication_plan = llm_result.get("data") or {}
                mode = str(llm_replication_plan.get("mode") or "score_consistency").strip()
                if mode:
                    replication_payload["mode"] = mode[:80]
                protocol_name = llm_replication_plan.get("protocol_name")
                if isinstance(protocol_name, str) and protocol_name.strip():
                    replication_payload["protocol_name"] = protocol_name.strip()[:120]
                pass_criteria = llm_replication_plan.get("pass_criteria")
                if isinstance(pass_criteria, dict):
                    safe_criteria = {}
                    for key, value in pass_criteria.items():
                        if isinstance(value, (str, int, float, bool)):
                            safe_criteria[str(key)] = value
                    if safe_criteria:
                        replication_payload["pass_criteria"] = safe_criteria
                stress_tests = self.plugin._safe_text_list(llm_replication_plan.get("stress_tests"), limit=6, item_limit=180)
                failure_signals = self.plugin._safe_text_list(llm_replication_plan.get("failure_signals"), limit=6, item_limit=180)
                notes = self.plugin._safe_text_list(llm_replication_plan.get("notes"), limit=6, item_limit=180)
                if stress_tests:
                    replication_payload["stress_tests"] = stress_tests
                if failure_signals:
                    replication_payload["failure_signals"] = failure_signals
                if notes:
                    replication_payload["notes"] = notes

        replication_submit = await self.plugin.controller.run_environment(
            "science",
            "submit_replication",
            paper_id=target_paper_id,
            agent_id=agent_id,
            replication=replication_payload,
            source="agent_replicate",
        )

        reward = -0.01
        paper_metrics_after = None
        replication_signal = 0.0
        contradiction_bonus = 0.0
        confirmation_bonus = 0.0
        support_ratio = 0.0
        if bool((replication_submit or {}).get("ok")):
            support = (replication_submit or {}).get("support") or {}
            support_ratio = self.plugin._clamp01(support.get("support_ratio", 0.0))
            replication_signal = abs(support_ratio - 0.5) * 2.0
            contradiction_bonus = max(0.0, 0.5 - support_ratio) / 0.5
            confirmation_bonus = max(0.0, support_ratio - 0.5) / 0.5
            reward = 0.01 + 0.02 * replication_signal + 0.02 * contradiction_bonus + 0.01 * confirmation_bonus
            if support_ratio >= self.plugin._replicate_support_threshold:
                reward = max(float(reward), float(self.plugin._replicate_high_support_reward))
            paper_obj = await self.plugin.controller.run_environment("science", "get_paper", paper_id=target_paper_id)
            if paper_obj:
                paper_metrics_after = await self.plugin.controller.run_environment(
                    "science",
                    "evaluate_paper",
                    paper=paper_obj,
                    paper_id=target_paper_id,
                )
                author_id = (paper_obj or {}).get("author_id")
                if author_id and isinstance(paper_metrics_after, dict):
                    await self.plugin._set_state(author_id, "last_fitness", paper_metrics_after)

        replications = await self.plugin._get_state(agent_id, "replications") or []
        replications.append(
            {
                "paper_id": target_paper_id,
                "support": (replication_submit or {}).get("support") if isinstance(replication_submit, dict) else None,
                "ok": bool((replication_submit or {}).get("ok", False)),
            }
        )
        await self.plugin._set_state(agent_id, "replications", replications)
        await self.plugin._inc_state_number(agent_id, "replication_count", 1)

        ar = ActionResult.success(
            method_name="replicate",
            message="Replication executed.",
            data={
                "paper_id": target_paper_id,
                "replication_submit": replication_submit,
                "replication_payload": replication_payload,
                "llm_replication_plan": llm_replication_plan,
                "paper_metrics_after_replication": paper_metrics_after,
                "reward": reward,
                "effective_action": "replicate",
                "reward_components": {
                    "learning_reward": float(reward),
                    "replicate_reward": float(reward),
                    "replication_support_reward": float(reward),
                    "replication_signal": float(replication_signal),
                    "replication_contradiction_bonus": float(contradiction_bonus),
                    "replication_confirmation_bonus": float(confirmation_bonus),
                    "replication_support_ratio": float(support_ratio),
                    "replication_high_support_reward": float(
                        self.plugin._replicate_high_support_reward
                        if support_ratio >= self.plugin._replicate_support_threshold
                        else 0.0
                    ),
                },
            },
        )
        await self.plugin._append_trace(agent_id, "replicate", reward, ar.data or {})
        return ar
