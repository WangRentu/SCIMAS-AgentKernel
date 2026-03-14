import json
from typing import Any, Dict, List, Optional

from agentkernel_standalone.toolkit.logger import get_logger
from agentkernel_standalone.types.schemas.action import ActionResult

logger = get_logger(__name__)


class ReviewOperator:
    """Encapsulates review execution path with backward-compatible outputs."""

    def __init__(self, plugin: Any):
        self.plugin = plugin

    async def execute(
        self,
        agent_id: str,
        paper_id: Optional[str] = None,
        run_id: Optional[str] = None,
        submission: Optional[Dict[str, Any]] = None,
    ) -> ActionResult:
        if self.plugin._strict_review_mode and not paper_id and not run_id:
            local_runs = await self.plugin._get_state(agent_id, "observations") or []
            inferred_run_id = ""
            for run in reversed(local_runs):
                if not isinstance(run, dict):
                    continue
                candidate_run_id = str(run.get("run_id") or "").strip()
                if not candidate_run_id:
                    continue
                scientific_ok = bool(run.get("scientific_ok", run.get("evidence_ok", run.get("ok", False))))
                if scientific_ok:
                    inferred_run_id = candidate_run_id
                    break
            if inferred_run_id:
                run_id = inferred_run_id
            else:
                ar = ActionResult.success(
                    method_name="review",
                    message="Review deferred: no paper_id or scientific run_id available in strict mode.",
                    data={
                        "ok": False,
                        "review_deferred": True,
                        "reason": "no_artifact_to_review",
                        "reward": 0.0,
                        "effective_action": "review",
                        "reward_components": {
                            "review_reward": 0.0,
                            "learning_reward": 0.0,
                        },
                    },
                )
                await self.plugin._append_trace(agent_id, "review", 0.0, ar.data or {})
                return ar

        review_world_spec: Dict[str, Any] = {}
        review_local_runs: List[Dict[str, Any]] = []
        review_plan_spec: Dict[str, Any] = {}
        review_hypothesis: List[str] = []
        if self.plugin._orchestrator_v2:
            ctx = await self.plugin._load_research_context(agent_id=agent_id, include_shared=True)
            if isinstance(ctx, dict):
                review_world_spec = dict(ctx.get("world_spec") or {})
                review_local_runs = list(ctx.get("local_observations") or ctx.get("observations") or [])
                review_plan_spec = dict(ctx.get("plan_spec") or {})
                review_hypothesis = list(ctx.get("hypothesis") or [])
            else:
                review_world_spec = dict(getattr(ctx, "world_spec", {}) or {})
                review_local_runs = list(getattr(ctx, "local_observations", []) or getattr(ctx, "observations", []) or [])
                review_plan_spec = dict(getattr(ctx, "plan_spec", {}) or {})
                review_hypothesis = list(getattr(ctx, "hypothesis", []) or [])

        if paper_id:
            paper = await self.plugin.controller.run_environment("science", "get_paper", paper_id=paper_id)
            if paper:
                metrics = await self.plugin.controller.run_environment("science", "evaluate_paper", paper=paper, paper_id=paper_id)
                author_id = (paper or {}).get("author_id")
                self_review = bool(author_id and str(author_id) == str(agent_id))

                llm_review_note: Optional[Dict[str, Any]] = None
                if self.plugin._llm_ready("review"):
                    prompt = self.plugin._build_review_prompt(paper=paper, metrics=metrics)
                    llm_result = await self.plugin._call_llm_json(agent_id=agent_id, action_name="review", prompt=prompt)
                    if llm_result.get("ok") and isinstance(llm_result.get("data"), dict):
                        llm_review_note = llm_result.get("data") or {}

                review_note = llm_review_note if isinstance(llm_review_note, dict) else {}
                heuristic_note = self.plugin._heuristic_review_note(paper=paper, metrics=metrics)
                if not review_note:
                    review_note = heuristic_note

                strengths = review_note.get("strengths")
                if not isinstance(strengths, list) or not strengths:
                    strengths = heuristic_note.get("strengths") or []
                issues = self.plugin._normalize_review_issues(review_note)
                if not issues:
                    issues = self.plugin._normalize_review_issues(heuristic_note)
                review_note["strengths"] = strengths[:6]
                review_note["issues"] = issues[:10]
                if not review_note.get("revision_actions"):
                    review_note["revision_actions"] = heuristic_note.get("revision_actions") or []
                if not review_note.get("summary"):
                    review_note["summary"] = heuristic_note.get("summary")
                review_note["paper_id"] = paper_id
                review_note["reviewer_id"] = agent_id

                critique_quality = self.plugin._score_review_quality(
                    review_note=review_note,
                    issues=issues,
                    self_review=self_review,
                    replication_ok=bool(metrics.get("replication_ok", False)),
                )
                review_score = float(critique_quality.get("critique_score", 0.0) or 0.0)
                paper_context = json.dumps(
                    {
                        "paper": {
                            "title": (paper or {}).get("title"),
                            "abstract": (paper or {}).get("abstract"),
                            "citations": (paper or {}).get("citations"),
                            "observation_refs": (paper or {}).get("observation_refs"),
                        },
                        "metrics": metrics,
                    },
                    ensure_ascii=False,
                )
                qgr_gate = await self.plugin._qgr_validate_review(
                    review_note=review_note,
                    issues=issues,
                    context_text=paper_context,
                    stage="prewrite",
                )
                await self.plugin._log_review_gate(agent_id=agent_id, paper_id=paper_id, run_id=None, gate=qgr_gate)
                if not bool(qgr_gate.get("valid", False)):
                    ar = self.plugin._action_error(
                        "review",
                        "QGR gate rejected review: quality below threshold.",
                        effective_action="review",
                        detail={
                            "precondition_failed": True,
                            "reason": "review_quality_below_threshold",
                            "paper_id": paper_id,
                            "qgr_gate": qgr_gate,
                            "review_note": review_note,
                        },
                    )
                    await self.plugin._append_trace(agent_id, "review", 0.0, ar.data or {})
                    return ar

                reward = float(self.plugin._qgr_base_reward)
                quality_bonus = 0.0
                if int((qgr_gate.get("metrics") or {}).get("citation_count", 0) or 0) >= self.plugin._qgr_min_citations:
                    quality_bonus = float(self.plugin._qgr_quality_bonus)
                    reward += quality_bonus
                local_runs = review_local_runs if review_local_runs else (await self.plugin._get_state(agent_id, "observations") or [])
                pred = self.plugin._qgr_predictive_bonus(
                    issues=issues,
                    run_history=local_runs,
                    target_run_id=((paper or {}).get("claimed_results") or {}).get("run_id") if isinstance((paper or {}).get("claimed_results"), dict) else None,
                )
                predictive_bonus = float(pred.get("bonus", 0.0) or 0.0)
                reward += predictive_bonus
                reward += max(-0.08, min(0.10, 0.10 * review_score - 0.02))
                reward = max(-0.3, min(2.5, reward))

                validation_tasks = await self.plugin._spawn_review_validation_tasks(
                    paper_id=str(paper_id),
                    reviewer_id=agent_id,
                    review_note=review_note,
                    critique_quality=critique_quality,
                )
                followup_tasks = await self.plugin._spawn_qgr_followup_tasks(
                    paper_id=str(paper_id),
                    run_id=((paper or {}).get("claimed_results") or {}).get("run_id") if isinstance((paper or {}).get("claimed_results"), dict) else None,
                    score=review_score,
                    issues=issues,
                )

                if bool(metrics.get("replication_ok", False)) and int(critique_quality.get("issue_count", 0) or 0) <= 1:
                    if author_id:
                        await self.plugin._grant_credits({str(author_id): 0.03}, source="review_verified_strength", reference_id=paper_id)
                if author_id and isinstance(metrics, dict):
                    await self.plugin._set_state(str(author_id), "last_fitness", metrics)

                await self.plugin._set_state(
                    agent_id,
                    "last_review_quality",
                    {
                        "paper_id": paper_id,
                        "critique_quality": critique_quality,
                        "review_score": review_score,
                        "self_review": self_review,
                    },
                )
                await self.plugin._log_paper_result(agent_id, paper_id, paper, metrics, source="review")
                await self.plugin._inc_state_number(agent_id, "review_count", 1)
                ar = ActionResult.success(
                    method_name="review",
                    message="Paper reviewed with evidence-backed strengths and falsifiable critiques.",
                    data={
                        "paper_id": paper_id,
                        "score": review_score,
                        "metrics": metrics,
                        "reward": reward,
                        "effective_action": "review",
                        "reward_components": {
                            "review_reward": float(reward),
                            "learning_reward": float(reward),
                            "review_critique_score": float(review_score),
                            "review_flattery_penalty": float(critique_quality.get("flattery_penalty", 0.0) or 0.0),
                            "qgr_base_reward": float(self.plugin._qgr_base_reward),
                            "qgr_quality_bonus": float(quality_bonus),
                            "qgr_predictive_bonus": float(predictive_bonus),
                        },
                        "review_note": review_note,
                        "critique_quality": critique_quality,
                        "qgr_gate": qgr_gate,
                        "predictive_match": pred,
                        "validation_tasks": validation_tasks,
                        "followup_tasks": followup_tasks,
                        "llm_used": llm_review_note is not None,
                        "self_review": self_review,
                    },
                )
                await self.plugin._append_trace(agent_id, "review", reward, ar.data or {})
                return ar

            if self.plugin._strict_task_dependencies:
                ar = self.plugin._action_error(
                    "review",
                    f"Paper {paper_id} not found.",
                    effective_action="review",
                    detail={"precondition_failed": True, "paper_id": paper_id, "reason": "paper_not_found"},
                )
                await self.plugin._append_trace(agent_id, "review", 0.0, ar.data or {})
                return ar
            if self.plugin._strict_review_mode:
                ar = ActionResult.success(
                    method_name="review",
                    message=f"Review deferred: paper {paper_id} not found.",
                    data={
                        "ok": False,
                        "review_deferred": True,
                        "reason": "paper_not_found",
                        "paper_id": paper_id,
                        "reward": 0.0,
                        "effective_action": "review",
                        "reward_components": {"review_reward": 0.0, "learning_reward": 0.0},
                    },
                )
                await self.plugin._append_trace(agent_id, "review", 0.0, ar.data or {})
                return ar

        if run_id and submission is None:
            run_history = review_local_runs if review_local_runs else (await self.plugin._get_state(agent_id, "observations") or [])
            plan_spec = review_plan_spec if review_plan_spec else (await self.plugin._get_state(agent_id, "plan_spec") or {})
            world_spec = review_world_spec if review_world_spec else await self.plugin.controller.run_environment("science", "get_world_spec")
            latest = None
            for run in reversed(run_history):
                if str(run.get("run_id") or "") == str(run_id):
                    latest = run
                    break
            if latest is None and run_history:
                latest = run_history[-1]
            if latest is None:
                latest = {}

            latest_scientific_ok = bool(latest.get("scientific_ok", latest.get("evidence_ok", latest.get("ok", False))))
            if not latest_scientific_ok:
                ar = ActionResult.success(
                    method_name="review",
                    message="Review deferred: run lacks scientific evidence (missing dev proxy).",
                    data={
                        "ok": False,
                        "review_deferred": True,
                        "reason": "run_scientific_not_ready",
                        "run_id": latest.get("run_id") or run_id,
                        "reward": 0.0,
                        "effective_action": "review",
                        "reward_components": {"review_reward": 0.0, "learning_reward": 0.0},
                    },
                )
                await self.plugin._append_trace(agent_id, "review", 0.0, ar.data or {})
                return ar

            next_plan = self.plugin._derive_next_solver_plan_from_history(plan_spec=plan_spec, run_history=run_history)
            await self.plugin._set_state(agent_id, "plan_spec", next_plan)

            review_note = {
                "summary": "Iterative run-level diagnosis generated from run history.",
                "strengths": [
                    {
                        "id": "S-001",
                        "claim": "Latest run produced an evaluable submission artifact.",
                        "evidence": [f"RUN@{latest.get('run_id')}", f"score_norm={float(latest.get('score_norm', 0.0) or 0.0):.4f}"],
                        "confidence": 0.6,
                        "verification": {"kind": "replicate", "params": {"mode": "score_consistency"}},
                    }
                ],
                "issues": [
                    {
                        "id": "I-001",
                        "type": "iterative_refinement",
                        "severity": 0.65,
                        "claim": "Current configuration may still be suboptimal; schedule next experiment with updated solver plan.",
                        "evidence_refs": [f"RUN@{latest.get('run_id')}"] if latest.get("run_id") else [],
                        "proposed_test": {"kind": "ablation", "params": {"focus": "solver_hyperparams"}},
                        "suggested_fix": "Execute next experiment using updated plan_spec to validate improvement.",
                    }
                ],
                "revision_actions": ["Launch next experiment with updated solver_spec and compare dev_score_norm trend."],
                "run_id": latest.get("run_id"),
            }

            enqueued_task_id = None
            try:
                create_res = await self.plugin.controller.run_environment(
                    "science",
                    "task_create",
                    task_type="experiment",
                    payload={
                        "config": {
                            "strategy": next_plan.get("strategy"),
                            "solver_spec": next_plan.get("solver_spec"),
                            "source": "review_iteration",
                        },
                        "from_run_id": latest.get("run_id"),
                        "revision_reason": "iterative_review",
                    },
                    priority=7,
                )
                if isinstance(create_res, dict) and create_res.get("ok"):
                    enqueued_task_id = (create_res.get("task") or {}).get("task_id")
            except Exception as e:
                logger.warning(f"Failed to enqueue iterative experiment task: {e}")

            reward = 0.02
            run_context = json.dumps({"latest": latest, "next_plan": next_plan}, ensure_ascii=False)
            qgr_gate = await self.plugin._qgr_validate_review(
                review_note=review_note,
                issues=self.plugin._normalize_review_issues(review_note),
                context_text=run_context,
                stage="early",
            )
            await self.plugin._log_review_gate(
                agent_id=agent_id,
                paper_id=None,
                run_id=str(latest.get("run_id") or run_id or ""),
                gate=qgr_gate,
            )
            if not bool(qgr_gate.get("valid", False)):
                ar = self.plugin._action_error(
                    "review",
                    "QGR gate rejected run-level review.",
                    effective_action="review",
                    detail={
                        "precondition_failed": True,
                        "reason": "review_quality_below_threshold",
                        "run_id": latest.get("run_id") or run_id,
                        "qgr_gate": qgr_gate,
                        "review_note": review_note,
                    },
                )
                await self.plugin._append_trace(agent_id, "review", 0.0, ar.data or {})
                return ar

            ar = ActionResult.success(
                method_name="review",
                message="Run-level review completed and next experiment queued.",
                data={
                    "run_id": latest.get("run_id"),
                    "task_name": world_spec.get("task_name"),
                    "review_note": review_note,
                    "next_plan_spec": next_plan,
                    "enqueued_experiment_task_id": enqueued_task_id,
                    "reward": reward,
                    "effective_action": "review",
                    "reward_components": {
                        "review_reward": float(reward),
                        "learning_reward": float(reward),
                        "review_iterative_planning_reward": float(reward),
                    },
                    "qgr_gate": qgr_gate,
                },
            )
            await self.plugin._append_trace(agent_id, "review", reward, ar.data or {})
            return ar

        if self.plugin._strict_review_mode:
            ar = ActionResult.success(
                method_name="review",
                message="Review skipped in strict mode: no valid artifact context.",
                data={
                    "ok": False,
                    "review_deferred": True,
                    "reason": "strict_mode_no_valid_artifact",
                    "reward": 0.0,
                    "effective_action": "review",
                    "reward_components": {"review_reward": 0.0, "learning_reward": 0.0},
                },
            )
            await self.plugin._append_trace(agent_id, "review", 0.0, ar.data or {})
            return ar

        if submission is None:
            submission = {
                "author_id": agent_id,
                "hypothesis": review_hypothesis if review_hypothesis else (await self.plugin._get_state(agent_id, "hypothesis") or []),
            }
        review_score = min(1.0, len(submission.get("hypothesis") or []) / 4.0)
        review_note = {
            "summary": "Fallback review without paper object.",
            "strengths": [
                {
                    "id": "S-001",
                    "claim": "Hypothesis is explicitly stated.",
                    "evidence": [f"hypothesis_count={len(submission.get('hypothesis') or [])}"],
                    "confidence": 0.5,
                    "verification": {"kind": "static_check", "params": {"focus": "hypothesis_presence"}},
                }
            ],
            "issues": [
                {
                    "id": "I-001",
                    "type": "evidence_missing",
                    "severity": 0.6,
                    "claim": "No full paper artifact available for scientific review.",
                    "evidence_refs": [],
                    "proposed_test": {"kind": "static_check", "params": {"focus": "paper_availability"}},
                    "suggested_fix": "Generate paper artifact before formal peer review.",
                }
            ],
            "revision_actions": ["Produce a paper object and rerun review with evidence map."],
        }
        reward = max(-0.05, min(0.08, 0.08 * review_score - 0.01))
        ar = ActionResult.success(
            method_name="review",
            message="Review completed.",
            data={
                "score": review_score,
                "reward": reward,
                "effective_action": "review",
                "reward_components": {
                    "review_reward": float(reward),
                    "learning_reward": float(reward),
                },
                "review_note": review_note,
            },
        )
        await self.plugin._append_trace(agent_id, "review", reward, ar.data or {})
        return ar
