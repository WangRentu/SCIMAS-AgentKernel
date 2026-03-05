import csv
import hashlib
import json
import os
from typing import Any, Dict

from agentkernel_standalone.toolkit.logger import get_logger
from agentkernel_standalone.types.schemas.action import ActionResult

logger = get_logger(__name__)


class WriteOperator:
    """Encapsulates write execution path with backward-compatible outputs."""

    def __init__(self, plugin: Any):
        self.plugin = plugin

    def _write_precondition_failures(
        self,
        *,
        hypothesis,
        notes,
        observations,
    ):
        failures = []
        if len(hypothesis or []) < self.plugin._write_min_hypothesis:
            failures.append(f"need_hypothesis>={self.plugin._write_min_hypothesis}")
        if len(notes or []) < self.plugin._write_min_notes:
            failures.append(f"need_notes>={self.plugin._write_min_notes}")
        if len(observations or []) < self.plugin._write_min_observations:
            failures.append(f"need_observations>={self.plugin._write_min_observations}")
        return failures

    def _local_submission_format_check(self, submission_path: str) -> Dict[str, Any]:
        if not submission_path or not os.path.exists(submission_path):
            return {
                "ok": False,
                "error_code": "submission_not_found",
                "message": f"submission missing: {submission_path}",
            }
        rows = 0
        try:
            with open(submission_path, "r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                fieldnames = list(reader.fieldnames or [])
                if not fieldnames:
                    return {
                        "ok": False,
                        "error_code": "missing_header",
                        "message": "submission.csv missing header row",
                    }
                for _ in reader:
                    rows += 1
            if rows <= 0:
                return {
                    "ok": False,
                    "error_code": "empty_submission",
                    "message": "submission.csv has no data rows",
                    "columns": fieldnames,
                }
            try:
                digest = hashlib.sha256()
                with open(submission_path, "rb") as f:
                    while True:
                        chunk = f.read(1024 * 1024)
                        if not chunk:
                            break
                        digest.update(chunk)
                file_hash = digest.hexdigest()
            except Exception:
                file_hash = ""
            return {
                "ok": True,
                "columns": fieldnames,
                "row_count": rows,
                "submission_hash": file_hash,
            }
        except Exception as e:
            return {
                "ok": False,
                "error_code": "csv_parse_error",
                "message": f"failed to parse submission.csv: {e}",
            }

    async def execute(self, agent_id: str) -> ActionResult:
        ctx = await self.plugin._load_research_context(agent_id=agent_id, include_shared=True)
        if isinstance(ctx, dict):
            world_spec = dict(ctx.get("world_spec") or {})
            hypothesis = list(ctx.get("hypothesis") or [])
            local_notes = list(ctx.get("local_notes") or [])
            shared_notes = list(ctx.get("shared_notes") or [])
            notes = list(ctx.get("notes") or [])
            local_observations = list(ctx.get("local_observations") or [])
            shared_observations = list(ctx.get("shared_observations") or [])
            observations = list(ctx.get("observations") or [])
            plan_spec = dict(ctx.get("plan_spec") or {})
        else:
            world_spec = dict(getattr(ctx, "world_spec", {}) or {})
            hypothesis = list(getattr(ctx, "hypothesis", []) or [])
            local_notes = list(getattr(ctx, "local_notes", []) or [])
            shared_notes = list(getattr(ctx, "shared_notes", []) or [])
            notes = list(getattr(ctx, "notes", []) or [])
            local_observations = list(getattr(ctx, "local_observations", []) or [])
            shared_observations = list(getattr(ctx, "shared_observations", []) or [])
            observations = list(getattr(ctx, "observations", []) or [])
            plan_spec = dict(getattr(ctx, "plan_spec", {}) or {})
        if not world_spec:
            world_spec = await self.plugin.controller.run_environment("science", "get_world_spec")
        task_name = str(world_spec.get("task_name") or "unknown_task")

        exp_count = await self.plugin._get_state(agent_id, "exp_count") or 0

        precondition_failures = self._write_precondition_failures(
            hypothesis=hypothesis,
            notes=notes,
            observations=observations,
        )
        if precondition_failures:
            ar = self.plugin._action_error(
                "write",
                "Write preconditions not satisfied.",
                effective_action="write",
                detail={
                    "precondition_failed": True,
                    "precondition_failures": precondition_failures,
                    "counts": {
                        "hypothesis": len(hypothesis),
                        "notes": len(notes),
                        "observations": len(observations),
                    },
                },
            )
            await self.plugin._append_trace(agent_id, "write", 0.0, ar.data or {})
            return ar

        valid_runs = [
            obs
            for obs in observations
            if (
                bool(obs.get("evidence_ok", obs.get("ok", False)))
                and obs.get("run_id")
            )
        ]
        if not valid_runs:
            ar = self.plugin._action_error(
                "write",
                "No scientifically successful experiment run available for submission.",
                effective_action="write",
                detail={"precondition_failed": True, "reason": "no_evidence_backed_run"},
            )
            await self.plugin._append_trace(agent_id, "write", 0.0, ar.data or {})
            return ar

        best_run = sorted(
            valid_runs,
            key=lambda r: float(r.get("dev_score_norm", r.get("score_norm", 0.0)) or 0.0),
            reverse=True,
        )[0]
        best_run_for_write = dict(best_run)
        local_preflight = self._local_submission_format_check(str(best_run_for_write.get("submission_path") or ""))
        if not bool(local_preflight.get("ok")):
            ar = self.plugin._action_error(
                "write",
                "Write preflight failed before environment evaluation.",
                effective_action="write",
                detail={
                    "precondition_failed": True,
                    "reason": "local_format_check_failed",
                    "local_preflight": local_preflight,
                    "best_run_id": best_run_for_write.get("run_id"),
                },
            )
            await self.plugin._append_trace(agent_id, "write", 0.0, ar.data or {})
            return ar

        official_eval = await self.plugin.controller.run_environment(
            "science",
            "evaluate_submission",
            submission_path=best_run_for_write.get("submission_path"),
        )
        if not isinstance(official_eval, dict) or not bool(official_eval.get("ok")):
            eval_reason = str((official_eval or {}).get("reason") or "")
            eval_error_type = str((official_eval or {}).get("error_type") or "")
            cache_hit = bool((official_eval or {}).get("cache_hit"))
            cache_repeat_penalty = self.plugin._write_cache_repeat_penalty if cache_hit else 0.0
            write_reward = -float(cache_repeat_penalty)
            reward_components = {
                "terminal_quality_reward": float(write_reward),
                "learning_reward": float(write_reward),
                "paper_write_reward": float(write_reward),
                "write_eval_success_bonus": 0.0,
                "write_format_pass_reward": 0.0,
                "write_cache_repeat_penalty": float(-cache_repeat_penalty),
            }
            if self.plugin._write_defer_on_system_error and (
                eval_error_type == "system_error" or eval_reason.startswith("system_error:")
            ):
                ar = ActionResult.success(
                    method_name="write",
                    message="Write deferred: evaluation environment precheck failed.",
                    data={
                        "ok": False,
                        "deferred": True,
                        "reason": "evaluation_system_error",
                        "best_run_id": best_run_for_write.get("run_id"),
                        "evaluate_submission": official_eval,
                        "local_preflight": local_preflight,
                        "reward": float(write_reward),
                        "effective_action": "write",
                        "reward_components": reward_components,
                    },
                )
                await self.plugin._append_trace(agent_id, "write", write_reward, ar.data or {})
                return ar
            ar = self.plugin._action_error(
                "write",
                "Final write requires successful test evaluation but evaluate_submission failed.",
                effective_action="write",
                detail={
                    "precondition_failed": True,
                    "reason": "evaluate_submission_failed",
                    "evaluate_submission": official_eval,
                    "local_preflight": local_preflight,
                    "best_run_id": best_run_for_write.get("run_id"),
                    "reward": float(write_reward),
                    "reward_components": reward_components,
                },
            )
            await self.plugin._append_trace(agent_id, "write", write_reward, ar.data or {})
            return ar

        best_run_for_write["metric_name"] = official_eval.get("metric_name") or best_run_for_write.get("metric_name")
        best_run_for_write["raw_score"] = official_eval.get("raw_score")
        best_run_for_write["score_norm"] = official_eval.get("score_norm")
        best_run_for_write["official_eval"] = official_eval
        metric_name = str(best_run_for_write.get("metric_name") or world_spec.get("metric") or "score")
        citations_for_prompt = []
        for note in notes:
            for card in (note or {}).get("cards", []) or []:
                cid = card.get("citation_id")
                if cid and cid not in citations_for_prompt:
                    citations_for_prompt.append(str(cid))
        obs_refs_for_prompt = []
        for obs in observations:
            run_id = obs.get("run_id")
            if run_id:
                obs_refs_for_prompt.append(f"RUN@{run_id}")

        llm_write_spec = None
        if self.plugin._llm_ready("write"):
            write_data_card = await self.plugin._get_state(agent_id, "data_card")
            write_method_card = await self.plugin._get_state(agent_id, "method_card")
            rag_query = " | ".join(
                [
                    f"task={world_spec.get('task_name')}",
                    f"metric={metric_name}",
                    (
                        f"best_run={json.dumps({'run_id': best_run_for_write.get('run_id'), 'score_norm': best_run_for_write.get('score_norm'), 'strategy': best_run_for_write.get('strategy')}, ensure_ascii=False)}"
                    ),
                    f"paper_claims_seed={json.dumps(hypothesis[:6], ensure_ascii=False)}",
                ]
            )
            rag_result = await self.plugin._rag_retrieve_context(
                agent_id=agent_id,
                action="write",
                run_id=str(best_run_for_write.get("run_id") or ""),
                paper_id=None,
                query_text=rag_query,
                quotas={"observation": 3, "data_card": 2, "method_card": 2, "paper": 1, "review": 1, "note": 1},
                notes=notes,
                observations=observations,
                data_card=write_data_card if isinstance(write_data_card, dict) else None,
                method_card=write_method_card if isinstance(write_method_card, dict) else None,
                paper=None,
            )
            rag_block = self.plugin._format_rag_prompt_block(result=rag_result)
            prompt = self.plugin._build_write_prompt(
                world_spec=world_spec,
                best_run=best_run_for_write,
                hypothesis=hypothesis,
                plan_spec=plan_spec,
                citations=citations_for_prompt,
                observation_refs=obs_refs_for_prompt,
                notes=notes,
                observations=observations,
                rag_context=rag_block.get("context", ""),
                rag_refs=rag_block.get("refs", []),
                rag_status=rag_block.get("status", ""),
            )
            llm_result = await self.plugin._call_llm_json(agent_id=agent_id, action_name="write", prompt=prompt)
            if llm_result.get("ok") and isinstance(llm_result.get("data"), dict):
                llm_write_spec = llm_result.get("data") or {}

        paper = self.plugin._build_paper_payload(
            task_name=task_name,
            metric_name=metric_name,
            best_run=best_run_for_write,
            notes=notes,
            observations=observations,
            hypothesis=hypothesis,
            plan_spec=plan_spec,
            exp_count=int(exp_count or 0),
            llm_write_spec=llm_write_spec,
        )
        paper["author_id"] = agent_id
        paper["citation_owner_map"] = self.plugin._build_citation_owner_map(
            agent_id=agent_id,
            local_notes=local_notes,
            shared_notes=shared_notes,
        )

        submit_info = await self.plugin.controller.run_environment("science", "submit_paper", paper=paper)
        paper_id = (submit_info or {}).get("paper_id")
        metrics = await self.plugin.controller.run_environment("science", "evaluate_paper", paper=paper, paper_id=paper_id)
        reward = float(metrics.get("fitness", 0.0) or 0.0)
        eval_success_bonus = self.plugin._write_eval_success_bonus if bool(official_eval.get("ok", False)) else 0.0
        format_pass_reward = self.plugin._write_format_pass_reward if bool((official_eval.get("preflight") or {}).get("ok", True)) else 0.0
        cache_repeat_penalty = self.plugin._write_cache_repeat_penalty if (
            bool(official_eval.get("cache_hit")) and not bool(official_eval.get("ok"))
        ) else 0.0
        reward += float(eval_success_bonus) + float(format_pass_reward) - float(cache_repeat_penalty)

        contribution_credit = self.plugin._compute_contribution_credit(
            agent_id=agent_id,
            paper=paper,
            metrics=metrics,
            shared_observations=shared_observations,
        )
        await self.plugin._grant_credits(contribution_credit, source="paper_write", reference_id=paper_id)

        await self.plugin._set_state(agent_id, "last_fitness", metrics)
        await self.plugin._set_state(agent_id, "last_paper_id", paper_id)
        await self.plugin._inc_state_number(agent_id, "paper_write_count", 1)

        try:
            await self.plugin.controller.run_environment(
                "science",
                "task_create",
                task_type="review",
                payload={"paper_id": paper_id},
                priority=7,
            )
            await self.plugin.controller.run_environment(
                "science",
                "task_create",
                task_type="replicate",
                payload={"paper_id": paper_id},
                priority=8,
            )
        except Exception as e:
            logger.warning(f"Failed to create follow-up tasks for {paper_id}: {e}")

        await self.plugin._log_paper_result(agent_id, paper_id, paper, metrics, source="write")
        if self.plugin._rag_index_on_write:
            rag_paper_docs = self.plugin._rag_docs_from_paper(
                world_spec=world_spec,
                agent_id=agent_id,
                paper=paper,
                paper_id=paper_id,
                action="write",
            )
            if rag_paper_docs:
                await self.plugin._rag_index_documents(
                    agent_id=agent_id,
                    action="write",
                    docs=rag_paper_docs,
                    run_id=str(best_run_for_write.get("run_id") or ""),
                    paper_id=str(paper_id or ""),
                )
        ar = ActionResult.success(
            method_name="write",
            message="AIRS submission written and evaluated.",
            data={
                "metrics": metrics,
                "paper_id": paper_id,
                "paper": paper,
                "llm_write_spec": llm_write_spec,
                "official_eval": official_eval,
                "credit_by_agent": contribution_credit,
                "reward": reward,
                "effective_action": "write",
                "reward_components": {
                    "terminal_quality_reward": float(reward),
                    "learning_reward": float(reward),
                    "paper_write_reward": float(reward),
                    "write_eval_success_bonus": float(eval_success_bonus),
                    "write_format_pass_reward": float(format_pass_reward),
                    "write_cache_repeat_penalty": float(-cache_repeat_penalty),
                },
            },
        )
        await self.plugin._append_trace(agent_id, "write", reward, ar.data or {})
        return ar
