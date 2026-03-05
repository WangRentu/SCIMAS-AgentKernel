import json
from typing import Any, Dict, List, Optional

from agentkernel_standalone.toolkit.logger import get_logger
from agentkernel_standalone.types.schemas.action import ActionResult

logger = get_logger(__name__)


class ExperimentOperator:
    """Encapsulates experiment execution path with full backward-compatible behavior."""

    def __init__(self, plugin: Any):
        self.plugin = plugin

    @staticmethod
    def _normalize_columns_map(raw: Any) -> Dict[str, List[str]]:
        out: Dict[str, List[str]] = {"train": [], "test": []}
        if not isinstance(raw, dict):
            return out
        for split in ("train", "test"):
            vals = raw.get(split)
            if not isinstance(vals, list):
                continue
            seen = set()
            cleaned: List[str] = []
            for v in vals:
                s = str(v or "").strip()
                if not s or s in seen:
                    continue
                seen.add(s)
                cleaned.append(s)
                if len(cleaned) >= 256:
                    break
            out[split] = cleaned
        return out

    def _columns_from_data_card(self, data_card: Optional[Dict[str, Any]]) -> Dict[str, List[str]]:
        out: Dict[str, List[str]] = {"train": [], "test": []}
        if not isinstance(data_card, dict):
            return out
        schema_cols: List[str] = []
        schema = data_card.get("schema")
        if isinstance(schema, list):
            for item in schema:
                if not isinstance(item, dict):
                    continue
                name = str(item.get("name") or item.get("column") or item.get("field") or "").strip()
                if name:
                    schema_cols.append(name)
        focus_cols = [str(x).strip() for x in (data_card.get("focus_columns") or []) if str(x).strip()]
        sample_rows = data_card.get("sample_rows")
        sample_keys: List[str] = []
        if isinstance(sample_rows, list):
            for row in sample_rows:
                if isinstance(row, dict):
                    sample_keys.extend([str(k).strip() for k in row.keys() if str(k).strip()])
        merged = []
        seen = set()
        for col in schema_cols + focus_cols + sample_keys:
            if col and col not in seen:
                seen.add(col)
                merged.append(col)
            if len(merged) >= 256:
                break
        out["train"] = list(merged)
        out["test"] = list(merged)
        return out

    @staticmethod
    def _merge_columns_map(base: Dict[str, List[str]], incoming: Dict[str, List[str]]) -> Dict[str, List[str]]:
        merged: Dict[str, List[str]] = {"train": [], "test": []}
        for split in ("train", "test"):
            vals = []
            seen = set()
            for src in (base.get(split, []), incoming.get(split, [])):
                for v in src:
                    s = str(v or "").strip()
                    if not s or s in seen:
                        continue
                    seen.add(s)
                    vals.append(s)
                    if len(vals) >= 256:
                        break
                if len(vals) >= 256:
                    break
            merged[split] = vals
        return merged

    @staticmethod
    def _derive_evidence_ok(result: Dict[str, Any]) -> bool:
        if not isinstance(result, dict):
            return False
        if "evidence_ok" in result:
            return bool(result.get("evidence_ok", False))
        exec_ok = bool(result.get("exec_ok", result.get("ok", False)))
        has_dev_proxy = bool(
            isinstance(result.get("dev_score"), (int, float))
            or isinstance(result.get("dev_score_norm"), (int, float))
            or (
                isinstance(result.get("dev_eval"), dict)
                and (
                    bool((result.get("dev_eval") or {}).get("ok", False))
                    or isinstance((result.get("dev_eval") or {}).get("raw_score"), (int, float))
                    or isinstance((result.get("dev_eval") or {}).get("score_norm"), (int, float))
                )
            )
        )
        preflight = result.get("submission_preflight")
        preflight_ok = bool(preflight.get("ok", False)) if isinstance(preflight, dict) else True
        return bool(exec_ok and has_dev_proxy and preflight_ok)

    async def _run_code_research_loop(
        self,
        *,
        agent_id: str,
        world_spec: Dict[str, Any],
        hypothesis: List[str],
        plan_spec: Dict[str, Any],
        notes: List[Dict[str, Any]],
        prior_observations: List[Dict[str, Any]],
        data_card: Optional[Dict[str, Any]],
        method_card: Optional[Dict[str, Any]],
        exp_count: int,
        budget: int,
        base_run_config: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        if not self.plugin._code_loop_enabled or not self.plugin._llm_ready("experiment"):
            return None
        if not bool(world_spec.get("code_agent_enable", False)):
            return None

        attempts: List[Dict[str, Any]] = []
        best_result: Optional[Dict[str, Any]] = None
        best_score = -1.0
        best_plan: Dict[str, Any] = {}
        previous_plan: Dict[str, Any] = {}
        failure_context = ""
        last_diagnosis: Dict[str, Any] = {}
        last_template_fix: Dict[str, Any] = {}
        no_improve_rounds = 0
        had_code_agent_success = False
        require_code_agent_success = bool(getattr(self.plugin, "_experiment_require_code_agent_success", True))
        treat_fallback_as_repair = bool(getattr(self.plugin, "_experiment_treat_fallback_as_repair", True))
        require_evidence_ok = bool(getattr(self.plugin, "_experiment_require_evidence_ok", True))
        observed_data_columns = self._columns_from_data_card(data_card if isinstance(data_card, dict) else None)

        max_rounds = min(self.plugin._code_debug_rounds, max(1, int(budget or self.plugin._code_debug_rounds)))

        for idx in range(max_rounds):
            has_success = best_result is not None
            if not has_success and idx == 0:
                phase = "generate"
            elif not has_success:
                phase = "repair"
            else:
                phase = "optimize"
                if not self.plugin._code_optimize_after_success:
                    break
                if not self.plugin._should_enter_optimize(attempts):
                    phase = "repair"

            recent_errors = [
                self.plugin._truncate((a.get("result") or {}).get("error"), 180)
                for a in attempts[-5:]
                if isinstance(a, dict)
                and isinstance(a.get("result"), dict)
                and str(((a.get("result") or {}).get("error") or "")).strip()
            ]
            rag_query = " | ".join(
                [
                    f"task={world_spec.get('task_name')}",
                    f"phase={phase}",
                    f"strategy={plan_spec.get('strategy')}",
                    f"round={idx + 1}/{max_rounds}",
                    f"failure_codes={json.dumps((last_diagnosis or {}).get('error_codes', []), ensure_ascii=False)}",
                    f"recent_errors={json.dumps(recent_errors, ensure_ascii=False)}",
                ]
            )
            rag_result = await self.plugin._rag_retrieve_context(
                agent_id=agent_id,
                action="experiment",
                run_id=str((best_result or {}).get("run_id") or ""),
                paper_id=None,
                query_text=rag_query,
                quotas={"observation": 5, "diagnosis": 5, "method_card": 2, "data_card": 1, "note": 1},
                notes=notes,
                observations=prior_observations + [a.get("result") or {} for a in attempts],
                data_card=data_card,
                method_card=method_card,
            )
            rag_block = self.plugin._format_rag_prompt_block(result=rag_result)

            prompt = self.plugin._build_code_experiment_prompt(
                world_spec=world_spec,
                hypothesis=hypothesis,
                notes=notes,
                observations=prior_observations + [a.get("result") or {} for a in attempts],
                data_card=data_card,
                method_card=method_card,
                plan_spec=plan_spec,
                exp_count=exp_count + idx,
                budget=budget,
                phase=phase,
                round_idx=idx + 1,
                max_rounds=max_rounds,
                previous_plan=previous_plan,
                failure_context=failure_context,
                failure_diagnosis=last_diagnosis,
                template_fix=last_template_fix,
                best_dev_score_norm=best_score if best_score >= 0.0 else None,
                rag_context=rag_block.get("context", ""),
                rag_refs=rag_block.get("refs", []),
                rag_status=rag_block.get("status", ""),
                data_columns=observed_data_columns,
            )
            llm_result = await self.plugin._call_llm_json(agent_id=agent_id, action_name="experiment", prompt=prompt)
            if not llm_result.get("ok") or not isinstance(llm_result.get("data"), dict):
                attempts.append(
                    {
                        "round": idx + 1,
                        "phase": phase,
                        "llm_ok": False,
                        "llm_reason": llm_result.get("reason"),
                    }
                )
                break

            code_plan = self.plugin._normalize_code_plan(llm_result.get("data") or {})
            llm_fixed_error_codes = self.plugin._safe_text_list(
                (llm_result.get("data") or {}).get("fixed_error_codes"),
                limit=10,
                item_limit=120,
            )
            llm_risk_left = self.plugin._safe_text_list((llm_result.get("data") or {}).get("risk_left"), limit=10, item_limit=120)
            previous_plan = code_plan
            if not bool(code_plan.get("files")):
                attempts.append(
                    {
                        "round": idx + 1,
                        "phase": phase,
                        "llm_ok": True,
                        "llm_reason": "code_plan_files_empty",
                        "diagnosis": {"error_class": "runtime", "error_codes": ["code_plan_files_empty"]},
                    }
                )
                failure_context = "LLM returned empty files list; provide full runnable files."
                last_diagnosis = {
                    "error_class": "runtime",
                    "error_codes": ["code_plan_files_empty"],
                    "severity": "medium",
                    "retryable": True,
                    "root_cause": "llm_output_invalid",
                    "repair_hints": ["return_non_empty_files"],
                    "evidence": {"error": "code_plan_files_empty"},
                }
                last_template_fix = {"applied": False, "rules_hit": [], "mutated_files": [], "summary": "no_file_to_fix"}
                if self.plugin._code_diag_enable:
                    await self.plugin._log_code_diagnosis(
                        agent_id=agent_id,
                        phase=phase,
                        run_id=None,
                        diagnosis=last_diagnosis,
                        template_fix=last_template_fix,
                        decision="repair",
                    )
                continue

            current_tick = int(await self.plugin.controller.run_system("timer", "get_tick"))
            run_config = dict(base_run_config or {})
            run_config["strategy"] = "code_agent_iterative"
            run_config["code_phase"] = phase
            run_config["code_round"] = idx + 1
            run_config["code_plan"] = code_plan

            result = await self.plugin.controller.run_environment(
                "science",
                "run_experiment",
                config=run_config,
                agent_id=agent_id,
                current_tick=current_tick,
            )
            attempts.append(
                {
                    "round": idx + 1,
                    "phase": phase,
                    "llm_ok": True,
                    "code_plan": code_plan,
                    "llm_fixed_error_codes": llm_fixed_error_codes,
                    "llm_risk_left": llm_risk_left,
                    "result": result,
                }
            )

            result_payload = result or {}
            round_exec_ok = bool(result_payload.get("exec_ok", result_payload.get("ok", False)))
            round_evidence_ok = self._derive_evidence_ok(result_payload)
            run_columns = self._normalize_columns_map(result_payload.get("data_columns"))
            if run_columns.get("train") or run_columns.get("test"):
                observed_data_columns = self._merge_columns_map(observed_data_columns, run_columns)

            code_agent_attempted = bool(result_payload.get("code_agent_attempted", False))
            code_agent_ok = bool(result_payload.get("code_agent_ok", not code_agent_attempted))
            code_agent_failed = bool(code_agent_attempted and not code_agent_ok)
            fallback_masked_failure = bool(round_exec_ok and code_agent_failed)
            round_success = bool(round_exec_ok)
            if code_agent_failed and (require_code_agent_success or treat_fallback_as_repair):
                round_success = False
            if require_evidence_ok and not round_evidence_ok:
                round_success = False

            if not round_success:
                diagnosis_result = dict(result_payload)
                if fallback_masked_failure:
                    diagnosis_result["ok"] = False
                if not str(diagnosis_result.get("error") or "").strip() and str(result_payload.get("code_agent_error") or "").strip():
                    diagnosis_result["error"] = str(result_payload.get("code_agent_error") or "")
                if not str(diagnosis_result.get("stderr_tail") or "").strip() and str(result_payload.get("code_agent_stderr_tail") or "").strip():
                    diagnosis_result["stderr_tail"] = str(result_payload.get("code_agent_stderr_tail") or "")
                if not str(diagnosis_result.get("stdout_tail") or "").strip() and str(result_payload.get("code_agent_stdout_tail") or "").strip():
                    diagnosis_result["stdout_tail"] = str(result_payload.get("code_agent_stdout_tail") or "")
                if round_exec_ok and require_evidence_ok and not round_evidence_ok:
                    diag_reason = str(result_payload.get("evidence_reason") or "")
                    preflight = result_payload.get("submission_preflight") or {}
                    if not isinstance(preflight, dict):
                        preflight = {}
                    preflight_code = str(preflight.get("error_code") or "")
                    if not diag_reason:
                        diag_reason = "missing_dev_proxy_or_submission_preflight"
                    diagnosis_result["ok"] = False
                    diagnosis_result["error"] = f"evidence_not_ready:{diag_reason}"
                    if not str(diagnosis_result.get("stderr_tail") or "").strip():
                        diagnosis_result["stderr_tail"] = (
                            f"submission_preflight_error={preflight_code}; has_dev_proxy={bool(result_payload.get('has_dev_proxy', False))}"
                        )
                diagnosis = self.plugin._classify_experiment_failure(
                    result=diagnosis_result,
                    code_plan=code_plan,
                    world_spec=world_spec,
                )
                template_fix = self.plugin._apply_failure_template_fixes(
                    code_plan=code_plan,
                    diagnosis=diagnosis,
                    world_spec=world_spec,
                )
                if isinstance(template_fix.get("code_plan"), dict):
                    previous_plan = template_fix.get("code_plan") or previous_plan
                if fallback_masked_failure:
                    diagnosis = dict(diagnosis or {})
                    error_codes = list(diagnosis.get("error_codes") or [])
                    if "code_agent_failed_with_solver_fallback" not in error_codes:
                        error_codes.append("code_agent_failed_with_solver_fallback")
                    diagnosis["error_codes"] = error_codes
                    diagnosis["root_cause"] = "code_agent_failed_solver_fallback_masked"
                    diagnosis["retryable"] = True
                if round_exec_ok and require_evidence_ok and not round_evidence_ok:
                    diagnosis = dict(diagnosis or {})
                    error_codes = list(diagnosis.get("error_codes") or [])
                    if "scientific_evidence_not_ready" not in error_codes:
                        error_codes.append("scientific_evidence_not_ready")
                    diagnosis["error_codes"] = error_codes
                    diagnosis["root_cause"] = "missing_dev_proxy_or_submission_preflight"
                    diagnosis["retryable"] = True
                failure_context = self.plugin._build_repair_context(
                    diagnosis=diagnosis,
                    template_fix=template_fix,
                    prev_plan=previous_plan,
                )
                if fallback_masked_failure:
                    code_err = str(result_payload.get("code_agent_error") or "")
                    code_stderr = str(result_payload.get("code_agent_stderr_tail") or "")
                    ctx_extra = f"Code-agent failed but fallback solver returned ok.\ncode_agent_error={code_err}\ncode_agent_stderr_tail={code_stderr}"
                    failure_context = f"{ctx_extra}\n\n{failure_context}".strip()
                last_diagnosis = diagnosis
                last_template_fix = template_fix
                attempts[-1]["diagnosis"] = diagnosis
                attempts[-1]["template_fix"] = {
                    "applied": template_fix.get("applied"),
                    "rules_hit": template_fix.get("rules_hit"),
                    "mutated_files": template_fix.get("mutated_files"),
                    "summary": template_fix.get("summary"),
                }
                attempts[-1]["fallback_masked_failure"] = fallback_masked_failure
                if self.plugin._code_diag_enable:
                    await self.plugin._log_code_diagnosis(
                        agent_id=agent_id,
                        phase=phase,
                        run_id=str((result or {}).get("run_id") or ""),
                        diagnosis=diagnosis,
                        template_fix=template_fix,
                        decision="repair",
                        score_norm=(result or {}).get("score_norm"),
                        dev_score_norm=(result or {}).get("dev_score_norm"),
                        code_agent_ok=(result or {}).get("code_agent_ok"),
                        fallback_solver_used=(result or {}).get("fallback_solver_used"),
                        execution_path=(result or {}).get("execution_path"),
                    )
                if not bool(diagnosis.get("retryable", True)):
                    break
                continue

            had_code_agent_success = had_code_agent_success or bool(code_agent_ok)
            score = (result or {}).get("dev_score_norm")
            if not isinstance(score, (int, float)):
                score = (result or {}).get("score_norm")
            score_f = self.plugin._clamp01(score or 0.0)
            if best_result is None or score_f > best_score + 1e-6:
                best_result = result
                best_score = score_f
                best_plan = code_plan
                no_improve_rounds = 0
            else:
                no_improve_rounds += 1

            if self.plugin._code_diag_enable:
                await self.plugin._log_code_diagnosis(
                    agent_id=agent_id,
                    phase=phase,
                    run_id=str((result or {}).get("run_id") or ""),
                    diagnosis={
                        "error_class": "none",
                        "error_codes": [],
                        "severity": "low",
                        "retryable": True,
                        "root_cause": "success",
                        "repair_hints": [],
                        "evidence": {},
                    },
                    template_fix={"applied": False, "rules_hit": [], "mutated_files": [], "summary": "success"},
                    decision="optimize" if self.plugin._should_enter_optimize(attempts) else "repair",
                    score_norm=(result or {}).get("score_norm"),
                    dev_score_norm=(result or {}).get("dev_score_norm"),
                    code_agent_ok=(result or {}).get("code_agent_ok"),
                    fallback_solver_used=(result or {}).get("fallback_solver_used"),
                    execution_path=(result or {}).get("execution_path"),
                )

            if has_success and no_improve_rounds >= self.plugin._code_optimize_patience:
                break

        if not attempts:
            return None

        await self.plugin._log_code_loop(
            agent_id=agent_id,
            attempts=attempts,
            best_dev_score_norm=(best_score if best_score >= 0.0 else None),
        )

        final_result = best_result
        final_plan = best_plan
        if final_result is None:
            last = attempts[-1]
            if isinstance(last.get("result"), dict):
                final_result = last.get("result")
            final_plan = last.get("code_plan") if isinstance(last.get("code_plan"), dict) else {}
        if not isinstance(final_result, dict):
            return None
        final_result = dict(final_result)

        if require_code_agent_success and not had_code_agent_success:
            final_code_attempted = bool(final_result.get("code_agent_attempted", False))
            final_code_ok = bool(final_result.get("code_agent_ok", not final_code_attempted))
            if final_code_attempted and not final_code_ok:
                code_err = str(final_result.get("code_agent_error") or "code_agent_failed")
                existing_err = str(final_result.get("error") or "").strip()
                merged_err = f"code_agent_failed_after_retries:{code_err}"
                if existing_err and merged_err not in existing_err:
                    merged_err = f"{merged_err} | solver_result={existing_err}"
                final_result["ok"] = False
                final_result["error"] = merged_err
                final_result["failure_stage"] = str(final_result.get("failure_stage") or "execute")
                final_result["fallback_reason"] = str(final_result.get("fallback_reason") or f"code_agent:{code_err}")
                final_result["fallback_masked_failure"] = True

        return {
            "result": final_result,
            "run_config": {
                "strategy": "code_agent_iterative",
                "code_plan": final_plan,
            },
            "llm_experiment_plan": {
                "mode": "code_agent_loop",
                "attempts": attempts,
                "best_dev_score_norm": best_score if best_score >= 0.0 else None,
            },
            "code_attempts": attempts,
        }

    async def execute(
        self,
        agent_id: str,
        config: Optional[Dict[str, Any]] = None,
        intervention: Optional[Dict[str, float]] = None,
        n_samples: int = 50,
    ) -> ActionResult:
        del intervention, n_samples
        ctx = await self.plugin._load_research_context(agent_id=agent_id, include_shared=True)
        if isinstance(ctx, dict):
            plan_spec = dict(ctx.get("plan_spec") or {})
            world_spec = dict(ctx.get("world_spec") or {})
            notes = list(ctx.get("notes") or [])
            prior_observations = list(ctx.get("observations") or [])
            hypothesis = list(ctx.get("hypothesis") or [])
            data_card = ctx.get("data_card")
            method_card = ctx.get("method_card")
        else:
            plan_spec = dict(getattr(ctx, "plan_spec", {}) or {})
            world_spec = dict(getattr(ctx, "world_spec", {}) or {})
            notes = list(getattr(ctx, "notes", []) or [])
            prior_observations = list(getattr(ctx, "observations", []) or [])
            hypothesis = list(getattr(ctx, "hypothesis", []) or [])
            data_card = getattr(ctx, "data_card", None)
            method_card = getattr(ctx, "method_card", None)
        if not world_spec:
            world_spec = await self.plugin.controller.run_environment("science", "get_world_spec")

        exp_count = int((await self.plugin._get_state(agent_id, "exp_count")) or 0) + 1
        budget = int((await self.plugin._get_state(agent_id, "budget")) or world_spec.get("budget") or 10)
        precondition_failures = self.plugin._experiment_precondition_failures(
            hypothesis=hypothesis,
            notes=notes,
            data_card=data_card if isinstance(data_card, dict) else None,
            method_card=method_card if isinstance(method_card, dict) else None,
        )
        plan_spec_missing = (
            not isinstance(plan_spec, dict)
            or not str((plan_spec or {}).get("strategy") or "").strip()
            or not isinstance((plan_spec or {}).get("solver_spec"), dict)
        )
        if plan_spec_missing:
            precondition_failures.append("need_plan_spec")
        if precondition_failures:
            await self.plugin._log_precondition_gate(
                agent_id=agent_id,
                action="experiment",
                phase="initial",
                failures=precondition_failures,
                summary={
                    "hypothesis_count": len(hypothesis or []),
                    "notes_count": len(notes or []),
                    "observation_count": len(prior_observations or []),
                },
            )
            hydrate_summary = await self.plugin._hydrate_experiment_prerequisites(
                agent_id=agent_id,
                hypothesis=hypothesis,
                notes=notes,
                data_card=data_card if isinstance(data_card, dict) else None,
                method_card=method_card if isinstance(method_card, dict) else None,
                failures=precondition_failures,
            )
            precondition_failures = list(hydrate_summary.get("remaining_failures") or [])
            await self.plugin._log_precondition_gate(
                agent_id=agent_id,
                action="experiment",
                phase="post_hydrate",
                failures=precondition_failures,
                summary=hydrate_summary if isinstance(hydrate_summary, dict) else {},
            )
            if precondition_failures:
                if "need_plan_spec" in precondition_failures:
                    try:
                        await self.plugin.controller.run_environment(
                            "science",
                            "task_create",
                            task_type="hypothesize",
                            payload={"reason": "missing_plan_spec"},
                            priority=9,
                        )
                    except Exception as e:
                        logger.warning(f"Failed to create hypothesize recovery task: {e}")
                recovery = await self.plugin._enqueue_prereq_recovery_tasks(failures=precondition_failures)
                ar = ActionResult.success(
                    method_name="experiment",
                    message="Experiment deferred: prerequisites pending.",
                    data={
                        "ok": False,
                        "pending_prereq": True,
                        "precondition_failed": True,
                        "precondition_failures": precondition_failures,
                        "hydrate_summary": hydrate_summary,
                        "recovery_tasks": recovery,
                        "counts": {
                            "hypothesis": len(hypothesis),
                            "notes": len(notes),
                            "observations": len(prior_observations),
                        },
                        "reward": 0.0,
                        "effective_action": "experiment",
                        "reward_components": {"learning_reward": 0.0, "experiment_reward": 0.0},
                    },
                )
                await self.plugin._append_trace(agent_id, "experiment", 0.0, ar.data or {})
                return ar
            hypothesis = await self.plugin._get_state(agent_id, "hypothesis") or hypothesis
            notes = (await self.plugin._get_state(agent_id, "notes") or []) + (
                await self.plugin._get_state(agent_id, "shared_notes") or []
            )
            data_card = await self.plugin._get_state(agent_id, "data_card")
            method_card = await self.plugin._get_state(agent_id, "method_card")

        run_config = dict(config or {})
        solver_spec = plan_spec.get("solver_spec") if isinstance(plan_spec.get("solver_spec"), dict) else {}
        run_config.setdefault("strategy", plan_spec.get("strategy", "iterative_solver_baseline"))
        if solver_spec:
            run_config.setdefault("solver_spec", json.loads(json.dumps(solver_spec)))
        llm_experiment_plan: Optional[Dict[str, Any]] = None
        code_attempts = None

        code_loop_payload = await self._run_code_research_loop(
            agent_id=agent_id,
            world_spec=world_spec,
            hypothesis=hypothesis,
            plan_spec=plan_spec,
            notes=notes,
            prior_observations=prior_observations,
            data_card=data_card if isinstance(data_card, dict) else None,
            method_card=method_card if isinstance(method_card, dict) else None,
            exp_count=exp_count,
            budget=budget,
            base_run_config=run_config,
        )

        if isinstance(code_loop_payload, dict) and isinstance(code_loop_payload.get("result"), dict):
            result = code_loop_payload.get("result") or {}
            run_config = code_loop_payload.get("run_config") or run_config
            llm_experiment_plan = code_loop_payload.get("llm_experiment_plan")
            code_attempts = code_loop_payload.get("code_attempts")
        else:
            if self.plugin._llm_ready("experiment"):
                rag_query = " | ".join(
                    [
                        f"task={world_spec.get('task_name')}",
                        f"metric={world_spec.get('metric')}",
                        f"strategy={plan_spec.get('strategy')}",
                        f"solver_spec={json.dumps(solver_spec or {}, ensure_ascii=False)}",
                        (
                            f"recent_errors={json.dumps([self.plugin._truncate((o or {}).get('error'), 160) for o in prior_observations[-6:]], ensure_ascii=False)}"
                        ),
                    ]
                )
                rag_result = await self.plugin._rag_retrieve_context(
                    agent_id=agent_id,
                    action="experiment",
                    run_id=None,
                    paper_id=None,
                    query_text=rag_query,
                    quotas={"observation": 5, "diagnosis": 5, "method_card": 2, "data_card": 1, "note": 1},
                    notes=notes,
                    observations=prior_observations,
                    data_card=data_card if isinstance(data_card, dict) else None,
                    method_card=method_card if isinstance(method_card, dict) else None,
                )
                rag_block = self.plugin._format_rag_prompt_block(result=rag_result)
                prompt = self.plugin._build_experiment_prompt(
                    world_spec=world_spec,
                    hypothesis=hypothesis,
                    plan_spec=plan_spec,
                    notes=notes,
                    observations=prior_observations,
                    data_card=data_card if isinstance(data_card, dict) else None,
                    method_card=method_card if isinstance(method_card, dict) else None,
                    exp_count=exp_count,
                    budget=budget,
                    rag_context=rag_block.get("context", ""),
                    rag_refs=rag_block.get("refs", []),
                    rag_status=rag_block.get("status", ""),
                )
                llm_result = await self.plugin._call_llm_json(agent_id=agent_id, action_name="experiment", prompt=prompt)
                if llm_result.get("ok") and isinstance(llm_result.get("data"), dict):
                    llm_experiment_plan = llm_result.get("data") or {}
                    strategy = llm_experiment_plan.get("strategy")
                    if isinstance(strategy, str) and strategy.strip():
                        run_config["strategy"] = strategy.strip()[:120]
                    cfg = llm_experiment_plan.get("config")
                    if isinstance(cfg, dict):
                        run_config["solver_spec"] = self.plugin._merge_solver_plan(
                            {"solver_spec": run_config.get("solver_spec", {})},
                            {"solver_spec": cfg},
                        ).get("solver_spec")
                    validity_checks = self.plugin._safe_text_list(
                        llm_experiment_plan.get("validity_checks"), limit=6, item_limit=180
                    )
                    failure_modes = self.plugin._safe_text_list(
                        llm_experiment_plan.get("failure_modes"), limit=6, item_limit=180
                    )
                    if validity_checks:
                        run_config["validity_checks"] = validity_checks
                    if failure_modes:
                        run_config["failure_modes"] = failure_modes

            current_tick = int(await self.plugin.controller.run_system("timer", "get_tick"))
            result = await self.plugin.controller.run_environment(
                "science",
                "run_experiment",
                config=run_config,
                agent_id=agent_id,
                current_tick=current_tick,
            )

        observations = await self.plugin._get_state(agent_id, "observations") or []
        observation = {
            "run_id": (result or {}).get("run_id"),
            "task_name": (result or {}).get("task_name"),
            "metric_name": (result or {}).get("metric_name"),
            "raw_score": (result or {}).get("raw_score"),
            "score_norm": (result or {}).get("score_norm"),
            "dev_score": (result or {}).get("dev_score"),
            "dev_score_norm": (result or {}).get("dev_score_norm"),
            "submission_path": (result or {}).get("submission_path"),
            "model_path": (result or {}).get("model_path"),
            "solver_log_path": (result or {}).get("solver_log_path"),
            "solver_mode": (result or {}).get("solver_mode"),
            "fallback_reason": (result or {}).get("fallback_reason"),
            "eval_split": (result or {}).get("eval_split", "dev"),
            "stderr_tail": (result or {}).get("stderr_tail"),
            "code_workspace": (result or {}).get("code_workspace"),
            "code_log_path": (result or {}).get("code_log_path"),
            "code_artifacts": (result or {}).get("code_artifacts"),
            "dev_eval": (result or {}).get("dev_eval"),
            "executor_used": (result or {}).get("executor_used"),
            "executor_fallback_used": (result or {}).get("executor_fallback_used"),
            "code_agent_attempted": bool((result or {}).get("code_agent_attempted", False)),
            "code_agent_ok": bool((result or {}).get("code_agent_ok", False)),
            "code_agent_error": (result or {}).get("code_agent_error"),
            "code_agent_exit_code": (result or {}).get("code_agent_exit_code"),
            "code_agent_stdout_tail": (result or {}).get("code_agent_stdout_tail"),
            "code_agent_stderr_tail": (result or {}).get("code_agent_stderr_tail"),
            "fallback_solver_used": bool((result or {}).get("fallback_solver_used", False)),
            "fallback_solver_ok": bool((result or {}).get("fallback_solver_ok", False)),
            "execution_path": (result or {}).get("execution_path"),
            "data_columns": (result or {}).get("data_columns"),
            "exec_ok": bool((result or {}).get("exec_ok", (result or {}).get("ok", False))),
            "evidence_ok": self._derive_evidence_ok(result or {}),
            "evidence_reason": (result or {}).get("evidence_reason"),
            "has_dev_proxy": bool((result or {}).get("has_dev_proxy", False)),
            "submission_preflight": (result or {}).get("submission_preflight"),
            "ok": self._derive_evidence_ok(result or {}),
            "error": (result or {}).get("error"),
            "elapsed_s": (result or {}).get("elapsed_s"),
            "strategy": run_config.get("strategy"),
            "config": run_config.get("solver_spec") or run_config,
            "llm_experiment_plan": llm_experiment_plan,
            "code_attempts": code_attempts,
        }
        observations.append(observation)
        await self.plugin._set_state(agent_id, "observations", observations)
        await self.plugin._set_state(agent_id, "run_history", observations)
        await self.plugin._set_state(agent_id, "exp_count", exp_count)
        if self.plugin._rag_index_on_experiment:
            rag_obs_docs = self.plugin._rag_docs_from_observation(
                world_spec=world_spec,
                agent_id=agent_id,
                observation=observation,
                action="experiment",
            )
            if rag_obs_docs:
                await self.plugin._rag_index_documents(
                    agent_id=agent_id,
                    action="experiment",
                    docs=rag_obs_docs,
                    run_id=str(observation.get("run_id") or ""),
                )

        score_norm = float((result or {}).get("score_norm", 0.0) or 0.0)
        dev_score_norm = (result or {}).get("dev_score_norm")
        dev_score_norm = float(dev_score_norm) if isinstance(dev_score_norm, (int, float)) else score_norm
        exec_ok = bool(observation.get("exec_ok", False))
        evidence_ok = bool(observation.get("evidence_ok", False))
        cost = float((result or {}).get("cost", 0.0) or 0.0)
        prev_best_dev = max(
            [float(o.get("dev_score_norm", 0.0) or 0.0) for o in prior_observations if bool(o.get("ok"))] or [0.0]
        )
        improvement = dev_score_norm - prev_best_dev
        reward = max(-0.06, min(0.12, 0.05 * dev_score_norm + 0.08 * improvement - 0.03 * cost))
        flags = self.plugin._experiment_error_flags(result or {})
        first_pass = self.plugin._is_first_pass_success(
            code_attempts=code_attempts,
            ok=bool((result or {}).get("ok", False)),
        )
        vram_eff = self.plugin._estimate_vram_efficiency(result=result or {}, world_spec=world_spec)
        if evidence_ok:
            reward += self.plugin._experiment_success_reward
            if first_pass:
                reward += self.plugin._experiment_first_pass_bonus
            if isinstance(vram_eff, (int, float)):
                reward += self.plugin._experiment_vram_reward_weight * float(vram_eff)
        elif exec_ok:
            reward = min(reward, -0.01)
        if flags.get("oom", False):
            reward -= self.plugin._experiment_oom_penalty
        if flags.get("typeerror", False):
            reward -= self.plugin._experiment_typeerror_penalty
        if not evidence_ok:
            reward = min(reward, -0.02)
        reward = max(-1.0, min(2.0, float(reward)))

        post_experiment_recovery = None
        if evidence_ok:
            try:
                await self.plugin.controller.run_environment(
                    "science",
                    "task_create",
                    task_type="review",
                    payload={
                        "run_id": observation.get("run_id"),
                        "revision_reason": "post_experiment_diagnosis",
                    },
                    priority=6,
                )
            except Exception as e:
                logger.warning(f"Failed to enqueue post-experiment review task: {e}")
        elif exec_ok:
            recovery_service = getattr(self.plugin, "_recovery_service", None)
            if recovery_service is not None and hasattr(recovery_service, "ensure_experiment_spawned"):
                try:
                    post_experiment_recovery = await recovery_service.ensure_experiment_spawned(
                        reason="missing_scientific_evidence"
                    )
                except Exception as e:
                    logger.warning(f"Failed to ensure experiment spawn after evidence miss: {e}")
            else:
                try:
                    created = await self.plugin.controller.run_environment(
                        "science",
                        "task_create",
                        task_type="experiment",
                        payload={
                            "reason": "missing_scientific_evidence",
                            "from_run_id": observation.get("run_id"),
                        },
                        priority=4,
                    )
                    post_experiment_recovery = {"spawned": bool((created or {}).get("ok", False)), "raw": created}
                except Exception as e:
                    logger.warning(f"Failed to create evidence-recovery experiment task: {e}")

        ar = ActionResult.success(
            method_name="experiment",
            message="AIRS experiment executed.",
            data={
                "ok": bool(evidence_ok),
                "exec_ok": bool(exec_ok),
                "evidence_ok": bool(evidence_ok),
                "evidence_reason": str(observation.get("evidence_reason") or ""),
                "observation": observation,
                "exp_count": exp_count,
                "reward": reward,
                "effective_action": "experiment",
                "reward_components": {
                    "learning_reward": float(reward),
                    "experiment_reward": float(reward),
                    "experiment_score_norm": float(score_norm),
                    "experiment_dev_score_norm": float(dev_score_norm),
                    "experiment_improvement": float(improvement),
                    "experiment_first_pass_bonus": float(self.plugin._experiment_first_pass_bonus if first_pass else 0.0),
                    "experiment_vram_efficiency": float(vram_eff) if isinstance(vram_eff, (int, float)) else 0.0,
                    "experiment_oom_penalty": float(-self.plugin._experiment_oom_penalty if flags.get("oom", False) else 0.0),
                    "experiment_typeerror_penalty": float(
                        -self.plugin._experiment_typeerror_penalty if flags.get("typeerror", False) else 0.0
                    ),
                },
                "run_config": run_config,
                "llm_experiment_plan": llm_experiment_plan,
                "engineering_diagnostics": {
                    "first_pass_success": bool(first_pass),
                    "oom_flag": bool(flags.get("oom", False)),
                    "typeerror_flag": bool(flags.get("typeerror", False)),
                    "vram_efficiency": float(vram_eff) if isinstance(vram_eff, (int, float)) else None,
                    "exec_ok": bool(exec_ok),
                    "evidence_ok": bool(evidence_ok),
                    "evidence_reason": str(observation.get("evidence_reason") or ""),
                    "code_agent_attempted": bool((result or {}).get("code_agent_attempted", False)),
                    "code_agent_ok": bool((result or {}).get("code_agent_ok", False)),
                    "fallback_solver_used": bool((result or {}).get("fallback_solver_used", False)),
                    "execution_path": (result or {}).get("execution_path"),
                },
                "post_experiment_recovery": post_experiment_recovery,
            },
        )
        await self.plugin._append_trace(agent_id, "experiment", reward, ar.data or {})
        return ar
