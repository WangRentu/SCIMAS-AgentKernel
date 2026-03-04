import json
import os
import re
from typing import Any, Dict, List, Optional


class DiagnosisService:
    """Structured failure diagnosis and template-fix logic for code loop."""

    def __init__(self, plugin: Any):
        self.plugin = plugin

    def normalize_code_plan(self, payload: Any) -> Dict[str, Any]:
        if not isinstance(payload, dict):
            return {}
        files_raw = payload.get("files")
        files: List[Dict[str, str]] = []
        if isinstance(files_raw, list):
            for item in files_raw[: self.plugin._code_max_files]:
                if not isinstance(item, dict):
                    continue
                rel_path = str(item.get("path") or "").replace("\\", "/").strip()
                content = item.get("content")
                if not rel_path or not isinstance(content, str):
                    continue
                if rel_path.startswith("/") or rel_path.startswith("../") or "/../" in rel_path:
                    continue
                files.append({"path": rel_path[:220], "content": content[: self.plugin._code_max_file_chars]})
        run_cmd = str(payload.get("run_cmd") or "").strip()
        if not run_cmd:
            run_cmd = "python src/main.py --data-dir ./data --output-dir ./outputs --task-manifest ./.task_manifest.json"
        plan = {
            "run_cmd": run_cmd[:600],
            "files": files,
        }
        if isinstance(payload.get("notes"), str):
            plan["notes"] = self.plugin._truncate(payload.get("notes"), 500)
        return plan

    def classify_experiment_failure(
        self,
        result: Dict[str, Any],
        code_plan: Optional[Dict[str, Any]] = None,
        world_spec: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        error = str((result or {}).get("error") or "")
        stderr_tail = str((result or {}).get("stderr_tail") or "")
        stdout_tail = str((result or {}).get("stdout_tail") or "")
        merged = "\n".join([error, stderr_tail, stdout_tail]).lower()
        fallback_reason = str((result or {}).get("fallback_reason") or "").lower()
        world = world_spec or {}

        error_class = "unknown"
        error_codes: List[str] = []
        severity = "medium"
        retryable = True
        root_cause = "unknown_failure"
        repair_hints: List[str] = []

        if "hard_stop:" in merged:
            error_class = "runtime"
            error_codes.append("hard_stop")
            severity = "fatal"
            retryable = False
            root_cause = "episode_budget_or_step_hard_stop"
            repair_hints.append("wait_for_new_episode_or_reduce_compute_cost")
        elif "timed out" in merged or "timeout" in merged:
            error_class = "timeout"
            error_codes.append("execution_timeout")
            severity = "high"
            root_cause = "run_timeout"
            repair_hints.extend(["reduce_data_scale", "reduce_model_complexity", "add_fast_path"])
        elif "killed" in merged or "out of memory" in merged or "oom" in merged:
            error_class = "oom"
            error_codes.append("process_killed_or_oom")
            severity = "high"
            root_cause = "memory_pressure"
            repair_hints.extend(["enable_sampling", "reduce_batch_size", "reduce_feature_count"])
        elif "no such file or directory" in merged or "filenotfounderror" in merged:
            error_class = "io"
            severity = "medium"
            root_cause = "input_or_output_path_missing"
            if "./data/train.csv" in merged:
                error_codes.append("missing_train_csv")
                repair_hints.append("prefer_load_from_disk_train_with_csv_fallback")
            if "submission.csv" in merged:
                error_codes.append("missing_submission_csv")
                repair_hints.append("ensure_outputs_submission_csv_written")
            if not error_codes:
                error_codes.append("missing_path")
                repair_hints.append("check_manifest_and_runtime_paths")
        elif "modulenotfounderror" in merged or "no module named" in merged:
            error_class = "dependency"
            severity = "medium"
            root_cause = "missing_runtime_dependency"
            m = re.search(r"no module named ['\"]([^'\"]+)['\"]", merged)
            if m:
                error_codes.append(f"module_missing_{m.group(1)}")
            else:
                error_codes.append("module_missing_unknown")
            repair_hints.append("avoid_optional_dependency_or_use_stdlib_fallback")
        elif "column object has no attribute tolist" in merged:
            error_class = "schema"
            error_codes.append("column_tolist_misuse")
            root_cause = "dataset_column_type_mismatch"
            repair_hints.append("convert_column_with_list_or_numpy_array")
        elif "unhashable type: 'list'" in merged or "unhashable type: \"list\"" in merged:
            error_class = "schema"
            error_codes.append("unhashable_list_schema")
            root_cause = "list_used_in_hash_context"
            repair_hints.append("avoid_value_counts_on_list_column")
        elif "scoring_column" in merged and ("list" in merged or "[0]" in merged):
            error_class = "schema"
            error_codes.append("scoring_column_list_misuse")
            root_cause = "scoring_column_list_not_indexed"
            repair_hints.append("read_manifest_and_use_scoring_column_index0")
        elif "evaluate.py failed" in merged or ("submission" in merged and "format" in merged):
            error_class = "format"
            error_codes.append("submission_format_invalid")
            root_cause = "submission_schema_mismatch"
            repair_hints.append("align_submission_columns_with_manifest")
        elif "syntaxerror" in merged or "typeerror" in merged or "valueerror" in merged:
            error_class = "runtime"
            error_codes.append("python_runtime_exception")
            root_cause = "python_exception"
            repair_hints.append("fix_exception_using_traceback_line")

        if "code_agent:" in fallback_reason and not error_codes:
            error_codes.append("code_agent_fallback")

        return {
            "error_class": error_class,
            "error_codes": error_codes or ["unknown_failure"],
            "severity": severity,
            "retryable": bool(retryable),
            "root_cause": root_cause,
            "evidence": {
                "error": self.plugin._truncate(error, 800),
                "stderr_tail": self.plugin._truncate(stderr_tail, self.plugin._code_error_tail_chars),
                "stdout_tail": self.plugin._truncate(stdout_tail, self.plugin._code_error_tail_chars),
                "run_meta": {
                    "task_name": world.get("task_name"),
                    "metric": world.get("metric"),
                    "code_memory_mb": world.get("code_memory_mb"),
                    "solver_mode": (result or {}).get("solver_mode"),
                    "fallback_reason": (result or {}).get("fallback_reason"),
                },
            },
            "repair_hints": repair_hints[:8],
            "code_plan_brief": {
                "has_files": bool((code_plan or {}).get("files")),
                "run_cmd": self.plugin._truncate((code_plan or {}).get("run_cmd"), 240),
            },
        }

    def apply_failure_template_fixes(
        self,
        code_plan: Dict[str, Any],
        diagnosis: Dict[str, Any],
        world_spec: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if not self.plugin._code_template_fix_enable:
            return {
                "applied": False,
                "rules_hit": [],
                "mutated_files": [],
                "summary": "template_fix_disabled",
                "code_plan": code_plan,
            }
        plan = json.loads(json.dumps(code_plan or {}))
        files = plan.get("files")
        if not isinstance(files, list):
            return {"applied": False, "rules_hit": [], "mutated_files": [], "summary": "no_files", "code_plan": plan}

        rules_hit: List[str] = []
        mutated_files: List[str] = []
        error_codes = {str(x) for x in (diagnosis or {}).get("error_codes", [])}
        error_class = str((diagnosis or {}).get("error_class") or "")

        for file_obj in files:
            if not isinstance(file_obj, dict):
                continue
            path = str(file_obj.get("path") or "")
            content = file_obj.get("content")
            if not isinstance(content, str):
                continue
            updated = content

            if "missing_train_csv" in error_codes or error_class == "io":
                if "./data/train.csv" in updated and "load_from_disk('./data/train')" not in updated:
                    updated = updated.replace(
                        "pd.read_csv('./data/train.csv')",
                        "load_from_disk('./data/train').to_pandas() if os.path.exists('./data/train') else pd.read_csv('./data/train.csv')",
                    )
                    rules_hit.append("io_train_csv_to_load_from_disk")

            if "scoring_column_list_misuse" in error_codes:
                if "scoring_column" in updated and "[0]" not in updated:
                    updated = updated.replace(
                        "manifest.get('scoring_column')",
                        "(manifest.get('scoring_column') or ['prediction'])[0]",
                    )
                    rules_hit.append("schema_scoring_column_index0")

            if "column_tolist_misuse" in error_codes and ".tolist()" in updated:
                updated = updated.replace(".tolist()", " if hasattr(train_data['target'], '__iter__') else []")
                rules_hit.append("schema_column_tolist_guard")

            if error_class == "oom":
                if "max_features" in updated and "50000" in updated:
                    updated = updated.replace("50000", "15000")
                    rules_hit.append("oom_reduce_max_features")
                if "sample_ratio" not in updated:
                    updated += "\n\n# template fix: OOM guard\nsample_ratio = 0.1\n"
                    rules_hit.append("oom_add_sample_ratio")

            if error_class == "format":
                if "outputs/submission.csv" not in updated:
                    updated += (
                        "\n\n# template fix: enforce submission path\n"
                        "submission_path = os.path.join('./outputs', 'submission.csv')\n"
                    )
                    rules_hit.append("format_force_submission_path")

            if updated != content:
                file_obj["content"] = updated[: self.plugin._code_max_file_chars]
                mutated_files.append(path or "unknown_path")

        applied = bool(mutated_files)
        return {
            "applied": applied,
            "rules_hit": sorted(set(rules_hit)),
            "mutated_files": mutated_files,
            "summary": "template_fix_applied" if applied else "no_rule_matched",
            "code_plan": plan,
            "task_name": (world_spec or {}).get("task_name"),
        }

    def build_repair_context(
        self,
        diagnosis: Dict[str, Any],
        template_fix: Optional[Dict[str, Any]] = None,
        prev_plan: Optional[Dict[str, Any]] = None,
    ) -> str:
        payload = {
            "diagnosis": {
                "error_class": diagnosis.get("error_class"),
                "error_codes": diagnosis.get("error_codes"),
                "severity": diagnosis.get("severity"),
                "retryable": diagnosis.get("retryable"),
                "root_cause": diagnosis.get("root_cause"),
                "repair_hints": diagnosis.get("repair_hints"),
                "evidence": diagnosis.get("evidence"),
            },
            "template_fix": {
                "applied": (template_fix or {}).get("applied"),
                "rules_hit": (template_fix or {}).get("rules_hit"),
                "mutated_files": (template_fix or {}).get("mutated_files"),
                "summary": (template_fix or {}).get("summary"),
            },
            "previous_plan_brief": {
                "run_cmd": (prev_plan or {}).get("run_cmd"),
                "file_count": len((prev_plan or {}).get("files") or []),
            },
        }
        return json.dumps(payload, ensure_ascii=False)

    def should_enter_optimize(self, attempts: List[Dict[str, Any]]) -> bool:
        if not self.plugin._code_optimize_guard_enable:
            return True
        require_code_agent_success = bool(getattr(self.plugin, "_experiment_require_code_agent_success", True))
        treat_fallback_as_repair = bool(getattr(self.plugin, "_experiment_treat_fallback_as_repair", True))

        def _is_masked_fallback_success(r: Dict[str, Any]) -> bool:
            if not isinstance(r, dict):
                return False
            attempted = bool(r.get("code_agent_attempted", False))
            if not attempted:
                return False
            code_ok = bool(r.get("code_agent_ok", not attempted))
            return bool(r.get("ok", False)) and (not code_ok)

        if not isinstance(attempts, list) or len(attempts) < 2:
            return False
        successes = [
            a
            for a in attempts
            if isinstance(a, dict)
            and isinstance(a.get("result"), dict)
            and bool((a.get("result") or {}).get("ok", False))
            and not (
                _is_masked_fallback_success(a.get("result") or {})
                and (require_code_agent_success or treat_fallback_as_repair)
            )
        ]
        if len(successes) < 1:
            return False
        last_two = attempts[-2:]
        for item in last_two:
            if not isinstance(item, dict) or not isinstance(item.get("result"), dict):
                return False
            r = item.get("result") or {}
            if not bool(r.get("ok", False)):
                return False
            if _is_masked_fallback_success(r) and (require_code_agent_success or treat_fallback_as_repair):
                return False
            if not isinstance(r.get("dev_score_norm"), (int, float)) and not isinstance(r.get("score_norm"), (int, float)):
                return False
        return True

    async def log_code_diagnosis(
        self,
        agent_id: str,
        task_name: str,
        run_id: str,
        phase: str,
        diagnosis: Dict[str, Any],
        template_fix: Dict[str, Any],
        decision: str,
    ) -> None:
        await self.plugin._log_code_diagnosis(
            agent_id=agent_id,
            task_name=task_name,
            run_id=run_id,
            phase=phase,
            diagnosis=diagnosis,
            template_fix=template_fix,
            decision=decision,
        )
