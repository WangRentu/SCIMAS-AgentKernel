import json
from typing import Any, Dict, List, Optional


class RecoveryService:
    """Prerequisite and recovery task orchestration for research actions."""

    def __init__(self, plugin: Any):
        self.plugin = plugin

    def experiment_precondition_failures(
        self,
        hypothesis: List[str],
        notes: List[Dict[str, Any]],
        data_card: Optional[Dict[str, Any]],
        method_card: Optional[Dict[str, Any]],
    ) -> List[str]:
        failures: List[str] = []
        if self.plugin._experiment_require_data_card and not isinstance(data_card, dict):
            failures.append("need_data_card")
        if self.plugin._experiment_require_method_card and not isinstance(method_card, dict):
            failures.append("need_method_card")
        if (
            self.plugin._experiment_require_method_card
            and self.plugin._experiment_require_method_card_quality
            and isinstance(method_card, dict)
            and not self.method_card_quality_ok(method_card)
        ):
            failures.append("need_method_card_quality_pass")
        if self.plugin._experiment_min_notes > 0 and len(notes or []) < self.plugin._experiment_min_notes:
            failures.append(f"need_notes>={self.plugin._experiment_min_notes}")
        if self.plugin._experiment_min_hypothesis > 0 and len(hypothesis or []) < self.plugin._experiment_min_hypothesis:
            failures.append(f"need_hypothesis>={self.plugin._experiment_min_hypothesis}")
        return failures

    def method_card_quality_report(self, method_card: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        report = {
            "schema_valid": False,
            "citation_coverage": 0.0,
            "executable_minimum": False,
            "level": "L2",
            "degraded": True,
            "fail_reasons": [],
            "baseline_count": 0,
            "evidence_count": 0,
        }
        if not isinstance(method_card, dict):
            report["fail_reasons"].append("method_card_missing")
            return report

        def _add_ref(ref_set: set[str], value: Any) -> None:
            text = str(value or "").strip()
            if not text:
                return
            lower = text.lower()
            if lower in {"unknown", "none", "null", "n/a", "-"}:
                return
            ref_set.add(text)

        quality = method_card.get("quality") if isinstance(method_card.get("quality"), dict) else {}
        baseline_candidates = (
            method_card.get("baseline_candidates") if isinstance(method_card.get("baseline_candidates"), list) else []
        )
        recommended_baselines = (
            method_card.get("recommended_baselines")
            if isinstance(method_card.get("recommended_baselines"), list)
            else []
        )
        baselines = baseline_candidates if baseline_candidates else recommended_baselines
        report["baseline_count"] = len(baselines)
        if len(baselines) < self.plugin._retrieve_min_baselines:
            report["fail_reasons"].append(f"insufficient_baselines<{self.plugin._retrieve_min_baselines}")

        citation_map = method_card.get("citation_map") if isinstance(method_card.get("citation_map"), dict) else {}
        evidence_ref_set: set[str] = set()
        for refs in citation_map.values():
            if isinstance(refs, list):
                for ref in refs:
                    _add_ref(evidence_ref_set, ref)

        # Merge evidence references from all structured fields to avoid under-counting.
        for baseline in baselines:
            if not isinstance(baseline, dict):
                continue
            for ref in (baseline.get("evidence_refs") or []):
                _add_ref(evidence_ref_set, ref)

        for item in (method_card.get("failure_playbook") or []):
            if not isinstance(item, dict):
                continue
            for ref in (item.get("evidence_refs") or []):
                _add_ref(evidence_ref_set, ref)

        for item in (method_card.get("ablation_plan") or []):
            if not isinstance(item, dict):
                continue
            for ref in (item.get("evidence_refs") or []):
                _add_ref(evidence_ref_set, ref)

        for item in (method_card.get("task_evidence_refs") or []):
            if not isinstance(item, dict):
                continue
            _add_ref(evidence_ref_set, item.get("citation_id"))

        for item in (method_card.get("evidence_pack") or []):
            if not isinstance(item, dict):
                continue
            _add_ref(evidence_ref_set, item.get("evidence_id"))

        report["evidence_count"] = len(evidence_ref_set)
        if report["evidence_count"] < int(self.plugin._retrieve_min_evidence):
            report["fail_reasons"].append(f"insufficient_evidence<{int(self.plugin._retrieve_min_evidence)}")

        explicit_coverage = quality.get("citation_coverage") if isinstance(quality, dict) else None
        if isinstance(explicit_coverage, (int, float)):
            citation_coverage = max(0.0, min(1.0, float(explicit_coverage)))
        else:
            suggestion_count = max(1, len(baselines))
            if len(evidence_ref_set) <= 0:
                citation_coverage = 0.0
            else:
                citation_coverage = min(1.0, float(len(evidence_ref_set)) / float(suggestion_count))
        report["citation_coverage"] = citation_coverage
        if citation_coverage < self.plugin._retrieve_citation_threshold:
            report["fail_reasons"].append(
                f"citation_coverage<{self.plugin._retrieve_citation_threshold:.2f}"
            )

        executable_minimum = False
        if isinstance(quality, dict) and isinstance(quality.get("executable_minimum"), bool):
            executable_minimum = bool(quality.get("executable_minimum"))
        else:
            for baseline in baselines:
                if not isinstance(baseline, dict):
                    continue
                steps = baseline.get("implementation_steps") if isinstance(baseline.get("implementation_steps"), list) else []
                key_steps = baseline.get("key_steps") if isinstance(baseline.get("key_steps"), list) else []
                check_blob = " ".join([str(x).lower() for x in (steps + key_steps)])
                if (
                    ("data" in check_blob or "load" in check_blob)
                    and ("train" in check_blob or "fit" in check_blob)
                    and ("dev" in check_blob or "valid" in check_blob)
                    and ("submission" in check_blob or "format" in check_blob)
                ):
                    executable_minimum = True
                    break
        report["executable_minimum"] = executable_minimum
        if not executable_minimum:
            report["fail_reasons"].append("missing_executable_minimum_baseline")

        required_fields = (
            "task_summary",
            "baseline_candidates",
            "experiment_roadmap",
            "failure_playbook",
            "ablation_plan",
            "citation_map",
        )
        missing_required = [
            key
            for key in required_fields
            if key not in method_card
        ]
        if missing_required:
            report["fail_reasons"].append("missing_required_fields:" + ",".join(missing_required))

        schema_valid = bool(
            method_card.get("ok", True)
            and str(method_card.get("card_type") or "") == "method_card"
            and isinstance(method_card.get("task_name"), str)
            and isinstance(method_card.get("metric"), str)
            and isinstance(method_card.get("category"), str)
            and not missing_required
        )
        if not schema_valid:
            report["fail_reasons"].append("schema_invalid")
        report["schema_valid"] = schema_valid

        degraded = bool((quality or {}).get("degraded", False))
        if not report["fail_reasons"] and report["citation_coverage"] >= self.plugin._retrieve_citation_threshold:
            level = "L0" if not degraded else "L1"
        elif executable_minimum and schema_valid:
            level = "L1"
        else:
            level = "L2"
        report["level"] = level
        report["degraded"] = degraded or level != "L0"
        return report

    def method_card_quality_ok(self, method_card: Optional[Dict[str, Any]]) -> bool:
        report = self.method_card_quality_report(method_card)
        strict_ok = bool(
            report.get("schema_valid")
            and report.get("executable_minimum")
            and int(report.get("evidence_count", 0) or 0) >= int(self.plugin._retrieve_min_evidence)
            and float(report.get("citation_coverage", 0.0) or 0.0) >= self.plugin._retrieve_citation_threshold
            and str(report.get("level") or "L2") in {"L0", "L1"}
        )
        if strict_ok:
            return True

        # Bootstrap mode: allow L1 cards through when the only gap is evidence-count,
        # so experiment can run once and generate new run_memory evidence.
        if not bool(getattr(self.plugin, "_experiment_allow_l1_bootstrap", False)):
            return False
        fail_reasons = [str(x or "") for x in (report.get("fail_reasons") or [])]
        evidence_only_failure = bool(fail_reasons) and all(
            r.startswith("insufficient_evidence<") for r in fail_reasons
        )
        return bool(
            evidence_only_failure
            and report.get("schema_valid")
            and report.get("executable_minimum")
            and int(report.get("baseline_count", 0) or 0) >= int(self.plugin._retrieve_min_baselines)
            and float(report.get("citation_coverage", 0.0) or 0.0) >= self.plugin._retrieve_citation_threshold
            and str(report.get("level") or "L2") == "L1"
        )

    def has_notes_failure(self, failures: List[str]) -> bool:
        return any(str(item).startswith("need_notes>=") for item in (failures or []))

    def _build_method_note_from_card(self, method_card: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if not isinstance(method_card, dict):
            return None
        baselines = (
            method_card.get("baseline_candidates")
            if isinstance(method_card.get("baseline_candidates"), list)
            else method_card.get("recommended_baselines")
        )
        baselines = baselines if isinstance(baselines, list) else []
        cards: List[Dict[str, Any]] = []
        hints: List[str] = []
        for idx, baseline in enumerate(baselines[: self.plugin._llm_max_cards], start=1):
            if not isinstance(baseline, dict):
                continue
            citation_id = f"M{idx:04d}"
            title = str(baseline.get("name") or f"baseline_{idx}")
            steps = self.plugin._safe_text_list(
                baseline.get("implementation_steps") or baseline.get("key_steps"),
                limit=3,
                item_limit=120,
            )
            pitfalls = self.plugin._safe_text_list(
                baseline.get("risks") or baseline.get("pitfalls"),
                limit=2,
                item_limit=120,
            )
            text_parts = [f"use_when={self.plugin._truncate(baseline.get('use_when'), 120)}"]
            if steps:
                text_parts.append("steps=" + "; ".join(steps))
            if pitfalls:
                text_parts.append("pitfalls=" + "; ".join(pitfalls))
            cards.append(
                {
                    "citation_id": citation_id,
                    "kind": "method_card",
                    "title": title,
                    "text": " | ".join(text_parts),
                }
            )
            hints.append(f"[{citation_id}] {title}")
        if not cards and isinstance(method_card.get("task_evidence_refs"), list):
            for ref in method_card.get("task_evidence_refs")[: self.plugin._llm_max_cards]:
                if not isinstance(ref, dict):
                    continue
                cid = str(ref.get("citation_id") or "")
                title = str(ref.get("title") or "method_ref")
                cards.append(
                    {
                        "citation_id": cid or f"MR{len(cards) + 1:04d}",
                        "kind": "method_ref",
                        "title": title,
                        "text": self.plugin._truncate(ref.get("snippet"), 220),
                    }
                )
                hints.append(f"[{cards[-1]['citation_id']}] {title}")
        if not cards:
            return None
        return {
            "topic": "task_baselines",
            "hints": hints,
            "cards": cards,
            "task_name": method_card.get("task_name"),
            "source": "method_card",
        }

    def _build_data_note_from_card(self, data_card: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if not isinstance(data_card, dict):
            return None
        split_stats = data_card.get("split_stats") if isinstance(data_card.get("split_stats"), dict) else {}
        schema = data_card.get("schema") if isinstance(data_card.get("schema"), list) else []
        cards: List[Dict[str, Any]] = []
        hints: List[str] = []
        cards.append(
            {
                "citation_id": "D0001",
                "kind": "data_profile",
                "title": "split_stats",
                "text": self.plugin._truncate(json.dumps(split_stats, ensure_ascii=False), 320),
            }
        )
        hints.append("[D0001] split_stats")
        for idx, col in enumerate(schema[:6], start=2):
            if not isinstance(col, dict):
                continue
            name = str(col.get("name") or f"col_{idx}")
            desc = {
                "dtype": col.get("dtype"),
                "missing_ratio": col.get("missing_ratio"),
                "unique": col.get("unique"),
            }
            cards.append(
                {
                    "citation_id": f"D{idx:04d}",
                    "kind": "data_schema",
                    "title": name,
                    "text": self.plugin._truncate(json.dumps(desc, ensure_ascii=False), 220),
                }
            )
            hints.append(f"[D{idx:04d}] {name}")
        return {
            "topic": "data_profile",
            "hints": hints,
            "cards": cards,
            "task_name": data_card.get("task_name"),
            "source": "data_card",
        }

    async def hydrate_experiment_prerequisites(
        self,
        agent_id: str,
        hypothesis: List[str],
        notes: List[Dict[str, Any]],
        data_card: Optional[Dict[str, Any]],
        method_card: Optional[Dict[str, Any]],
        failures: List[str],
    ) -> Dict[str, Any]:
        summary: Dict[str, Any] = {
            "used_shared_artifacts": False,
            "hydrate_steps": [],
            "remaining_failures": list(failures or []),
        }
        if not failures:
            return summary

        local_notes = await self.plugin._get_state(agent_id, "notes") or []
        changed_notes = False
        shared_artifacts: Dict[str, Any] = {}

        need_data = "need_data_card" in failures and not isinstance(data_card, dict)
        need_method = (
            ("need_method_card" in failures and not isinstance(method_card, dict))
            or "need_method_card_quality_pass" in failures
        )
        need_notes = self.has_notes_failure(failures)

        if need_data or need_method or need_notes:
            try:
                shared_artifacts = await self.plugin.controller.run_environment(
                    "science",
                    "get_shared_artifacts",
                    include_cards=True,
                    max_refs=max(4, min(12, self.plugin._llm_max_cards)),
                )
                if isinstance(shared_artifacts, dict) and shared_artifacts.get("ok"):
                    summary["used_shared_artifacts"] = True
                    summary["hydrate_steps"].append("fetched_shared_artifacts")
            except Exception as e:
                summary["hydrate_steps"].append(f"shared_artifacts_failed:{self.plugin._truncate(str(e), 160)}")

        if need_data and not isinstance(data_card, dict):
            shared_data = (
                shared_artifacts.get("data_card") if isinstance(shared_artifacts.get("data_card"), dict) else None
            )
            if isinstance(shared_data, dict):
                data_card = dict(shared_data)
                await self.plugin._set_state(agent_id, "data_card", data_card)
                summary["hydrate_steps"].append("data_card_from_shared_cache")
            else:
                prof = await self.plugin.controller.run_environment("science", "profile_data", agent_id=agent_id, refresh=False)
                if isinstance(prof, dict) and bool(prof.get("ok", False)):
                    data_card = prof
                    await self.plugin._set_state(agent_id, "data_card", data_card)
                    summary["hydrate_steps"].append("data_card_from_profile_data")
                else:
                    summary["hydrate_steps"].append(
                        "data_card_unavailable:" + self.plugin._truncate((prof or {}).get("reason"), 140)
                    )

        if need_method and (not isinstance(method_card, dict) or not self.method_card_quality_ok(method_card)):
            shared_method = (
                shared_artifacts.get("method_card") if isinstance(shared_artifacts.get("method_card"), dict) else None
            )
            if isinstance(shared_method, dict) and self.method_card_quality_ok(shared_method):
                method_card = dict(shared_method)
                await self.plugin._set_state(agent_id, "method_card", method_card)
                summary["hydrate_steps"].append("method_card_from_shared_cache")
            else:
                method = await self.plugin.controller.run_environment(
                    "science",
                    "retrieve_method_card",
                    agent_id=agent_id,
                    topic="task_baselines",
                    refresh=False,
                )
                if isinstance(method, dict) and bool(method.get("ok", False)):
                    method_card = method
                    await self.plugin._set_state(agent_id, "method_card", method_card)
                    summary["hydrate_steps"].append("method_card_from_retrieve")
                else:
                    summary["hydrate_steps"].append(
                        "method_card_unavailable:" + self.plugin._truncate((method or {}).get("reason"), 140)
                    )

        if need_notes:
            note_count = len((local_notes or []) + (await self.plugin._get_state(agent_id, "shared_notes") or []))
            if note_count < self.plugin._experiment_min_notes:
                shared_template = (
                    shared_artifacts.get("notes_template")
                    if isinstance(shared_artifacts.get("notes_template"), dict)
                    else None
                )
                if isinstance(shared_template, dict):
                    local_notes.append(shared_template)
                    changed_notes = True
                    summary["hydrate_steps"].append("notes_from_shared_template")
            note_count = len((local_notes or []) + (await self.plugin._get_state(agent_id, "shared_notes") or []))
            if note_count < self.plugin._experiment_min_notes:
                method_note = self._build_method_note_from_card(method_card)
                if isinstance(method_note, dict):
                    local_notes.append(method_note)
                    changed_notes = True
                    summary["hydrate_steps"].append("notes_from_method_card")
            note_count = len((local_notes or []) + (await self.plugin._get_state(agent_id, "shared_notes") or []))
            if note_count < self.plugin._experiment_min_notes:
                data_note = self._build_data_note_from_card(data_card)
                if isinstance(data_note, dict):
                    local_notes.append(data_note)
                    changed_notes = True
                    summary["hydrate_steps"].append("notes_from_data_card")
            note_count = len((local_notes or []) + (await self.plugin._get_state(agent_id, "shared_notes") or []))
            if note_count < self.plugin._experiment_min_notes:
                lit = await self.plugin.controller.run_environment(
                    "science",
                    "read_literature",
                    agent_id=agent_id,
                    topic="task_requirements",
                )
                if isinstance(lit, dict):
                    local_notes.append(lit)
                    changed_notes = True
                    summary["hydrate_steps"].append("notes_from_read_literature")

        if changed_notes:
            await self.plugin._set_state(agent_id, "notes", local_notes)

        refreshed_hypothesis = await self.plugin._get_state(agent_id, "hypothesis") or hypothesis
        refreshed_notes = (await self.plugin._get_state(agent_id, "notes") or []) + (
            await self.plugin._get_state(agent_id, "shared_notes") or []
        )
        refreshed_data = await self.plugin._get_state(agent_id, "data_card")
        refreshed_method = await self.plugin._get_state(agent_id, "method_card")
        summary["remaining_failures"] = self.experiment_precondition_failures(
            hypothesis=refreshed_hypothesis,
            notes=refreshed_notes,
            data_card=refreshed_data if isinstance(refreshed_data, dict) else None,
            method_card=refreshed_method if isinstance(refreshed_method, dict) else None,
        )
        summary["counts"] = {
            "hypothesis": len(refreshed_hypothesis or []),
            "notes": len(refreshed_notes or []),
            "has_data_card": isinstance(refreshed_data, dict),
            "has_method_card": isinstance(refreshed_method, dict),
        }
        return summary

    async def enqueue_prereq_recovery_tasks(self, failures: List[str]) -> Dict[str, Any]:
        open_list = await self.plugin.controller.run_environment("science", "task_list", status="open")
        claimed_list = await self.plugin.controller.run_environment("science", "task_list", status="claimed")
        known_types = set()
        for listing in (open_list, claimed_list):
            tasks = (listing or {}).get("tasks", []) if isinstance(listing, dict) else []
            for task in tasks:
                known_types.add(str((task or {}).get("task_type") or ""))

        created: List[str] = []
        required_task_types: List[str] = []
        if "need_data_card" in failures:
            required_task_types.append("profile_data")
        if "need_method_card" in failures:
            required_task_types.append("retrieve_literature")
        if "need_method_card_quality_pass" in failures:
            required_task_types.append("retrieve_literature")
            required_task_types.append("read")
        if self.has_notes_failure(failures):
            required_task_types.append("read")
        if any(str(f).startswith("need_hypothesis>=") for f in failures):
            required_task_types.append("hypothesize")
        if "need_plan_spec" in failures:
            required_task_types.append("hypothesize")

        for task_type in required_task_types:
            if task_type in known_types:
                continue
            created_task = await self.plugin.controller.run_environment(
                "science",
                "task_create",
                task_type=task_type,
                payload={},
                priority=8,
            )
            if isinstance(created_task, dict) and created_task.get("ok"):
                tid = str((created_task.get("task") or {}).get("task_id") or "")
                if tid:
                    created.append(tid)
                known_types.add(task_type)

        return {"created_task_ids": created, "required_task_types": required_task_types}

    async def enqueue_vdh_recovery_tasks(self, vdh_report: Dict[str, Any]) -> Dict[str, Any]:
        return await self.plugin._enqueue_vdh_recovery_tasks(vdh_report=vdh_report)

    async def ensure_experiment_spawned(
        self,
        *,
        reason: str = "",
        priority: int = 4,
    ) -> Dict[str, Any]:
        """Ensure at least one schedulable experiment task exists.

        Used by retrieve->L0 recovery path: if no open/claimed experiment task is present,
        create one so the pipeline can continue instead of idling after quality recovery.
        """
        known_experiment_ids: List[str] = []
        for status in ("open", "claimed"):
            listing = await self.plugin.controller.run_environment("science", "task_list", status=status)
            tasks = (listing or {}).get("tasks", []) if isinstance(listing, dict) else []
            for task in tasks:
                if str((task or {}).get("task_type") or "") != "experiment":
                    continue
                tid = str((task or {}).get("task_id") or "")
                if tid:
                    known_experiment_ids.append(tid)
        if known_experiment_ids:
            return {
                "spawned": False,
                "reason": "experiment_task_already_pending",
                "existing_task_ids": known_experiment_ids,
            }

        payload = {"trigger": "retrieve_l0_ready"}
        if str(reason or "").strip():
            payload["reason"] = str(reason).strip()
        created = await self.plugin.controller.run_environment(
            "science",
            "task_create",
            task_type="experiment",
            payload=payload,
            priority=int(priority),
        )
        task = (created or {}).get("task") if isinstance(created, dict) else {}
        task_id = str((task or {}).get("task_id") or "")
        return {
            "spawned": bool(task_id),
            "reason": "spawned" if task_id else "task_create_failed",
            "task_id": task_id,
            "raw": created if isinstance(created, dict) else {"ok": False},
        }

    def rag_pause_active(self, current_tick: int) -> bool:
        return current_tick < int(getattr(self.plugin, "_rag_degraded_pause_until_tick", 0) or 0)
