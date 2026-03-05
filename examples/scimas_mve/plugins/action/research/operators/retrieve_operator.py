import json
from datetime import datetime
from typing import Any, Dict, List, Optional

from agentkernel_standalone.types.schemas.action import ActionResult


class RetrieveOperator:
    """LLM-driven retrieve operator with RAG evidence, guardrails and graded fallback."""

    def __init__(self, plugin: Any):
        self.plugin = plugin

    def _context_bundle(
        self,
        *,
        world_spec: Dict[str, Any],
        data_card: Optional[Dict[str, Any]],
        notes: List[Dict[str, Any]],
        observations: List[Dict[str, Any]],
        topic: Optional[str],
    ) -> Dict[str, Any]:
        metric = str(world_spec.get("metric") or "").strip()
        category = str(world_spec.get("category") or "").strip()
        problem = str(world_spec.get("research_problem") or "").strip()
        task_name = str(world_spec.get("task_name") or "").strip()

        risk_flags = self.plugin._safe_text_list(
            (data_card or {}).get("risk_flags") if isinstance(data_card, dict) else [],
            limit=8,
            item_limit=80,
        )
        trigger_terms = list(risk_flags)
        lower_blob = " ".join(x.lower() for x in risk_flags)
        if any(tok in lower_blob for tok in ("imbalance", "skew")):
            trigger_terms.append("class imbalance")
        if any(tok in lower_blob for tok in ("long", "length", "token")):
            trigger_terms.append("long text")
        if "time series" in category.lower():
            trigger_terms.append("irregular time series")
        recent_errors = [
            self.plugin._truncate((o or {}).get("error"), 180)
            for o in observations[-8:]
            if isinstance(o, dict) and str((o or {}).get("error") or "").strip()
        ]
        query_text = " | ".join(
            [
                f"task={task_name}",
                f"metric={metric}",
                f"category={category}",
                f"problem={problem}",
                f"topic={topic or 'task_baselines'}",
                f"risk_flags={json.dumps(risk_flags[:6], ensure_ascii=False)}",
                f"recent_errors={json.dumps(recent_errors[:5], ensure_ascii=False)}",
            ]
        )
        return {
            "query_text": query_text,
            "filters": {
                "task_name": task_name,
                "metric": metric,
                "category": category,
            },
            "trigger_terms": trigger_terms[:12],
            "recent_errors": recent_errors[:6],
            "notes_count": len(notes or []),
            "observations_count": len(observations or []),
        }

    def _build_evidence_pack(self, *, selected_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        evidence: List[Dict[str, Any]] = []
        for idx, row in enumerate(selected_rows or [], start=1):
            source_type = str(row.get("source_type") or "unknown")
            source_id = str(row.get("source_id") or f"unknown_{idx}")
            evidence_id = f"EVID-{idx:04d}"
            evidence.append(
                {
                    "evidence_id": evidence_id,
                    "source_collection": str(row.get("source_collection") or ""),
                    "source_type": source_type,
                    "source_id": source_id,
                    "tags": list(row.get("tags") or []),
                    "chunk_text": str(row.get("text") or ""),
                    "match_score": float(row.get("score", 0.0) or 0.0),
                    "retrieval_mode": str(row.get("retrieval_mode") or "vector"),
                }
            )
        return evidence

    def _cold_start_seed_evidence(
        self,
        *,
        world_spec: Dict[str, Any],
        data_card: Optional[Dict[str, Any]],
        template_card: Optional[Dict[str, Any]],
        topic: str,
        start_index: int,
        needed: int,
    ) -> List[Dict[str, Any]]:
        if needed <= 0:
            return []
        task_name = str(world_spec.get("task_name") or "").strip()
        metric = str(world_spec.get("metric") or "").strip()
        category = str(world_spec.get("category") or "").strip()
        problem = str(world_spec.get("research_problem") or "").strip()
        desc = ""
        for key in ("project_description", "project_description_md", "task_description", "description", "objective"):
            text = str(world_spec.get(key) or "").strip()
            if text:
                desc = text
                break
        if not desc and isinstance(world_spec.get("task_meta"), dict):
            meta = world_spec.get("task_meta") or {}
            for key in ("project_description", "task_description", "description", "objective"):
                text = str(meta.get(key) or "").strip()
                if text:
                    desc = text
                    break

        candidates: List[Dict[str, Any]] = []
        # Evidence 1: task spec / description.
        task_text_lines = [
            f"task={task_name}",
            f"metric={metric}",
            f"category={category}",
            f"problem={problem}",
            f"topic={topic or 'task_baselines'}",
        ]
        if desc:
            task_text_lines.append(f"description={self.plugin._truncate(desc, 1400)}")
        candidates.append(
            {
                "source_type": "task_spec",
                "source_id": f"seed:task_spec:{task_name or 'unknown'}",
                "tags": [metric, category, "task_spec", "seed"],
                "text": "\n".join([x for x in task_text_lines if str(x).strip()]),
            }
        )

        # Evidence 2+: baseline guidance from template card.
        template_baselines = (
            (template_card or {}).get("recommended_baselines")
            if isinstance((template_card or {}).get("recommended_baselines"), list)
            else []
        )
        for idx, baseline in enumerate(template_baselines[:4], start=1):
            if not isinstance(baseline, dict):
                continue
            steps = self.plugin._safe_text_list(
                baseline.get("key_steps") or baseline.get("implementation_steps"),
                limit=5,
                item_limit=180,
            )
            pitfalls = self.plugin._safe_text_list(
                baseline.get("pitfalls") or baseline.get("risks"),
                limit=4,
                item_limit=160,
            )
            text_parts = [
                f"name={self.plugin._truncate(baseline.get('name'), 120)}",
                f"use_when={self.plugin._truncate(baseline.get('use_when'), 220)}",
            ]
            if steps:
                text_parts.append("steps=" + "; ".join(steps))
            if pitfalls:
                text_parts.append("pitfalls=" + "; ".join(pitfalls))
            candidates.append(
                {
                    "source_type": "method_card",
                    "source_id": f"seed:baseline:{task_name or 'unknown'}:{idx}",
                    "tags": [metric, category, "baseline", "seed"],
                    "text": " | ".join([x for x in text_parts if str(x).strip()]),
                }
            )

        # Evidence: evaluation protocol.
        protocol = self.plugin._safe_text_list(
            (template_card or {}).get("evaluation_protocol"),
            limit=8,
            item_limit=180,
        )
        if not protocol:
            protocol = [
                "align optimization with metric direction on dev/proxy split",
                "validate submission schema and row count before evaluate",
                "track seed stability before promoting best run",
            ]
        candidates.append(
            {
                "source_type": "method_card",
                "source_id": f"seed:eval_protocol:{task_name or 'unknown'}",
                "tags": [metric, category, "evaluation_protocol", "seed"],
                "text": "evaluation_protocol=" + " ; ".join(protocol[:6]),
            }
        )

        # Optional evidence: data-card risk summary.
        if isinstance(data_card, dict):
            risk_flags = self.plugin._safe_text_list(data_card.get("risk_flags"), limit=6, item_limit=100)
            target_col = str(data_card.get("target_column") or "")
            if risk_flags or target_col:
                dc_text = f"target_column={target_col}; risk_flags={'; '.join(risk_flags)}"
                candidates.append(
                    {
                        "source_type": "data_card",
                        "source_id": f"seed:data_card:{task_name or 'unknown'}",
                        "tags": [metric, category, "data_card", "seed"],
                        "text": dc_text,
                    }
                )

        # Deduplicate and cut.
        seen_text = set()
        selected: List[Dict[str, Any]] = []
        for item in candidates:
            text = str(item.get("text") or "").strip()
            if not text:
                continue
            text_key = text[:1200]
            if text_key in seen_text:
                continue
            seen_text.add(text_key)
            selected.append(item)
            if len(selected) >= max(needed, int(self.plugin._retrieve_min_evidence)):
                break

        seeded: List[Dict[str, Any]] = []
        for offset, item in enumerate(selected, start=0):
            seeded.append(
                {
                    "evidence_id": f"EVID-{start_index + offset:04d}",
                    "source_collection": str(getattr(self.plugin, "_rag_collection_literature", "") or "literature_chunks"),
                    "source_type": str(item.get("source_type") or "seed"),
                    "source_id": str(item.get("source_id") or f"seed:{offset + 1}"),
                    "tags": list(item.get("tags") or []),
                    "chunk_text": str(item.get("text") or ""),
                    "match_score": 0.31,
                    "retrieval_mode": "seed",
                }
            )
        return seeded

    def _cold_start_seed_docs(
        self,
        *,
        world_spec: Dict[str, Any],
        agent_id: str,
        data_card: Optional[Dict[str, Any]],
        template_card: Optional[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        docs: List[Dict[str, Any]] = []
        docs.extend(
            self.plugin._rag_docs_from_world_spec(
                world_spec=world_spec,
                agent_id=agent_id,
                action="retrieve_seed",
            )
        )
        if isinstance(data_card, dict):
            docs.extend(
                self.plugin._rag_docs_from_data_card(
                    world_spec=world_spec,
                    agent_id=agent_id,
                    data_card=data_card,
                    action="retrieve_seed",
                )
            )
        if isinstance(template_card, dict):
            docs.extend(
                self.plugin._rag_docs_from_method_card(
                    world_spec=world_spec,
                    agent_id=agent_id,
                    method_card=template_card,
                    action="retrieve_seed",
                )
            )
            protocol = self.plugin._safe_text_list(
                template_card.get("evaluation_protocol"),
                limit=8,
                item_limit=180,
            )
            if protocol:
                source_id = f"eval_protocol:{self.plugin._rag_hash(str(world_spec.get('task_name') or 'task'))[:12]}"
                doc = self.plugin._rag_doc_base(
                    world_spec=world_spec,
                    agent_id=agent_id,
                    source_type="method_card",
                    source_id=source_id,
                    action="retrieve_seed",
                    tags=["evaluation_protocol", str(world_spec.get("metric") or ""), str(world_spec.get("category") or "")],
                    quality=0.78,
                )
                doc["text"] = "evaluation_protocol=" + " ; ".join(protocol[:6])
                docs.append(doc)
        return [d for d in docs if isinstance(d, dict) and str(d.get("text") or "").strip()]

    def _sanitize_method_card(self, card: Dict[str, Any], *, world_spec: Dict[str, Any], topic: str) -> Dict[str, Any]:
        result = dict(card or {})
        result["ok"] = bool(result.get("ok", True))
        result["card_type"] = "method_card"
        result["version"] = str(result.get("version") or "v2")
        result["task_name"] = str(result.get("task_name") or world_spec.get("task_name") or "")
        result["topic"] = str(result.get("topic") or topic or "task_baselines")
        result["metric"] = str(result.get("metric") or world_spec.get("metric") or "")
        result["category"] = str(result.get("category") or world_spec.get("category") or "")
        result["research_problem"] = str(
            result.get("research_problem") or world_spec.get("research_problem") or ""
        )

        if not isinstance(result.get("task_summary"), dict):
            result["task_summary"] = {
                "problem_type": str(world_spec.get("category") or ""),
                "data_signals": [],
                "risk_flags": [],
            }
        for key in ("baseline_candidates", "failure_playbook", "ablation_plan"):
            if not isinstance(result.get(key), list):
                result[key] = []
        if not isinstance(result.get("experiment_roadmap"), dict):
            result["experiment_roadmap"] = {"stage_1": [], "stage_2": [], "stop_conditions": []}
        if not isinstance(result.get("citation_map"), dict):
            result["citation_map"] = {}
        if not isinstance(result.get("quality"), dict):
            result["quality"] = {}
        return result

    def _inject_template_baselines(
        self,
        *,
        method_card: Dict[str, Any],
        template_card: Dict[str, Any],
        evidence_pack: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        card = dict(method_card or {})
        baseline_candidates = (
            card.get("baseline_candidates") if isinstance(card.get("baseline_candidates"), list) else []
        )
        if len(baseline_candidates) >= self.plugin._retrieve_min_baselines:
            return card
        template_baselines = (
            template_card.get("recommended_baselines")
            if isinstance(template_card.get("recommended_baselines"), list)
            else []
        )
        for idx, baseline in enumerate(template_baselines, start=1):
            if not isinstance(baseline, dict):
                continue
            evidence_refs = [str(e.get("evidence_id")) for e in evidence_pack[:2] if isinstance(e, dict)]
            baseline_candidates.append(
                {
                    "name": str(baseline.get("name") or f"template_baseline_{idx}"),
                    "priority": len(baseline_candidates) + 1,
                    "implementation_steps": self.plugin._safe_text_list(
                        baseline.get("key_steps"), limit=6, item_limit=160
                    ),
                    "hyperparam_ranges": {},
                    "expected_gain": self.plugin._truncate(
                        baseline.get("use_when"), 180
                    ),
                    "risks": self.plugin._safe_text_list(baseline.get("pitfalls"), limit=4, item_limit=140),
                    "evidence_refs": evidence_refs or ["unknown"],
                }
            )
            if len(baseline_candidates) >= self.plugin._retrieve_min_baselines:
                break
        card["baseline_candidates"] = baseline_candidates[:8]
        return card

    def _ensure_compat_fields(self, card: Dict[str, Any], evidence_pack: List[Dict[str, Any]]) -> Dict[str, Any]:
        result = dict(card or {})
        baseline_candidates = (
            result.get("baseline_candidates") if isinstance(result.get("baseline_candidates"), list) else []
        )
        recommended_baselines = []
        for item in baseline_candidates[:8]:
            if not isinstance(item, dict):
                continue
            recommended_baselines.append(
                {
                    "name": item.get("name"),
                    "use_when": item.get("expected_gain"),
                    "key_steps": self.plugin._safe_text_list(
                        item.get("implementation_steps"), limit=5, item_limit=140
                    ),
                    "pitfalls": self.plugin._safe_text_list(item.get("risks"), limit=4, item_limit=140),
                }
            )
        result["recommended_baselines"] = recommended_baselines
        refs = []
        for ev in evidence_pack[: max(1, self.plugin._retrieve_min_evidence)]:
            if not isinstance(ev, dict):
                continue
            refs.append(
                {
                    "citation_id": ev.get("evidence_id"),
                    "title": f"{ev.get('source_type')}:{ev.get('source_id')}",
                    "snippet": self.plugin._truncate(ev.get("chunk_text"), 180),
                }
            )
        result["task_evidence_refs"] = refs
        result.setdefault(
            "evaluation_protocol",
            [
                "align dev optimization with official metric direction",
                "enforce submission schema checks each run",
                "track seed stability before promoting best run",
            ],
        )
        if not isinstance(result.get("common_pitfalls"), list):
            pitfalls = []
            for p in (result.get("failure_playbook") or [])[:6]:
                if isinstance(p, dict):
                    pitfalls.append(str(p.get("error_type") or "runtime_risk"))
            result["common_pitfalls"] = pitfalls
        return result

    def _strengthen_evidence_links(self, *, card: Dict[str, Any], evidence_pack: List[Dict[str, Any]]) -> Dict[str, Any]:
        result = dict(card or {})

        evidence_ids = []
        for ev in (evidence_pack or []):
            if not isinstance(ev, dict):
                continue
            eid = str(ev.get("evidence_id") or "").strip()
            if not eid:
                continue
            evidence_ids.append(eid)
        evidence_ids = list(dict.fromkeys(evidence_ids))

        if not isinstance(result.get("citation_map"), dict):
            result["citation_map"] = {}
        citation_map = dict(result.get("citation_map") or {})

        existing_refs = set()

        def _add_ref(value: Any) -> None:
            text = str(value or "").strip()
            if not text:
                return
            if text.lower() in {"unknown", "none", "null", "n/a", "-"}:
                return
            existing_refs.add(text)

        for refs in citation_map.values():
            if isinstance(refs, list):
                for ref in refs:
                    _add_ref(ref)

        baseline_candidates = (
            result.get("baseline_candidates") if isinstance(result.get("baseline_candidates"), list) else []
        )
        for baseline in baseline_candidates:
            if not isinstance(baseline, dict):
                continue
            refs = baseline.get("evidence_refs") if isinstance(baseline.get("evidence_refs"), list) else []
            refs = [str(x or "").strip() for x in refs if str(x or "").strip()]
            refs = [x for x in refs if x.lower() not in {"unknown", "none", "null", "n/a", "-"}]
            if not refs and evidence_ids:
                refs = evidence_ids[: max(1, min(2, len(evidence_ids)))]
                baseline["evidence_refs"] = refs
            for ref in refs:
                _add_ref(ref)

        for key in ("failure_playbook", "ablation_plan"):
            items = result.get(key) if isinstance(result.get(key), list) else []
            for item in items:
                if not isinstance(item, dict):
                    continue
                refs = item.get("evidence_refs") if isinstance(item.get("evidence_refs"), list) else []
                if not refs and evidence_ids:
                    item["evidence_refs"] = evidence_ids[:1]
                    refs = item["evidence_refs"]
                for ref in refs:
                    _add_ref(ref)

        task_refs = result.get("task_evidence_refs") if isinstance(result.get("task_evidence_refs"), list) else []
        if not task_refs and evidence_ids:
            task_refs = []
        known_task_ref_ids = set()
        for item in task_refs:
            if not isinstance(item, dict):
                continue
            cid = str(item.get("citation_id") or "").strip()
            if cid:
                known_task_ref_ids.add(cid)
                _add_ref(cid)
        if evidence_ids:
            for eid in evidence_ids:
                if eid in known_task_ref_ids:
                    continue
                if len(task_refs) >= max(1, self.plugin._retrieve_min_evidence):
                    break
                task_refs.append(
                    {
                        "citation_id": eid,
                        "title": "evidence:auto",
                        "snippet": "",
                    }
                )
                known_task_ref_ids.add(eid)
                _add_ref(eid)
        if task_refs:
            result["task_evidence_refs"] = task_refs

        target_count = max(1, int(self.plugin._retrieve_min_evidence))
        if evidence_ids:
            missing = [eid for eid in evidence_ids if eid not in existing_refs]
            idx = 0
            while len(existing_refs) < target_count and idx < len(missing):
                key = f"evidence:auto_{idx + 1}"
                while key in citation_map:
                    idx += 1
                    key = f"evidence:auto_{idx + 1}"
                citation_map[key] = [missing[idx]]
                existing_refs.add(missing[idx])
                idx += 1
        result["citation_map"] = citation_map
        if evidence_pack:
            result["evidence_pack"] = list(evidence_pack[:24])
        return result

    def _quality_report(
        self,
        *,
        method_card: Dict[str, Any],
    ) -> Dict[str, Any]:
        recovery = self.plugin._get_service("_recovery_service", None)
        if recovery is not None and hasattr(recovery, "method_card_quality_report"):
            return recovery.method_card_quality_report(method_card)
        return {
            "schema_valid": False,
            "citation_coverage": 0.0,
            "executable_minimum": False,
            "level": "L2",
            "degraded": True,
            "fail_reasons": ["recovery_service_unavailable"],
            "baseline_count": 0,
            "evidence_count": 0,
        }

    async def _log_retrieve_pipeline(
        self,
        *,
        agent_id: str,
        phase: str,
        payload: Dict[str, Any],
    ) -> None:
        await self.plugin._append_jsonl(
            self.plugin._retrieve_pipeline_log_path,
            {
                "ts": datetime.utcnow().isoformat() + "Z",
                "agent_id": agent_id,
                "phase": phase,
                **(payload or {}),
            },
        )

    async def _log_retrieve_guardrail(
        self,
        *,
        agent_id: str,
        quality: Dict[str, Any],
        source: str,
    ) -> None:
        await self.plugin._append_jsonl(
            self.plugin._retrieve_guardrail_log_path,
            {
                "ts": datetime.utcnow().isoformat() + "Z",
                "agent_id": agent_id,
                "source": source,
                "quality": quality,
            },
        )

    async def _log_retrieve_evidence(
        self,
        *,
        agent_id: str,
        evidence_pack: List[Dict[str, Any]],
        rag_status: str,
    ) -> None:
        await self.plugin._append_jsonl(
            self.plugin._retrieve_evidence_log_path,
            {
                "ts": datetime.utcnow().isoformat() + "Z",
                "agent_id": agent_id,
                "rag_status": rag_status,
                "evidence_count": len(evidence_pack or []),
                "evidence": [
                    {
                        "evidence_id": e.get("evidence_id"),
                        "source_collection": e.get("source_collection"),
                        "source_type": e.get("source_type"),
                        "source_id": e.get("source_id"),
                        "match_score": e.get("match_score"),
                        "retrieval_mode": e.get("retrieval_mode"),
                        "tags": list(e.get("tags") or []),
                        "chunk_text": self.plugin._truncate(e.get("chunk_text"), 220),
                    }
                    for e in (evidence_pack or [])[:24]
                    if isinstance(e, dict)
                ],
            },
        )

    def _build_method_note(self, *, method_card: Dict[str, Any], topic: str) -> Dict[str, Any]:
        baselines = (
            method_card.get("baseline_candidates")
            if isinstance(method_card.get("baseline_candidates"), list)
            else method_card.get("recommended_baselines")
        ) or []
        cards = []
        hints = []
        for idx, baseline in enumerate(baselines[: self.plugin._llm_max_cards], start=1):
            if not isinstance(baseline, dict):
                continue
            cid = f"M{idx:04d}"
            title = str(baseline.get("name") or f"baseline_{idx}")
            steps = self.plugin._safe_text_list(
                baseline.get("implementation_steps") or baseline.get("key_steps"),
                limit=3,
                item_limit=140,
            )
            risks = self.plugin._safe_text_list(
                baseline.get("risks") or baseline.get("pitfalls"),
                limit=3,
                item_limit=140,
            )
            refs = self.plugin._safe_text_list(baseline.get("evidence_refs"), limit=4, item_limit=80)
            text_parts = []
            if steps:
                text_parts.append("steps=" + "; ".join(steps))
            if risks:
                text_parts.append("risks=" + "; ".join(risks))
            if refs:
                text_parts.append("refs=" + ",".join(refs))
            cards.append(
                {
                    "citation_id": cid,
                    "kind": "method_card",
                    "title": title,
                    "text": " | ".join(text_parts) or "method baseline",
                }
            )
            hints.append(f"[{cid}] {title}")
        return {
            "topic": topic or "task_baselines",
            "hints": hints,
            "cards": cards,
            "task_name": method_card.get("task_name"),
            "source": "method_card_v2",
        }

    async def execute(self, agent_id: str, topic: Optional[str] = None, refresh: bool = False) -> ActionResult:
        ctx = await self.plugin._load_research_context(agent_id=agent_id, include_shared=True)
        if isinstance(ctx, dict):
            world_spec = dict(ctx.get("world_spec") or {})
            notes = list(ctx.get("notes") or [])
            observations = list(ctx.get("observations") or [])
            data_card = ctx.get("data_card")
        else:
            world_spec = dict(getattr(ctx, "world_spec", {}) or {})
            notes = list(getattr(ctx, "notes", []) or [])
            observations = list(getattr(ctx, "observations", []) or [])
            data_card = getattr(ctx, "data_card", None)
        if not world_spec:
            world_spec = await self.plugin.controller.run_environment("science", "get_world_spec")

        topic_val = topic or "task_baselines"
        fallback_card: Dict[str, Any] = {}
        if self.plugin._retrieve_template_fallback:
            fallback_card = await self.plugin.controller.run_environment(
                "science",
                "retrieve_method_card",
                agent_id=agent_id,
                topic=topic_val,
                refresh=bool(refresh),
            )

        # Bootstrap episode-level RAG knowledge before retrieval.
        await self.plugin._rag_bootstrap_episode_knowledge(
            agent_id=agent_id,
            world_spec=world_spec,
            data_card=data_card if isinstance(data_card, dict) else None,
            method_card=fallback_card if isinstance(fallback_card, dict) else None,
        )

        query_bundle = self._context_bundle(
            world_spec=world_spec,
            data_card=data_card if isinstance(data_card, dict) else None,
            notes=notes,
            observations=observations,
            topic=topic_val,
        )
        await self._log_retrieve_pipeline(
            agent_id=agent_id,
            phase="context_builder",
            payload={"query_bundle": query_bundle, "refresh": bool(refresh)},
        )

        rag_result = await self.plugin._rag_retrieve_context(
            agent_id=agent_id,
            action="retrieve_literature",
            run_id=None,
            paper_id=None,
            query_text=str(query_bundle.get("query_text") or ""),
            quotas={"method_card": 4, "diagnosis": 3, "observation": 3, "data_card": 2, "note": 2},
            notes=notes,
            observations=observations,
            data_card=data_card if isinstance(data_card, dict) else None,
            method_card=None,
        )
        selected_rows = list((rag_result or {}).get("selected") or [])
        evidence_pack = self._build_evidence_pack(selected_rows=selected_rows)
        if len(evidence_pack) < int(self.plugin._retrieve_min_evidence):
            needed = int(self.plugin._retrieve_min_evidence) - len(evidence_pack)
            seed_evidence = self._cold_start_seed_evidence(
                world_spec=world_spec,
                data_card=data_card if isinstance(data_card, dict) else None,
                template_card=fallback_card if isinstance(fallback_card, dict) else None,
                topic=topic_val,
                start_index=len(evidence_pack) + 1,
                needed=max(needed, 3),
            )
            if seed_evidence:
                evidence_pack.extend(seed_evidence)
                seed_docs = self._cold_start_seed_docs(
                    world_spec=world_spec,
                    agent_id=agent_id,
                    data_card=data_card if isinstance(data_card, dict) else None,
                    template_card=fallback_card if isinstance(fallback_card, dict) else None,
                )
                if seed_docs:
                    await self.plugin._rag_index_documents(
                        agent_id=agent_id,
                        action="retrieve_literature",
                        docs=seed_docs,
                    )
                await self._log_retrieve_pipeline(
                    agent_id=agent_id,
                    phase="cold_start_seed",
                    payload={
                        "seeded_count": len(seed_evidence),
                        "evidence_count_after_seed": len(evidence_pack),
                        "trigger": "insufficient_evidence_for_l0",
                    },
                )
        await self._log_retrieve_evidence(
            agent_id=agent_id,
            evidence_pack=evidence_pack,
            rag_status=str((rag_result or {}).get("status") or ""),
        )
        await self._log_retrieve_pipeline(
            agent_id=agent_id,
            phase="retriever",
            payload={
                "rag_status": (rag_result or {}).get("status"),
                "fallback_reason": (rag_result or {}).get("fallback_reason"),
                "selected_count": len(selected_rows),
            },
        )

        method_card_candidate: Dict[str, Any] = {}
        candidate_source = "none"
        if self.plugin._retrieve_v2_enable and self.plugin._llm_ready("retrieve_literature"):
            prompt = self.plugin._build_retrieve_method_prompt(
                world_spec=world_spec,
                data_card=data_card if isinstance(data_card, dict) else None,
                notes=notes,
                observations=observations,
                evidence_pack=evidence_pack,
                query_bundle=query_bundle,
                rag_status=str((rag_result or {}).get("status") or ""),
            )
            llm_result = await self.plugin._call_llm_json(
                agent_id=agent_id,
                action_name="retrieve_literature",
                prompt=prompt,
            )
            if llm_result.get("ok") and isinstance(llm_result.get("data"), dict):
                method_card_candidate = dict(llm_result.get("data") or {})
                candidate_source = "retrieve_v2_llm"
            else:
                await self._log_retrieve_pipeline(
                    agent_id=agent_id,
                    phase="composer",
                    payload={"llm_ok": False, "llm_reason": llm_result.get("reason")},
                )

        if not method_card_candidate and isinstance(fallback_card, dict):
            method_card_candidate = dict(fallback_card)
            candidate_source = "retrieve_v2_template"

        method_card_candidate = self._sanitize_method_card(
            method_card_candidate or {},
            world_spec=world_spec,
            topic=topic_val,
        )
        method_card_candidate = self._strengthen_evidence_links(
            card=method_card_candidate,
            evidence_pack=evidence_pack,
        )
        method_card_candidate["source"] = candidate_source or "retrieve_v2_template"

        quality_report = self._quality_report(method_card=method_card_candidate)
        await self._log_retrieve_guardrail(
            agent_id=agent_id,
            quality=quality_report,
            source=str(method_card_candidate.get("source") or ""),
        )

        final_card = dict(method_card_candidate)
        if quality_report.get("level") == "L2" and self.plugin._retrieve_degrade_allow:
            if isinstance(fallback_card, dict):
                final_card = self._sanitize_method_card(
                    fallback_card,
                    world_spec=world_spec,
                    topic=topic_val,
                )
                final_card = self._strengthen_evidence_links(card=final_card, evidence_pack=evidence_pack)
                final_card["source"] = "retrieve_v2_template"
                quality_report = self._quality_report(method_card=final_card)
            else:
                final_card = self._inject_template_baselines(
                    method_card=final_card,
                    template_card={},
                    evidence_pack=evidence_pack,
                )
                final_card = self._strengthen_evidence_links(card=final_card, evidence_pack=evidence_pack)
                quality_report = self._quality_report(method_card=final_card)
                final_card.setdefault("quality", {})
                final_card["quality"]["degraded"] = True
                final_card["quality"]["level"] = "L1"
                final_card["source"] = "retrieve_v2_hybrid"
        elif quality_report.get("level") == "L1":
            final_card = self._inject_template_baselines(
                method_card=final_card,
                template_card=fallback_card if isinstance(fallback_card, dict) else {},
                evidence_pack=evidence_pack,
            )
            final_card = self._strengthen_evidence_links(card=final_card, evidence_pack=evidence_pack)
            final_card["source"] = "retrieve_v2_hybrid"
            quality_report = self._quality_report(method_card=final_card)

        final_card["quality"] = {
            "schema_valid": bool(quality_report.get("schema_valid")),
            "citation_coverage": float(quality_report.get("citation_coverage", 0.0) or 0.0),
            "executable_minimum": bool(quality_report.get("executable_minimum")),
            "level": str(quality_report.get("level") or "L2"),
            "degraded": bool(quality_report.get("degraded")),
            "fail_reasons": list(quality_report.get("fail_reasons") or []),
        }
        final_card = self._ensure_compat_fields(final_card, evidence_pack)
        quality_level = str((final_card.get("quality") or {}).get("level") or "L2").upper()
        experiment_spawn_result: Dict[str, Any] = {
            "spawned": False,
            "reason": "not_triggered",
        }
        recovery_service = self.plugin._get_service("_recovery_service", None)
        method_quality_ok = bool(
            recovery_service is not None
            and hasattr(recovery_service, "method_card_quality_ok")
            and recovery_service.method_card_quality_ok(final_card)
        )

        if quality_level == "L2" or (quality_level == "L1" and not method_quality_ok):
            try:
                recovery_tasks = await self.plugin._enqueue_prereq_recovery_tasks(
                    failures=["need_method_card_quality_pass"]
                )
                await self._log_retrieve_pipeline(
                    agent_id=agent_id,
                    phase="recovery_tasks",
                    payload={
                        "trigger": "retrieve_quality_not_ready",
                        "quality_level": quality_level,
                        "method_quality_ok": method_quality_ok,
                        "recovery_tasks": recovery_tasks,
                    },
                )
            except Exception:
                try:
                    await self.plugin.controller.run_environment(
                        "science",
                        "task_create",
                        task_type="read",
                        payload={"reason": "retrieve_quality_not_ready", "topic": topic_val},
                        priority=9,
                    )
                except Exception:
                    pass
        elif quality_level in {"L0", "L1"} and method_quality_ok:
            if recovery_service is not None and hasattr(recovery_service, "ensure_experiment_spawned"):
                try:
                    experiment_spawn_result = await recovery_service.ensure_experiment_spawned(
                        reason=f"method_card_quality_{quality_level.lower()}",
                        priority=4,
                    )
                except Exception as e:
                    experiment_spawn_result = {
                        "spawned": False,
                        "reason": "spawn_exception",
                        "error": self.plugin._truncate(str(e), 200),
                    }
                await self._log_retrieve_pipeline(
                    agent_id=agent_id,
                    phase="spawn_experiment",
                    payload={
                        "quality_level": quality_level,
                        "method_quality_ok": method_quality_ok,
                        "spawn_result": experiment_spawn_result,
                    },
                )

        await self.plugin._set_state(agent_id, "method_card", final_card)
        try:
            await self.plugin.controller.run_environment(
                "science",
                "publish_method_card",
                agent_id=agent_id,
                method_card=final_card,
                refresh=bool(refresh),
            )
        except Exception:
            pass

        local_notes = list(getattr(ctx, "local_notes", []) or []) if not isinstance(ctx, dict) else list(ctx.get("local_notes") or [])
        method_note = self._build_method_note(method_card=final_card, topic=topic_val)
        local_notes.append(method_note)
        await self.plugin._set_state(agent_id, "notes", local_notes)
        await self.plugin._log_evidence_cards(agent_id, method_note, source="retrieve_literature")

        if self.plugin._rag_index_on_read:
            rag_docs = self.plugin._rag_docs_from_method_card(
                world_spec=world_spec,
                agent_id=agent_id,
                method_card=final_card,
                action="retrieve_literature",
            )
            rag_docs.extend(
                self.plugin._rag_docs_from_note(
                    world_spec=world_spec,
                    agent_id=agent_id,
                    note=method_note,
                    action="retrieve_literature",
                )
            )
            if rag_docs:
                await self.plugin._rag_index_documents(
                    agent_id=agent_id,
                    action="retrieve_literature",
                    docs=rag_docs,
                )

        level = str(((final_card.get("quality") or {}).get("level") or "L2")).upper()
        ok = bool(final_card.get("ok", False))
        if not ok:
            reward = -0.01
        elif level == "L0":
            reward = 0.02
        elif level == "L1":
            reward = 0.01
        else:
            reward = 0.002

        await self._log_retrieve_pipeline(
            agent_id=agent_id,
            phase="finalize",
            payload={
                "ok": ok,
                "quality": final_card.get("quality"),
                "source": final_card.get("source"),
                "reward": reward,
                "rag_status": rag_result.get("status"),
            },
        )

        ar = ActionResult.success(
            method_name="retrieve_literature",
            message="Method card generated by Retrieve V2." if ok else "Method retrieval failed.",
            data={
                "ok": ok,
                "method_card": final_card,
                "quality": final_card.get("quality"),
                "evidence_pack": evidence_pack,
                "rag_status": rag_result.get("status"),
                "experiment_spawn": experiment_spawn_result,
                "reward": reward,
                "effective_action": "retrieve_literature",
                "reward_components": {
                    "learning_reward": float(reward),
                    "retrieve_literature_reward": float(reward),
                },
            },
        )
        await self.plugin._append_trace(agent_id, "retrieve_literature", reward, ar.data or {})
        return ar
