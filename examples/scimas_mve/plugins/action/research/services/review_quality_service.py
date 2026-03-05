import asyncio
import json
import urllib.request
from collections import Counter
from typing import Any, Dict, List, Optional


class ReviewQualityService:
    """QGR helpers extracted from ResearchActionsPlugin."""

    def __init__(self, plugin: Any):
        self.plugin = plugin

    def _qgr_thresholds_for_stage(self, stage: Optional[str]) -> Dict[str, Any]:
        stage_raw = str(stage or "").strip().lower()
        if stage_raw in {"early", "run", "run_level", "post_experiment", "cold_start"}:
            return {
                "stage": "early",
                "min_issues": int(getattr(self.plugin, "_qgr_early_min_issue_count", 1)),
                "min_citations": int(getattr(self.plugin, "_qgr_early_min_citations", 1)),
                "min_relevance": float(getattr(self.plugin, "_qgr_early_relevance_threshold", 0.2)),
            }
        return {
            "stage": "prewrite",
            "min_issues": int(getattr(self.plugin, "_qgr_min_issue_count", 2)),
            "min_citations": int(getattr(self.plugin, "_qgr_min_citations", 3)),
            "min_relevance": float(getattr(self.plugin, "_qgr_relevance_threshold", 0.75)),
        }

    def normalize_review_issues(self, review_note: Dict[str, Any]) -> List[Dict[str, Any]]:
        issues: List[Dict[str, Any]] = []
        raw = review_note.get("issues")
        if isinstance(raw, list):
            for idx, item in enumerate(raw[:10], start=1):
                if not isinstance(item, dict):
                    continue
                evidence_refs = item.get("evidence_refs")
                if isinstance(evidence_refs, str):
                    evidence_refs = self.plugin._extract_evidence_refs(evidence_refs)
                elif isinstance(evidence_refs, list):
                    evidence_refs = [str(ref) for ref in evidence_refs if isinstance(ref, (str, int, float))]
                else:
                    evidence_refs = self.plugin._extract_evidence_refs(item.get("claim"))
                issue = {
                    "id": str(item.get("id") or f"I-{idx:03d}"),
                    "type": self.plugin._truncate(item.get("type"), 80) or "generic_issue",
                    "severity": self.plugin._clamp01(item.get("severity", 0.5)),
                    "claim": self.plugin._truncate(item.get("claim"), 420),
                    "evidence_refs": evidence_refs[:8],
                    "proposed_test": item.get("proposed_test") if isinstance(item.get("proposed_test"), dict) else {},
                    "suggested_fix": self.plugin._truncate(item.get("suggested_fix"), 320),
                }
                if issue["claim"]:
                    issues.append(issue)
        if not issues:
            summary = str(review_note.get("summary") or "")
            if summary:
                issues.append(
                    {
                        "id": "I-001",
                        "type": "summary_only",
                        "severity": 0.4,
                        "claim": self.plugin._truncate(summary, 420),
                        "evidence_refs": self.plugin._extract_evidence_refs(summary),
                        "proposed_test": {},
                        "suggested_fix": "",
                    }
                )
        return issues

    def heuristic_review_note(self, *, paper: Dict[str, Any], metrics: Dict[str, Any]) -> Dict[str, Any]:
        replication_ok = bool((metrics or {}).get("replication_ok", False))
        support_ratio = float((metrics or {}).get("replication_support_ratio", 0.0) or 0.0)
        dev_score = float((paper.get("claimed_results") or {}).get("dev_score_norm", 0.0) or 0.0)
        metric_name = str((paper.get("claimed_results") or {}).get("metric_name") or "score")
        run_id = str((paper.get("claimed_results") or {}).get("run_id") or "")
        citations = [str(x) for x in (paper.get("citations") or [])[:8]]
        observation_refs = [str(x) for x in (paper.get("observation_refs") or [])[:8]]
        baseline_refs = citations[:2] + observation_refs[:2]

        strengths = [
            {
                "id": "S-001",
                "claim": f"Reported normalized {metric_name} score is {dev_score:.4f} on tracked run.",
                "evidence": [f"RUN@{run_id}"] if run_id else [],
                "confidence": min(0.9, 0.45 + 0.45 * dev_score),
                "verification": {"kind": "replicate", "params": {"mode": "score_consistency", "target_run_id": run_id}},
            },
            {
                "id": "S-002",
                "claim": "Paper includes explicit evidence references for key claims.",
                "evidence": baseline_refs,
                "confidence": 0.6 if baseline_refs else 0.35,
                "verification": {"kind": "static_check", "params": {"focus": "evidence_map_non_empty"}},
            },
        ]
        issues: List[Dict[str, Any]] = []
        if not replication_ok:
            issues.append(
                {
                    "id": "I-001",
                    "type": "replication_gap",
                    "severity": 0.9,
                    "claim": (
                        f"Replication support ratio is {support_ratio:.2f}, below robust threshold "
                        f"{self.plugin._replicate_support_threshold:.2f}."
                    ),
                    "evidence_refs": [f"RUN@{run_id}"] if run_id else [],
                    "proposed_test": {
                        "kind": "replicate",
                        "params": {
                            "mode": "score_consistency",
                            "target_run_id": run_id,
                            "required_support_ratio": self.plugin._replicate_support_threshold,
                        },
                    },
                    "suggested_fix": "Run additional replication with varied seeds and compare support ratio.",
                }
            )
        if dev_score < 0.4:
            issues.append(
                {
                    "id": "I-002",
                    "type": "performance_ceiling",
                    "severity": 0.65,
                    "claim": "Current dev score remains low; solver and feature settings likely underfit.",
                    "evidence_refs": [f"RUN@{run_id}"] if run_id else [],
                    "proposed_test": {"kind": "ablation", "params": {"focus": "solver_hyperparams"}},
                    "suggested_fix": "Trigger another experiment with stronger feature capacity and regularization sweep.",
                }
            )
        if not citations:
            issues.append(
                {
                    "id": "I-003",
                    "type": "citation_gap",
                    "severity": 0.55,
                    "claim": "Paper lacks citation links to method/data evidence cards.",
                    "evidence_refs": [],
                    "proposed_test": {"kind": "static_check", "params": {"focus": "citations_present"}},
                    "suggested_fix": "Attach method/data citations to each main claim.",
                }
            )

        if not strengths:
            strengths = [
                {
                    "id": "S-001",
                    "claim": "Submission pipeline executed and produced evaluable artifact.",
                    "evidence": [f"raw_score={float(metrics.get('raw_score', 0.0) or 0.0):.6f}"],
                    "confidence": 0.55,
                    "verification": {"kind": "static_check", "params": {"focus": "submission_validity"}},
                }
            ]

        revision_actions = [item.get("suggested_fix") for item in issues if item.get("suggested_fix")]
        return {
            "summary": "Evidence-based review with both validated strengths and falsifiable concerns.",
            "stance": "weak_accept" if not issues else "weak_reject",
            "strengths": strengths[:4],
            "issues": issues[:8],
            "revision_actions": [x for x in revision_actions[:6] if x],
            "replication_focus": "score_consistency_under_new_seeds",
            "anti_flattery": {"non_evidence_praise": False},
            "paper_id": paper.get("paper_id"),
        }

    def score_review_quality(
        self,
        *,
        review_note: Dict[str, Any],
        issues: List[Dict[str, Any]],
        self_review: bool,
        replication_ok: bool,
    ) -> Dict[str, Any]:
        strengths = review_note.get("strengths")
        if not isinstance(strengths, list):
            strengths = []

        revision_actions = self.plugin._safe_text_list(review_note.get("revision_actions"), limit=10, item_limit=220)
        issue_count = len(issues)
        major_count = sum(1 for i in issues if self.plugin._clamp01(i.get("severity")) >= 0.8)
        evidence_linked_count = sum(1 for i in issues if bool(i.get("evidence_refs")))
        actionable_count = len(revision_actions)
        severity_avg = (
            sum(self.plugin._clamp01(i.get("severity")) for i in issues) / max(1, issue_count) if issue_count else 0.0
        )
        evidence_ratio = evidence_linked_count / max(1, issue_count)
        issue_density = min(1.0, issue_count / max(1, self.plugin._review_min_issue_count))
        actionability = min(1.0, actionable_count / max(1, self.plugin._review_min_revision_actions))

        praise_words = (
            "excellent",
            "outstanding",
            "impressive",
            "great",
            "fantastic",
            "solid",
            "well done",
            "novel",
            "strong",
        )
        praise_blob = " ".join(
            [
                str(review_note.get("summary") or ""),
                " ".join(str(x) for x in strengths if not isinstance(x, dict)),
                " ".join(str((x or {}).get("claim") or "") for x in strengths if isinstance(x, dict)),
            ]
        ).lower()
        praise_hits = sum(praise_blob.count(word) for word in praise_words)

        flattery_penalty = 0.0
        if praise_hits > 0 and issue_count == 0:
            flattery_penalty += self.plugin._review_flattery_penalty
        elif praise_hits > (issue_count + 1):
            flattery_penalty += self.plugin._review_flattery_penalty * 0.5

        shallow_penalty = 0.0
        if issue_count < self.plugin._review_min_issue_count:
            shallow_penalty += self.plugin._review_shallow_penalty
        if actionable_count < self.plugin._review_min_revision_actions:
            shallow_penalty += self.plugin._review_shallow_penalty * 0.5

        self_penalty = self.plugin._review_self_review_penalty if self_review else 0.0
        no_issue_but_failed_replication = (issue_count == 0) and (not replication_ok)
        if no_issue_but_failed_replication:
            shallow_penalty += self.plugin._review_shallow_penalty

        critique_score = (
            0.40 * issue_density
            + 0.20 * severity_avg
            + 0.20 * evidence_ratio
            + 0.20 * actionability
            - flattery_penalty
            - shallow_penalty
            - self_penalty
        )
        critique_score = max(0.0, min(1.0, critique_score))
        needs_revision = bool(
            (major_count > 0) or (critique_score < self.plugin._review_revision_trigger_score) or (not replication_ok)
        )
        return {
            "issue_count": issue_count,
            "major_issue_count": major_count,
            "evidence_linked_issue_count": evidence_linked_count,
            "actionable_count": actionable_count,
            "severity_avg": severity_avg,
            "praise_hits": praise_hits,
            "flattery_penalty": flattery_penalty,
            "shallow_penalty": shallow_penalty,
            "self_review_penalty": self_penalty,
            "no_issue_but_failed_replication": no_issue_but_failed_replication,
            "critique_score": critique_score,
            "needs_revision": needs_revision,
        }

    async def qdrant_search_similarity(self, *, vector: List[float], collection: str) -> Optional[float]:
        if not self.plugin._vdh_qdrant_url or not collection or not vector:
            return None
        url = f"{self.plugin._vdh_qdrant_url}/collections/{collection}/points/search"
        headers = {"Content-Type": "application/json"}
        if self.plugin._vdh_qdrant_api_key:
            headers["api-key"] = self.plugin._vdh_qdrant_api_key
        payload = {"vector": vector, "limit": 1, "with_payload": False, "with_vector": False}
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )

        def _send() -> Optional[float]:
            with urllib.request.urlopen(req, timeout=6) as resp:
                body = resp.read().decode("utf-8", errors="replace")
            data = json.loads(body)
            result = data.get("result")
            if isinstance(result, list) and result:
                score = (result[0] or {}).get("score")
                if isinstance(score, (int, float)):
                    return float(score)
            return None

        try:
            return await asyncio.to_thread(_send)
        except Exception:
            return None

    async def qgr_relevance_score(self, *, review_note: Dict[str, Any], context_text: str) -> Dict[str, Any]:
        review_text = " ".join(
            [
                str(review_note.get("summary") or ""),
                " ".join(
                    str((i or {}).get("claim") or "")
                    for i in (review_note.get("issues") or [])
                    if isinstance(i, dict)
                ),
                " ".join(self.plugin._safe_text_list(review_note.get("revision_actions"), limit=10, item_limit=220)),
            ]
        ).strip()
        token_score = self.plugin._counter_cosine(
            Counter(self.plugin._text_tokens(review_text)),
            Counter(self.plugin._text_tokens(context_text)),
        )
        vector_score = None
        if self.plugin._vdh_tei_enable and (self.plugin._vdh_tei_url or self.plugin._reward_tei_url):
            rv = await self.plugin._vdh_embed_text(review_text)
            cv = await self.plugin._vdh_embed_text(context_text)
            vector_score = self.plugin._vector_cosine(rv, cv)
        if isinstance(vector_score, (int, float)):
            score = 0.7 * float(vector_score) + 0.3 * float(token_score)
            source = "tei+token"
        else:
            score = float(token_score)
            source = "token_fallback"
        return {
            "score": max(0.0, min(1.0, score)),
            "source": source,
            "token_score": float(token_score),
            "vector_score": float(vector_score) if isinstance(vector_score, (int, float)) else None,
        }

    async def qgr_fact_check(self, issues: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not self.plugin._vdh_qdrant_enable or not self.plugin._vdh_qdrant_url:
            return {"hallucinated_count": 0, "checked": 0, "source": "disabled"}
        checked = 0
        hallucinated = 0
        details: List[Dict[str, Any]] = []
        for issue in (issues or [])[:8]:
            if not isinstance(issue, dict):
                continue
            issue_type = str(issue.get("type") or "").lower()
            claim = str(issue.get("claim") or "")
            if not claim.strip():
                continue
            if not any(k in issue_type for k in ("schema", "format", "type", "data", "replication")):
                continue
            checked += 1
            vec = await self.plugin._vdh_embed_text(claim)
            if not isinstance(vec, list) or not vec:
                details.append(
                    {
                        "issue_id": issue.get("id"),
                        "issue_type": issue.get("type"),
                        "support_score": None,
                        "supported": True,
                        "skipped": "embedding_unavailable",
                    }
                )
                continue
            score = await self.qdrant_search_similarity(vector=vec or [], collection=self.plugin._vdh_qdrant_collection)
            score_val = float(score) if isinstance(score, (int, float)) else 0.0
            supported = score_val >= self.plugin._qgr_fact_support_threshold
            if not supported:
                hallucinated += 1
            details.append(
                {
                    "issue_id": issue.get("id"),
                    "issue_type": issue.get("type"),
                    "support_score": score_val,
                    "supported": supported,
                }
            )
        return {
            "hallucinated_count": hallucinated,
            "checked": checked,
            "details": details,
            "source": "qdrant_schema_collection",
        }

    async def qgr_validate_review(
        self,
        *,
        review_note: Dict[str, Any],
        issues: List[Dict[str, Any]],
        context_text: str,
        stage: Optional[str] = None,
    ) -> Dict[str, Any]:
        thresholds = self._qgr_thresholds_for_stage(stage=stage)
        citation_set = set()
        for issue in issues:
            if not isinstance(issue, dict):
                continue
            for ref in (issue.get("evidence_refs") or []):
                ref_s = str(ref)
                if ref_s:
                    citation_set.add(ref_s)
        min_issue_ok = len(issues) >= int(thresholds.get("min_issues", 1))
        min_citation_ok = len(citation_set) >= int(thresholds.get("min_citations", 1))
        relevance = await self.qgr_relevance_score(review_note=review_note, context_text=context_text)
        relevance_ok = float(relevance.get("score", 0.0) or 0.0) >= float(thresholds.get("min_relevance", 0.0))
        fact_check = await self.qgr_fact_check(issues)
        fact_ok = int(fact_check.get("hallucinated_count", 0) or 0) == 0
        valid = bool(min_issue_ok and min_citation_ok and relevance_ok and fact_ok)
        return {
            "valid": valid,
            "stage": str(thresholds.get("stage") or "prewrite"),
            "thresholds": {
                "min_issues": int(thresholds.get("min_issues", 1)),
                "min_citations": int(thresholds.get("min_citations", 1)),
                "min_relevance": float(thresholds.get("min_relevance", 0.0)),
            },
            "metrics": {
                "issue_count": len(issues),
                "citation_count": len(citation_set),
                "relevance_score": float(relevance.get("score", 0.0) or 0.0),
                "relevance_source": relevance.get("source"),
                "hallucinated_count": int(fact_check.get("hallucinated_count", 0) or 0),
                "fact_checked_count": int(fact_check.get("checked", 0) or 0),
            },
            "checks": {
                "issue_count_ok": min_issue_ok,
                "citation_count_ok": min_citation_ok,
                "relevance_ok": relevance_ok,
                "fact_ok": fact_ok,
            },
            "fact_check": fact_check,
        }

    def qgr_predictive_bonus(
        self,
        *,
        issues: List[Dict[str, Any]],
        run_history: List[Dict[str, Any]],
        target_run_id: Optional[str],
    ) -> Dict[str, Any]:
        target = None
        if target_run_id:
            for run in reversed(run_history or []):
                if str(run.get("run_id") or "") == str(target_run_id):
                    target = run
                    break
        if target is None and run_history:
            target = run_history[-1]
        merged = ""
        if isinstance(target, dict):
            merged = "\n".join(
                [
                    str(target.get("error") or ""),
                    str(target.get("stderr_tail") or ""),
                    str(target.get("fallback_reason") or ""),
                ]
            ).lower()
        matched = []
        for issue in issues:
            if not isinstance(issue, dict):
                continue
            claim = str(issue.get("claim") or "").lower()
            issue_type = str(issue.get("type") or "").lower()
            if ("oom" in claim or "memory" in claim or "oom" in issue_type) and ("killed" in merged or "out of memory" in merged):
                matched.append(str(issue.get("id") or ""))
            if ("schema" in claim or "format" in claim or "type" in claim) and (
                "typeerror" in merged or "unhashable" in merged or "submission" in merged
            ):
                matched.append(str(issue.get("id") or ""))
        matched = [m for m in matched if m]
        return {
            "matched_issue_ids": sorted(set(matched)),
            "bonus": float(self.plugin._qgr_predictive_bonus_reward if matched else 0.0),
        }

    async def spawn_qgr_followup_tasks(
        self,
        *,
        paper_id: Optional[str],
        run_id: Optional[str],
        score: float,
        issues: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        created: List[str] = []
        if score >= 0.6 or not issues:
            return {"created": created, "reason": "score_ok_or_no_issues"}
        if paper_id:
            try:
                write_res = await self.plugin.controller.run_environment(
                    "science",
                    "task_create",
                    task_type="write",
                    payload={"paper_id": paper_id, "revision_reason": "qgr_low_score"},
                    priority=9,
                )
                if isinstance(write_res, dict) and write_res.get("ok"):
                    tid = ((write_res.get("task") or {}).get("task_id"))
                    if tid:
                        created.append(str(tid))
            except Exception:
                pass
        if run_id:
            try:
                exp_res = await self.plugin.controller.run_environment(
                    "science",
                    "task_create",
                    task_type="experiment",
                    payload={"from_run_id": run_id, "revision_reason": "qgr_low_score"},
                    priority=8,
                )
                if isinstance(exp_res, dict) and exp_res.get("ok"):
                    tid = ((exp_res.get("task") or {}).get("task_id"))
                    if tid:
                        created.append(str(tid))
            except Exception:
                pass
        return {"created": created, "reason": "low_score_spawned"}

    async def spawn_review_validation_tasks(
        self,
        *,
        paper_id: str,
        reviewer_id: str,
        review_note: Dict[str, Any],
        critique_quality: Dict[str, Any],
    ) -> Dict[str, Any]:
        listed = await self.plugin.controller.run_environment("science", "task_list")
        tasks = (listed or {}).get("tasks", []) if isinstance(listed, dict) else []
        existing = []
        for task in tasks:
            payload = task.get("payload") or {}
            if str(payload.get("paper_id") or "") != str(paper_id):
                continue
            if str(payload.get("revision_reason") or "") != "review_validation":
                continue
            if str(task.get("task_type") or "") in {"verify_strength", "verify_issue", "write"}:
                existing.append(task.get("task_id"))
        if existing:
            return {"created": [], "skipped_existing": existing}

        created: List[str] = []
        strengths = review_note.get("strengths")
        if not isinstance(strengths, list):
            strengths = []
        issues = review_note.get("issues")
        if not isinstance(issues, list):
            issues = []

        for strength in strengths[:6]:
            if not isinstance(strength, dict):
                continue
            test = strength.get("verification")
            if not isinstance(test, dict):
                continue
            created_task = await self.plugin.controller.run_environment(
                "science",
                "task_create",
                task_type="verify_strength",
                payload={
                    "paper_id": paper_id,
                    "reviewer_id": reviewer_id,
                    "strength": strength,
                    "test": test,
                    "revision_reason": "review_validation",
                },
                priority=8,
            )
            if isinstance(created_task, dict):
                tid = ((created_task.get("task") or {}).get("task_id")) if created_task.get("ok") else None
                if tid:
                    created.append(tid)

        for issue in issues[:8]:
            if not isinstance(issue, dict):
                continue
            test = issue.get("proposed_test")
            if not isinstance(test, dict):
                continue
            sev = self.plugin._clamp01(issue.get("severity", 0.5))
            created_task = await self.plugin.controller.run_environment(
                "science",
                "task_create",
                task_type="verify_issue",
                payload={
                    "paper_id": paper_id,
                    "reviewer_id": reviewer_id,
                    "issue": issue,
                    "test": test,
                    "revision_reason": "review_validation",
                },
                priority=10 if sev >= 0.8 else 9,
            )
            if isinstance(created_task, dict):
                tid = ((created_task.get("task") or {}).get("task_id")) if created_task.get("ok") else None
                if tid:
                    created.append(tid)

        write_task_id = None
        if bool(critique_quality.get("needs_revision")):
            deps = created[: min(3, len(created))]
            write_task = await self.plugin.controller.run_environment(
                "science",
                "task_create",
                task_type="write",
                payload={
                    "paper_id": paper_id,
                    "revision_reason": "review_validation",
                    "reviewer_id": reviewer_id,
                    "critique_score": float(critique_quality.get("critique_score", 0.0) or 0.0),
                },
                priority=10,
                depends_on=deps,
            )
            if isinstance(write_task, dict) and write_task.get("ok"):
                write_task_id = (write_task.get("task") or {}).get("task_id")
                if write_task_id:
                    created.append(write_task_id)

        return {"created": created, "write_task_id": write_task_id, "skipped_existing": []}
