import asyncio
import json
import os
import urllib.request
from collections import Counter
from typing import Any, Dict, List, Optional


class VDHService:
    """Validation-Driven Hypothesis gate service."""

    def __init__(self, plugin: Any):
        self.plugin = plugin

    def hypothesis_feasibility(self, world_spec: Dict[str, Any], plan_spec: Dict[str, Any]) -> Dict[str, Any]:
        plan_text = json.dumps(plan_spec or {}, ensure_ascii=False).lower()
        schema_markers = ("scoring_column", "task_manifest", "list", "target_column_hint", "submission csv")
        schema_safe = sum(1 for marker in schema_markers if marker in plan_text) >= 2
        resource_markers = ("sample", "batch", "chunk", "window", "stream")
        resource_by_text = any(marker in plan_text for marker in resource_markers)
        solver_spec = (plan_spec or {}).get("solver_spec") if isinstance((plan_spec or {}).get("solver_spec"), dict) else {}
        preprocess = solver_spec.get("preprocess") if isinstance(solver_spec.get("preprocess"), dict) else {}
        max_features = preprocess.get("max_features")
        resource_by_config = isinstance(max_features, (int, float)) and float(max_features) <= 50000
        schema_bonus = self.plugin._hypothesis_schema_bonus if schema_safe else 0.0
        resource_bonus = self.plugin._hypothesis_resource_bonus if (resource_by_text or resource_by_config) else 0.0
        feasibility = schema_bonus + resource_bonus
        reward = max(-0.02, min(0.08, 0.10 * feasibility - 0.01))
        return {
            "feasibility_score": float(feasibility),
            "schema_safe": bool(schema_safe),
            "resource_safe": bool(resource_by_text or resource_by_config),
            "schema_bonus": float(schema_bonus),
            "resource_bonus": float(resource_bonus),
            "reward": float(reward),
            "code_memory_mb": world_spec.get("code_memory_mb"),
        }

    async def vdh_qdrant_schema_constraints(self, world_spec: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if not self.plugin._vdh_qdrant_enable or not self.plugin._vdh_qdrant_url or not self.plugin._vdh_qdrant_collection:
            return None
        task_name = str(world_spec.get("task_name") or "").strip()
        if not task_name:
            return None
        url = f"{self.plugin._vdh_qdrant_url}/collections/{self.plugin._vdh_qdrant_collection}/points/scroll"
        headers = {"Content-Type": "application/json"}
        if self.plugin._vdh_qdrant_api_key:
            headers["api-key"] = self.plugin._vdh_qdrant_api_key
        payload = {
            "with_payload": True,
            "with_vector": False,
            "limit": 1,
            "filter": {"must": [{"key": "task_name", "match": {"value": task_name}}]},
        }
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )

        def _send() -> Optional[Dict[str, Any]]:
            with urllib.request.urlopen(req, timeout=6) as resp:
                body = resp.read().decode("utf-8", errors="replace")
            data = json.loads(body)
            result = data.get("result")
            points = result.get("points") if isinstance(result, dict) else result
            if isinstance(points, list) and points:
                payload_obj = (points[0] or {}).get("payload")
                if isinstance(payload_obj, dict):
                    return payload_obj
            return None

        try:
            return await asyncio.to_thread(_send)
        except Exception:
            return None

    def vdh_constraints_from_manifest_file(self, world_spec: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        task_path = str(world_spec.get("task_path") or "").strip()
        if not task_path:
            return None
        candidates = [
            os.path.join(task_path, ".task_manifest.json"),
            os.path.join(task_path, "task_manifest.json"),
            os.path.join(task_path, "metadata.yaml"),
            os.path.join(task_path, "metadata.yml"),
        ]
        for path in candidates:
            if not os.path.exists(path):
                continue
            try:
                if path.endswith(".json"):
                    with open(path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                else:
                    import yaml  # type: ignore

                    with open(path, "r", encoding="utf-8") as f:
                        data = yaml.safe_load(f) or {}
                if isinstance(data, dict):
                    return data
            except Exception:
                continue
        return None

    def vdh_normalize_constraints(
        self,
        *,
        source: str,
        raw: Optional[Dict[str, Any]],
        world_spec: Dict[str, Any],
        notes: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        warnings: List[str] = []
        raw = raw if isinstance(raw, dict) else {}
        scoring = raw.get("scoring_column")
        if not isinstance(scoring, list):
            info = raw.get("logging_info") if isinstance(raw.get("logging_info"), dict) else {}
            scoring = info.get("scoring_column")
        if not isinstance(scoring, list):
            scoring = []
        scoring_is_list = bool(isinstance(scoring, list) and len(scoring) > 0)

        target_hint = ""
        if isinstance(raw.get("target_column"), str):
            target_hint = str(raw.get("target_column"))
        elif isinstance(raw.get("label_column"), str):
            target_hint = str(raw.get("label_column"))
        elif isinstance(raw.get("submission"), dict) and isinstance((raw.get("submission") or {}).get("target"), str):
            target_hint = str((raw.get("submission") or {}).get("target"))
        elif scoring_is_list:
            target_hint = str(scoring[0])
        metric = str(raw.get("metric") or world_spec.get("metric") or "").strip()
        submission_requirements = []
        if isinstance(raw.get("submission_requirements"), list):
            submission_requirements = [str(x) for x in raw.get("submission_requirements")[:8]]
        if not submission_requirements and isinstance(raw.get("submission"), dict):
            submission_requirements = [self.plugin._truncate(json.dumps(raw.get("submission"), ensure_ascii=False), 280)]
        if not scoring_is_list and notes:
            notes_text = " ".join(self.plugin._note_to_text(n) for n in notes[-6:] if isinstance(n, dict)).lower()
            if "scoring_column" in notes_text and "list" in notes_text:
                scoring_is_list = True
                warnings.append("inferred_scoring_column_list_from_notes")
        if not target_hint:
            target_hint = "target"
            warnings.append("target_column_hint_fallback")
        return {
            "ok": True,
            "source": source,
            "constraints": {
                "scoring_column_is_list": bool(scoring_is_list),
                "target_column_hint": target_hint,
                "metric": metric,
                "submission_requirements": submission_requirements,
            },
            "warnings": warnings,
        }

    async def vdh_metadata_alignment(
        self,
        *,
        world_spec: Dict[str, Any],
        notes: Optional[List[Dict[str, Any]]],
        plan_spec: Dict[str, Any],
    ) -> Dict[str, Any]:
        source = "fallback"
        raw = None
        if self.plugin._vdh_qdrant_enable:
            raw = await self.vdh_qdrant_schema_constraints(world_spec)
            if isinstance(raw, dict):
                source = "qdrant"
        if not isinstance(raw, dict):
            raw = self.vdh_constraints_from_manifest_file(world_spec)
            if isinstance(raw, dict):
                source = "manifest"
        normalized = self.vdh_normalize_constraints(source=source, raw=raw, world_spec=world_spec, notes=notes)
        constraints = normalized.get("constraints") if isinstance(normalized.get("constraints"), dict) else {}
        scoring_is_list = bool(constraints.get("scoring_column_is_list", False))
        plan_text = json.dumps(plan_spec or {}, ensure_ascii=False).lower()
        target_cols = (plan_spec or {}).get("target_cols")
        solver_spec = (plan_spec or {}).get("solver_spec") if isinstance((plan_spec or {}).get("solver_spec"), dict) else {}
        solver_target_cols = solver_spec.get("target_cols")
        handles_list = False
        if isinstance(target_cols, list) and target_cols:
            handles_list = True
        if isinstance(solver_target_cols, list) and solver_target_cols:
            handles_list = True
        markers = (
            "scoring_column[0]",
            "scoring_column'][0]",
            "manifest['scoring_column'][0]",
            "manifest[\"scoring_column\"][0]",
            "target_cols",
        )
        if any(m in plan_text for m in markers):
            handles_list = True
        errors: List[str] = []
        if scoring_is_list and not handles_list:
            errors.append("scoring_column_is_list_but_plan_not_handled")
        normalized["ok"] = len(errors) == 0
        normalized["errors"] = errors
        normalized["plan_handles_scoring_list"] = bool(handles_list)
        return normalized

    def vdh_plan_validator(
        self,
        *,
        world_spec: Dict[str, Any],
        plan_spec: Dict[str, Any],
        data_card: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        errors: List[str] = []
        solver_spec = (plan_spec or {}).get("solver_spec")
        if not isinstance(solver_spec, dict):
            errors.append("solver_spec must be object")
            solver_spec = {}
        for key in ("input_columns", "target_cols"):
            if key in plan_spec and not isinstance(plan_spec.get(key), list):
                errors.append(f"{key} must be list")
            if key in solver_spec and not isinstance(solver_spec.get(key), list):
                errors.append(f"solver_spec.{key} must be list")

        batch_size = plan_spec.get("batch_size", solver_spec.get("batch_size", 32))
        if not isinstance(batch_size, (int, float)) or float(batch_size) <= 0:
            errors.append("batch_size must be positive number")
            batch_size = 32
        sample_ratio = plan_spec.get("sample_ratio", solver_spec.get("sample_ratio"))
        if sample_ratio is not None and (not isinstance(sample_ratio, (int, float)) or not (0 < float(sample_ratio) <= 1.0)):
            errors.append("sample_ratio must be in (0,1]")

        preprocess = solver_spec.get("preprocess") if isinstance(solver_spec.get("preprocess"), dict) else {}
        if "max_features" in preprocess:
            max_features = preprocess.get("max_features")
            if not isinstance(max_features, (int, float)) or float(max_features) <= 0:
                errors.append("preprocess.max_features must be positive number")

        model_family = str(solver_spec.get("model_family") or plan_spec.get("model_family") or "").strip().lower()
        model_param_defaults = {"tfidf_logreg": 6_000_000, "linear_svc": 5_000_000, "tfidf_ridge": 4_000_000, "naive_series": 500_000}
        model_params = plan_spec.get("model_params", solver_spec.get("model_params"))
        if not isinstance(model_params, (int, float)):
            model_params = model_param_defaults.get(model_family, 3_000_000)
        seq_len = plan_spec.get("sequence_length", solver_spec.get("sequence_length", 128))
        if not isinstance(seq_len, (int, float)) or float(seq_len) <= 0:
            seq_len = 128

        split_stats = (data_card or {}).get("split_stats") if isinstance((data_card or {}).get("split_stats"), dict) else {}
        approx_rows = 0
        for info in split_stats.values():
            if isinstance(info, dict) and isinstance(info.get("rows"), (int, float)):
                approx_rows += int(info.get("rows"))
        if approx_rows <= 0:
            approx_rows = 145_000
        effective_rows = int(approx_rows * float(sample_ratio if isinstance(sample_ratio, (int, float)) else 1.0))
        estimated_mb = (
            512.0
            + float(batch_size) * (8.0 + 0.02 * float(seq_len))
            + (float(model_params) / 1_000_000.0) * 18.0
            + (float(effective_rows) / 1000.0) * 0.04
        )
        if not isinstance(sample_ratio, (int, float)) and approx_rows >= 100_000:
            estimated_mb += 1400.0

        limit_mb = world_spec.get("code_memory_mb")
        if not isinstance(limit_mb, (int, float)) or float(limit_mb) <= 0:
            limit_mb = 8192.0
        ratio = float(estimated_mb) / float(limit_mb)
        risk = "low"
        if ratio >= self.plugin._vdh_oom_ratio_threshold:
            risk = "high"
            errors.append("potential_oom")
        elif ratio >= 0.7:
            risk = "medium"

        return {
            "ok": len(errors) == 0,
            "errors": errors,
            "resource_estimate": {
                "estimated_mb": round(float(estimated_mb), 2),
                "limit_mb": round(float(limit_mb), 2),
                "ratio": round(float(ratio), 4),
                "risk": risk,
                "rows": int(approx_rows),
                "effective_rows": int(effective_rows),
            },
        }

    async def vdh_embed_text(self, text: str) -> Optional[List[float]]:
        endpoint = self.plugin._vdh_tei_url or self.plugin._reward_tei_url
        if not self.plugin._vdh_tei_enable or not endpoint:
            return None
        payload = json.dumps({"inputs": text}).encode("utf-8")
        req = urllib.request.Request(endpoint, data=payload, headers={"Content-Type": "application/json"}, method="POST")

        def _send() -> Optional[List[float]]:
            with urllib.request.urlopen(req, timeout=6) as resp:
                body = resp.read().decode("utf-8", errors="replace")
            data = json.loads(body)
            if isinstance(data, list) and data and isinstance(data[0], (int, float)):
                return [float(x) for x in data]
            if isinstance(data, list) and data and isinstance(data[0], list):
                return [float(x) for x in data[0]]
            return None

        try:
            return await asyncio.to_thread(_send)
        except Exception:
            return None

    @staticmethod
    def vector_cosine(a: Optional[List[float]], b: Optional[List[float]]) -> float:
        if not isinstance(a, list) or not isinstance(b, list) or not a or not b:
            return 0.0
        n = min(len(a), len(b))
        if n <= 0:
            return 0.0
        dot = na = nb = 0.0
        for i in range(n):
            x = float(a[i])
            y = float(b[i])
            dot += x * y
            na += x * x
            nb += y * y
        if na <= 0 or nb <= 0:
            return 0.0
        return max(0.0, min(1.0, dot / ((na ** 0.5) * (nb ** 0.5))))

    def _dynamic_coverage_threshold(
        self,
        *,
        evidence_chunks_count: int,
        vector_similarity: Optional[float],
    ) -> Dict[str, Any]:
        base = float(self.plugin._vdh_evidence_threshold)
        if not bool(getattr(self.plugin, "_vdh_dynamic_gating_enable", True)):
            return {
                "threshold": base,
                "base_threshold": base,
                "dynamic_enabled": False,
                "relaxations": [],
            }
        min_threshold = float(getattr(self.plugin, "_vdh_dynamic_min_threshold", 0.20))
        min_threshold = max(0.0, min(1.0, min_threshold))
        relaxations: List[str] = []
        relax_value = 0.0

        low_evidence_chunks = int(getattr(self.plugin, "_vdh_dynamic_low_evidence_chunks", 8) or 8)
        if int(evidence_chunks_count) < max(1, low_evidence_chunks):
            relax_value += float(getattr(self.plugin, "_vdh_dynamic_low_evidence_relax", 0.20) or 0.20)
            relaxations.append("low_evidence_chunks")

        rag_degraded = False
        rag_status = str(getattr(self.plugin, "_rag_degraded_last_status", "") or "").strip().lower()
        rag_streak = int(getattr(self.plugin, "_rag_degraded_streak", 0) or 0)
        if rag_streak > 0 or rag_status.startswith("degraded") or rag_status.startswith("disabled"):
            rag_degraded = True
        if rag_degraded:
            relax_value += float(getattr(self.plugin, "_vdh_dynamic_rag_degraded_relax", 0.15) or 0.15)
            relaxations.append("rag_degraded")

        if not isinstance(vector_similarity, (int, float)):
            relax_value += float(getattr(self.plugin, "_vdh_dynamic_no_vector_relax", 0.10) or 0.10)
            relaxations.append("vector_unavailable")

        effective = max(min_threshold, base - max(0.0, relax_value))
        effective = min(base, effective)
        return {
            "threshold": float(round(effective, 4)),
            "base_threshold": float(round(base, 4)),
            "dynamic_enabled": True,
            "min_threshold": float(round(min_threshold, 4)),
            "relaxations": relaxations,
            "rag_degraded_streak": rag_streak,
            "rag_status": rag_status,
            "evidence_chunks_count": int(evidence_chunks_count),
        }

    async def vdh_evidence_coverage(
        self,
        *,
        hypothesis: List[str],
        plan_spec: Dict[str, Any],
        notes: List[Dict[str, Any]],
        observations: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        hypo_parts: List[str] = []
        for tag in (hypothesis or [])[:10]:
            hypo_parts.append(str(tag))
        for key in ("strategy", "rationale", "risk", "schema_assumptions", "memory_safety", "evidence_refs"):
            val = (plan_spec or {}).get(key)
            if isinstance(val, str):
                hypo_parts.append(val)
            elif isinstance(val, list):
                hypo_parts.extend([str(x) for x in val[:8]])
            elif isinstance(val, dict):
                hypo_parts.append(json.dumps(val, ensure_ascii=False))
        hypothesis_text = "\n".join([x for x in hypo_parts if str(x).strip()]).strip()

        evidence_chunks: List[str] = []
        for note in (notes or [])[-12:]:
            if isinstance(note, dict):
                evidence_chunks.append(self.plugin._note_to_text(note))
        for obs in (observations or [])[-10:]:
            if not isinstance(obs, dict):
                continue
            blob = {
                "strategy": obs.get("strategy"),
                "ok": obs.get("ok"),
                "error": obs.get("error"),
                "dev_score_norm": obs.get("dev_score_norm"),
                "score_norm": obs.get("score_norm"),
            }
            evidence_chunks.append(json.dumps(blob, ensure_ascii=False))
        evidence_text = "\n".join([x for x in evidence_chunks if x.strip()])

        h_counter = Counter(self.plugin._text_tokens(hypothesis_text))
        e_counter = Counter(self.plugin._text_tokens(evidence_text))
        token_overlap = self.plugin._counter_cosine(h_counter, e_counter)
        keyword_cov = 0.0
        h_tokens = set(h_counter.keys())
        if h_tokens:
            keyword_cov = len(h_tokens & set(e_counter.keys())) / max(1, len(h_tokens))
        fallback_score = max(0.0, min(1.0, 0.5 * token_overlap + 0.5 * keyword_cov))

        vector_score = None
        h_vec = await self.vdh_embed_text(hypothesis_text) if hypothesis_text else None
        if isinstance(h_vec, list) and h_vec:
            sims: List[float] = []
            for chunk in evidence_chunks[-8:]:
                e_vec = await self.vdh_embed_text(chunk)
                if isinstance(e_vec, list) and e_vec:
                    sims.append(self.vector_cosine(h_vec, e_vec))
            if sims:
                vector_score = max(sims)

        if isinstance(vector_score, (int, float)):
            coverage_score = max(0.0, min(1.0, 0.6 * float(vector_score) + 0.4 * fallback_score))
            source = "tei+token"
        else:
            coverage_score = fallback_score
            source = "token_fallback"

        threshold_info = self._dynamic_coverage_threshold(
            evidence_chunks_count=len(evidence_chunks),
            vector_similarity=float(vector_score) if isinstance(vector_score, (int, float)) else None,
        )
        effective_threshold = float(threshold_info.get("threshold", self.plugin._vdh_evidence_threshold))
        ok = bool(coverage_score >= effective_threshold)

        return {
            "ok": ok,
            "coverage_score": float(round(coverage_score, 4)),
            "threshold": float(effective_threshold),
            "base_threshold": float(threshold_info.get("base_threshold", self.plugin._vdh_evidence_threshold)),
            "source": source,
            "token_overlap": float(round(token_overlap, 4)),
            "keyword_coverage": float(round(keyword_cov, 4)),
            "vector_similarity": float(round(float(vector_score), 4)) if isinstance(vector_score, (int, float)) else None,
            "dynamic": threshold_info,
            "errors": [] if ok else ["evidence_coverage_below_threshold"],
        }

    async def evaluate_vdh_gates(
        self,
        *,
        world_spec: Dict[str, Any],
        hypothesis: List[str],
        plan_spec: Dict[str, Any],
        notes: List[Dict[str, Any]],
        observations: List[Dict[str, Any]],
        data_card: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        gate_a = await self.vdh_metadata_alignment(world_spec=world_spec, notes=notes, plan_spec=plan_spec)
        gate_b = self.vdh_plan_validator(world_spec=world_spec, plan_spec=plan_spec, data_card=data_card)
        gate_c = await self.vdh_evidence_coverage(
            hypothesis=hypothesis,
            plan_spec=plan_spec,
            notes=notes,
            observations=observations,
        )
        errors: List[str] = []
        for gate_name, gate_obj in (("gate_a", gate_a), ("gate_b", gate_b), ("gate_c", gate_c)):
            if not bool((gate_obj or {}).get("ok", False)):
                for err in (gate_obj or {}).get("errors", []) or [f"{gate_name}_failed"]:
                    errors.append(f"{gate_name}:{err}")
        return {
            "gate_a": gate_a,
            "gate_b": gate_b,
            "gate_c": gate_c,
            "final_ok": len(errors) == 0,
            "failures": errors,
            "policy": self.plugin._vdh_gate_policy,
        }

    async def enqueue_vdh_recovery_tasks(self, *, vdh_report: Dict[str, Any]) -> Dict[str, Any]:
        failures = [str(x) for x in (vdh_report or {}).get("failures", [])]
        required: List[Dict[str, Any]] = []
        if any("gate_a" in f for f in failures):
            required.append({"task_type": "read", "payload": {"topic": "task_requirements"}})
        if any("gate_c" in f for f in failures):
            required.append({"task_type": "retrieve_literature", "payload": {"topic": "task_baselines"}})
            required.append({"task_type": "read", "payload": {"topic": "task_requirements"}})
        if any("potential_oom" in f or "gate_b" in f for f in failures):
            required.append({"task_type": "read", "payload": {"topic": "memory_safety"}})

        open_list = await self.plugin.controller.run_environment("science", "task_list", status="open")
        claimed_list = await self.plugin.controller.run_environment("science", "task_list", status="claimed")
        known_types = set()
        for listing in (open_list, claimed_list):
            tasks = (listing or {}).get("tasks", []) if isinstance(listing, dict) else []
            for task in tasks:
                known_types.add(str((task or {}).get("task_type") or ""))

        created: List[str] = []
        skipped: List[Dict[str, Any]] = []
        dedup_types: set[str] = set()
        for item in required:
            task_type = str(item.get("task_type") or "")
            if not task_type or task_type in dedup_types:
                continue
            if task_type == "retrieve_literature":
                paused = await self.plugin._rag_retrieve_recovery_paused()
                if paused:
                    skipped.append(
                        {
                            "task_type": task_type,
                            "reason": "rag_degraded_pause_active",
                            "pause_until_tick": int(self.plugin._rag_degraded_pause_until_tick or 0),
                        }
                    )
                    continue
            dedup_types.add(task_type)
            if task_type in known_types:
                continue
            created_task = await self.plugin.controller.run_environment(
                "science",
                "task_create",
                task_type=task_type,
                payload=dict(item.get("payload") or {}),
                priority=9,
            )
            if isinstance(created_task, dict) and created_task.get("ok"):
                tid = str((created_task.get("task") or {}).get("task_id") or "")
                if tid:
                    created.append(tid)
                known_types.add(task_type)
        return {"created_task_ids": created, "requested": required, "skipped": skipped}
