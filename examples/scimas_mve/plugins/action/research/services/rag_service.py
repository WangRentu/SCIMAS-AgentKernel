import hashlib
import json
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

try:
    from ..rag_retriever import RagRetriever
    from ..rag_store import RagStore, RagStoreConfig
except Exception:  # pragma: no cover
    RagRetriever = None  # type: ignore
    RagStore = None  # type: ignore
    RagStoreConfig = None  # type: ignore


class RagService:
    """RAG orchestration and persistence helpers extracted from plugin."""

    def __init__(self, plugin: Any):
        self.plugin = plugin

    def init_rag_clients(self) -> None:
        self.plugin._rag_store = None
        self.plugin._rag_retriever = None
        if not self.plugin._rag_enable:
            return
        if RagStoreConfig is None or RagStore is None or RagRetriever is None:
            return
        cfg = RagStoreConfig(
            enable=bool(self.plugin._rag_enable),
            qdrant_url=self.plugin._rag_qdrant_url,
            qdrant_api_key=self.plugin._rag_qdrant_api_key,
            collection=self.plugin._rag_collection_literature or self.plugin._rag_collection,
            embed_url=self.plugin._rag_embed_url,
            embed_model=self.plugin._rag_embed_model,
            timeout_s=float(self.plugin._rag_timeout_s),
            chunk_chars=int(self.plugin._rag_chunk_chars),
            chunk_overlap=int(self.plugin._rag_chunk_overlap),
            batch_size=int(self.plugin._rag_batch_size),
        )
        self.plugin._rag_store = RagStore(cfg)
        self.plugin._rag_retriever = RagRetriever(
            self.plugin._rag_store,
            max_context_chars=self.plugin._rag_max_context_chars,
        )

    async def rag_startup_health_check(self, *, world_spec: Dict[str, Any]) -> None:
        if self.plugin._rag_health_checked:
            return
        self.plugin._rag_health_checked = True

        ts = datetime.utcnow().isoformat() + "Z"
        request_payload = {
            "rag_enable": bool(self.plugin._rag_enable),
            "qdrant_url": self.plugin._rag_qdrant_url,
            "embed_url": self.plugin._rag_embed_url,
            "collection_default": self.plugin._rag_collection,
            "collection_literature": self.plugin._rag_collection_literature,
            "collection_run_memory": self.plugin._rag_collection_run_memory,
            "embed_model": self.plugin._rag_embed_model,
            "timeout_s": float(self.plugin._rag_timeout_s),
        }

        if not self.plugin._rag_enable:
            response_payload = {"ok": False, "status": "disabled", "reason": "SCIMAS_RAG_ENABLE=0"}
        elif self.plugin._rag_store is None:
            response_payload = {"ok": False, "status": "degraded:init_failed", "reason": "rag_clients_unavailable"}
        else:
            try:
                response_payload = await self.plugin._rag_store.health_check()
                # Cold-start friendly: ensure required collections exist even when currently empty.
                if bool(response_payload.get("ok")):
                    ensure = await self.plugin._rag_store.ensure_collections(
                        [
                            self.plugin._rag_collection_literature,
                            self.plugin._rag_collection_run_memory,
                        ]
                    )
                    response_payload["bootstrap"] = ensure
                    if not bool(ensure.get("ok")):
                        response_payload["ok"] = False
                        response_payload["status"] = str(ensure.get("status") or "degraded:ensure_collection_failed")
            except Exception as e:  # pragma: no cover
                response_payload = {"ok": False, "status": "degraded:health_check_exception", "error": str(e)}

        await self.plugin._append_jsonl(
            self.plugin._rag_health_log_path,
            {
                "ts": ts,
                "episode_id": (world_spec or {}).get("episode_id"),
                "task_name": (world_spec or {}).get("task_name"),
                "request": request_payload,
                "result": response_payload,
            },
        )

        if self.plugin._audit_io_enable:
            await self.log_rag_audit(
                world_spec=world_spec or {},
                agent_id="system",
                action="system",
                operation="health_check",
                run_id=None,
                paper_id=None,
                request_payload=request_payload,
                response_payload=response_payload,
            )

    async def rag_track_runtime_health(
        self,
        *,
        world_spec: Dict[str, Any],
        agent_id: str,
        action: str,
        status: str,
        source: str,
    ) -> None:
        st = str(status or "").strip().lower()
        degraded = st.startswith("degraded:")
        if not degraded:
            if self.plugin._rag_degraded_streak > 0 or self.plugin._rag_degraded_pause_until_tick > 0:
                self.plugin._rag_degraded_streak = 0
                self.plugin._rag_degraded_last_status = st
                self.plugin._rag_degraded_pause_until_tick = 0
            return

        self.plugin._rag_degraded_streak += 1
        self.plugin._rag_degraded_last_status = st

        try:
            tick = int(await self.plugin.controller.run_system("timer", "get_tick") or 0)
        except Exception:
            tick = 0
        if self.plugin._rag_degraded_streak < self.plugin._rag_degraded_alert_threshold:
            return
        if tick > 0 and tick < self.plugin._rag_degraded_pause_until_tick:
            return

        self.plugin._rag_degraded_pause_until_tick = max(
            self.plugin._rag_degraded_pause_until_tick,
            tick + self.plugin._rag_degraded_pause_ticks,
        )
        alert_record = {
            "ts": datetime.utcnow().isoformat() + "Z",
            "episode_id": (world_spec or {}).get("episode_id"),
            "task_name": (world_spec or {}).get("task_name"),
            "agent_id": agent_id,
            "action": action,
            "source": source,
            "rag_status": st,
            "degraded_streak": int(self.plugin._rag_degraded_streak),
            "threshold": int(self.plugin._rag_degraded_alert_threshold),
            "pause_until_tick": int(self.plugin._rag_degraded_pause_until_tick),
            "reason": "rag_degraded_streak_threshold_reached",
        }
        await self.plugin._append_jsonl(self.plugin._rag_alert_log_path, alert_record)

    async def rag_retrieve_recovery_paused(self) -> bool:
        until = int(self.plugin._rag_degraded_pause_until_tick or 0)
        if until <= 0:
            return False
        try:
            tick = int(await self.plugin.controller.run_system("timer", "get_tick") or 0)
        except Exception:
            tick = 0
        if tick <= 0:
            return True
        return tick < until

    @staticmethod
    def rag_hash(text: str) -> str:
        return hashlib.sha1((text or "").encode("utf-8")).hexdigest()

    def rag_doc_base(
        self,
        *,
        world_spec: Dict[str, Any],
        agent_id: str,
        source_type: str,
        source_id: str,
        action: str,
        tags: Optional[List[str]] = None,
        quality: Optional[float] = None,
    ) -> Dict[str, Any]:
        return {
            "episode_id": world_spec.get("episode_id"),
            "task_name": world_spec.get("task_name"),
            "agent_id": agent_id,
            "source_type": source_type,
            "source_id": source_id,
            "action": action,
            "tags": list(tags or []),
            "quality": quality,
            "ts": datetime.utcnow().isoformat() + "Z",
            "version": "v1",
        }

    def rag_collection_for_source_type(self, source_type: str) -> str:
        st = str(source_type or "").strip().lower()
        if st in {"observation", "diagnosis"}:
            return self.plugin._rag_collection_run_memory or self.plugin._rag_collection
        if st in {"paper", "review"}:
            return self.plugin._rag_collection_run_memory or self.plugin._rag_collection
        return self.plugin._rag_collection_literature or self.plugin._rag_collection

    def rag_query_collections_for_action(self, action: str) -> List[str]:
        act = str(action or "").strip().lower()
        if act == "retrieve_literature":
            ordered = [
                self.plugin._rag_collection_literature,
                self.plugin._rag_collection_run_memory,
            ]
        elif act in {"experiment", "write", "review", "replicate"}:
            ordered = [
                self.plugin._rag_collection_run_memory,
                self.plugin._rag_collection_literature,
            ]
        else:
            ordered = [
                self.plugin._rag_collection_literature,
                self.plugin._rag_collection_run_memory,
            ]
        seen = set()
        result: List[str] = []
        for name in ordered:
            n = str(name or "").strip()
            if not n or n in seen:
                continue
            seen.add(n)
            result.append(n)
        if not result:
            result.append(self.plugin._rag_collection)
        return result

    def rag_docs_from_note(
        self,
        *,
        world_spec: Dict[str, Any],
        agent_id: str,
        note: Dict[str, Any],
        action: str,
    ) -> List[Dict[str, Any]]:
        if not isinstance(note, dict):
            return []
        text = self.plugin._note_to_text(note).strip()
        if not text:
            return []
        source_id = f"note:{self.rag_hash(text)[:16]}"
        doc = self.rag_doc_base(
            world_spec=world_spec,
            agent_id=agent_id,
            source_type="note",
            source_id=source_id,
            action=action,
            tags=self.plugin._safe_text_list(note.get("hints"), limit=6, item_limit=80),
            quality=0.5,
        )
        doc["text"] = text
        return [doc]

    def rag_docs_from_method_card(
        self,
        *,
        world_spec: Dict[str, Any],
        agent_id: str,
        method_card: Dict[str, Any],
        action: str,
    ) -> List[Dict[str, Any]]:
        if not isinstance(method_card, dict):
            return []
        docs: List[Dict[str, Any]] = []
        topic = str(method_card.get("topic") or method_card.get("task_name") or "method").strip() or "method"
        baselines = (
            method_card.get("baseline_candidates")
            if isinstance(method_card.get("baseline_candidates"), list)
            else method_card.get("recommended_baselines")
        )
        baselines = baselines if isinstance(baselines, list) else []
        if baselines:
            for idx, baseline in enumerate(baselines[:8], start=1):
                if not isinstance(baseline, dict):
                    continue
                steps = baseline.get("implementation_steps") if isinstance(baseline.get("implementation_steps"), list) else baseline.get("key_steps")
                pitfalls = baseline.get("risks") if isinstance(baseline.get("risks"), list) else baseline.get("pitfalls")
                text = " | ".join(
                    [
                        f"name={self.plugin._truncate(baseline.get('name'), 120)}",
                        f"use_when={self.plugin._truncate(baseline.get('use_when'), 180)}",
                        "key_steps=" + "; ".join(self.plugin._safe_text_list(steps, limit=5, item_limit=120)),
                        "pitfalls=" + "; ".join(self.plugin._safe_text_list(pitfalls, limit=4, item_limit=120)),
                    ]
                ).strip(" |")
                if not text:
                    continue
                doc = self.rag_doc_base(
                    world_spec=world_spec,
                    agent_id=agent_id,
                    source_type="method_card",
                    source_id=f"method:{topic}:{idx}",
                    action=action,
                    tags=[topic, str(method_card.get("metric") or "")],
                    quality=0.8,
                )
                doc["text"] = text
                docs.append(doc)
        else:
            text = self.plugin._truncate(json.dumps(method_card, ensure_ascii=False), 4000)
            if text:
                doc = self.rag_doc_base(
                    world_spec=world_spec,
                    agent_id=agent_id,
                    source_type="method_card",
                    source_id=f"method:{topic}",
                    action=action,
                    tags=[topic],
                    quality=0.7,
                )
                doc["text"] = text
                docs.append(doc)
        return docs

    def rag_docs_from_data_card(
        self,
        *,
        world_spec: Dict[str, Any],
        agent_id: str,
        data_card: Dict[str, Any],
        action: str,
    ) -> List[Dict[str, Any]]:
        if not isinstance(data_card, dict):
            return []
        summary = self.plugin._compact_data_card(data_card)
        text = self.plugin._truncate(json.dumps(summary, ensure_ascii=False), 8000)
        if not text:
            return []
        doc = self.rag_doc_base(
            world_spec=world_spec,
            agent_id=agent_id,
            source_type="data_card",
            source_id=f"data_card:{self.rag_hash(text)[:16]}",
            action=action,
            tags=[str(summary.get("target_column") or "target")],
            quality=0.85,
        )
        doc["text"] = text
        return [doc]

    def rag_docs_from_world_spec(
        self,
        *,
        world_spec: Dict[str, Any],
        agent_id: str,
        action: str,
    ) -> List[Dict[str, Any]]:
        if not isinstance(world_spec, dict):
            return []
        task_name = str(world_spec.get("task_name") or "").strip()
        metric = str(world_spec.get("metric") or "").strip()
        category = str(world_spec.get("category") or "").strip()
        problem = str(world_spec.get("research_problem") or "").strip()
        desc_candidates: List[str] = []
        for key in ("project_description", "project_description_md", "task_description", "description", "objective"):
            value = world_spec.get(key)
            text = str(value or "").strip()
            if text:
                desc_candidates.append(text)
        task_meta = world_spec.get("task_meta")
        if isinstance(task_meta, dict):
            for key in ("project_description", "task_description", "description", "objective"):
                text = str(task_meta.get(key) or "").strip()
                if text:
                    desc_candidates.append(text)
        description = self.plugin._truncate("\n\n".join(desc_candidates), 5000)
        lines = [
            f"task_name={task_name}",
            f"metric={metric}",
            f"category={category}",
            f"research_problem={problem}",
        ]
        if description:
            lines.append(f"description={description}")
        text = "\n".join([line for line in lines if str(line or "").strip()]).strip()
        if not text:
            return []
        source_key = task_name or f"episode_{world_spec.get('episode_id') or 0}"
        doc = self.rag_doc_base(
            world_spec=world_spec,
            agent_id=agent_id,
            source_type="task_spec",
            source_id=f"task_spec:{self.rag_hash(source_key)[:16]}",
            action=action,
            tags=[metric, category, "task_spec"],
            quality=0.7,
        )
        doc["text"] = text
        return [doc]

    def rag_docs_from_observation(
        self,
        *,
        world_spec: Dict[str, Any],
        agent_id: str,
        observation: Dict[str, Any],
        action: str,
    ) -> List[Dict[str, Any]]:
        if not isinstance(observation, dict):
            return []
        run_id = str(observation.get("run_id") or "").strip()
        source_id = run_id or f"obs:{self.rag_hash(json.dumps(observation, ensure_ascii=False))[:16]}"
        obs_blob = {
            "run_id": observation.get("run_id"),
            "ok": observation.get("ok"),
            "metric_name": observation.get("metric_name"),
            "score_norm": observation.get("score_norm"),
            "dev_score_norm": observation.get("dev_score_norm"),
            "strategy": observation.get("strategy"),
            "error": self.plugin._truncate(observation.get("error"), 300),
            "stderr_tail": self.plugin._truncate(observation.get("stderr_tail"), 600),
        }
        docs: List[Dict[str, Any]] = []
        d_obs = self.rag_doc_base(
            world_spec=world_spec,
            agent_id=agent_id,
            source_type="observation",
            source_id=source_id,
            action=action,
            tags=[str(observation.get("metric_name") or ""), str(observation.get("solver_mode") or "")],
            quality=0.75 if bool(observation.get("ok")) else 0.55,
        )
        d_obs["text"] = self.plugin._truncate(json.dumps(obs_blob, ensure_ascii=False), 2000)
        docs.append(d_obs)

        error_text = str(observation.get("error") or "")
        if error_text or observation.get("stderr_tail"):
            d_diag = self.rag_doc_base(
                world_spec=world_spec,
                agent_id=agent_id,
                source_type="diagnosis",
                source_id=source_id,
                action=action,
                tags=["failure" if not bool(observation.get("ok")) else "success"],
                quality=0.65,
            )
            d_diag["text"] = self.plugin._truncate(
                f"error={error_text}\nstderr={observation.get('stderr_tail')}\nstrategy={observation.get('strategy')}",
                2400,
            )
            docs.append(d_diag)
        return docs

    def rag_docs_from_paper(
        self,
        *,
        world_spec: Dict[str, Any],
        agent_id: str,
        paper: Dict[str, Any],
        paper_id: Optional[str],
        action: str,
    ) -> List[Dict[str, Any]]:
        if not isinstance(paper, dict):
            return []
        source_id = str(paper_id or paper.get("paper_id") or f"paper:{self.rag_hash(json.dumps(paper, ensure_ascii=False))[:16]}")
        summary = {
            "title": paper.get("title"),
            "abstract": paper.get("abstract"),
            "key_claims": paper.get("key_claims"),
            "limitations": paper.get("limitations"),
            "evidence_map": paper.get("evidence_map"),
            "observation_refs": paper.get("observation_refs"),
        }
        doc = self.rag_doc_base(
            world_spec=world_spec,
            agent_id=agent_id,
            source_type="paper",
            source_id=source_id,
            action=action,
            tags=self.plugin._safe_text_list(paper.get("citations"), limit=6, item_limit=40),
            quality=0.8,
        )
        doc["text"] = self.plugin._truncate(json.dumps(summary, ensure_ascii=False), 6000)
        return [doc]

    async def rag_log_query(
        self,
        *,
        agent_id: str,
        action: str,
        run_id: Optional[str],
        paper_id: Optional[str],
        query_text: str,
        topk: int,
        result: Dict[str, Any],
        latency_ms: float,
    ) -> None:
        world_spec = await self.plugin.controller.run_environment("science", "get_world_spec")
        await self.plugin._append_jsonl(
            self.plugin._rag_query_log_path,
            {
                "ts": datetime.utcnow().isoformat() + "Z",
                "episode_id": world_spec.get("episode_id"),
                "task_name": world_spec.get("task_name"),
                "agent_id": agent_id,
                "action": action,
                "run_id": run_id,
                "paper_id": paper_id,
                "query_text_hash": self.rag_hash(query_text),
                "topk": int(topk),
                "latency_ms": float(round(latency_ms, 2)),
                "rag_status": result.get("status"),
                "retrieval_mode": result.get("retrieval_mode"),
                "fallback_reason": result.get("fallback_reason"),
                "result_count": len(result.get("all_results") or []),
                "selected_count": len(result.get("selected") or []),
                "collections": self.rag_query_collections_for_action(action),
            },
        )

    async def log_rag_audit(
        self,
        *,
        world_spec: Dict[str, Any],
        agent_id: str,
        action: str,
        operation: str,
        run_id: Optional[str],
        paper_id: Optional[str],
        request_payload: Dict[str, Any],
        response_payload: Dict[str, Any],
    ) -> None:
        if not self.plugin._audit_io_enable:
            return
        ts_info = self.plugin._audit_timestamp_fields()
        ts = ts_info["ts"]
        meta = {
            "ts": ts,
            "episode_id": world_spec.get("episode_id"),
            "task_name": world_spec.get("task_name"),
            "agent_id": agent_id,
            "action": action,
            "operation": operation,
            "run_id": run_id,
            "paper_id": paper_id,
            "kind": "rag_io",
        }
        record = {
            "meta": meta,
            "input": self.plugin._safe_jsonable(request_payload),
            "output": self.plugin._safe_jsonable(response_payload),
        }
        await self.plugin._append_jsonl(self.plugin._rag_audit_jsonl_path, record)
        await self.plugin._append_markdown_audit(
            path=self.plugin._rag_audit_md_path,
            title=f"[{ts}] action={action} op={operation} agent={agent_id}",
            meta=meta,
            request_payload=request_payload,
            response_payload=response_payload,
        )

    async def rag_log_usage(
        self,
        *,
        agent_id: str,
        action: str,
        run_id: Optional[str],
        paper_id: Optional[str],
        result: Dict[str, Any],
    ) -> None:
        world_spec = await self.plugin.controller.run_environment("science", "get_world_spec")
        await self.plugin._append_jsonl(
            self.plugin._rag_usage_log_path,
            {
                "ts": datetime.utcnow().isoformat() + "Z",
                "episode_id": world_spec.get("episode_id"),
                "task_name": world_spec.get("task_name"),
                "agent_id": agent_id,
                "action": action,
                "run_id": run_id,
                "paper_id": paper_id,
                "rag_status": result.get("status"),
                "used_refs": result.get("refs") or [],
                "used_count": len(result.get("refs") or []),
            },
        )

    async def rag_index_documents(
        self,
        *,
        agent_id: str,
        action: str,
        docs: List[Dict[str, Any]],
        run_id: Optional[str] = None,
        paper_id: Optional[str] = None,
        collection: Optional[str] = None,
    ) -> Dict[str, Any]:
        world_spec = await self.plugin.controller.run_environment("science", "get_world_spec")
        if not self.plugin._rag_enable or self.plugin._rag_store is None:
            payload = {"ok": False, "status": "disabled", "indexed_points": 0}
        else:
            if collection:
                payload = await self.plugin._rag_store.upsert_documents(docs, collection=collection)
            else:
                grouped: Dict[str, List[Dict[str, Any]]] = {}
                for doc in (docs or []):
                    if not isinstance(doc, dict):
                        continue
                    target = self.rag_collection_for_source_type(str(doc.get("source_type") or ""))
                    grouped.setdefault(target, []).append(doc)
                payload = {"ok": True, "status": "empty", "indexed_points": 0, "documents": 0, "collections": {}}
                for target_collection, group_docs in grouped.items():
                    ret = await self.plugin._rag_store.upsert_documents(group_docs, collection=target_collection)
                    payload["collections"][target_collection] = {
                        "ok": bool(ret.get("ok")),
                        "status": ret.get("status"),
                        "indexed_points": int(ret.get("indexed_points", 0) or 0),
                        "documents": int(ret.get("documents", 0) or 0),
                        "error": ret.get("error"),
                    }
                    payload["indexed_points"] += int(ret.get("indexed_points", 0) or 0)
                    payload["documents"] += int(ret.get("documents", 0) or 0)
                    if not bool(ret.get("ok")):
                        payload["ok"] = False
                if grouped:
                    payload["status"] = "ok" if bool(payload.get("ok")) else "degraded:partial_index_failure"
        await self.plugin._append_jsonl(
            self.plugin._rag_index_log_path,
            {
                "ts": datetime.utcnow().isoformat() + "Z",
                "episode_id": world_spec.get("episode_id"),
                "task_name": world_spec.get("task_name"),
                "agent_id": agent_id,
                "action": action,
                "run_id": run_id,
                "paper_id": paper_id,
                "doc_count": len(docs or []),
                "indexed_points": int((payload or {}).get("indexed_points", 0) or 0),
                "rag_status": (payload or {}).get("status"),
                "ok": bool((payload or {}).get("ok", False)),
                "collection": collection,
                "collections": (payload or {}).get("collections", {}),
            },
        )
        sample_docs = []
        for doc in (docs or [])[: self.plugin._audit_rag_max_rows]:
            sample_docs.append(
                {
                    "source_type": doc.get("source_type"),
                    "source_id": doc.get("source_id"),
                    "tags": list(doc.get("tags") or []),
                    "quality": doc.get("quality"),
                    "text": self.plugin._clip_text_for_audit(doc.get("text"), self.plugin._audit_rag_max_chars),
                }
            )
        await self.log_rag_audit(
            world_spec=world_spec,
            agent_id=agent_id,
            action=action,
            operation="index",
            run_id=run_id,
            paper_id=paper_id,
            request_payload={
                "collection": collection or self.plugin._rag_collection,
                "doc_count": len(docs or []),
                "docs_sample": sample_docs,
            },
            response_payload={
                "ok": bool((payload or {}).get("ok", False)),
                "status": (payload or {}).get("status"),
                "indexed_points": int((payload or {}).get("indexed_points", 0) or 0),
                "documents": int((payload or {}).get("documents", 0) or 0),
                "collections": (payload or {}).get("collections", {}),
                "error": (payload or {}).get("error"),
            },
        )
        await self.rag_track_runtime_health(
            world_spec=world_spec,
            agent_id=agent_id,
            action=action,
            status=str((payload or {}).get("status") or ""),
            source="index",
        )
        return payload

    def rag_local_docs(
        self,
        *,
        world_spec: Dict[str, Any],
        agent_id: str,
        notes: List[Dict[str, Any]],
        observations: List[Dict[str, Any]],
        data_card: Optional[Dict[str, Any]],
        method_card: Optional[Dict[str, Any]],
        paper: Optional[Dict[str, Any]] = None,
        paper_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        docs: List[Dict[str, Any]] = []
        if isinstance(data_card, dict):
            docs.extend(self.rag_docs_from_data_card(world_spec=world_spec, agent_id=agent_id, data_card=data_card, action="local"))
        if isinstance(method_card, dict):
            docs.extend(
                self.rag_docs_from_method_card(world_spec=world_spec, agent_id=agent_id, method_card=method_card, action="local")
            )
        for note in (notes or [])[-12:]:
            if isinstance(note, dict):
                docs.extend(self.rag_docs_from_note(world_spec=world_spec, agent_id=agent_id, note=note, action="local"))
        for observation in (observations or [])[-12:]:
            if isinstance(observation, dict):
                docs.extend(
                    self.rag_docs_from_observation(
                        world_spec=world_spec,
                        agent_id=agent_id,
                        observation=observation,
                        action="local",
                    )
                )
        if isinstance(paper, dict):
            docs.extend(self.rag_docs_from_paper(world_spec=world_spec, agent_id=agent_id, paper=paper, paper_id=paper_id, action="local"))
        return docs

    async def rag_retrieve_context(
        self,
        *,
        agent_id: str,
        action: str,
        run_id: Optional[str],
        paper_id: Optional[str],
        query_text: str,
        quotas: Dict[str, int],
        notes: List[Dict[str, Any]],
        observations: List[Dict[str, Any]],
        data_card: Optional[Dict[str, Any]],
        method_card: Optional[Dict[str, Any]],
        paper: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        world_spec = await self.plugin.controller.run_environment("science", "get_world_spec")
        local_docs = self.rag_local_docs(
            world_spec=world_spec,
            agent_id=agent_id,
            notes=notes,
            observations=observations,
            data_card=data_card,
            method_card=method_card,
            paper=paper,
            paper_id=paper_id,
        )
        t0 = time.perf_counter()
        if self.plugin._rag_retriever is None:
            result = {
                "status": "disabled",
                "fallback_reason": "retriever_unavailable",
                "all_results": [],
                "selected": [],
                "context": "",
                "refs": [],
            }
        else:
            vector_rows: List[Dict[str, Any]] = []
            vector_statuses: List[str] = []
            vector_errors: List[str] = []
            query_collections = self.rag_query_collections_for_action(action)
            if self.plugin._rag_store is not None:
                for collection_name in query_collections:
                    ret = await self.plugin._rag_store.search(
                        query_text=query_text,
                        topk=max(self.plugin._rag_topk * 2, self.plugin._rag_topk),
                        min_score=self.plugin._rag_min_score,
                        collection=collection_name,
                    )
                    vector_statuses.append(str(ret.get("status") or ""))
                    if bool(ret.get("ok")):
                        for row in list(ret.get("results") or []):
                            row = dict(row or {})
                            row["source_collection"] = collection_name
                            row["retrieval_mode"] = "vector"
                            vector_rows.append(row)
                    else:
                        err = str(ret.get("error") or ret.get("status") or "")
                        if err:
                            vector_errors.append(err)
            result = await self.plugin._rag_retriever.retrieve(
                action=action,
                query_text=query_text,
                topk=self.plugin._rag_topk,
                min_score=self.plugin._rag_min_score,
                quotas=quotas,
                local_docs=local_docs,
                mode=self.plugin._rag_retrieve_mode,
                vector_rows=vector_rows,
            )
            if vector_statuses:
                result["vector_statuses"] = vector_statuses
            if vector_errors:
                result["vector_errors"] = vector_errors
            for row in (result.get("all_results") or []):
                if "source_collection" not in row:
                    row["source_collection"] = self.rag_collection_for_source_type(str(row.get("source_type") or ""))
                if "retrieval_mode" not in row:
                    row["retrieval_mode"] = "lexical"
            for row in (result.get("selected") or []):
                if "source_collection" not in row:
                    row["source_collection"] = self.rag_collection_for_source_type(str(row.get("source_type") or ""))
                if "retrieval_mode" not in row:
                    row["retrieval_mode"] = "lexical"
        latency_ms = (time.perf_counter() - t0) * 1000.0
        await self.rag_log_query(
            agent_id=agent_id,
            action=action,
            run_id=run_id,
            paper_id=paper_id,
            query_text=query_text,
            topk=self.plugin._rag_topk,
            result=result,
            latency_ms=latency_ms,
        )
        await self.rag_log_usage(
            agent_id=agent_id,
            action=action,
            run_id=run_id,
            paper_id=paper_id,
            result=result,
        )
        selected_rows = []
        for row in (result.get("selected") or [])[: self.plugin._audit_rag_max_rows]:
            selected_rows.append(
                {
                    "source_type": row.get("source_type"),
                    "source_id": row.get("source_id"),
                    "source_collection": row.get("source_collection"),
                    "retrieval_mode": row.get("retrieval_mode"),
                    "score": row.get("score"),
                    "tags": list(row.get("tags") or []),
                    "text": self.plugin._clip_text_for_audit(row.get("text"), self.plugin._audit_rag_max_chars),
                }
            )
        await self.log_rag_audit(
            world_spec=world_spec,
            agent_id=agent_id,
            action=action,
            operation="query",
            run_id=run_id,
            paper_id=paper_id,
            request_payload={
                "query_text": self.plugin._clip_text_for_audit(query_text, self.plugin._audit_rag_max_chars),
                "topk": int(self.plugin._rag_topk),
                "min_score": float(self.plugin._rag_min_score),
                "retrieve_mode": self.plugin._rag_retrieve_mode,
                "collections": self.rag_query_collections_for_action(action),
                "quotas": dict(quotas or {}),
                "local_docs_count": len(local_docs or []),
            },
            response_payload={
                "status": result.get("status"),
                "fallback_reason": result.get("fallback_reason"),
                "retrieval_mode": result.get("retrieval_mode"),
                "result_count": len(result.get("all_results") or []),
                "selected_count": len(result.get("selected") or []),
                "refs": list(result.get("refs") or []),
                "context": self.plugin._clip_text_for_audit(result.get("context"), self.plugin._audit_rag_max_chars),
                "selected_rows": selected_rows,
                "vector_statuses": list(result.get("vector_statuses") or []),
            },
        )
        await self.rag_track_runtime_health(
            world_spec=world_spec,
            agent_id=agent_id,
            action=action,
            status=str(result.get("status") or ""),
            source="query",
        )
        return result

    async def rag_bootstrap_episode_knowledge(
        self,
        *,
        agent_id: str,
        world_spec: Dict[str, Any],
        data_card: Optional[Dict[str, Any]],
        method_card: Optional[Dict[str, Any]],
    ) -> None:
        ep = int(world_spec.get("episode_id") or 0)
        if ep <= 0 or self.plugin._rag_bootstrap_episode == ep:
            return
        docs: List[Dict[str, Any]] = []
        if isinstance(data_card, dict):
            docs.extend(self.rag_docs_from_data_card(world_spec=world_spec, agent_id=agent_id, data_card=data_card, action="bootstrap"))
        docs.extend(self.rag_docs_from_world_spec(world_spec=world_spec, agent_id=agent_id, action="bootstrap"))
        if isinstance(method_card, dict):
            docs.extend(
                self.rag_docs_from_method_card(
                    world_spec=world_spec,
                    agent_id=agent_id,
                    method_card=method_card,
                    action="bootstrap",
                )
            )
        elif bool(self.plugin._retrieve_template_fallback):
            # Cold start: seed literature collection from template method card once per episode.
            try:
                fallback_card = await self.plugin.controller.run_environment(
                    "science",
                    "retrieve_method_card",
                    agent_id=agent_id,
                    topic="task_baselines",
                    refresh=False,
                )
                if isinstance(fallback_card, dict):
                    docs.extend(
                        self.rag_docs_from_method_card(
                            world_spec=world_spec,
                            agent_id=agent_id,
                            method_card=fallback_card,
                            action="bootstrap_template",
                        )
                    )
            except Exception:
                pass
        if docs:
            await self.rag_index_documents(agent_id=agent_id, action="bootstrap", docs=docs)
        self.plugin._rag_bootstrap_episode = ep

    @staticmethod
    def format_rag_prompt_block(*, result: Dict[str, Any]) -> Dict[str, Any]:
        refs = list(result.get("refs") or [])
        context = str(result.get("context") or "").strip()
        if not context:
            context = "(no high-confidence retrieval results)"
        return {
            "context": context,
            "refs": refs,
            "status": str(result.get("status") or "empty"),
            "usage_constraint": "Prioritize retrieved evidence. Do not fabricate citations or run references.",
        }
