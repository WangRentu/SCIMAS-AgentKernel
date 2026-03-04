import asyncio
import hashlib
import json
import math
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class RagStoreConfig:
    enable: bool = True
    qdrant_url: str = "http://127.0.0.1:6333"
    qdrant_api_key: str = ""
    collection: str = "scimas_local_knowledge_v1"
    embed_url: str = "http://127.0.0.1:8001/v1/embeddings"
    embed_model: str = "Qwen/Qwen3-Embedding-4B"
    timeout_s: float = 8.0
    chunk_chars: int = 1200
    chunk_overlap: int = 180
    batch_size: int = 32


class RagStore:
    def __init__(self, config: RagStoreConfig):
        self.cfg = config
        self._vector_size: Optional[int] = None

    @staticmethod
    def text_hash(text: str) -> str:
        return hashlib.sha1((text or "").encode("utf-8")).hexdigest()

    @staticmethod
    def point_id(
        *,
        episode_id: Any,
        task_name: Any,
        source_type: Any,
        source_id: Any,
        chunk_idx: int,
        text_hash: str,
    ) -> str:
        raw = f"{episode_id}|{task_name}|{source_type}|{source_id}|{chunk_idx}|{text_hash}"
        # Qdrant point IDs must be uint or UUID. Use deterministic UUIDv5 for idempotent upsert.
        return str(uuid.uuid5(uuid.NAMESPACE_URL, raw))

    @staticmethod
    def chunk_text(text: str, chunk_chars: int, overlap: int) -> List[str]:
        value = str(text or "").strip()
        if not value:
            return []
        c = max(200, int(chunk_chars))
        o = max(0, min(int(overlap), c - 1))
        step = max(1, c - o)
        chunks: List[str] = []
        i = 0
        n = len(value)
        while i < n:
            part = value[i : i + c].strip()
            if part:
                chunks.append(part)
            if i + c >= n:
                break
            i += step
        return chunks

    def _http_json(
        self,
        *,
        method: str,
        url: str,
        payload: Optional[Dict[str, Any]],
        headers: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        req_headers = {"Content-Type": "application/json"}
        if headers:
            req_headers.update(headers)
        body = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(url=url, data=body, headers=req_headers, method=method.upper())
        try:
            with urllib.request.urlopen(req, timeout=float(self.cfg.timeout_s)) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
                data = json.loads(raw) if raw else {}
                return {"ok": True, "status": getattr(resp, "status", 200), "data": data}
        except urllib.error.HTTPError as e:
            raw = e.read().decode("utf-8", errors="replace")
            try:
                data = json.loads(raw) if raw else {}
            except Exception:
                data = {"raw": raw}
            return {"ok": False, "status": int(e.code), "error": str(e), "data": data}
        except Exception as e:
            return {"ok": False, "status": None, "error": str(e), "data": {}}

    async def _http_json_async(
        self,
        *,
        method: str,
        url: str,
        payload: Optional[Dict[str, Any]],
        headers: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        return await asyncio.to_thread(self._http_json, method=method, url=url, payload=payload, headers=headers)

    async def _embed_texts(self, texts: List[str]) -> Optional[List[List[float]]]:
        if not self.cfg.embed_url or not texts:
            return None
        payload = {
            "model": self.cfg.embed_model,
            "input": texts,
        }
        ret = await self._http_json_async(method="POST", url=self.cfg.embed_url, payload=payload)
        if not ret.get("ok"):
            return None
        data = (ret.get("data") or {}).get("data")
        if not isinstance(data, list) or not data:
            return None
        vectors: List[List[float]] = []
        for item in data:
            emb = (item or {}).get("embedding") if isinstance(item, dict) else None
            if not isinstance(emb, list) or not emb:
                return None
            vectors.append([float(x) for x in emb])
        return vectors

    async def _ensure_collection(self, vector_size: int, *, collection: Optional[str] = None) -> bool:
        base = self.cfg.qdrant_url.rstrip("/")
        name = str(collection or self.cfg.collection).strip()
        if not name:
            return False
        headers = {}
        if self.cfg.qdrant_api_key:
            headers["api-key"] = self.cfg.qdrant_api_key

        get_ret = await self._http_json_async(
            method="GET",
            url=f"{base}/collections/{name}",
            payload=None,
            headers=headers,
        )
        if get_ret.get("ok"):
            self._vector_size = vector_size
            return True

        create_payload = {
            "vectors": {
                "size": int(vector_size),
                "distance": "Cosine",
            }
        }
        put_ret = await self._http_json_async(
            method="PUT",
            url=f"{base}/collections/{name}",
            payload=create_payload,
            headers=headers,
        )
        if put_ret.get("ok"):
            self._vector_size = vector_size
            return True
        return False

    async def health_check(self) -> Dict[str, Any]:
        if not self.cfg.enable:
            return {"ok": False, "status": "disabled", "qdrant": {}, "embed": {}}

        headers = {}
        if self.cfg.qdrant_api_key:
            headers["api-key"] = self.cfg.qdrant_api_key
        base = self.cfg.qdrant_url.rstrip("/")

        qdrant_ret = await self._http_json_async(
            method="GET",
            url=f"{base}/collections",
            payload=None,
            headers=headers,
        )
        qdrant_ok = bool(qdrant_ret.get("ok"))

        embed_ret = await self._http_json_async(
            method="POST",
            url=self.cfg.embed_url,
            payload={"model": self.cfg.embed_model, "input": ["scimas_rag_health_check"]},
            headers=None,
        )
        embed_data = (embed_ret.get("data") or {}).get("data")
        embed_ok = bool(embed_ret.get("ok")) and isinstance(embed_data, list) and bool(embed_data)
        embed_dim = None
        if embed_ok:
            first = embed_data[0] or {}
            emb = first.get("embedding") if isinstance(first, dict) else None
            if isinstance(emb, list):
                embed_dim = len(emb)

        status_parts: List[str] = []
        if not qdrant_ok:
            status_parts.append("qdrant_unavailable")
        if not embed_ok:
            status_parts.append("embed_unavailable")
        status = "ok" if not status_parts else f"degraded:{'+'.join(status_parts)}"

        return {
            "ok": qdrant_ok and embed_ok,
            "status": status,
            "qdrant": {
                "ok": qdrant_ok,
                "url": self.cfg.qdrant_url,
                "status_code": qdrant_ret.get("status"),
                "error": qdrant_ret.get("error"),
            },
            "embed": {
                "ok": embed_ok,
                "url": self.cfg.embed_url,
                "model": self.cfg.embed_model,
                "status_code": embed_ret.get("status"),
                "dim": embed_dim,
                "error": embed_ret.get("error"),
            },
        }

    async def ensure_collections(self, collections: List[str]) -> Dict[str, Any]:
        if not self.cfg.enable:
            return {"ok": False, "status": "disabled", "collections": {}}
        names = []
        seen = set()
        for name in collections or []:
            n = str(name or "").strip()
            if not n or n in seen:
                continue
            seen.add(n)
            names.append(n)
        if not names:
            return {"ok": False, "status": "empty", "collections": {}}
        probe = await self._embed_texts(["scimas_rag_bootstrap"])
        if not probe or not probe[0]:
            return {"ok": False, "status": "degraded:embed_unavailable", "collections": {}, "error": "embed_probe_failed"}
        vector_size = len(probe[0])
        details: Dict[str, Dict[str, Any]] = {}
        all_ok = True
        for name in names:
            ok = await self._ensure_collection(vector_size=vector_size, collection=name)
            details[name] = {"ok": bool(ok), "vector_size": int(vector_size)}
            if not ok:
                all_ok = False
        return {
            "ok": all_ok,
            "status": "ok" if all_ok else "degraded:ensure_collection_failed",
            "collections": details,
        }

    def build_points(self, docs: List[Dict[str, Any]], vectors: List[List[float]]) -> List[Dict[str, Any]]:
        points: List[Dict[str, Any]] = []
        for doc, vec in zip(docs, vectors):
            text = str(doc.get("text") or "")
            th = self.text_hash(text)
            source_id = str(doc.get("source_id") or f"unknown:{th[:8]}")
            pid = self.point_id(
                episode_id=doc.get("episode_id"),
                task_name=doc.get("task_name"),
                source_type=doc.get("source_type"),
                source_id=source_id,
                chunk_idx=int(doc.get("chunk_idx", 0) or 0),
                text_hash=th,
            )
            payload = {
                "episode_id": doc.get("episode_id"),
                "task_name": doc.get("task_name"),
                "agent_id": doc.get("agent_id"),
                "source_type": doc.get("source_type"),
                "source_id": source_id,
                "action": doc.get("action"),
                "text": text,
                "tags": list(doc.get("tags") or []),
                "quality": doc.get("quality"),
                "ts": doc.get("ts"),
                "version": "v1",
            }
            points.append({"id": pid, "vector": vec, "payload": payload})
        return points

    async def upsert_documents(
        self,
        docs: List[Dict[str, Any]],
        *,
        collection: Optional[str] = None,
    ) -> Dict[str, Any]:
        if not self.cfg.enable:
            return {"ok": False, "status": "disabled", "indexed_points": 0}
        if not docs:
            return {"ok": True, "status": "empty", "indexed_points": 0}

        chunked_docs: List[Dict[str, Any]] = []
        for doc in docs:
            text = str((doc or {}).get("text") or "").strip()
            if not text:
                continue
            chunks = self.chunk_text(text, self.cfg.chunk_chars, self.cfg.chunk_overlap)
            for idx, chunk in enumerate(chunks):
                item = dict(doc)
                item["text"] = chunk
                item["chunk_idx"] = idx
                chunked_docs.append(item)
        if not chunked_docs:
            return {"ok": True, "status": "empty", "indexed_points": 0}

        vectors: List[List[float]] = []
        bs = max(1, int(self.cfg.batch_size))
        for i in range(0, len(chunked_docs), bs):
            batch = chunked_docs[i : i + bs]
            batch_texts = [str(x.get("text") or "") for x in batch]
            emb = await self._embed_texts(batch_texts)
            if not isinstance(emb, list) or len(emb) != len(batch):
                return {
                    "ok": False,
                    "status": "degraded:embed_unavailable",
                    "indexed_points": 0,
                    "documents": len(docs),
                }
            vectors.extend(emb)

        vector_size = len(vectors[0]) if vectors else 0
        if vector_size <= 0:
            return {"ok": False, "status": "degraded:empty_vector", "indexed_points": 0}
        target_collection = str(collection or self.cfg.collection).strip()
        collection_ok = await self._ensure_collection(vector_size, collection=target_collection)
        if not collection_ok:
            return {"ok": False, "status": "degraded:qdrant_unavailable", "indexed_points": 0}

        points = self.build_points(chunked_docs, vectors)
        headers = {}
        if self.cfg.qdrant_api_key:
            headers["api-key"] = self.cfg.qdrant_api_key
        base = self.cfg.qdrant_url.rstrip("/")
        payload = {"points": points}
        ret = await self._http_json_async(
            method="PUT",
            url=f"{base}/collections/{target_collection}/points?wait=true",
            payload=payload,
            headers=headers,
        )
        if not ret.get("ok"):
            return {
                "ok": False,
                "status": "degraded:qdrant_upsert_failed",
                "indexed_points": 0,
                "error": ret.get("error"),
            }
        return {
            "ok": True,
            "status": "ok",
            "indexed_points": len(points),
            "documents": len(docs),
            "collection": target_collection,
        }

    async def search(
        self,
        *,
        query_text: str,
        topk: int,
        min_score: float = 0.0,
        collection: Optional[str] = None,
    ) -> Dict[str, Any]:
        if not self.cfg.enable:
            return {"ok": False, "status": "disabled", "results": []}
        q = str(query_text or "").strip()
        if not q:
            return {"ok": True, "status": "empty_query", "results": []}

        emb = await self._embed_texts([q])
        if not isinstance(emb, list) or not emb or not emb[0]:
            return {"ok": False, "status": "degraded:embed_unavailable", "results": []}
        vector = emb[0]

        headers = {}
        if self.cfg.qdrant_api_key:
            headers["api-key"] = self.cfg.qdrant_api_key
        base = self.cfg.qdrant_url.rstrip("/")
        payload = {
            "vector": vector,
            "limit": max(1, int(topk)),
            "with_payload": True,
            "with_vector": False,
        }
        target_collection = str(collection or self.cfg.collection).strip()
        if not target_collection:
            return {"ok": False, "status": "degraded:collection_missing", "results": []}
        if not await self._ensure_collection(len(vector), collection=target_collection):
            return {
                "ok": False,
                "status": "degraded:qdrant_collection_init_failed",
                "results": [],
                "collection": target_collection,
            }

        ret = await self._http_json_async(
            method="POST",
            url=f"{base}/collections/{target_collection}/points/search",
            payload=payload,
            headers=headers,
        )
        if not ret.get("ok"):
            return {"ok": False, "status": "degraded:qdrant_unavailable", "results": [], "error": ret.get("error")}

        rows = []
        for item in ((ret.get("data") or {}).get("result") or []):
            score = float((item or {}).get("score", 0.0) or 0.0)
            if score < float(min_score):
                continue
            payload_obj = (item or {}).get("payload") if isinstance((item or {}).get("payload"), dict) else {}
            rows.append(
                {
                    "id": (item or {}).get("id"),
                    "score": score,
                    "source_type": payload_obj.get("source_type"),
                    "source_id": payload_obj.get("source_id"),
                    "action": payload_obj.get("action"),
                    "text": payload_obj.get("text"),
                    "tags": payload_obj.get("tags") or [],
                    "payload": payload_obj,
                }
            )
        return {"ok": True, "status": "ok", "results": rows}
