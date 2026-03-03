import math
import re
from collections import Counter
from typing import Any, Dict, List, Optional


class RagRetriever:
    def __init__(self, store: Any, *, max_context_chars: int = 9000):
        self.store = store
        self.max_context_chars = int(max(800, max_context_chars))
        self._source_priority = {
            "data_card": 0,
            "method_card": 1,
            "diagnosis": 2,
            "observation": 3,
            "paper": 4,
            "review": 5,
            "note": 6,
        }

    def _tokens(self, text: str) -> List[str]:
        return [tok for tok in re.split(r"[^a-z0-9_]+", (text or "").lower()) if len(tok) >= 2]

    def _counter_cosine(self, a: Counter, b: Counter) -> float:
        if not a or not b:
            return 0.0
        dot = 0.0
        for k, v in a.items():
            dot += float(v) * float(b.get(k, 0.0))
        if dot <= 0:
            return 0.0
        na = math.sqrt(sum(float(v) * float(v) for v in a.values()))
        nb = math.sqrt(sum(float(v) * float(v) for v in b.values()))
        if na <= 0 or nb <= 0:
            return 0.0
        return max(0.0, min(1.0, dot / (na * nb)))

    def token_fallback_search(self, *, query_text: str, local_docs: List[Dict[str, Any]], topk: int) -> List[Dict[str, Any]]:
        q = Counter(self._tokens(query_text))
        rows: List[Dict[str, Any]] = []
        for doc in local_docs or []:
            text = str((doc or {}).get("text") or "")
            if not text.strip():
                continue
            score = self._counter_cosine(q, Counter(self._tokens(text)))
            rows.append(
                {
                    "score": float(score),
                    "source_type": (doc or {}).get("source_type"),
                    "source_id": (doc or {}).get("source_id"),
                    "text": text,
                    "tags": list((doc or {}).get("tags") or []),
                    "payload": dict(doc or {}),
                }
            )
        rows.sort(key=lambda x: float(x.get("score", 0.0) or 0.0), reverse=True)
        return rows[: max(1, int(topk))]

    def rerank_results(
        self,
        *,
        rows: List[Dict[str, Any]],
        quotas: Dict[str, int],
        min_score: float,
    ) -> List[Dict[str, Any]]:
        filtered = [r for r in (rows or []) if float((r or {}).get("score", 0.0) or 0.0) >= float(min_score)]
        filtered.sort(
            key=lambda r: (
                self._source_priority.get(str(r.get("source_type") or ""), 99),
                -float(r.get("score", 0.0) or 0.0),
            )
        )
        selected: List[Dict[str, Any]] = []
        used_source_ids = set()
        by_type_count: Dict[str, int] = {}
        for item in filtered:
            source_type = str(item.get("source_type") or "")
            source_id = str(item.get("source_id") or "")
            if source_id and source_id in used_source_ids:
                continue
            limit = int((quotas or {}).get(source_type, 0) or 0)
            if limit > 0 and int(by_type_count.get(source_type, 0) or 0) >= limit:
                continue
            selected.append(item)
            if source_id:
                used_source_ids.add(source_id)
            by_type_count[source_type] = int(by_type_count.get(source_type, 0) or 0) + 1
        return selected

    def build_context(self, *, rows: List[Dict[str, Any]], max_chars: Optional[int] = None) -> Dict[str, Any]:
        cap = int(max_chars or self.max_context_chars)
        lines: List[str] = []
        refs: List[str] = []
        total = 0
        for idx, item in enumerate(rows or [], start=1):
            source_type = str(item.get("source_type") or "unknown")
            source_id = str(item.get("source_id") or f"unknown_{idx}")
            ref = f"{source_type}/{source_id}"
            if ref not in refs:
                refs.append(ref)
            text = str(item.get("text") or "").strip()
            score = float(item.get("score", 0.0) or 0.0)
            snippet = f"[{ref}] (score={score:.3f}) {text}"
            if total + len(snippet) + 1 > cap:
                break
            lines.append(snippet)
            total += len(snippet) + 1
        return {"context": "\n".join(lines), "refs": refs}

    async def retrieve(
        self,
        *,
        action: str,
        query_text: str,
        topk: int,
        min_score: float,
        quotas: Dict[str, int],
        local_docs: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        source_rows: List[Dict[str, Any]] = []
        status = "ok"
        fallback_reason = ""

        if self.store is not None:
            ret = await self.store.search(query_text=query_text, topk=max(topk * 3, topk), min_score=min_score)
            if bool(ret.get("ok")):
                source_rows = list(ret.get("results") or [])
            else:
                status = str(ret.get("status") or "degraded")
                fallback_reason = str(ret.get("error") or ret.get("status") or "search_failed")

        if not source_rows:
            fallback_rows = self.token_fallback_search(query_text=query_text, local_docs=local_docs, topk=max(topk * 3, topk))
            if fallback_rows:
                source_rows = fallback_rows
                status = "ok" if status == "ok" else f"{status}+token_fallback"
            elif status == "ok":
                status = "empty"

        selected = self.rerank_results(rows=source_rows, quotas=quotas, min_score=min_score)
        block = self.build_context(rows=selected)
        return {
            "action": action,
            "status": status,
            "fallback_reason": fallback_reason,
            "query_text": query_text,
            "all_results": source_rows,
            "selected": selected,
            "context": block.get("context", ""),
            "refs": block.get("refs", []),
        }
