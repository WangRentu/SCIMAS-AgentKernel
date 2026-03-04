import asyncio
import json
import math
import re
import urllib.request
from collections import Counter
from typing import Any, Dict, List, Optional


class UtilityService:
    """General-purpose helpers extracted from ResearchActionsPlugin."""

    def __init__(self, plugin: Any):
        self.plugin = plugin

    @staticmethod
    def truncate(text: Any, limit: int = 280) -> str:
        value = str(text or "")
        if len(value) <= limit:
            return value
        return value[: limit - 3] + "..."

    @staticmethod
    def text_tokens(text: str) -> List[str]:
        return [tok for tok in re.split(r"[^a-z0-9_]+", (text or "").lower()) if len(tok) >= 2]

    def safe_task_types(self, values: Any, *, fallback: Optional[List[str]] = None) -> List[str]:
        allowed = {
            "read",
            "prepare_data",
            "profile_data",
            "retrieve_literature",
            "hypothesize",
            "experiment",
            "write",
            "review",
            "replicate",
            "verify_strength",
            "verify_issue",
        }
        result: List[str] = []
        if isinstance(values, list):
            for item in values:
                name = str(item or "").strip().lower()
                if name in allowed and name not in result:
                    result.append(name)
        if result:
            return result
        return list(fallback or [])

    def safe_text_list(self, values: Any, *, limit: int = 5, item_limit: int = 220) -> List[str]:
        if not isinstance(values, list):
            return []
        out: List[str] = []
        for item in values[: max(0, limit)]:
            text = str(item or "").strip()
            if text:
                out.append(self.truncate(text, item_limit))
        return out

    @staticmethod
    def counter_cosine(a: Counter, b: Counter) -> float:
        if not a or not b:
            return 0.0
        dot = 0.0
        for k, v in a.items():
            dot += float(v) * float(b.get(k, 0.0))
        if dot <= 0:
            return 0.0
        na = math.sqrt(sum(float(v) * float(v) for v in a.values()))
        nb = math.sqrt(sum(float(v) * float(v) for v in b.values()))
        if na <= 0.0 or nb <= 0.0:
            return 0.0
        return max(0.0, min(1.0, dot / (na * nb)))

    def note_to_text(self, note: Dict[str, Any]) -> str:
        parts: List[str] = [str(note.get("topic") or "")]
        for hint in (note.get("hints") or [])[:6]:
            parts.append(str(hint))
        for card in (note.get("cards") or [])[:8]:
            if not isinstance(card, dict):
                continue
            parts.append(str(card.get("title") or ""))
            parts.append(str(card.get("text") or ""))
        return "\n".join(parts)

    @staticmethod
    def has_method_signal(text: str) -> bool:
        s = (text or "").lower()
        keywords = (
            "baseline",
            "method",
            "ablation",
            "protocol",
            "pitfall",
            "mase",
            "timeseriessplit",
            "submission",
            "evaluation",
            "scoring_column",
        )
        return any(k in s for k in keywords)

    async def tei_embed(self, text: str) -> Optional[List[float]]:
        if not self.plugin._reward_tei_url:
            return None
        payload = json.dumps({"inputs": text}).encode("utf-8")
        req = urllib.request.Request(
            self.plugin._reward_tei_url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

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

    async def qdrant_max_similarity(self, vector: List[float]) -> Optional[float]:
        if not self.plugin._reward_qdrant_url or not self.plugin._reward_qdrant_collection or not vector:
            return None
        url = f"{self.plugin._reward_qdrant_url}/collections/{self.plugin._reward_qdrant_collection}/points/search"
        headers = {"Content-Type": "application/json"}
        if self.plugin._reward_qdrant_api_key:
            headers["api-key"] = self.plugin._reward_qdrant_api_key
        payload = {
            "vector": vector,
            "limit": 1,
            "with_payload": False,
            "with_vector": False,
        }
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

    async def compute_read_reward(
        self,
        *,
        existing_notes: List[Dict[str, Any]],
        new_note: Dict[str, Any],
    ) -> Dict[str, Any]:
        new_text = self.note_to_text(new_note)
        new_counter = Counter(self.text_tokens(new_text))
        local_max_sim = 0.0
        for note in (existing_notes or [])[-24:]:
            if not isinstance(note, dict):
                continue
            sim = self.counter_cosine(new_counter, Counter(self.text_tokens(self.note_to_text(note))))
            if sim > local_max_sim:
                local_max_sim = sim

        remote_max_sim = None
        if self.plugin._dense_reward_enable and self.plugin._reward_tei_url and self.plugin._reward_qdrant_url:
            vec = await self.tei_embed(new_text)
            if isinstance(vec, list) and vec:
                remote_max_sim = await self.qdrant_max_similarity(vec)

        max_sim = max(local_max_sim, float(remote_max_sim or 0.0))
        novelty = max(0.0, 1.0 - max_sim)
        method_bonus = self.plugin._read_method_bonus if self.has_method_signal(new_text) else 0.0
        if max_sim >= self.plugin._read_dup_threshold:
            reward = 0.0
        else:
            reward = self.plugin._read_reward_base + (self.plugin._read_reward_alpha * novelty) + method_bonus
            reward = max(0.0, min(self.plugin._read_reward_max, reward))
        return {
            "reward": float(reward),
            "novelty": float(novelty),
            "local_similarity": float(local_max_sim),
            "qdrant_similarity": float(remote_max_sim) if isinstance(remote_max_sim, (int, float)) else None,
            "method_bonus": float(method_bonus),
            "duplicate": bool(max_sim >= self.plugin._read_dup_threshold),
        }

    @staticmethod
    def experiment_error_flags(result: Dict[str, Any]) -> Dict[str, bool]:
        err = str((result or {}).get("error") or "")
        stderr = str((result or {}).get("stderr_tail") or "")
        merged = (err + "\n" + stderr).lower()
        return {
            "oom": ("killed" in merged) or ("out of memory" in merged) or ("oom" in merged),
            "typeerror": "typeerror" in merged,
        }

    @staticmethod
    def is_first_pass_success(*, code_attempts: Any, ok: bool) -> bool:
        if not ok:
            return False
        if not isinstance(code_attempts, list) or not code_attempts:
            return False
        first = code_attempts[0] if isinstance(code_attempts[0], dict) else {}
        first_result = first.get("result") if isinstance(first.get("result"), dict) else {}
        first_ok = bool(first_result.get("ok", False))
        failed_before_success = any(
            isinstance(a, dict)
            and isinstance(a.get("result"), dict)
            and not bool((a.get("result") or {}).get("ok", False))
            for a in code_attempts[:1]
        )
        return bool(first_ok and not failed_before_success)

    @staticmethod
    def estimate_vram_efficiency(*, result: Dict[str, Any], world_spec: Dict[str, Any]) -> Optional[float]:
        limit_mb = world_spec.get("code_memory_mb")
        peak_mb = (result or {}).get("peak_memory_mb")
        if not isinstance(limit_mb, (int, float)) or float(limit_mb) <= 0:
            return None
        if not isinstance(peak_mb, (int, float)):
            return None
        eff = (float(limit_mb) - float(peak_mb)) / float(limit_mb)
        return max(0.0, min(1.0, eff))
