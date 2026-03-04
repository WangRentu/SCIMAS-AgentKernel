import asyncio
import json
import re
from typing import Any, Dict


class LlmService:
    """LLM JSON call, parsing and audit helpers."""

    def __init__(self, plugin: Any):
        self.plugin = plugin

    def extract_json_candidate(self, text: str) -> str:
        if not text:
            return ""
        stripped = text.strip()
        fenced = re.search(r"```(?:json)?\s*(\{.*\}|\[.*\])\s*```", stripped, re.DOTALL)
        if fenced:
            return fenced.group(1).strip()
        first_obj = stripped.find("{")
        last_obj = stripped.rfind("}")
        if first_obj != -1 and last_obj != -1 and last_obj > first_obj:
            return stripped[first_obj : last_obj + 1]
        first_arr = stripped.find("[")
        last_arr = stripped.rfind("]")
        if first_arr != -1 and last_arr != -1 and last_arr > first_arr:
            return stripped[first_arr : last_arr + 1]
        return stripped

    def safe_json_loads(self, text: str) -> Any:
        candidate = self.extract_json_candidate(text)
        return json.loads(candidate)

    async def log_llm_call(self, record: Dict[str, Any]) -> None:
        if not self.plugin._llm_log_enabled:
            return
        await self.plugin._append_jsonl(self.plugin._llm_log_path, record)

    async def log_llm_audit(
        self,
        *,
        ts: str,
        tick: int,
        world_spec: Dict[str, Any],
        agent_id: str,
        action_name: str,
        prompt: str,
        raw_response: Any,
        parsed_data: Any,
        ok: bool,
        reason: str = "",
    ) -> None:
        if not self.plugin._audit_io_enable:
            return
        req = {
            "prompt": self.plugin._clip_text_for_audit(prompt, self.plugin._audit_llm_max_chars),
        }
        raw_text = raw_response if isinstance(raw_response, str) else str(raw_response or "")
        resp = {
            "ok": bool(ok),
            "reason": str(reason or ""),
            "raw_response": self.plugin._clip_text_for_audit(raw_text, self.plugin._audit_llm_max_chars),
            "parsed_json": self.plugin._safe_jsonable(parsed_data),
        }
        meta = {
            "ts": ts,
            "tick": tick,
            "episode_id": world_spec.get("episode_id"),
            "task_name": world_spec.get("task_name"),
            "agent_id": agent_id,
            "action": action_name,
            "kind": "llm_chat_json",
        }
        record = {
            "meta": meta,
            "input": req,
            "output": resp,
        }
        await self.plugin._append_jsonl(self.plugin._llm_audit_jsonl_path, record)
        await self.plugin._append_markdown_audit(
            path=self.plugin._llm_audit_md_path,
            title=f"[{ts}] action={action_name} agent={agent_id} ok={bool(ok)}",
            meta=meta,
            request_payload=req,
            response_payload=resp,
        )

    async def call_llm_json(self, *, agent_id: str, action_name: str, prompt: str) -> Dict[str, Any]:
        tick = await self.plugin.controller.run_system("timer", "get_tick")
        world_spec = await self.plugin.controller.run_environment("science", "get_world_spec")
        ts_info = self.plugin._audit_timestamp_fields()
        ts = ts_info["ts"]

        if not self.plugin._llm_ready(action_name):
            await self.log_llm_audit(
                ts=ts,
                tick=tick,
                world_spec=world_spec,
                agent_id=agent_id,
                action_name=action_name,
                prompt=prompt,
                raw_response="",
                parsed_data={},
                ok=False,
                reason="llm_disabled_or_missing",
            )
            return {"ok": False, "reason": "llm_disabled_or_missing"}
        try:
            raw = await asyncio.wait_for(self.plugin.model.chat(prompt), timeout=self.plugin._llm_timeout_s)
            parsed = self.safe_json_loads(raw if isinstance(raw, str) else str(raw))
            await self.log_llm_call(
                {
                    "ts": ts,
                    "tick": tick,
                    "episode_id": world_spec.get("episode_id"),
                    "task_name": world_spec.get("task_name"),
                    "agent_id": agent_id,
                    "action": action_name,
                    "ok": True,
                    "prompt_chars": len(prompt),
                    "response_chars": len(raw if isinstance(raw, str) else str(raw)),
                    "prompt_preview": prompt[:500],
                    "response_preview": (raw if isinstance(raw, str) else str(raw))[:500],
                }
            )
            await self.log_llm_audit(
                ts=ts,
                tick=tick,
                world_spec=world_spec,
                agent_id=agent_id,
                action_name=action_name,
                prompt=prompt,
                raw_response=raw,
                parsed_data=parsed,
                ok=True,
            )
            return {"ok": True, "raw": raw, "data": parsed}
        except Exception as e:
            await self.log_llm_call(
                {
                    "ts": ts,
                    "tick": tick,
                    "episode_id": world_spec.get("episode_id"),
                    "task_name": world_spec.get("task_name"),
                    "agent_id": agent_id,
                    "action": action_name,
                    "ok": False,
                    "error": str(e),
                    "prompt_chars": len(prompt),
                    "prompt_preview": prompt[:500],
                }
            )
            await self.log_llm_audit(
                ts=ts,
                tick=tick,
                world_spec=world_spec,
                agent_id=agent_id,
                action_name=action_name,
                prompt=prompt,
                raw_response="",
                parsed_data={},
                ok=False,
                reason=str(e),
            )
            return {"ok": False, "reason": str(e)}
