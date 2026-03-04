from typing import Any, Dict, List, Optional

from agentkernel_standalone.types.schemas.action import ActionResult


class DataOpsOperator:
    """Encapsulates read/profile_data/prepare_data actions."""

    def __init__(self, plugin: Any):
        self.plugin = plugin

    async def execute_read(self, agent_id: str, topic: Optional[str] = None) -> ActionResult:
        existing_notes = await self.plugin._get_state(agent_id, "notes") or []
        literature = await self.plugin.controller.run_environment("science", "read_literature", agent_id=agent_id, topic=topic)
        notes = existing_notes
        notes.append(literature)
        await self.plugin._set_state(agent_id, "notes", notes)
        await self.plugin._log_evidence_cards(agent_id, literature, source="read")
        if self.plugin._rag_index_on_read:
            world_spec = await self.plugin.controller.run_environment("science", "get_world_spec")
            rag_docs = self.plugin._rag_docs_from_note(
                world_spec=world_spec,
                agent_id=agent_id,
                note=literature if isinstance(literature, dict) else {},
                action="read",
            )
            if rag_docs:
                await self.plugin._rag_index_documents(agent_id=agent_id, action="read", docs=rag_docs)
        read_reward_info = await self.plugin._compute_read_reward(existing_notes=existing_notes, new_note=literature)
        read_reward = float(read_reward_info.get("reward", 0.0) or 0.0)

        ar = ActionResult.success(
            method_name="read",
            message="Task cards retrieved.",
            data={
                "note": literature,
                "reward": read_reward,
                "effective_action": "read",
                "reward_components": {
                    "learning_reward": float(read_reward),
                    "read_reward": float(read_reward),
                    "read_novelty": float(read_reward_info.get("novelty", 0.0) or 0.0),
                    "read_method_bonus": float(read_reward_info.get("method_bonus", 0.0) or 0.0),
                },
                "read_reward_info": read_reward_info,
            },
        )
        await self.plugin._append_trace(agent_id, "read", read_reward, ar.data or {})
        return ar

    async def execute_profile_data(
        self,
        agent_id: str,
        focus_cols: Optional[List[str]] = None,
        refresh: bool = False,
    ) -> ActionResult:
        data_card = await self.plugin.controller.run_environment(
            "science",
            "profile_data",
            agent_id=agent_id,
            focus_cols=focus_cols,
            refresh=bool(refresh),
        )
        ok = isinstance(data_card, dict) and bool(data_card.get("ok", False))
        reward = 0.01 if ok else -0.01
        if ok:
            await self.plugin._set_state(agent_id, "data_card", data_card)

        ar = ActionResult.success(
            method_name="profile_data",
            message="Data card generated." if ok else "Data profiling failed.",
            data={
                "ok": ok,
                "data_card": data_card,
                "reward": reward,
                "effective_action": "profile_data",
                "reward_components": {
                    "learning_reward": float(reward),
                    "profile_data_reward": float(reward),
                },
            },
        )
        await self.plugin._append_trace(agent_id, "profile_data", reward, ar.data or {})
        return ar

    async def execute_prepare_data(
        self,
        agent_id: str,
        refresh: bool = False,
    ) -> ActionResult:
        prep = await self.plugin.controller.run_environment(
            "science",
            "prepare_data",
            agent_id=agent_id,
            refresh=bool(refresh),
        )
        ok = isinstance(prep, dict) and bool(prep.get("ok", False))
        reward = 0.008 if ok else -0.01
        if ok:
            await self.plugin._set_state(agent_id, "prepare_data_ready", True)

        ar = ActionResult.success(
            method_name="prepare_data",
            message="Prepared AIRS data cache is ready." if ok else "prepare_data failed.",
            data={
                "ok": ok,
                "prepare_data": prep,
                "reward": reward,
                "effective_action": "prepare_data",
                "reward_components": {
                    "learning_reward": float(reward),
                    "prepare_data_reward": float(reward),
                },
            },
        )
        await self.plugin._append_trace(agent_id, "prepare_data", reward, ar.data or {})
        return ar
