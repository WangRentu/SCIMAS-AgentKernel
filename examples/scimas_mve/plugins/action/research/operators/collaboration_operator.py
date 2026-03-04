import json
from typing import Any, Optional

from agentkernel_standalone.types.schemas.action import ActionResult


class CollaborationOperator:
    """Encapsulates collaboration share actions."""

    def __init__(self, plugin: Any):
        self.plugin = plugin

    def _pick_recipient(self, agent_id: str, to_agent_id: Optional[str] = None) -> Optional[str]:
        if to_agent_id and to_agent_id != agent_id:
            return to_agent_id
        others = [aid for aid in self.plugin.controller.get_agent_ids() if aid != agent_id]
        if not others:
            return None
        return self.plugin._rng.choice(others)

    async def execute_share_evidence(
        self,
        agent_id: str,
        to_agent_id: Optional[str] = None,
        max_hints: int = 3,
    ) -> ActionResult:
        notes = await self.plugin._get_state(agent_id, "notes") or []
        if not notes:
            ar = ActionResult.success(method_name="share_evidence", message="No local evidence to share.", data={"reward": 0.0})
            await self.plugin._append_trace(agent_id, "share_evidence", 0.0, ar.data or {})
            return ar

        recipient = self._pick_recipient(agent_id, to_agent_id)
        if not recipient:
            ar = ActionResult.success(method_name="share_evidence", message="No recipient available.", data={"reward": 0.0})
            await self.plugin._append_trace(agent_id, "share_evidence", 0.0, ar.data or {})
            return ar

        latest_note = notes[-1]
        payload = {
            "type": "evidence_share",
            "from_agent": agent_id,
            "evidence": {
                "topic": latest_note.get("topic"),
                "hints": (latest_note.get("hints") or [])[: max(1, max_hints)],
                "cards": (latest_note.get("cards") or [])[: max(1, max_hints)],
            },
        }
        await self.plugin._log_evidence_cards(agent_id, payload["evidence"], source="share_evidence")
        content = json.dumps(payload, ensure_ascii=False)
        send_result = await self.plugin.controller.run_action(
            "communication",
            "send_message",
            from_id=agent_id,
            to_id=recipient,
            content=content,
        )
        ok = isinstance(send_result, ActionResult) and send_result.is_successful()
        reward = 0.02 if ok else 0.0
        if ok:
            await self.plugin._inc_state_number(agent_id, "share_sent_evidence_count", 1)

        ar = ActionResult.success(
            method_name="share_evidence",
            message="Evidence shared." if ok else "Evidence share attempted.",
            data={
                "to_agent_id": recipient,
                "shared_type": "note",
                "reward": reward,
                "effective_action": "share_evidence",
                "reward_components": {
                    "collaboration_reward": float(reward),
                    "learning_reward": float(reward),
                    "share_evidence_reward": float(reward),
                },
            },
        )
        await self.plugin._append_trace(agent_id, "share_evidence", reward, ar.data or {})
        return ar

    async def execute_share_observation(
        self,
        agent_id: str,
        to_agent_id: Optional[str] = None,
    ) -> ActionResult:
        observations = await self.plugin._get_state(agent_id, "observations") or []
        if not observations:
            ar = ActionResult.success(
                method_name="share_observation",
                message="No local observations to share.",
                data={"reward": 0.0},
            )
            await self.plugin._append_trace(agent_id, "share_observation", 0.0, ar.data or {})
            return ar

        recipient = self._pick_recipient(agent_id, to_agent_id)
        if not recipient:
            ar = ActionResult.success(method_name="share_observation", message="No recipient available.", data={"reward": 0.0})
            await self.plugin._append_trace(agent_id, "share_observation", 0.0, ar.data or {})
            return ar

        latest_obs = observations[-1]
        payload = {
            "type": "observation_share",
            "from_agent": agent_id,
            "observation": latest_obs,
        }
        content = json.dumps(payload, ensure_ascii=False)
        send_result = await self.plugin.controller.run_action(
            "communication",
            "send_message",
            from_id=agent_id,
            to_id=recipient,
            content=content,
        )
        ok = isinstance(send_result, ActionResult) and send_result.is_successful()
        reward = 0.02 if ok else 0.0
        if ok:
            await self.plugin._inc_state_number(agent_id, "share_sent_observation_count", 1)

        ar = ActionResult.success(
            method_name="share_observation",
            message="Observation shared." if ok else "Observation share attempted.",
            data={
                "to_agent_id": recipient,
                "shared_type": "observation",
                "reward": reward,
                "effective_action": "share_observation",
                "reward_components": {
                    "collaboration_reward": float(reward),
                    "learning_reward": float(reward),
                    "share_observation_reward": float(reward),
                },
            },
        )
        await self.plugin._append_trace(agent_id, "share_observation", reward, ar.data or {})
        return ar
