from typing import Any

from ..models import ResearchContext


class ContextService:
    """State/context loader for research actions."""

    def __init__(self, plugin: Any):
        self.plugin = plugin

    async def load_context(self, agent_id: str, include_shared: bool = True) -> ResearchContext:
        world_spec = await self.plugin.controller.run_environment("science", "get_world_spec")

        local_notes = await self.plugin._get_state(agent_id, "notes") or []
        shared_notes = (await self.plugin._get_state(agent_id, "shared_notes") or []) if include_shared else []

        local_observations = await self.plugin._get_state(agent_id, "observations") or []
        shared_observations = (
            (await self.plugin._get_state(agent_id, "shared_observations") or []) if include_shared else []
        )

        hypothesis = await self.plugin._get_state(agent_id, "hypothesis") or []
        data_card = await self.plugin._get_state(agent_id, "data_card")
        method_card = await self.plugin._get_state(agent_id, "method_card")
        plan_spec = await self.plugin._get_state(agent_id, "plan_spec") or {}

        return ResearchContext(
            agent_id=agent_id,
            world_spec=world_spec if isinstance(world_spec, dict) else {},
            notes=list(local_notes) + list(shared_notes),
            observations=list(local_observations) + list(shared_observations),
            hypothesis=list(hypothesis),
            data_card=data_card if isinstance(data_card, dict) else None,
            method_card=method_card if isinstance(method_card, dict) else None,
            plan_spec=plan_spec if isinstance(plan_spec, dict) else {},
            local_notes=list(local_notes),
            shared_notes=list(shared_notes),
            local_observations=list(local_observations),
            shared_observations=list(shared_observations),
        )
