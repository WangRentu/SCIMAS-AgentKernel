from agentkernel_standalone.mas.agent.components import StateComponent
from agentkernel_standalone.toolkit.logger import get_logger

logger = get_logger(__name__)


class ScimasStateComponent(StateComponent):
    """
    State component wrapper that exposes plugin methods for older runtimes.
    """

    async def get_state(self, key: str):
        if not self._plugin:
            logger.warning("No plugin found in ScimasStateComponent.")
            return None
        return await self._plugin.get_state(key)

    async def set_state(self, key: str, value):
        if not self._plugin:
            logger.warning("No plugin found in ScimasStateComponent.")
            return None
        return await self._plugin.set_state(key, value)
