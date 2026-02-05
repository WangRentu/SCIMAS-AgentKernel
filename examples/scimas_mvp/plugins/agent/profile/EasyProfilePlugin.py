from typing import Dict, Any, Optional
from agentkernel_standalone.mas.agent.base.plugin_base import ProfilePlugin
from agentkernel_standalone.toolkit.logger import get_logger

logger = get_logger(__name__)

class EasyProfilePlugin(ProfilePlugin):
    def __init__(self, profile_data: Optional[Dict[str, Any]] = None):
        self._profile_data = profile_data if profile_data is not None else {}
        self.agent_id = None
    async def init(self):
        self.agent_id = self._component.agent.agent_id
    async def execute(self, current_tick: int):
        logger.info(f"Agent {self.agent_id} profile snapshot: {self._profile_data}")
    
    async def set_profile(self, key: str, value: Any):
        self._profile_data[key] = value
        logger.info(f"Agent {self.agent_id} profile update: {key} -> {value}")

    async def get_profile(self, key: str) -> Any:
        return self._profile_data.get(key)


