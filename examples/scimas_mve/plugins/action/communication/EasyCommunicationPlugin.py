import textwrap
import json
import inspect
import os
from typing import List, Dict, Any, Optional
from agentkernel_standalone.mas.action.base.plugin_base import CommunicationPlugin
from agentkernel_standalone.types.schemas.message import Message, MessageKind
from agentkernel_standalone.toolkit.logger import get_logger
from agentkernel_standalone.toolkit.utils.annotation import ServiceCall
from agentkernel_standalone.toolkit.utils.annotation import AgentCall

# Import the standardized ActionResult and ActionStatus
from agentkernel_standalone.types.schemas.action import ActionResult

logger = get_logger(__name__)


class EasyCommunicationPlugin(CommunicationPlugin):
    """
    EasyCommunicationPlugin chieve the communication function bwtween two agents.
    """

    def __init__(self):
        super().__init__()
        log_mode = (os.getenv("SCIMAS_LOG_MODE", "compact") or "compact").strip().lower()
        self._verbose_comm_logs = os.getenv(
            "SCIMAS_VERBOSE_COMMUNICATION_LOGS",
            "1" if log_mode == "verbose" else "0",
        ).lower() in {"1", "true", "yes"}

    async def init(self, controller, model_router):
        self.controller = controller
        self.model = model_router

    async def _log_action(self, *args, **kwargs):
        return None

    @ServiceCall
    async def save_to_db(self):
        return ActionResult.success(method_name="save_to_db", message="No state to save.")

    @ServiceCall
    async def load_from_db(self):
        return ActionResult.success(method_name="load_from_db", message="No state to load.")

    @AgentCall
    async def send_message(self, from_id: str, to_id: str, content: str):
        if self._verbose_comm_logs:
            logger.info(f"{from_id} send message to {to_id}")
        message = Message(
            from_id=from_id,
            to_id=to_id,
            kind=MessageKind.FROM_AGENT_TO_AGENT,
            content=content,
            created_at=await self.controller.run_system("timer", "get_tick"),
            extra={},
        )

        try:
            await self.controller.run_system("messager", "send_message", message=message)
            if self._verbose_comm_logs:
                logger.info(f"{from_id} send message to {to_id}")
            return ActionResult.success(
                method_name="send_message",
                message="Message sent.",
                data={"from_id": from_id, "to_id": to_id},
            )
        except Exception as e:
            logger.error(f"{from_id} Error sending message: {e}")
            return ActionResult.error(
                method_name="send_message",
                message=str(e),
            )
