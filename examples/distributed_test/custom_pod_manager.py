"""Example custom pod manager with convenience helpers."""

from __future__ import annotations

import asyncio
from typing import Any, Dict, Iterable, List, Optional

import ray

from agentkernel_distributed.mas.pod import PodManagerImpl
from agentkernel_distributed.toolkit.logger import get_logger

logger = get_logger(__name__)


@ray.remote
class CustomPodManager(PodManagerImpl):
    """Pod manager extension that exposes broadcast helpers for examples."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # In-memory storage for group discussions (shared across all pods)
        
    async def update_agents_status(self) -> None:
        """
        Trigger each pod to refresh agent status within the environment.

        Returns:
            None
        """
        try:
            await asyncio.gather(*(pod.forward.remote("update_agents_status") for pod in self._pod_id_to_pod.values()))
            logger.info("Update agent status completed across all pods.")
        except Exception as exc:
            logger.error("Failed to update agent status: %s", exc, exc_info=True)