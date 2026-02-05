import random
from typing import Any, Dict, List, Optional

from agentkernel_standalone.mas.action.base.plugin_base import OtherActionsPlugin
from agentkernel_standalone.toolkit.logger import get_logger
from agentkernel_standalone.toolkit.utils.annotation import AgentCall
from agentkernel_standalone.types.schemas.action import ActionResult

logger = get_logger(__name__)


class ResearchActionsPlugin(OtherActionsPlugin):
    """
    Minimal research action plugin for causal discovery.
    """

    def __init__(self):
        super().__init__()
        self._rng = random.Random()

    async def init(self, model_router=None, controller=None):
        self.model = model_router
        self.controller = controller

    async def _log_action(self, *args, **kwargs):
        return None

    async def _get_state(self, agent_id: str, key: str) -> Any:
        return await self.controller.run_agent_method(agent_id, "state", "get_state", key)

    async def _set_state(self, agent_id: str, key: str, value: Any) -> None:
        await self.controller.run_agent_method(agent_id, "state", "set_state", key, value)

    @AgentCall
    async def read(self, agent_id: str, topic: Optional[str] = None) -> ActionResult:
        literature = await self.controller.run_environment("science", "read_literature", agent_id=agent_id, topic=topic)
        notes = await self._get_state(agent_id, "notes") or []
        notes.append(literature)
        await self._set_state(agent_id, "notes", notes)
        logger.info(f"Agent {agent_id} read literature: {literature}")
        return ActionResult.success(
            method_name="read",
            message="Literature retrieved.",
            data={"note": literature, "reward": 0.0},
        )

    @AgentCall
    async def hypothesize(self, agent_id: str, hypothesis: Optional[List[str]] = None) -> ActionResult:
        if hypothesis is None:
            spec = await self.controller.run_environment("science", "get_world_spec")
            candidates = [v for v in spec.get("variables", []) if v != spec.get("target")]
            if not candidates:
                hypothesis = []
            else:
                k = self._rng.randint(1, max(1, len(candidates) // 2))
                hypothesis = self._rng.sample(candidates, k=k)
        await self._set_state(agent_id, "hypothesis", hypothesis)
        logger.info(f"Agent {agent_id} proposed hypothesis: {hypothesis}")
        return ActionResult.success(
            method_name="hypothesize",
            message="Hypothesis proposed.",
            data={"hypothesis": hypothesis, "reward": 0.0},
        )

    @AgentCall
    async def experiment(
        self,
        agent_id: str,
        intervention: Optional[Dict[str, float]] = None,
        n_samples: int = 50,
    ) -> ActionResult:
        exp_count = (await self._get_state(agent_id, "exp_count")) or 0
        exp_count += 1
        result = await self.controller.run_environment(
            "science",
            "run_experiment",
            intervention=intervention or {},
            n_samples=n_samples,
        )
        await self._set_state(agent_id, "exp_count", exp_count)
        logger.info(f"Agent {agent_id} ran experiment #{exp_count}: {result}")
        return ActionResult.success(
            method_name="experiment",
            message="Experiment executed.",
            data={"observation": result, "exp_count": exp_count, "reward": 0.0},
        )

    @AgentCall
    async def write(self, agent_id: str) -> ActionResult:
        hypothesis = await self._get_state(agent_id, "hypothesis") or []
        exp_count = await self._get_state(agent_id, "exp_count") or 0
        metrics = await self.controller.run_environment(
            "science",
            "evaluate_hypothesis",
            hypothesis=hypothesis,
            exp_count=exp_count,
        )
        reward = metrics.get("fitness", 0.0)
        await self._set_state(agent_id, "last_fitness", metrics)
        logger.info(f"Agent {agent_id} write submission: {metrics}")
        return ActionResult.success(
            method_name="write",
            message="Submission evaluated.",
            data={"metrics": metrics, "reward": reward},
        )

    @AgentCall
    async def review(
        self,
        agent_id: str,
        submission: Optional[Dict[str, Any]] = None,
    ) -> ActionResult:
        if submission is None:
            submission = {
                "author_id": agent_id,
                "hypothesis": await self._get_state(agent_id, "hypothesis") or [],
            }
        score = await self.controller.run_environment(
            "science",
            "score_hypothesis",
            hypothesis=submission.get("hypothesis", []),
        )
        review_score = score.get("f1", 0.0) if isinstance(score, dict) else 0.0
        logger.info(f"Agent {agent_id} review score: {review_score}")
        return ActionResult.success(
            method_name="review",
            message="Review completed.",
            data={"score": review_score, "reward": 0.1 * review_score},
        )
