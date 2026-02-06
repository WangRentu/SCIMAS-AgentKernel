import json
import os
import random
from datetime import datetime
from typing import Any, Dict, List, Optional

from agentkernel_standalone.mas.action.base.plugin_base import OtherActionsPlugin
from agentkernel_standalone.toolkit.logger import get_logger
from agentkernel_standalone.toolkit.utils.annotation import AgentCall, ServiceCall
from agentkernel_standalone.types.schemas.action import ActionResult

logger = get_logger(__name__)


class ResearchActionsPlugin(OtherActionsPlugin):
    """
    Minimal research action plugin for causal discovery.
    """

    def __init__(self):
        super().__init__()
        self._rng = random.Random()
        base = os.getenv("MAS_PROJECT_ABS_PATH", ".")
        self._trace_path = os.path.join(base, "logs", "app", "action", "trace.jsonl")

    async def init(self, model_router=None, controller=None):
        self.model = model_router
        self.controller = controller
        os.makedirs(os.path.dirname(self._trace_path), exist_ok=True)

    async def _log_action(self, *args, **kwargs):
        return None

    @ServiceCall
    async def save_to_db(self):
        return ActionResult.success(method_name="save_to_db", message="No state to save.")

    @ServiceCall
    async def load_from_db(self):
        return ActionResult.success(method_name="load_from_db", message="No state to load.")

    async def _get_state(self, agent_id: str, key: str) -> Any:
        return await self.controller.run_agent_method(agent_id, "state", "get_state", key)

    async def _set_state(self, agent_id: str, key: str, value: Any) -> None:
        await self.controller.run_agent_method(agent_id, "state", "set_state", key, value)

    async def _append_trace(self, agent_id: str, action: str, reward: float, detail: Dict[str, Any]):
        tick = await self.controller.run_system("timer", "get_tick")
        fitness = await self._get_state(agent_id, "last_fitness")
        exp_count = await self._get_state(agent_id, "exp_count") or 0
        hypothesis = await self._get_state(agent_id, "hypothesis") or []
        record = {
            "ts": datetime.utcnow().isoformat() + "Z",
            "tick": tick,
            "agent_id": agent_id,
            "action": action,
            "reward": reward,
            "exp_count": exp_count,
            "hypothesis": hypothesis,
            "last_fitness": fitness,
            "detail": detail,
        }
        try:
            with open(self._trace_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.error(f"Failed to write trace: {e}")

    def _infer_hypothesis_rule_based(
        self,
        *,
        target: str,
        candidates: List[str],
        notes: List[Dict[str, Any]],
        observations: List[Dict[str, Any]],
        max_parents: int,
    ) -> Dict[str, Any]:
        """
        Explainable heuristic:
        - Literature prior: count +/- evidence per Xi.
        - Experiment evidence: average |effect| / |delta| for experiments intervening on Xi.
        """
        lit_score: Dict[str, float] = {c: 0.0 for c in candidates}
        for note in notes or []:
            for hint in (note or {}).get("hints", []) or []:
                for c in candidates:
                    if hint.startswith(f"{c} likely causes {target}."):
                        lit_score[c] += 1.0
                    elif hint.startswith(f"No evidence that {c} causes {target}."):
                        lit_score[c] -= 1.0

        exp_strength: Dict[str, List[float]] = {c: [] for c in candidates}
        for obs in observations or []:
            effects = (obs or {}).get("effects") or {}
            intervention = (obs or {}).get("intervention") or {}
            for c in candidates:
                delta = intervention.get(c)
                if delta is None or delta == 0:
                    continue
                effect = effects.get(c)
                if effect is None:
                    continue
                exp_strength[c].append(abs(float(effect)) / max(1e-6, abs(float(delta))))

        exp_score: Dict[str, float] = {
            c: (sum(vals) / len(vals) if vals else 0.0) for c, vals in exp_strength.items()
        }

        # Normalise literature to [-1, 1] roughly (cap extremes).
        lit_norm: Dict[str, float] = {c: max(-1.0, min(1.0, lit_score[c] / 3.0)) for c in candidates}

        # Combine: if we have any experimental evidence, trust it more.
        has_exp = any(exp_score[c] > 0 for c in candidates)
        w_exp = 0.8 if has_exp else 0.2
        w_lit = 1.0 - w_exp
        combined: Dict[str, float] = {c: w_lit * lit_norm[c] + w_exp * exp_score[c] for c in candidates}

        ranked = sorted(candidates, key=lambda c: combined[c], reverse=True)
        chosen = ranked[: max(1, max_parents)]

        explanation = [
            {
                "var": c,
                "literature_score": lit_score[c],
                "experiment_score": exp_score[c],
                "combined_score": combined[c],
            }
            for c in ranked
        ]

        return {"hypothesis": chosen, "explanation": explanation, "weights": {"w_lit": w_lit, "w_exp": w_exp}}

    @AgentCall
    async def read(self, agent_id: str, topic: Optional[str] = None) -> ActionResult:
        literature = await self.controller.run_environment("science", "read_literature", agent_id=agent_id, topic=topic)
        notes = await self._get_state(agent_id, "notes") or []
        notes.append(literature)
        await self._set_state(agent_id, "notes", notes)
        logger.info(f"Agent {agent_id} read literature: {literature}")
        ar = ActionResult.success(
            method_name="read",
            message="Literature retrieved.",
            data={"note": literature, "reward": 0.0},
        )
        await self._append_trace(agent_id, "read", 0.0, ar.data or {})
        return ar

    @AgentCall
    async def hypothesize(self, agent_id: str, hypothesis: Optional[List[str]] = None) -> ActionResult:
        inferred: Dict[str, Any] = {}
        if hypothesis is None:
            spec = await self.controller.run_environment("science", "get_world_spec")
            target = spec.get("target")
            candidates = [v for v in spec.get("variables", []) if v != target]
            notes = await self._get_state(agent_id, "notes") or []
            observations = await self._get_state(agent_id, "observations") or []
            max_parents = max(1, len(candidates) // 2)

            if not candidates:
                inferred = {"hypothesis": [], "explanation": [], "weights": {"w_lit": 1.0, "w_exp": 0.0}}
            else:
                inferred = self._infer_hypothesis_rule_based(
                    target=target,
                    candidates=candidates,
                    notes=notes,
                    observations=observations,
                    max_parents=max_parents,
                )
            hypothesis = inferred["hypothesis"]
        await self._set_state(agent_id, "hypothesis", hypothesis)
        logger.info(f"Agent {agent_id} proposed hypothesis: {hypothesis}")
        ar = ActionResult.success(
            method_name="hypothesize",
            message="Hypothesis proposed.",
            data={
                "hypothesis": hypothesis,
                "reward": 0.0,
                "explanation": inferred.get("explanation", []),
                "weights": inferred.get("weights", {}),
            },
        )
        await self._append_trace(agent_id, "hypothesize", 0.0, ar.data or {})
        return ar

    @AgentCall
    async def experiment(
        self,
        agent_id: str,
        intervention: Optional[Dict[str, float]] = None,
        n_samples: int = 50,
    ) -> ActionResult:
        exp_count = (await self._get_state(agent_id, "exp_count")) or 0
        exp_count += 1
        intervention = intervention or {}
        result = await self.controller.run_environment(
            "science",
            "run_experiment",
            intervention=intervention,
            n_samples=n_samples,
        )
        await self._set_state(agent_id, "exp_count", exp_count)
        observations = await self._get_state(agent_id, "observations") or []
        tick = await self.controller.run_system("timer", "get_tick")
        observations.append(
            {
                "tick": tick,
                "intervention": intervention,
                **(result or {}),
            }
        )
        await self._set_state(agent_id, "observations", observations)
        logger.info(f"Agent {agent_id} ran experiment #{exp_count}: {result}")
        ar = ActionResult.success(
            method_name="experiment",
            message="Experiment executed.",
            data={"observation": {"intervention": intervention, **(result or {})}, "exp_count": exp_count, "reward": 0.0},
        )
        await self._append_trace(agent_id, "experiment", 0.0, ar.data or {})
        return ar

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
        ar = ActionResult.success(
            method_name="write",
            message="Submission evaluated.",
            data={"metrics": metrics, "reward": reward},
        )
        await self._append_trace(agent_id, "write", reward, ar.data or {})
        return ar

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
        ar = ActionResult.success(
            method_name="review",
            message="Review completed.",
            data={"score": review_score, "reward": 0.1 * review_score},
        )
        await self._append_trace(agent_id, "review", 0.1 * review_score, ar.data or {})
        return ar
