import random
from typing import Any, Dict, List, Optional

from agentkernel_standalone.mas.environment.base.plugin_base import create_plugin_class
from agentkernel_standalone.toolkit.logger import get_logger

logger = get_logger(__name__)

SciencePluginBase = create_plugin_class("science")


class CausalWorldPlugin(SciencePluginBase):
    """
    Minimal causal discovery environment for MVP.
    Generates a hidden ground-truth parent set for Y and exposes simple queries.
    """

    def __init__(
        self,
        num_vars: int = 4,
        target: str = "Y",
        seed: Optional[int] = None,
        budget: int = 10,
        lambda_cost: float = 0.3,
        hint_accuracy: float = 0.7,
        template_mode: str = "random",
        templates: Optional[List[Dict[str, Any]]] = None,
    ):
        super().__init__()
        self.num_vars = max(2, num_vars)
        self.target = target
        self.seed = seed
        self.budget = budget
        self.lambda_cost = lambda_cost
        self.hint_accuracy = hint_accuracy
        self.template_mode = template_mode
        self.templates = templates or []
        self._rng = random.Random(seed)
        self._episode_id = 0
        self._weights: Dict[str, float] = {}
        self._parents: List[str] = []
        self._variables: List[str] = []
        self._generate_world()

    async def init(self) -> None:
        logger.info("CausalWorldPlugin initialized.")

    ### 重置剧情，开始新的一局
    async def reset_episode(self, seed: Optional[int] = None) -> Dict[str, Any]:
        if seed is not None:
            self.seed = seed
            self._rng.seed(seed)
        self._episode_id += 1
        self._generate_world()
        logger.info(f"Episode reset to {self._episode_id}, parents={self._parents}")
        return {"episode_id": self._episode_id, "parents": list(self._parents)}

    ### 生成因果发现的ground truth
    def _generate_world(self) -> None: 
        self._variables = [f"X{i+1}" for i in range(self.num_vars)] + [self.target]
        if self.template_mode == "template" and self.templates:
            template = self._rng.choice(self.templates)
            self._parents = list(template.get("parents", []))
        else:
            candidates = [v for v in self._variables if v != self.target]
            k = self._rng.randint(1, max(1, len(candidates) // 2))
            self._parents = self._rng.sample(candidates, k=k)
        self._weights = {p: self._rng.uniform(0.5, 1.5) for p in self._parents}

    ### 告知agent基本设定
    async def get_world_spec(self) -> Dict[str, Any]:
        return {
            "variables": list(self._variables),
            "target": self.target,
            "budget": self.budget,
            "episode_id": self._episode_id,
        }

    async def read_literature(self, agent_id: Optional[str] = None, topic: Optional[str] = None) -> Dict[str, Any]:
        hints = []
        for var in self._variables:
            if var == self.target:
                continue
            is_parent = var in self._parents
            if self._rng.random() < self.hint_accuracy:
                hint = f"{var} likely causes {self.target}." if is_parent else f"No evidence that {var} causes {self.target}."
            else:
                hint = f"{var} likely causes {self.target}." if not is_parent else f"No evidence that {var} causes {self.target}."
            hints.append(hint)
        return {"topic": topic or "causal discovery", "hints": hints}

    async def run_experiment(
        self,
        intervention: Optional[Dict[str, float]] = None,
        n_samples: int = 50,
    ) -> Dict[str, Any]:
        intervention = intervention or {}
        effects = {}
        for var in self._variables:
            if var == self.target:
                continue
            if var in self._parents:
                weight = self._weights.get(var, 1.0)
                delta = intervention.get(var, 0.0)
                effects[var] = weight * delta + self._rng.uniform(-0.05, 0.05)
            else:
                effects[var] = self._rng.uniform(-0.05, 0.05)
        return {"n_samples": n_samples, "effects": effects}

    ### 评估假设的精度、召回率、F1值
    async def score_hypothesis(self, hypothesis: List[str]) -> Dict[str, float]:
        predicted = {h for h in hypothesis if h != self.target}
        truth = set(self._parents)
        tp = len(predicted & truth)
        fp = len(predicted - truth)
        fn = len(truth - predicted)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        return {"precision": precision, "recall": recall, "f1": f1}

    ### 评估假设的精度、召回率、F1值
    async def evaluate_hypothesis(self, hypothesis: List[str], exp_count: int) -> Dict[str, Any]:
        score = await self.score_hypothesis(hypothesis)
        f1 = score["f1"]
        aucl = max(0.0, 1.0 - min(1.0, exp_count / max(1, self.budget)))
        cost_penalty = self.lambda_cost * (exp_count / max(1, self.budget))
        fitness = f1 + 0.2 * aucl - cost_penalty
        return {
            "f1": f1,
            "precision": score["precision"],
            "recall": score["recall"],
            "aucl": aucl,
            "exp_count": exp_count,
            "budget": self.budget,
            "fitness": fitness,
        }
