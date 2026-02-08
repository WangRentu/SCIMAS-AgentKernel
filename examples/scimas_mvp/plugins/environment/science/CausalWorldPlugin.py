import json
import os
import random
from datetime import datetime
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
        view_ratio: float = 0.35,
        cards_per_edge: int = 3,
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
        self.view_ratio = min(1.0, max(0.1, view_ratio))
        self.cards_per_edge = max(1, cards_per_edge)
        self.template_mode = template_mode
        self.templates = templates or []
        self._rng = random.Random(seed)
        self._episode_id = 0
        self._weights: Dict[str, float] = {}
        self._parents: List[str] = []
        self._variables: List[str] = []
        self._cards: List[Dict[str, Any]] = []
        self._agent_views: Dict[str, List[int]] = {}
        self._agent_hint_accuracy: Dict[str, float] = {}
        self._paper_bank: Dict[str, Dict[str, Any]] = {}
        self._paper_seq: int = 0
        self._task_board: Dict[str, Dict[str, Any]] = {}
        self._task_seq: int = 0
        self._task_priority = {
            "read": 1,
            "experiment": 2,
            "hypothesize": 3,
            "write": 4,
            "review": 5,
            "replicate": 6,
        }
        base = os.getenv("MAS_PROJECT_ABS_PATH", ".")
        self._task_log_path = os.path.join(base, "logs", "app", "environment", "taskboard.jsonl")
        self._generate_world()
        self._build_cards()
        self._init_taskboard()

    async def init(self) -> None:
        os.makedirs(os.path.dirname(self._task_log_path), exist_ok=True)
        logger.info("CausalWorldPlugin initialized.")

    ### 重置剧情，开始新的一局
    async def reset_episode(self, seed: Optional[int] = None) -> Dict[str, Any]:
        if seed is not None:
            self.seed = seed
            self._rng.seed(seed)
        self._episode_id += 1
        self._generate_world()
        self._build_cards()
        self._agent_views = {}
        self._agent_hint_accuracy = {}
        self._paper_bank = {}
        self._paper_seq = 0
        self._init_taskboard()
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

    def _build_cards(self) -> None:
        self._cards = []
        card_id = 0
        candidates = [v for v in self._variables if v != self.target]
        for var in candidates:
            is_parent = var in self._parents
            for _ in range(self.cards_per_edge):
                self._cards.append(
                    {
                        "id": card_id,
                        "citation_id": f"C{card_id:04d}",
                        "var": var,
                        "target": self.target,
                        "is_parent": is_parent,
                    }
                )
                card_id += 1

    def _ensure_agent_view(self, agent_id: str) -> None:
        if not agent_id:
            return
        if agent_id not in self._agent_views:
            ids = list(range(len(self._cards)))
            self._rng.shuffle(ids)
            take = max(1, int(len(ids) * self.view_ratio))
            self._agent_views[agent_id] = ids[:take]
        if agent_id not in self._agent_hint_accuracy:
            # Per-agent reliability introduces information asymmetry.
            jitter = self._rng.uniform(-0.15, 0.15)
            self._agent_hint_accuracy[agent_id] = min(0.95, max(0.5, self.hint_accuracy + jitter))

    def _estimate_agent_count(self) -> int:
        try:
            if hasattr(self, "controller") and self.controller is not None:
                return max(1, len(self.controller.get_agent_ids()))
        except Exception:
            return 20
        return 20

    def _next_task_id(self) -> str:
        self._task_seq += 1
        return f"T{self._episode_id:03d}-{self._task_seq:05d}"

    def _append_taskboard_log(self, event: str, task: Dict[str, Any], meta: Optional[Dict[str, Any]] = None) -> None:
        record = {
            "ts": datetime.utcnow().isoformat() + "Z",
            "event": event,
            "episode_id": self._episode_id,
            "task": task,
            "meta": meta or {},
        }
        try:
            os.makedirs(os.path.dirname(self._task_log_path), exist_ok=True)
            with open(self._task_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.error(f"Failed to write taskboard log: {e}")

    def _create_task_internal(
        self,
        task_type: str,
        payload: Optional[Dict[str, Any]] = None,
        priority: Optional[int] = None,
    ) -> Dict[str, Any]:
        task_id = self._next_task_id()
        task = {
            "task_id": task_id,
            "episode_id": self._episode_id,
            "task_type": task_type,
            "status": "open",
            "priority": int(self._task_priority.get(task_type, 0) if priority is None else priority),
            "payload": payload or {},
            "claimed_by": None,
            "completed_by": None,
            "result": None,
        }
        self._task_board[task_id] = task
        self._append_taskboard_log("create", task)
        return task

    def _init_taskboard(self) -> None:
        self._task_board = {}
        self._task_seq = 0
        self._bootstrap_taskboard()

    def _bootstrap_taskboard(self) -> None:
        n_agents = self._estimate_agent_count()
        read_count = max(6, n_agents // 2)
        experiment_count = max(self.budget, n_agents // 2)
        hypothesize_count = max(4, n_agents // 3)
        write_count = max(4, n_agents // 4)
        review_count = max(4, n_agents // 4)
        replicate_count = max(4, n_agents // 4)
        for _ in range(read_count):
            self._create_task_internal("read", payload={"topic": "causal discovery"})
        for _ in range(experiment_count):
            self._create_task_internal("experiment", payload={})
        for _ in range(hypothesize_count):
            self._create_task_internal("hypothesize", payload={})
        for _ in range(write_count):
            self._create_task_internal("write", payload={})
        for _ in range(review_count):
            self._create_task_internal("review", payload={})
        for _ in range(replicate_count):
            self._create_task_internal("replicate", payload={})

    ### 告知agent基本设定
    async def get_world_spec(self) -> Dict[str, Any]:
        summary = {"open": 0, "claimed": 0, "completed": 0}
        for task in self._task_board.values():
            st = task.get("status", "open")
            summary[st] = summary.get(st, 0) + 1
        return {
            "variables": list(self._variables),
            "target": self.target,
            "budget": self.budget,
            "episode_id": self._episode_id,
            "taskboard": summary,
        }

    async def task_create(
        self,
        task_type: str,
        payload: Optional[Dict[str, Any]] = None,
        priority: Optional[int] = None,
    ) -> Dict[str, Any]:
        task = self._create_task_internal(task_type=task_type, payload=payload, priority=priority)
        return {"ok": True, "task": task}

    async def task_list(
        self,
        status: Optional[str] = None,
        agent_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        tasks = list(self._task_board.values())
        if status:
            tasks = [t for t in tasks if t.get("status") == status]
        if agent_id:
            tasks = [t for t in tasks if t.get("claimed_by") == agent_id or t.get("completed_by") == agent_id]
        tasks = sorted(tasks, key=lambda t: (-int(t.get("priority", 0)), t.get("task_id", "")))
        summary = {"open": 0, "claimed": 0, "completed": 0}
        for task in self._task_board.values():
            st = task.get("status", "open")
            summary[st] = summary.get(st, 0) + 1
        return {"tasks": tasks, "summary": summary}

    async def task_claim(self, task_id: str, agent_id: str) -> Dict[str, Any]:
        task = self._task_board.get(task_id)
        if not task:
            return {"ok": False, "reason": "task_not_found"}
        if task.get("status") != "open":
            return {"ok": False, "reason": f"task_not_open:{task.get('status')}", "task": task}
        task["status"] = "claimed"
        task["claimed_by"] = agent_id
        self._append_taskboard_log("claim", task, {"agent_id": agent_id})
        return {"ok": True, "task": task}

    async def task_complete(
        self,
        task_id: str,
        agent_id: str,
        result: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        task = self._task_board.get(task_id)
        if not task:
            return {"ok": False, "reason": "task_not_found"}
        if task.get("status") == "completed":
            return {"ok": False, "reason": "task_already_completed", "task": task}
        if task.get("claimed_by") and task.get("claimed_by") != agent_id:
            return {"ok": False, "reason": "task_claimed_by_other", "task": task}
        task["status"] = "completed"
        task["completed_by"] = agent_id
        task["result"] = result or {}
        self._append_taskboard_log("complete", task, {"agent_id": agent_id})
        return {"ok": True, "task": task}

    async def read_literature(self, agent_id: Optional[str] = None, topic: Optional[str] = None) -> Dict[str, Any]:
        self._ensure_agent_view(agent_id or "")
        view_ids = self._agent_views.get(agent_id or "", list(range(len(self._cards))))
        accuracy = self._agent_hint_accuracy.get(agent_id or "", self.hint_accuracy)
        hints = []
        cards = []
        for cid in view_ids:
            card = self._cards[cid]
            var = card["var"]
            is_parent = card["is_parent"]
            observed_is_parent = is_parent
            if self._rng.random() > accuracy:
                observed_is_parent = not is_parent
            if observed_is_parent:
                hint = f"{var} likely causes {self.target}."
            else:
                hint = f"No evidence that {var} causes {self.target}."
            hints.append(hint)
            cards.append({"id": card["id"], "citation_id": card["citation_id"], "var": var, "hint": hint})
        return {"topic": topic or "causal discovery", "hints": hints, "cards": cards, "agent_view_size": len(cards)}

    def _normalize_edge(self, edge: Any) -> Optional[tuple[str, str]]:
        if isinstance(edge, (list, tuple)) and len(edge) == 2:
            return str(edge[0]), str(edge[1])
        if isinstance(edge, str) and "->" in edge:
            parts = edge.split("->", 1)
            return parts[0].strip(), parts[1].strip()
        return None

    def _claimed_parents_from_edges(self, claimed_edges: List[Any]) -> List[str]:
        parents = []
        for edge in claimed_edges or []:
            norm = self._normalize_edge(edge)
            if not norm:
                continue
            src, dst = norm
            if dst == self.target and src in self._variables:
                parents.append(src)
        return parents

    async def submit_paper(self, paper: Dict[str, Any]) -> Dict[str, Any]:
        self._paper_seq += 1
        paper_id = f"P{self._episode_id:03d}-{self._paper_seq:04d}"
        stored = {
            "paper_id": paper_id,
            "episode_id": self._episode_id,
            "paper": paper,
        }
        self._paper_bank[paper_id] = stored
        return {"paper_id": paper_id, "episode_id": self._episode_id}

    async def get_paper(self, paper_id: str) -> Optional[Dict[str, Any]]:
        stored = self._paper_bank.get(paper_id)
        if not stored:
            return None
        return stored["paper"]

    async def evaluate_paper(self, paper: Dict[str, Any]) -> Dict[str, Any]:
        claimed_edges = paper.get("claimed_edges") or []
        citations = set(paper.get("citations") or [])
        evidence_map = paper.get("evidence_map") or {}
        exp_count = int(paper.get("exp_count") or 0)
        observations = paper.get("observation_refs") or []

        claimed_parents = self._claimed_parents_from_edges(claimed_edges)
        score = await self.score_hypothesis(claimed_parents)
        graph_score = score["f1"]

        # Evidence completeness: each claimed edge should have at least one citation or one observation ref.
        complete = 0
        edge_count = max(1, len(claimed_edges))
        for edge in claimed_edges:
            edge_key = edge if isinstance(edge, str) else f"{edge[0]}->{edge[1]}"
            edge_evidence = evidence_map.get(edge_key) or []
            has_citation = any(cid in citations for cid in edge_evidence)
            has_obs = bool(observations)
            if has_citation or has_obs:
                complete += 1
        evidence_score = complete / edge_count

        # Replication gate: accept if claimed parents are reasonably aligned with truth.
        precision = score["precision"]
        recall = score["recall"]
        replication_ok = precision >= 0.5 and recall >= 0.5

        cost_penalty = self.lambda_cost * (exp_count / max(1, self.budget))
        fitness = graph_score + 0.2 * evidence_score - cost_penalty
        if not replication_ok:
            fitness *= 0.3

        return {
            "graph_score": graph_score,
            "precision": score["precision"],
            "recall": score["recall"],
            "f1": score["f1"],
            "evidence_score": evidence_score,
            "replication_ok": replication_ok,
            "exp_count": exp_count,
            "budget": self.budget,
            "fitness": fitness,
        }

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
