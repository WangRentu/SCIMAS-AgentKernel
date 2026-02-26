import json
import os
import random
import hashlib
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
        env_seed = os.getenv("SCIMAS_WORLD_SEED")
        if env_seed is not None:
            try:
                seed = int(env_seed)
            except Exception:
                logger.warning(f"Invalid SCIMAS_WORLD_SEED={env_seed}, keeping configured seed={seed}")
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
        self._shadow_rng = random.Random(None if seed is None else seed + 100003)
        self._episode_id = 0
        self._weights: Dict[str, float] = {}
        self._shadow_weights: Dict[str, float] = {}
        self._parents: List[str] = []
        self._variables: List[str] = []
        self._cards: List[Dict[str, Any]] = []
        self._agent_views: Dict[str, List[int]] = {}
        self._agent_hint_accuracy: Dict[str, float] = {}
        self._paper_bank: Dict[str, Dict[str, Any]] = {}
        self._paper_replications: Dict[str, List[Dict[str, Any]]] = {}
        self._paper_seq: int = 0
        self._replication_holdout_enabled = os.getenv("SCIMAS_REPLICATION_HOLDOUT_ENABLE", "1").lower() not in {
            "0",
            "false",
            "no",
        }
        self._replication_holdout_threshold = float(os.getenv("SCIMAS_REPLICATION_HOLDOUT_THRESHOLD", "0.5"))
        self._shadow_weight_jitter = float(os.getenv("SCIMAS_SHADOW_WEIGHT_JITTER", "0.25"))
        self._main_noise_scale = float(os.getenv("SCIMAS_MAIN_NOISE_SCALE", "0.05"))
        self._shadow_noise_scale = float(os.getenv("SCIMAS_SHADOW_NOISE_SCALE", "0.08"))
        # Evaluation protocol v4 (gate + smooth score) defaults.
        self._paper_gate_graph = float(os.getenv("SCIMAS_PAPER_GATE_GRAPH", "0.35"))
        self._paper_gate_evidence = float(os.getenv("SCIMAS_PAPER_GATE_EVIDENCE", "0.40"))
        self._paper_gate_replication = float(os.getenv("SCIMAS_PAPER_GATE_REPLICATION", "0.50"))
        self._paper_require_replication = os.getenv("SCIMAS_PAPER_REQUIRE_REPLICATION", "1").lower() not in {
            "0",
            "false",
            "no",
        }
        self._fit_w_graph = float(os.getenv("SCIMAS_FIT_W_GRAPH", "0.70"))
        self._fit_w_evidence = float(os.getenv("SCIMAS_FIT_W_EVIDENCE", "0.20"))
        self._fit_w_replication = float(os.getenv("SCIMAS_FIT_W_REPLICATION", "0.10"))
        self._fit_w_readiness = float(os.getenv("SCIMAS_FIT_W_READINESS", "0.15"))
        self._fit_unverified_penalty = float(os.getenv("SCIMAS_FIT_UNVERIFIED_PENALTY", "0.03"))
        self._fit_failed_replication_penalty = float(os.getenv("SCIMAS_FIT_FAILED_REPLICATION_PENALTY", "0.10"))
        self._fit_failed_holdout_penalty = float(os.getenv("SCIMAS_FIT_FAILED_HOLDOUT_PENALTY", "0.05"))
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
        self._strict_task_dependencies = os.getenv("SCIMAS_STRICT_TASK_DEPENDENCIES", "1").lower() not in {
            "0",
            "false",
            "no",
        }
        self._write_min_completed_read = int(os.getenv("SCIMAS_WRITE_MIN_COMPLETED_READ", "2"))
        self._write_min_completed_experiment = int(os.getenv("SCIMAS_WRITE_MIN_COMPLETED_EXPERIMENT", "2"))
        self._write_min_completed_hypothesize = int(os.getenv("SCIMAS_WRITE_MIN_COMPLETED_HYPOTHESIZE", "1"))
        base = os.getenv("MAS_PROJECT_ABS_PATH", ".")
        self._task_log_path = os.path.join(base, "logs", "app", "environment", "taskboard.jsonl")
        self._generate_world()
        self._generate_shadow_world()
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
            self._shadow_rng.seed(seed + 100003)
        self._episode_id += 1
        self._generate_world()
        self._generate_shadow_world()
        self._build_cards()
        self._agent_views = {}
        self._agent_hint_accuracy = {}
        self._paper_bank = {}
        self._paper_replications = {}
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

    def _generate_shadow_world(self) -> None:
        # Holdout/shadow world preserves the same structural truth (parents) but perturbs weights/noise.
        self._shadow_weights = {}
        for p in self._parents:
            base = float(self._weights.get(p, 1.0))
            jitter = self._shadow_rng.uniform(-self._shadow_weight_jitter, self._shadow_weight_jitter)
            self._shadow_weights[p] = max(0.1, base * (1.0 + jitter))

    def _stable_int_seed(self, *parts: Any) -> int:
        payload = "|".join(str(p) for p in parts)
        h = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        return int(h[:16], 16)

    """生成因果发现证据卡片"""
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
        depends_on: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        task_id = self._next_task_id()
        task = {
            "task_id": task_id,
            "episode_id": self._episode_id,
            "task_type": task_type,
            "status": "open",
            "priority": int(self._task_priority.get(task_type, 0) if priority is None else priority),
            "payload": payload or {},
            "depends_on": list(depends_on or []),
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
        
    """每个 episode 初始化时会自动塞一批任务（bootstrap）：read / experiment / hypothesize / write / review / replicate"""
    def _bootstrap_taskboard(self) -> None:
        n_agents = self._estimate_agent_count()
        read_count = max(6, n_agents // 2)
        experiment_count = max(self.budget, n_agents // 2)
        hypothesize_count = max(4, n_agents // 3)
        write_count = max(4, n_agents // 4)
        for _ in range(read_count):
            self._create_task_internal("read", payload={"topic": "causal discovery"})
        for _ in range(experiment_count):
            self._create_task_internal("experiment", payload={})
        for _ in range(hypothesize_count):
            self._create_task_internal("hypothesize", payload={})
        for _ in range(write_count):
            self._create_task_internal("write", payload={})
        if not self._strict_task_dependencies:
            review_count = max(4, n_agents // 4)
            replicate_count = max(4, n_agents // 4)
            for _ in range(review_count):
                self._create_task_internal("review", payload={})
            for _ in range(replicate_count):
                self._create_task_internal("replicate", payload={})

    def _completed_task_counts(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for task in self._task_board.values():
            if task.get("status") != "completed":
                continue
            t = str(task.get("task_type"))
            counts[t] = counts.get(t, 0) + 1
        return counts

    def _validate_task_payload(
        self,
        task_type: str,
        payload: Optional[Dict[str, Any]],
        *,
        allow_missing_paper_id_for_non_strict: bool = True,
    ) -> Optional[str]:
        data = payload or {}
        if task_type in ("review", "replicate"):
            paper_id = data.get("paper_id")
            if not paper_id:
                if self._strict_task_dependencies and not allow_missing_paper_id_for_non_strict:
                    return "paper_id_required"
                if self._strict_task_dependencies:
                    return "paper_id_required"
                return None
            if str(paper_id) not in self._paper_bank:
                return "paper_not_found"
        depends_on = data.get("depends_on")
        if depends_on is not None and not isinstance(depends_on, list):
            return "depends_on_must_be_list"
        return None

    def _task_blockers(self, task: Dict[str, Any]) -> List[str]:
        blockers: List[str] = []
        depends_on = list(task.get("depends_on") or [])
        for dep_id in depends_on:
            dep_task = self._task_board.get(str(dep_id))
            if not dep_task:
                blockers.append(f"missing_dep:{dep_id}")
                continue
            if dep_task.get("status") != "completed":
                blockers.append(f"dep_not_completed:{dep_id}")

        if not self._strict_task_dependencies:
            return blockers

        task_type = str(task.get("task_type") or "")
        payload = task.get("payload") or {}
        if task_type in ("review", "replicate"):
            paper_id = payload.get("paper_id")
            if not paper_id:
                blockers.append("paper_id_required")
            elif str(paper_id) not in self._paper_bank:
                blockers.append(f"paper_not_found:{paper_id}")
        if task_type == "write":
            completed = self._completed_task_counts()
            if completed.get("read", 0) < self._write_min_completed_read:
                blockers.append(f"need_completed_read>={self._write_min_completed_read}")
            if completed.get("experiment", 0) < self._write_min_completed_experiment:
                blockers.append(f"need_completed_experiment>={self._write_min_completed_experiment}")
            if completed.get("hypothesize", 0) < self._write_min_completed_hypothesize:
                blockers.append(f"need_completed_hypothesize>={self._write_min_completed_hypothesize}")
        return blockers

    def _decorate_task_view(self, task: Dict[str, Any]) -> Dict[str, Any]:
        view = dict(task)
        blockers = self._task_blockers(task)
        view["ready"] = len(blockers) == 0
        if blockers:
            view["blocked_by"] = blockers
        return view

    def _has_pending_task(self, task_type: str, paper_id: Optional[str] = None, reason: Optional[str] = None) -> bool:
        for task in self._task_board.values():
            if str(task.get("task_type")) != str(task_type):
                continue
            if task.get("status") == "completed":
                continue
            payload = task.get("payload") or {}
            if paper_id is not None and str(payload.get("paper_id")) != str(paper_id):
                continue
            if reason is not None and str(payload.get("revision_reason")) != str(reason):
                continue
            return True
        return False

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
            "seed": self.seed,
            "replication_holdout_enabled": self._replication_holdout_enabled,
            "episode_id": self._episode_id,
            "taskboard": summary,
        }

    async def task_create(
        self,
        task_type: str,
        payload: Optional[Dict[str, Any]] = None,
        priority: Optional[int] = None,
        depends_on: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        payload_dict = dict(payload or {})
        if depends_on is None and isinstance(payload_dict.get("depends_on"), list):
            depends_on = list(payload_dict.get("depends_on") or [])
        if "depends_on" in payload_dict:
            payload_dict.pop("depends_on", None)
        err = self._validate_task_payload(task_type, payload_dict)
        if err:
            return {"ok": False, "reason": err}
        task = self._create_task_internal(task_type=task_type, payload=payload_dict, priority=priority, depends_on=depends_on)
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
        tasks = [self._decorate_task_view(t) for t in tasks]
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
        blockers = self._task_blockers(task)
        if blockers:
            return {"ok": False, "reason": "task_blocked", "blocked_by": blockers, "task": self._decorate_task_view(task)}
        task["status"] = "claimed"
        task["claimed_by"] = agent_id
        self._append_taskboard_log("claim", task, {"agent_id": agent_id})
        return {"ok": True, "task": task}

    async def task_release(
        self,
        task_id: str,
        agent_id: str,
        reason: Optional[str] = None,
    ) -> Dict[str, Any]:
        task = self._task_board.get(task_id)
        if not task:
            return {"ok": False, "reason": "task_not_found"}
        if task.get("status") != "claimed":
            return {"ok": False, "reason": f"task_not_claimed:{task.get('status')}", "task": task}
        if task.get("claimed_by") and task.get("claimed_by") != agent_id:
            return {"ok": False, "reason": "task_claimed_by_other", "task": task}
        task["status"] = "open"
        task["claimed_by"] = None
        self._append_taskboard_log("release", task, {"agent_id": agent_id, "reason": reason or ""})
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

        # Auto-create a revision loop when replication completes but the paper is not empirically supported.
        try:
            if self._strict_task_dependencies and str(task.get("task_type")) == "replicate":
                action_data = ((task.get("result") or {}).get("action_data") or {})
                paper_id = (action_data.get("paper_id") or (task.get("payload") or {}).get("paper_id"))
                metrics_after = action_data.get("paper_metrics_after_replication") or {}
                repl_submit = action_data.get("replication_submit") or {}
                replication_failed = False
                if isinstance(metrics_after, dict) and metrics_after:
                    replication_failed = not bool(metrics_after.get("replication_ok", False))
                elif isinstance(repl_submit, dict) and repl_submit:
                    support = (repl_submit.get("support") or {})
                    replication_failed = float(support.get("support_ratio", 0.0) or 0.0) < 0.5

                if paper_id and replication_failed:
                    review_task_id: Optional[str] = None
                    if not self._has_pending_task("review", paper_id=paper_id, reason="replication_failed"):
                        review_task = self._create_task_internal(
                            "review",
                            payload={"paper_id": paper_id, "revision_reason": "replication_failed"},
                            priority=9,
                        )
                        review_task_id = str(review_task.get("task_id"))
                    if not self._has_pending_task("write", paper_id=paper_id, reason="replication_failed"):
                        self._create_task_internal(
                            "write",
                            payload={"paper_id": paper_id, "revision_reason": "replication_failed"},
                            priority=10,
                            depends_on=[review_task_id] if review_task_id else None,
                        )
        except Exception as e:
            logger.warning(f"Failed to auto-create revision tasks after replicate completion: {e}")

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
        paper_with_id = dict(paper or {})
        paper_with_id["paper_id"] = paper_id
        stored = {
            "paper_id": paper_id,
            "episode_id": self._episode_id,
            "paper": paper_with_id,
        }
        self._paper_bank[paper_id] = stored
        self._paper_replications.setdefault(paper_id, [])
        return {"paper_id": paper_id, "episode_id": self._episode_id}

    async def get_paper(self, paper_id: str) -> Optional[Dict[str, Any]]:
        stored = self._paper_bank.get(paper_id)
        if not stored:
            return None
        return stored["paper"]

    def _citation_var_map(self) -> Dict[str, str]:
        mapping: Dict[str, str] = {}
        for card in self._cards or []:
            cid = card.get("citation_id")
            var = card.get("var")
            if cid and var:
                mapping[str(cid)] = str(var)
        return mapping

    def _simulate_effects(
        self,
        intervention: Optional[Dict[str, float]] = None,
        *,
        use_shadow: bool = False,
        noise_scale: Optional[float] = None,
        rng_override: Optional[random.Random] = None,
    ) -> Dict[str, Any]:
        intervention = intervention or {}
        effects = {}
        weights = self._shadow_weights if use_shadow else self._weights
        rng = rng_override or (self._shadow_rng if use_shadow else self._rng)
        local_noise_scale = self._shadow_noise_scale if use_shadow else self._main_noise_scale
        if noise_scale is not None:
            local_noise_scale = float(noise_scale)
        for var in self._variables:
            if var == self.target:
                continue
            if var in self._parents:
                weight = float(weights.get(var, 1.0))
                delta = float(intervention.get(var, 0.0) or 0.0)
                effects[var] = weight * delta + rng.uniform(-local_noise_scale, local_noise_scale)
            else:
                effects[var] = rng.uniform(-local_noise_scale, local_noise_scale)
        return {"effects": effects}

    def _simulate_holdout_effects_for_replication(
        self,
        paper_id: str,
        intervention: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        intervention = intervention or {}
        # Deterministic holdout sampling per (episode, paper, intervention) to improve comparability.
        seed = self._stable_int_seed(
            "holdout",
            self.seed,
            self._episode_id,
            paper_id,
            json.dumps(intervention, ensure_ascii=False, sort_keys=True),
        )
        local_rng = random.Random(seed)
        simulated = self._simulate_effects(intervention=intervention, use_shadow=True, rng_override=local_rng)
        simulated["holdout_seed"] = seed
        return simulated

    def _calc_replication_support_for_paper(
        self,
        paper: Dict[str, Any],
        replication: Dict[str, Any],
        threshold: float = 0.15,
    ) -> Dict[str, Any]:
        claimed_edges = paper.get("claimed_edges") or []
        effects = (replication or {}).get("effects") or {}
        intervention = (replication or {}).get("intervention") or {}
        covered_edges = 0
        supported_edges = 0
        edge_support: Dict[str, float] = {}
        for edge in claimed_edges:
            edge_key = edge if isinstance(edge, str) else f"{edge[0]}->{edge[1]}"
            norm = self._normalize_edge(edge)
            if not norm:
                continue
            src, dst = norm
            if dst != self.target:
                continue
            delta = intervention.get(src)
            if delta is None or float(delta) == 0.0:
                continue
            covered_edges += 1
            effect_val = float(effects.get(src, 0.0) or 0.0)
            ratio = abs(effect_val) / max(1e-6, abs(float(delta)))
            edge_support[edge_key] = ratio
            if ratio >= threshold:
                supported_edges += 1
        support_ratio = (supported_edges / covered_edges) if covered_edges > 0 else 0.0
        return {
            "covered_edges": covered_edges,
            "supported_edges": supported_edges,
            "support_ratio": support_ratio,
            "edge_support": edge_support,
            "threshold": threshold,
        }

    def _merge_replication_support(
        self,
        *,
        support_main: Dict[str, Any],
        support_holdout: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        if not support_holdout:
            merged = dict(support_main or {})
            merged["mode"] = "main_only"
            return merged
        main_ratio = float((support_main or {}).get("support_ratio", 0.0) or 0.0)
        hold_ratio = float((support_holdout or {}).get("support_ratio", 0.0) or 0.0)
        main_cov = int((support_main or {}).get("covered_edges", 0) or 0)
        hold_cov = int((support_holdout or {}).get("covered_edges", 0) or 0)
        combined_ratio = min(main_ratio, hold_ratio)
        combined_cov = min(main_cov, hold_cov)
        combined_supported = int(round(combined_ratio * combined_cov)) if combined_cov > 0 else 0
        return {
            "covered_edges": combined_cov,
            "supported_edges": combined_supported,
            "support_ratio": combined_ratio,
            "threshold": max(
                float((support_main or {}).get("threshold", 0.0) or 0.0),
                float((support_holdout or {}).get("threshold", 0.0) or 0.0),
            ),
            "mode": "main_and_holdout_min",
            "main_support_ratio": main_ratio,
            "holdout_support_ratio": hold_ratio,
        }

    async def submit_replication(
        self,
        paper_id: str,
        agent_id: Optional[str] = None,
        replication: Optional[Dict[str, Any]] = None,
        source: str = "agent_replicate",
    ) -> Dict[str, Any]:
        stored = self._paper_bank.get(paper_id)
        if not stored:
            return {"ok": False, "reason": "paper_not_found", "paper_id": paper_id}
        paper = stored.get("paper") or {}
        rep = dict(replication or {})
        support_main = self._calc_replication_support_for_paper(paper, rep)
        holdout_result = None
        support_holdout = None
        if self._replication_holdout_enabled:
            holdout_sim = self._simulate_holdout_effects_for_replication(paper_id, (rep or {}).get("intervention") or {})
            holdout_result = {
                "n_samples": rep.get("n_samples"),
                "effects": holdout_sim.get("effects") or {},
                "holdout_seed": holdout_sim.get("holdout_seed"),
            }
            holdout_rep = dict(rep)
            holdout_rep["effects"] = holdout_result["effects"]
            support_holdout = self._calc_replication_support_for_paper(paper, holdout_rep)
        support = self._merge_replication_support(support_main=support_main, support_holdout=support_holdout)
        record = {
            "ts": datetime.utcnow().isoformat() + "Z",
            "episode_id": self._episode_id,
            "paper_id": paper_id,
            "agent_id": agent_id,
            "source": source,
            "replication": rep,
            "holdout_replication": holdout_result,
            "support": support,
            "support_main": support_main,
            "support_holdout": support_holdout,
        }
        self._paper_replications.setdefault(paper_id, []).append(record)
        return {
            "ok": True,
            "paper_id": paper_id,
            "support": support,
            "support_main": support_main,
            "support_holdout": support_holdout,
            "holdout_enabled": self._replication_holdout_enabled,
            "replication_count": len(self._paper_replications.get(paper_id) or []),
        }

    async def get_replications(self, paper_id: str) -> Dict[str, Any]:
        records = list(self._paper_replications.get(paper_id) or [])
        return {"paper_id": paper_id, "records": records, "count": len(records)}

    async def evaluate_paper(self, paper: Dict[str, Any], paper_id: Optional[str] = None) -> Dict[str, Any]:
        def _clamp01(v: float) -> float:
            return max(0.0, min(1.0, float(v)))

        claimed_edges = paper.get("claimed_edges") or []
        citations = set(paper.get("citations") or [])
        evidence_map = paper.get("evidence_map") or {}
        observation_evidence_map = paper.get("observation_evidence_map") or {}
        exp_count = int(paper.get("exp_count") or 0)
        observations = paper.get("observation_refs") or []
        paper_id = paper_id or paper.get("paper_id")

        claimed_parents = self._claimed_parents_from_edges(claimed_edges)
        score = await self.score_hypothesis(claimed_parents)
        graph_score = score["f1"]

        # Evidence completeness (strict v2): edge-level support only.
        citation_var = self._citation_var_map()
        complete = 0
        edge_count = max(1, len(claimed_edges))
        edge_evidence_details: Dict[str, Dict[str, Any]] = {}
        edge_evidence_quality_sum = 0.0
        for edge in claimed_edges:
            edge_key = edge if isinstance(edge, str) else f"{edge[0]}->{edge[1]}"
            norm = self._normalize_edge(edge)
            src = norm[0] if norm else None
            edge_evidence = evidence_map.get(edge_key) or []
            valid_citations = []
            for cid in edge_evidence:
                cid_s = str(cid)
                if cid_s in citations and citation_var.get(cid_s) == src:
                    valid_citations.append(cid_s)
            obs_refs_for_edge = observation_evidence_map.get(edge_key) or []
            valid_obs_refs = [str(ref) for ref in obs_refs_for_edge if str(ref) in set(str(x) for x in observations)]
            has_citation = bool(valid_citations)
            has_obs = bool(valid_obs_refs)
            if has_citation or has_obs:
                complete += 1
            # Weighted evidence quality (v4): avoids binary "one ref is enough" scoring.
            citation_score = min(1.0, len(valid_citations) / 2.0)
            obs_score = min(1.0, len(valid_obs_refs) / 1.0)
            diversity_bonus = 0.1 if (has_citation and has_obs) else 0.0
            edge_evidence_score = _clamp01((0.55 * citation_score) + (0.45 * obs_score) + diversity_bonus)
            edge_evidence_quality_sum += edge_evidence_score
            edge_evidence_details[edge_key] = {
                "valid_citations": valid_citations,
                "valid_observation_refs": valid_obs_refs,
                "supported": bool(has_citation or has_obs),
                "citation_score": citation_score,
                "observation_score": obs_score,
                "diversity_bonus": diversity_bonus,
                "edge_evidence_score": edge_evidence_score,
            }
        evidence_coverage_score = complete / edge_count
        evidence_score = edge_evidence_quality_sum / edge_count

        precision = score["precision"]
        recall = score["recall"]
        replication_structural_ok = precision >= 0.5 and recall >= 0.5
        replication_records = list(self._paper_replications.get(paper_id) or []) if paper_id else []
        replication_verified = bool(replication_records)
        replication_support_scores = [
            float(((rec or {}).get("support") or {}).get("support_ratio", 0.0) or 0.0) for rec in replication_records
        ]
        replication_main_support_scores = [
            float(((rec or {}).get("support_main") or {}).get("support_ratio", 0.0) or 0.0) for rec in replication_records
        ]
        replication_holdout_support_scores = [
            float(((rec or {}).get("support_holdout") or {}).get("support_ratio", 0.0) or 0.0)
            for rec in replication_records
            if (rec or {}).get("support_holdout") is not None
        ]
        replication_coverages = [
            int(((rec or {}).get("support") or {}).get("covered_edges", 0) or 0) for rec in replication_records
        ]
        replication_support_score = (
            sum(replication_support_scores) / len(replication_support_scores) if replication_support_scores else 0.0
        )
        replication_main_support_score = (
            sum(replication_main_support_scores) / len(replication_main_support_scores) if replication_main_support_scores else 0.0
        )
        replication_holdout_support_score = (
            sum(replication_holdout_support_scores) / len(replication_holdout_support_scores)
            if replication_holdout_support_scores
            else 0.0
        )
        replication_coverage_ok = any(cov > 0 for cov in replication_coverages)
        holdout_required = self._replication_holdout_enabled and replication_verified
        holdout_available = any((rec or {}).get("support_holdout") is not None for rec in replication_records)
        holdout_pass = (not holdout_required) or (
            holdout_available and replication_holdout_support_score >= self._replication_holdout_threshold
        )
        replication_empirical_ok = (
            replication_verified
            and replication_coverage_ok
            and replication_support_score >= 0.5
            and holdout_pass
        )
        replication_ok = replication_empirical_ok if replication_verified else replication_structural_ok

        cost_penalty = self.lambda_cost * (exp_count / max(1, self.budget))
        # Gate checks (v4): kept separate from score to support analysis and smoother learning signals.
        gate_graph_pass = graph_score >= self._paper_gate_graph
        gate_evidence_pass = evidence_score >= self._paper_gate_evidence
        gate_preprint_pass = bool(gate_graph_pass and gate_evidence_pass)
        if self._paper_require_replication:
            gate_replication_pass = bool(replication_empirical_ok)
        else:
            gate_replication_pass = bool(replication_empirical_ok or replication_structural_ok)
        publishable = bool(gate_preprint_pass and gate_replication_pass)

        graph_readiness = _clamp01(graph_score / max(1e-6, self._paper_gate_graph))
        evidence_readiness = _clamp01(evidence_score / max(1e-6, self._paper_gate_evidence))
        if replication_verified and replication_coverage_ok:
            repl_strength = replication_support_score
            if holdout_required and not holdout_pass:
                repl_strength *= 0.7
            replication_readiness = _clamp01(repl_strength / max(1e-6, self._paper_gate_replication))
        else:
            replication_readiness = 0.25 if replication_structural_ok else 0.0
        readiness_components = [graph_readiness, evidence_readiness]
        if self._paper_require_replication:
            readiness_components.append(replication_readiness)
        readiness_score = sum(readiness_components) / max(1, len(readiness_components))

        # Smooth replication signal for learning/evolution (not a hard gate-only reward).
        if replication_verified:
            rep_signal = 0.7 * replication_support_score + 0.3 * (1.0 if holdout_pass else 0.0)
        else:
            rep_signal = 0.2 if replication_structural_ok else 0.0
        rep_signal = _clamp01(rep_signal)

        quality_score = (
            self._fit_w_graph * graph_score
            + self._fit_w_evidence * evidence_score
            + self._fit_w_replication * rep_signal
        )
        fitness = quality_score + (self._fit_w_readiness * readiness_score) - cost_penalty

        # Mild penalties provide training signal without collapsing rewards to near-zero.
        if replication_verified and not replication_empirical_ok:
            fitness -= self._fit_failed_replication_penalty
        elif not replication_verified:
            fitness -= self._fit_unverified_penalty
        if holdout_required and replication_verified and not holdout_pass:
            fitness -= self._fit_failed_holdout_penalty
        fitness = max(-1.0, min(1.5, fitness))

        return {
            "graph_score": graph_score,
            "precision": score["precision"],
            "recall": score["recall"],
            "f1": score["f1"],
            "evidence_score": evidence_score,
            "evidence_coverage_score": evidence_coverage_score,
            "evidence_edges_supported": complete,
            "evidence_edges_total": edge_count,
            "evidence_details": edge_evidence_details,
            "replication_ok": replication_ok,
            "replication_structural_ok": replication_structural_ok,
            "replication_verified": replication_verified,
            "replication_empirical_ok": replication_empirical_ok,
            "replication_support_score": replication_support_score,
            "replication_main_support_score": replication_main_support_score,
            "replication_holdout_support_score": replication_holdout_support_score,
            "replication_holdout_required": holdout_required,
            "replication_holdout_available": holdout_available,
            "replication_holdout_pass": holdout_pass,
            "replication_records_count": len(replication_records),
            "gate_graph_pass": gate_graph_pass,
            "gate_evidence_pass": gate_evidence_pass,
            "gate_preprint_pass": gate_preprint_pass,
            "gate_replication_pass": gate_replication_pass,
            "publishable": publishable,
            "readiness_score": readiness_score,
            "graph_readiness": graph_readiness,
            "evidence_readiness": evidence_readiness,
            "replication_readiness": replication_readiness,
            "quality_score": quality_score,
            "fitness_components": {
                "quality_score": quality_score,
                "readiness_bonus": self._fit_w_readiness * readiness_score,
                "cost_penalty": cost_penalty,
                "rep_signal": rep_signal,
                "replication_verified": replication_verified,
                "replication_empirical_ok": replication_empirical_ok,
                "publishable": publishable,
            },
            "evaluation_protocol": {
                "name": "v4_gate_plus_smooth_score",
                "gate_graph": self._paper_gate_graph,
                "gate_evidence": self._paper_gate_evidence,
                "gate_replication": self._paper_gate_replication,
                "require_replication": self._paper_require_replication,
                "weights": {
                    "graph": self._fit_w_graph,
                    "evidence": self._fit_w_evidence,
                    "replication": self._fit_w_replication,
                    "readiness": self._fit_w_readiness,
                },
            },
            "exp_count": exp_count,
            "budget": self.budget,
            "fitness": fitness,
        }

    async def run_experiment(
        self,
        intervention: Optional[Dict[str, float]] = None,
        n_samples: int = 50,
    ) -> Dict[str, Any]:
        simulated = self._simulate_effects(intervention=intervention, use_shadow=False)
        return {"n_samples": n_samples, **simulated}

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
