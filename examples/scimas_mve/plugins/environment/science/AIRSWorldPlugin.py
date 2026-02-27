import json
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from agentkernel_standalone.mas.environment.base.plugin_base import create_plugin_class
from agentkernel_standalone.toolkit.logger import get_logger

logger = get_logger(__name__)

SciencePluginBase = create_plugin_class("science")


class AIRSWorldPlugin(SciencePluginBase):
    """AIRS-Bench backed research world for MVE.

    This plugin keeps the same surface APIs as the previous science world so the
    existing MAS loop, taskboard orchestration, and evolution logic can run
    without architectural changes.
    """

    def __init__(
        self,
        tasks_root: str = "data/airs_tasks/rad",
        shared_data_dir: str = "../../data/airs_raw_datasets",
        workspace_root: str = "runs/airs_workspace",
        seed: Optional[int] = 42,
        budget: int = 8,
        lambda_cost: float = 0.25,
        task_sampling: str = "random",
        fixed_tasks: Optional[List[str]] = None,
        view_ratio: float = 0.35,
        hint_accuracy: float = 0.9,
        strict_task_dependencies: bool = True,
        active_worker_count: int = 8,
        max_claims_per_agent: int = 1,
        task_lease_ttl: int = 4,
    ):
        super().__init__()
        self.seed = seed
        self._rng = random.Random(seed)
        self.budget = int(max(1, budget))
        self.lambda_cost = float(max(0.0, lambda_cost))
        self.task_sampling = str(task_sampling or "random").strip().lower()
        self.fixed_tasks = [str(x) for x in (fixed_tasks or []) if str(x).strip()]
        self.view_ratio = min(1.0, max(0.1, float(view_ratio)))
        self.hint_accuracy = min(1.0, max(0.5, float(hint_accuracy)))
        self._strict_task_dependencies = bool(strict_task_dependencies)
        self._active_worker_count = int(max(1, active_worker_count))
        self._max_claims_per_agent = int(max(1, max_claims_per_agent))
        self._task_lease_ttl = int(max(1, task_lease_ttl))
        self._log_mode = (os.getenv("SCIMAS_LOG_MODE", "compact") or "compact").strip().lower()
        self._taskboard_log_events = self._resolve_taskboard_log_events()

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
        self._replication_delta_tol = float(os.getenv("SCIMAS_REPLICATION_DELTA_TOL", "0.08"))
        self._python_cmd = os.getenv("SCIMAS_AIRS_PYTHON", sys.executable)
        self._forbid_self_review = os.getenv("SCIMAS_FORBID_SELF_REVIEW", "1").lower() not in {"0", "false", "no"}
        self._forbid_self_replicate = os.getenv("SCIMAS_FORBID_SELF_REPLICATE", "1").lower() not in {"0", "false", "no"}
        self._solver_enabled = os.getenv("SCIMAS_AIRS_SOLVER_ENABLE", "1").lower() not in {"0", "false", "no"}
        self._solver_fallback_template = os.getenv("SCIMAS_AIRS_SOLVER_FALLBACK_TEMPLATE", "1").lower() not in {
            "0",
            "false",
            "no",
        }
        self._solver_prepare_once = os.getenv("SCIMAS_AIRS_PREPARE_ONCE_PER_EPISODE", "1").lower() not in {
            "0",
            "false",
            "no",
        }

        self._write_min_completed_read = int(os.getenv("SCIMAS_WRITE_MIN_COMPLETED_READ", "1"))
        self._write_min_completed_experiment = int(os.getenv("SCIMAS_WRITE_MIN_COMPLETED_EXPERIMENT", "1"))
        self._write_min_completed_hypothesize = int(os.getenv("SCIMAS_WRITE_MIN_COMPLETED_HYPOTHESIZE", "1"))

        self._episode_id = 0
        self._task_seq = 0
        self._paper_seq = 0
        self._run_seq = 0
        self._round_robin_idx = 0
        self._logical_tick = 0

        project_root = os.getenv("MAS_PROJECT_ABS_PATH", ".")
        self._project_root = os.path.abspath(project_root)
        self.tasks_root = self._resolve_path(tasks_root, project_root=self._project_root)
        self.shared_data_dir = self._resolve_path(shared_data_dir, project_root=self._project_root)
        self.workspace_root = self._resolve_path(workspace_root, project_root=self._project_root)
        self._all_agent_ids = self._load_agent_ids()

        self._task_log_path = os.path.join(self._project_root, "logs", "app", "environment", "taskboard.jsonl")

        self._tasks_catalog: List[Dict[str, Any]] = []
        self._current_task: Optional[Dict[str, Any]] = None
        self._current_cards: List[Dict[str, Any]] = []
        self._current_workspace: Optional[str] = None
        self._prepared_data_cache_dir: Optional[str] = None

        self._task_board: Dict[str, Dict[str, Any]] = {}
        self._active_workers: set[str] = set()
        self._taskboard_event_counts: Dict[str, int] = {"create": 0, "claim": 0, "complete": 0, "release": 0}
        self._task_priority = {
            "read": 1,
            "experiment": 2,
            "hypothesize": 3,
            "write": 4,
            "review": 5,
            "replicate": 6,
            "verify_strength": 7,
            "verify_issue": 8,
        }

        self._agent_views: Dict[str, List[int]] = {}
        self._agent_hint_accuracy: Dict[str, float] = {}
        self._score_cache: List[Dict[str, Any]] = []
        self._paper_bank: Dict[str, Dict[str, Any]] = {}
        self._paper_replications: Dict[str, List[Dict[str, Any]]] = {}

        self._load_tasks_catalog()
        self._init_taskboard()

    async def init(self) -> None:
        os.makedirs(os.path.dirname(self._task_log_path), exist_ok=True)
        os.makedirs(self.workspace_root, exist_ok=True)
        logger.info(
            f"AIRSWorldPlugin initialized: tasks_root={self.tasks_root}, shared_data_dir={self.shared_data_dir}, "
            f"catalog_size={len(self._tasks_catalog)}"
        )

    def _resolve_path(self, path: str, project_root: str) -> str:
        p = Path(path)
        if p.is_absolute():
            return str(p)
        direct = Path(project_root) / p
        if direct.exists():
            return str(direct.resolve())
        repo_rel = Path(project_root).parents[1] / p
        return str(repo_rel.resolve())

    def _resolve_taskboard_log_events(self) -> set[str]:
        explicit = str(os.getenv("SCIMAS_TASKBOARD_LOG_EVENTS", "") or "").strip()
        if explicit:
            items = {x.strip().lower() for x in explicit.split(",") if x.strip()}
            return {x for x in items if x in {"create", "claim", "complete", "release"}}
        if self._log_mode in {"compact", "minimal"}:
            return {"claim", "complete", "release"}
        return {"create", "claim", "complete", "release"}

    def _load_tasks_catalog(self) -> None:
        root = Path(self.tasks_root)
        if not root.exists():
            raise RuntimeError(f"AIRS tasks root not found: {root}")

        require_raw = os.getenv("SCIMAS_AIRS_REQUIRE_RAW_DATA", "1").lower() not in {"0", "false", "no"}
        task_dirs = [d for d in sorted(root.iterdir()) if d.is_dir() and (d / "metadata.yaml").exists()]
        catalog: List[Dict[str, Any]] = []

        for task_dir in task_dirs:
            try:
                metadata = self._read_yaml(task_dir / "metadata.yaml")
            except Exception as e:
                logger.warning(f"Skip task {task_dir.name}: metadata read failed: {e}")
                continue

            logging_info = metadata.get("logging_info") or {}
            dataset_rel = self._parse_prepare_dataset_rel(task_dir / "prepare.py")
            raw_ready = True
            if dataset_rel:
                raw_ready = (Path(self.shared_data_dir) / dataset_rel).exists()
            elif require_raw:
                # Some task prepare.py constructs path differently; keep if dataset exists by logging_info dataset root.
                dataset_name = str(logging_info.get("dataset") or "").strip()
                dataset_root = (Path(self.shared_data_dir) / dataset_name) if dataset_name else None
                raw_ready = bool(dataset_root and dataset_root.exists())

            if require_raw and not raw_ready:
                logger.info(f"Skip task {task_dir.name}: raw data missing")
                continue

            project_desc = task_dir / "project_description.md"
            catalog.append(
                {
                    "task_name": task_dir.name,
                    "task_path": str(task_dir),
                    "metadata": metadata,
                    "logging_info": logging_info,
                    "dataset_rel": dataset_rel,
                    "metric": logging_info.get("metric"),
                    "metric_lower_is_better": bool(metadata.get("metric_lower_is_better", False)),
                    "project_description_path": str(project_desc) if project_desc.exists() else None,
                }
            )

        if not catalog:
            raise RuntimeError(
                "No AIRS tasks available after filtering. Check tasks_root/shared_data_dir and SCIMAS_AIRS_REQUIRE_RAW_DATA."
            )
        self._tasks_catalog = catalog

    def _parse_prepare_dataset_rel(self, prepare_path: Path) -> str:
        if not prepare_path.exists():
            return ""
        try:
            text = prepare_path.read_text(encoding="utf-8")
        except Exception:
            return ""
        m = re.search(r"dataset_source_fpath\s*=\s*os\.path\.join\(global_shared_data_dir,\s*['\"]([^'\"]+)['\"]\)", text)
        if m:
            return m.group(1)
        m2 = re.search(
            r"dataset_source_fpath\s*=\s*os\.path\.join\(global_shared_data_dir,\s*['\"]([^'\"]+)['\"]\s*,\s*['\"]([^'\"]+)['\"]\)",
            text,
        )
        if m2:
            return f"{m2.group(1)}/{m2.group(2)}"
        return ""

    def _read_yaml(self, path: Path) -> Dict[str, Any]:
        with path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    def _load_agent_ids(self) -> List[str]:
        map_path = Path(self._project_root) / "data" / "map" / "agents.jsonl"
        if not map_path.exists():
            return []
        ids: List[str] = []
        try:
            with map_path.open("r", encoding="utf-8") as f:
                for raw in f:
                    line = raw.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    aid = str(obj.get("id") or obj.get("agent_id") or "").strip()
                    if aid:
                        ids.append(aid)
        except Exception as e:
            logger.warning(f"Failed to load agent ids from {map_path}: {e}")
            return []
        return ids

    def _pick_task(self) -> Dict[str, Any]:
        if self.task_sampling == "fixed_list" and self.fixed_tasks:
            name_set = set(self.fixed_tasks)
            cands = [t for t in self._tasks_catalog if t["task_name"] in name_set]
            if cands:
                return self._rng.choice(cands)
        if self.task_sampling == "round_robin":
            idx = self._round_robin_idx % len(self._tasks_catalog)
            self._round_robin_idx += 1
            return self._tasks_catalog[idx]
        return self._rng.choice(self._tasks_catalog)

    def _build_cards_for_task(self, task: Dict[str, Any]) -> List[Dict[str, Any]]:
        cards: List[Dict[str, Any]] = []
        card_idx = 0
        info = task.get("logging_info") or {}
        metadata = task.get("metadata") or {}

        summary_pairs = [
            ("task", task.get("task_name")),
            ("dataset", info.get("dataset")),
            ("metric", info.get("metric")),
            ("research_problem", info.get("research_problem")),
            ("output_type", info.get("output_type")),
            ("input_columns", info.get("input_columns")),
            ("scoring_column", info.get("scoring_column")),
            ("shape", info.get("shape")),
            ("metric_lower_is_better", metadata.get("metric_lower_is_better")),
            ("estimated_worst_score", info.get("estimated_worst_score")),
            ("optimal_score", info.get("optimal_score")),
        ]
        for key, value in summary_pairs:
            if value is None:
                continue
            cards.append(
                {
                    "id": card_idx,
                    "citation_id": f"C{card_idx:04d}",
                    "kind": "metadata",
                    "title": key,
                    "text": f"{key}: {value}",
                }
            )
            card_idx += 1

        desc_path = task.get("project_description_path")
        if desc_path and os.path.exists(desc_path):
            text = Path(desc_path).read_text(encoding="utf-8", errors="ignore")
            blocks = [b.strip() for b in re.split(r"\n\s*\n", text) if b.strip()]
            for block in blocks[:24]:
                cards.append(
                    {
                        "id": card_idx,
                        "citation_id": f"C{card_idx:04d}",
                        "kind": "description",
                        "title": "project_description",
                        "text": block[:900],
                    }
                )
                card_idx += 1

        if not cards:
            cards.append(
                {
                    "id": 0,
                    "citation_id": "C0000",
                    "kind": "metadata",
                    "title": "fallback",
                    "text": f"AIRS task {task.get('task_name')} has no description cards.",
                }
            )
        return cards

    async def reset_episode(self, seed: Optional[int] = None) -> Dict[str, Any]:
        if seed is not None:
            self.seed = int(seed)
            self._rng.seed(self.seed)

        self._episode_id += 1
        self._task_seq = 0
        self._paper_seq = 0
        self._run_seq = 0
        self._logical_tick = 0
        self._taskboard_event_counts = {"create": 0, "claim": 0, "complete": 0, "release": 0}

        self._current_task = self._pick_task()
        self._current_cards = self._build_cards_for_task(self._current_task)
        self._agent_views = {}
        self._agent_hint_accuracy = {}
        self._score_cache = []
        self._paper_bank = {}
        self._paper_replications = {}
        self._prepared_data_cache_dir = None

        task_name = str(self._current_task.get("task_name"))
        episode_dir = Path(self.workspace_root) / f"episode_{self._episode_id:03d}__{task_name}"
        if episode_dir.exists():
            shutil.rmtree(episode_dir, ignore_errors=True)
        episode_dir.mkdir(parents=True, exist_ok=True)
        self._current_workspace = str(episode_dir)

        self._select_active_workers()
        self._init_taskboard()
        logger.info(f"AIRS episode reset: ep={self._episode_id}, task={task_name}")
        return {
            "episode_id": self._episode_id,
            "task_name": task_name,
            "task_path": self._current_task.get("task_path"),
        }

    def _next_task_id(self) -> str:
        self._task_seq += 1
        return f"T{self._episode_id:03d}-{self._task_seq:05d}"

    def _append_taskboard_log(self, event: str, task: Dict[str, Any], meta: Optional[Dict[str, Any]] = None) -> None:
        if event in self._taskboard_event_counts:
            self._taskboard_event_counts[event] = int(self._taskboard_event_counts.get(event, 0) or 0) + 1
        if event not in self._taskboard_log_events:
            return
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
            "claimed_tick": None,
            "lease_ttl": self._task_lease_ttl,
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

    def _estimate_agent_count(self) -> int:
        if self._all_agent_ids:
            return max(1, len(self._all_agent_ids))
        return 20

    def _select_active_workers(self) -> None:
        all_agents = list(self._all_agent_ids)

        if not all_agents:
            self._active_workers = set()
            return

        k = min(len(all_agents), max(1, self._active_worker_count))
        self._active_workers = set(self._rng.sample(all_agents, k=k))

    def _normalize_tick(self, now_tick: Optional[int] = None) -> int:
        if now_tick is None:
            return int(self._logical_tick)
        normalized = int(now_tick)
        if normalized > self._logical_tick:
            self._logical_tick = normalized
        return normalized

    async def _expire_task_leases(self, now_tick: Optional[int] = None) -> int:
        now_tick = self._normalize_tick(now_tick)
        expired = 0
        for task in self._task_board.values():
            if task.get("status") != "claimed":
                continue
            claimed_tick = int(task.get("claimed_tick", now_tick) or now_tick)
            ttl = int(task.get("lease_ttl", self._task_lease_ttl) or self._task_lease_ttl)
            if (now_tick - claimed_tick) < ttl:
                continue
            holder = task.get("claimed_by")
            task["status"] = "open"
            task["claimed_by"] = None
            task["claimed_tick"] = None
            self._append_taskboard_log(
                "release",
                task,
                {
                    "agent_id": holder,
                    "reason": "lease_expired",
                    "expired_at_tick": now_tick,
                },
            )
            expired += 1
        return expired

    def _bootstrap_taskboard(self) -> None:
        n_agents = self._estimate_agent_count()
        n_workers = len(self._active_workers) if self._active_workers else min(n_agents, self._active_worker_count)
        n_workers = max(1, n_workers)
        read_count = max(2, min(6, n_workers // 2 + 1))
        experiment_count = max(2, min(self.budget, n_workers))
        hypothesize_count = max(2, min(5, n_workers // 2 + 1))
        write_count = max(1, min(3, n_workers // 3 + 1))

        for _ in range(read_count):
            self._create_task_internal("read", payload={"topic": "task_requirements"})
        for _ in range(experiment_count):
            self._create_task_internal("experiment", payload={})
        for _ in range(hypothesize_count):
            self._create_task_internal("hypothesize", payload={})
        for _ in range(write_count):
            self._create_task_internal("write", payload={})

        if not self._strict_task_dependencies:
            review_count = max(1, min(3, n_workers // 3 + 1))
            replicate_count = max(1, min(3, n_workers // 3 + 1))
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

    def _task_blockers(self, task: Dict[str, Any]) -> List[str]:
        blockers: List[str] = []
        for dep_id in list(task.get("depends_on") or []):
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
            if task_type == "review" and not paper_id:
                # Support pre-write iterative review over run history.
                run_id = payload.get("run_id")
                if not run_id:
                    blockers.append("paper_id_or_run_id_required")
            else:
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

    async def task_create(
        self,
        task_type: str,
        payload: Optional[Dict[str, Any]] = None,
        priority: Optional[int] = None,
        depends_on: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        task = self._create_task_internal(task_type=task_type, payload=payload or {}, priority=priority, depends_on=depends_on)
        return {"ok": True, "task": task}

    async def task_list(
        self,
        status: Optional[str] = None,
        agent_id: Optional[str] = None,
        current_tick: Optional[int] = None,
    ) -> Dict[str, Any]:
        await self._expire_task_leases(now_tick=current_tick)
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

    async def task_claim(self, task_id: str, agent_id: str, current_tick: Optional[int] = None) -> Dict[str, Any]:
        now_tick = self._normalize_tick(current_tick)
        await self._expire_task_leases(now_tick=now_tick)
        if self._active_workers and agent_id not in self._active_workers:
            return {
                "ok": False,
                "reason": "not_active_worker",
                "agent_id": agent_id,
                "active_worker_count": len(self._active_workers),
            }
        my_claimed = [
            t
            for t in self._task_board.values()
            if t.get("status") == "claimed" and t.get("claimed_by") == agent_id
        ]
        if len(my_claimed) >= self._max_claims_per_agent:
            return {
                "ok": False,
                "reason": "claim_quota_exceeded",
                "quota": self._max_claims_per_agent,
                "claimed_count": len(my_claimed),
            }
        task = self._task_board.get(task_id)
        if not task:
            return {"ok": False, "reason": "task_not_found"}
        if task.get("status") != "open":
            return {"ok": False, "reason": f"task_not_open:{task.get('status')}", "task": task}

        task_type = str(task.get("task_type") or "")
        payload = task.get("payload") or {}
        if task_type in {"review", "replicate"}:
            paper_id = str(payload.get("paper_id") or "")
            if paper_id:
                stored = self._paper_bank.get(paper_id) or {}
                paper = stored.get("paper") or {}
                author_id = str(paper.get("author_id") or "")
                if author_id and author_id == str(agent_id):
                    if task_type == "review" and self._forbid_self_review:
                        return {
                            "ok": False,
                            "reason": "self_review_forbidden",
                            "paper_id": paper_id,
                            "task": self._decorate_task_view(task),
                        }
                    if task_type == "replicate" and self._forbid_self_replicate:
                        return {
                            "ok": False,
                            "reason": "self_replicate_forbidden",
                            "paper_id": paper_id,
                            "task": self._decorate_task_view(task),
                        }

        blockers = self._task_blockers(task)
        if blockers:
            return {"ok": False, "reason": "task_blocked", "blocked_by": blockers, "task": self._decorate_task_view(task)}
        task["status"] = "claimed"
        task["claimed_by"] = agent_id
        task["claimed_tick"] = now_tick
        task["lease_ttl"] = self._task_lease_ttl
        self._append_taskboard_log("claim", task, {"agent_id": agent_id})
        return {"ok": True, "task": task}

    async def task_release(
        self,
        task_id: str,
        agent_id: str,
        reason: Optional[str] = None,
        current_tick: Optional[int] = None,
    ) -> Dict[str, Any]:
        await self._expire_task_leases(now_tick=current_tick)
        task = self._task_board.get(task_id)
        if not task:
            return {"ok": False, "reason": "task_not_found"}
        if task.get("status") != "claimed":
            return {"ok": False, "reason": f"task_not_claimed:{task.get('status')}", "task": task}
        if task.get("claimed_by") and task.get("claimed_by") != agent_id:
            return {"ok": False, "reason": "task_claimed_by_other", "task": task}
        task["status"] = "open"
        task["claimed_by"] = None
        task["claimed_tick"] = None
        self._append_taskboard_log("release", task, {"agent_id": agent_id, "reason": reason or ""})
        return {"ok": True, "task": task}

    async def task_complete(
        self,
        task_id: str,
        agent_id: str,
        result: Optional[Dict[str, Any]] = None,
        current_tick: Optional[int] = None,
    ) -> Dict[str, Any]:
        await self._expire_task_leases(now_tick=current_tick)
        task = self._task_board.get(task_id)
        if not task:
            return {"ok": False, "reason": "task_not_found"}
        if task.get("status") == "completed":
            return {"ok": False, "reason": "task_already_completed", "task": task}
        if task.get("claimed_by") and task.get("claimed_by") != agent_id:
            return {"ok": False, "reason": "task_claimed_by_other", "task": task}
        task["status"] = "completed"
        task["completed_by"] = agent_id
        task["claimed_tick"] = None
        task["result"] = result or {}
        self._append_taskboard_log("complete", task, {"agent_id": agent_id})

        # If replication failed, inject revise loop.
        try:
            if self._strict_task_dependencies and str(task.get("task_type")) == "replicate":
                action_data = ((task.get("result") or {}).get("action_data") or {})
                paper_id = action_data.get("paper_id") or (task.get("payload") or {}).get("paper_id")
                metrics_after = action_data.get("paper_metrics_after_replication") or {}
                replication_failed = bool(metrics_after) and (not bool(metrics_after.get("replication_ok", False)))
                if paper_id and replication_failed:
                    review_task = self._create_task_internal(
                        "review",
                        payload={"paper_id": paper_id, "revision_reason": "replication_failed"},
                        priority=9,
                    )
                    self._create_task_internal(
                        "write",
                        payload={"paper_id": paper_id, "revision_reason": "replication_failed"},
                        priority=10,
                        depends_on=[review_task.get("task_id")],
                    )
        except Exception as e:
            logger.warning(f"Failed to enqueue revision loop: {e}")

        return {"ok": True, "task": task}

    async def get_world_spec(self, current_tick: Optional[int] = None) -> Dict[str, Any]:
        await self._expire_task_leases(now_tick=current_tick)
        summary = {"open": 0, "claimed": 0, "completed": 0}
        for task in self._task_board.values():
            st = task.get("status", "open")
            summary[st] = summary.get(st, 0) + 1

        info = (self._current_task or {}).get("logging_info") or {}
        return {
            "episode_id": self._episode_id,
            "budget": self.budget,
            "task_name": (self._current_task or {}).get("task_name"),
            "task_path": (self._current_task or {}).get("task_path"),
            "taskboard": summary,
            "metric": info.get("metric"),
            "dataset": info.get("dataset"),
            "category": info.get("category"),
            "research_problem": info.get("research_problem"),
            "active_worker_count": len(self._active_workers) if self._active_workers else self._active_worker_count,
            "max_claims_per_agent": self._max_claims_per_agent,
            "task_lease_ttl": self._task_lease_ttl,
            "forbid_self_review": self._forbid_self_review,
            "forbid_self_replicate": self._forbid_self_replicate,
            "solver_enabled": self._solver_enabled,
            "solver_prepare_once": self._solver_prepare_once,
            "taskboard_event_counts": dict(self._taskboard_event_counts),
            # Kept for backward compatibility with existing planner interfaces.
            "variables": [],
            "target": "score",
        }

    async def get_taskboard_metrics(self) -> Dict[str, Any]:
        summary = {"open": 0, "claimed": 0, "completed": 0}
        for task in self._task_board.values():
            st = str(task.get("status") or "open")
            summary[st] = int(summary.get(st, 0) or 0) + 1
        return {
            "episode_id": self._episode_id,
            "task_name": (self._current_task or {}).get("task_name"),
            "event_counts": dict(self._taskboard_event_counts),
            "summary": summary,
            "active_worker_count": len(self._active_workers) if self._active_workers else self._active_worker_count,
            "max_claims_per_agent": self._max_claims_per_agent,
            "task_lease_ttl": self._task_lease_ttl,
        }

    async def is_active_worker(self, agent_id: str) -> Dict[str, Any]:
        if not self._active_workers:
            return {"agent_id": agent_id, "is_active": True, "active_worker_count": 0}
        return {
            "agent_id": agent_id,
            "is_active": agent_id in self._active_workers,
            "active_worker_count": len(self._active_workers),
        }

    def _ensure_agent_view(self, agent_id: str) -> None:
        if not agent_id:
            return
        if agent_id not in self._agent_views:
            ids = list(range(len(self._current_cards)))
            self._rng.shuffle(ids)
            take = max(1, int(len(ids) * self.view_ratio))
            self._agent_views[agent_id] = ids[:take]
        if agent_id not in self._agent_hint_accuracy:
            jitter = self._rng.uniform(-0.1, 0.1)
            self._agent_hint_accuracy[agent_id] = min(0.99, max(0.6, self.hint_accuracy + jitter))

    async def read_literature(self, agent_id: Optional[str] = None, topic: Optional[str] = None) -> Dict[str, Any]:
        self._ensure_agent_view(agent_id or "")
        view_ids = self._agent_views.get(agent_id or "", list(range(len(self._current_cards))))
        accuracy = self._agent_hint_accuracy.get(agent_id or "", self.hint_accuracy)
        cards: List[Dict[str, Any]] = []
        hints: List[str] = []
        for cid in view_ids:
            card = dict(self._current_cards[cid])
            text = str(card.get("text") or "")
            if self._rng.random() > accuracy and len(text) > 24:
                text = text[: max(12, len(text) // 2)] + " ..."
                card["text"] = text
            cards.append(card)
            hints.append(f"[{card.get('citation_id')}] {str(card.get('title') or 'card')}")
        return {
            "topic": topic or "airs_task_understanding",
            "hints": hints,
            "cards": cards,
            "agent_view_size": len(cards),
            "task_name": (self._current_task or {}).get("task_name"),
        }

    def _next_run_id(self) -> str:
        self._run_seq += 1
        return f"RUN{self._episode_id:03d}-{self._run_seq:04d}"

    def _task_workspace(self) -> str:
        if not self._current_workspace:
            fallback = Path(self.workspace_root) / f"episode_{self._episode_id:03d}__fallback"
            fallback.mkdir(parents=True, exist_ok=True)
            self._current_workspace = str(fallback)
        return self._current_workspace

    def _run_script(self, script_path: str, args: List[str], cwd: Optional[str] = None, timeout_s: int = 180) -> Tuple[int, str, str]:
        cmd = [self._python_cmd, script_path] + args
        proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout_s)
        return proc.returncode, proc.stdout or "", proc.stderr or ""

    def _shape_second_dim(self, shape: Any) -> int:
        if isinstance(shape, (list, tuple)):
            if len(shape) >= 2 and isinstance(shape[1], int):
                return max(1, int(shape[1]))
            return 1
        if isinstance(shape, str):
            nums = [int(x) for x in re.findall(r"\d+", shape)]
            if len(nums) >= 2:
                return max(1, nums[1])
            return 1
        return 1

    def _default_prediction(self, metric: str, category: str) -> Any:
        metric = str(metric or "").lower()
        category = str(category or "").lower()
        if "accuracy" in metric:
            return 0
        if "exactmatch" in metric or "rouge" in metric:
            return "None"
        if "pass@" in metric:
            return "print('hello')"
        if "mrr" in metric:
            return 0
        if "mae" in metric or "mase" in metric or "meanabsoluteerror" in metric:
            return 0.0
        if "time series" in category:
            return 0.0
        return 0.0

    def _load_prepared_test(self, task: Dict[str, Any], data_mount_dir: str) -> Dict[str, Any]:
        from datasets import load_from_disk

        task_name = str(task.get("task_name") or "")
        base = Path(data_mount_dir) / "test"
        if not base.exists():
            raise RuntimeError("Prepared test split missing")
        if task_name == "CodeRetrievalCodeXGlueMRR":
            queries_path = base / "queries"
            corpus_path = base / "search_corpus"
            if not queries_path.exists() or not corpus_path.exists():
                raise RuntimeError("CodeRetrieval prepared test requires test/queries and test/search_corpus")
            queries = load_from_disk(str(queries_path))
            corpus = load_from_disk(str(corpus_path))
            return {"mode": "retrieval", "queries": queries, "corpus": corpus}

        dset = load_from_disk(str(base))
        return {"mode": "table", "test": dset}

    def _load_prepared_split_df(self, data_mount_dir: str, split_name: str):
        from datasets import load_from_disk
        import pandas as pd

        split_path = Path(data_mount_dir) / split_name
        if not split_path.exists():
            return None
        dset = load_from_disk(str(split_path))
        try:
            df = dset.to_pandas()
        except Exception:
            df = pd.DataFrame(dset[:])
        if df is None or df.empty:
            return None
        return df

    def _infer_target_column(self, task: Dict[str, Any], train_df) -> str:
        info = task.get("logging_info") or {}
        scoring_col = str(info.get("scoring_column") or "").strip()
        if scoring_col and scoring_col in train_df.columns:
            return scoring_col
        # Fallback for task packs where scoring_column is omitted or changed by prepare.py.
        candidates = ["label", "labels", "target", "y", "answer", "Answer", "relatedness_score"]
        for col in candidates:
            if col in train_df.columns:
                return col
        return ""

    def _safe_solver_spec(self, task: Dict[str, Any], config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        info = task.get("logging_info") or {}
        metric = str(info.get("metric") or "").lower()
        category = str(info.get("category") or "").lower()
        input_cols = [str(c) for c in (info.get("input_columns") or []) if str(c).strip()]

        default_model_family = "tfidf_logreg"
        if "mae" in metric or "mase" in metric or "spearman" in metric:
            default_model_family = "tfidf_ridge"
        if "time series" in category:
            default_model_family = "naive_series"

        solver_spec = {
            "model_family": default_model_family,
            "seed": 42,
            "preprocess": {
                "max_features": 50000,
                "ngram_range": [1, 2],
                "min_df": 1,
            },
            "hyperparams": {
                "C": 1.0,
                "max_iter": 2000,
                "class_weight": "balanced",
                "alpha": 1.0,
            },
            "input_columns": input_cols,
        }

        cfg = dict(config or {})
        candidate = cfg.get("solver_spec") if isinstance(cfg.get("solver_spec"), dict) else cfg
        if isinstance(candidate, dict):
            mf = candidate.get("model_family")
            if isinstance(mf, str) and mf.strip():
                solver_spec["model_family"] = mf.strip()[:80]
            seed = candidate.get("seed")
            if isinstance(seed, int):
                solver_spec["seed"] = int(seed)
            if isinstance(candidate.get("input_columns"), list):
                solver_spec["input_columns"] = [str(c) for c in candidate.get("input_columns") if str(c).strip()][:16]
            pre = candidate.get("preprocess")
            if isinstance(pre, dict):
                pp = dict(solver_spec["preprocess"])
                for key in ("max_features", "min_df"):
                    value = pre.get(key)
                    if isinstance(value, int) and value > 0:
                        pp[key] = int(value)
                ng = pre.get("ngram_range")
                if isinstance(ng, (list, tuple)) and len(ng) == 2:
                    try:
                        pp["ngram_range"] = [max(1, int(ng[0])), max(1, int(ng[1]))]
                    except Exception:
                        pass
                solver_spec["preprocess"] = pp
            hp = candidate.get("hyperparams")
            if isinstance(hp, dict):
                hh = dict(solver_spec["hyperparams"])
                for key in ("C", "alpha"):
                    value = hp.get(key)
                    if isinstance(value, (int, float)) and float(value) > 0:
                        hh[key] = float(value)
                max_iter = hp.get("max_iter")
                if isinstance(max_iter, int) and max_iter > 0:
                    hh["max_iter"] = int(max_iter)
                class_weight = hp.get("class_weight")
                if isinstance(class_weight, str):
                    hh["class_weight"] = class_weight[:40]
                solver_spec["hyperparams"] = hh
        return solver_spec

    def _solver_support_check(self, task: Dict[str, Any], train_df, test_df, target_col: str) -> Dict[str, Any]:
        info = task.get("logging_info") or {}
        metric = str(info.get("metric") or "").lower()
        category = str(info.get("category") or "").lower()
        if not target_col:
            return {"supported": False, "reason": "target_column_missing"}
        if target_col not in train_df.columns:
            return {"supported": False, "reason": "target_not_in_train"}
        if train_df.empty or test_df.empty:
            return {"supported": False, "reason": "train_or_test_empty"}
        # Retrieval/code generation/custom QA remain outside this generic baseline solver.
        unsupported_metric_tokens = ("pass@", "mrr", "rouge", "exactmatch")
        if any(tok in metric for tok in unsupported_metric_tokens):
            return {"supported": False, "reason": f"metric_not_supported:{metric}"}
        if "code" in category or "question answering" in category:
            return {"supported": False, "reason": f"category_not_supported:{category}"}
        return {"supported": True, "reason": "ok"}

    def _row_to_text(self, row: Dict[str, Any], columns: List[str]) -> str:
        chunks: List[str] = []
        for col in columns:
            if col not in row:
                continue
            value = row.get(col)
            if value is None:
                continue
            if isinstance(value, (dict, list, tuple)):
                try:
                    text = json.dumps(value, ensure_ascii=False)
                except Exception:
                    text = str(value)
            else:
                text = str(value)
            text = text.strip()
            if not text:
                continue
            chunks.append(f"{col}: {text[:500]}")
        return " | ".join(chunks)

    def _compute_dev_metric(self, metric_name: str, y_true, y_pred) -> Optional[float]:
        import numpy as np

        metric = str(metric_name or "").lower()
        try:
            if "accuracy" in metric:
                from sklearn.metrics import accuracy_score

                return float(accuracy_score(y_true, y_pred))
            if "mae" in metric or "mase" in metric or "meanabsoluteerror" in metric:
                from sklearn.metrics import mean_absolute_error

                return float(mean_absolute_error(y_true, y_pred))
            if "spearman" in metric:
                yt = np.asarray(y_true, dtype=float)
                yp = np.asarray(y_pred, dtype=float)
                if yt.size == 0 or yp.size == 0 or yt.size != yp.size:
                    return None
                rank_t = yt.argsort().argsort().astype(float)
                rank_p = yp.argsort().argsort().astype(float)
                denom = float(np.std(rank_t) * np.std(rank_p))
                if denom <= 1e-12:
                    return 0.0
                corr = float(np.cov(rank_t, rank_p, ddof=0)[0, 1] / denom)
                return max(-1.0, min(1.0, corr))
        except Exception:
            return None
        return None

    def _run_solver_submission(
        self,
        task: Dict[str, Any],
        data_mount_dir: str,
        agent_log_dir: str,
        config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        import pickle
        import pandas as pd
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression, Ridge
        from sklearn.model_selection import train_test_split
        from sklearn.pipeline import Pipeline

        info = task.get("logging_info") or {}
        metric = str(info.get("metric") or "")
        scoring_col = str(info.get("scoring_column") or "prediction")
        output_path = Path(agent_log_dir) / "submission.csv"
        model_path = Path(agent_log_dir) / "model.pkl"
        solver_log_path = Path(agent_log_dir) / "solver_run.json"

        train_df = self._load_prepared_split_df(data_mount_dir=data_mount_dir, split_name="train")
        test_df = self._load_prepared_split_df(data_mount_dir=data_mount_dir, split_name="test")
        val_df = self._load_prepared_split_df(data_mount_dir=data_mount_dir, split_name="validation")
        if val_df is None:
            val_df = self._load_prepared_split_df(data_mount_dir=data_mount_dir, split_name="val")
        if train_df is None or test_df is None:
            raise RuntimeError("solver_prepare_missing_train_or_test")

        target_col = self._infer_target_column(task=task, train_df=train_df)
        support = self._solver_support_check(task=task, train_df=train_df, test_df=test_df, target_col=target_col)
        if not bool(support.get("supported")):
            raise RuntimeError(str(support.get("reason") or "solver_not_supported"))

        solver_spec = self._safe_solver_spec(task=task, config=config)
        feature_cols = [c for c in solver_spec.get("input_columns") or [] if c in train_df.columns and c != target_col]
        if not feature_cols:
            feature_cols = [c for c in train_df.columns if c != target_col]
        if not feature_cols:
            raise RuntimeError("no_feature_columns")

        metric_lower = str(metric or "").lower()
        task_kind = "regression" if any(x in metric_lower for x in ("mae", "mase", "meanabsoluteerror", "spearman")) else "classification"

        train_df = train_df.dropna(subset=[target_col]).reset_index(drop=True)
        if val_df is None or target_col not in val_df.columns:
            split_seed = int(solver_spec.get("seed", 42) or 42)
            train_part, val_part = train_test_split(
                train_df,
                test_size=0.2,
                random_state=split_seed,
                shuffle=True,
            )
            train_df_used = train_part.reset_index(drop=True)
            dev_df = val_part.reset_index(drop=True)
        else:
            train_df_used = train_df
            dev_df = val_df.dropna(subset=[target_col]).reset_index(drop=True)

        if task_kind == "classification":
            # High-cardinality labels are typically generation/QA-like; skip generic classifier.
            uniq = int(train_df_used[target_col].nunique(dropna=True))
            if uniq <= 1 or uniq > 200:
                raise RuntimeError(f"class_label_cardinality_not_supported:{uniq}")

        train_records = train_df_used[feature_cols].to_dict(orient="records")
        dev_records = dev_df[feature_cols].to_dict(orient="records")
        test_records = test_df[feature_cols].to_dict(orient="records")

        x_train = [self._row_to_text(r, feature_cols) for r in train_records]
        x_dev = [self._row_to_text(r, feature_cols) for r in dev_records]
        x_test = [self._row_to_text(r, feature_cols) for r in test_records]
        y_train = train_df_used[target_col].tolist()
        y_dev = dev_df[target_col].tolist()

        preprocess = solver_spec.get("preprocess") or {}
        max_features = int(preprocess.get("max_features", 50000) or 50000)
        min_df = int(preprocess.get("min_df", 1) or 1)
        ngram_raw = preprocess.get("ngram_range") or [1, 2]
        ngram_low = max(1, int(ngram_raw[0]))
        ngram_high = max(ngram_low, int(ngram_raw[1]))
        hp = solver_spec.get("hyperparams") or {}
        model_family = str(solver_spec.get("model_family") or "")

        if task_kind == "classification":
            if model_family in {"linear_svc", "svm_linear"}:
                from sklearn.svm import LinearSVC

                model = LinearSVC(
                    C=float(hp.get("C", 1.0) or 1.0),
                    max_iter=int(hp.get("max_iter", 2000) or 2000),
                    random_state=int(solver_spec.get("seed", 42) or 42),
                )
            else:
                cw = str(hp.get("class_weight") or "balanced")
                model = LogisticRegression(
                    C=float(hp.get("C", 1.0) or 1.0),
                    max_iter=int(hp.get("max_iter", 2000) or 2000),
                    class_weight=(None if cw == "none" else cw),
                    random_state=int(solver_spec.get("seed", 42) or 42),
                    n_jobs=1,
                )
        else:
            model = Ridge(alpha=float(hp.get("alpha", 1.0) or 1.0))

        pipeline = Pipeline(
            steps=[
                (
                    "tfidf",
                    TfidfVectorizer(
                        max_features=max_features,
                        min_df=min_df,
                        ngram_range=(ngram_low, ngram_high),
                    ),
                ),
                ("model", model),
            ]
        )
        pipeline.fit(x_train, y_train)
        y_dev_pred = pipeline.predict(x_dev) if x_dev else []
        dev_score = self._compute_dev_metric(metric_name=metric, y_true=y_dev, y_pred=y_dev_pred) if x_dev else None
        dev_score_norm = self._normalize_metric(float(dev_score), task=task) if dev_score is not None else None
        y_test_pred = pipeline.predict(x_test) if x_test else []

        # Format submission.
        if scoring_col and scoring_col in test_df.columns:
            # Some task packs accidentally keep label column in test; never leak it.
            test_df = test_df.drop(columns=[scoring_col])
        preds = [float(x) if task_kind == "regression" else x for x in list(y_test_pred)]
        if not scoring_col:
            scoring_col = "prediction"
        pd.DataFrame([{scoring_col: p} for p in preds]).to_csv(output_path, index=False)

        with open(model_path, "wb") as f:
            pickle.dump(pipeline, f)
        solver_log = {
            "solver_spec": solver_spec,
            "target_col": target_col,
            "feature_cols": feature_cols[:20],
            "task_kind": task_kind,
            "train_rows": int(len(train_df_used)),
            "dev_rows": int(len(dev_df)),
            "test_rows": int(len(test_df)),
            "dev_score": dev_score,
            "dev_score_norm": dev_score_norm,
            "metric": metric,
        }
        with open(solver_log_path, "w", encoding="utf-8") as f:
            json.dump(solver_log, f, ensure_ascii=False, indent=2)
            f.write("\n")

        return {
            "submission_path": str(output_path),
            "model_path": str(model_path),
            "solver_log_path": str(solver_log_path),
            "dev_score": dev_score,
            "dev_score_norm": dev_score_norm,
            "solver_spec": solver_spec,
            "solver_mode": "iterative_solver",
        }

    def _write_submission(self, task: Dict[str, Any], data_mount_dir: str, agent_log_dir: str, strategy: str) -> str:
        del strategy
        import pandas as pd

        prepared = self._load_prepared_test(task=task, data_mount_dir=data_mount_dir)
        mode = prepared.get("mode")

        info = task.get("logging_info") or {}
        metric = str(info.get("metric") or "")
        category = str(info.get("category") or "")
        scoring_col = str(info.get("scoring_column") or "prediction")
        shape = info.get("shape")

        output_path = Path(agent_log_dir) / "submission.csv"
        task_name = str(task.get("task_name") or "")
        dim2 = self._shape_second_dim(shape)

        if mode == "retrieval":
            queries = prepared["queries"]
            corpus = prepared["corpus"]
            query_values = [str(q) for q in (queries["query"] if "query" in queries.column_names else [])]
            corpus_ids = [int(x) for x in (corpus["id"] if "id" in corpus.column_names else [])]
            if not query_values or not corpus_ids:
                raise RuntimeError("Retrieval submission generation failed due to empty queries/corpus ids")
            topk = int(max(1, int(os.getenv("SCIMAS_AIRS_RETRIEVAL_TOPK", "1"))))
            rows = []
            n_ids = len(corpus_ids)
            for q in query_values:
                start = abs(hash(q)) % n_ids
                ranking = [int(corpus_ids[(start + i) % n_ids]) for i in range(min(topk, n_ids))]
                rows.append({"query": q, "rankings": json.dumps(ranking, ensure_ascii=False)})
            pd.DataFrame(rows, columns=["query", "rankings"]).to_csv(output_path, index=False)
            return str(output_path)

        test_dset = prepared["test"]
        n_rows = int(len(test_dset))
        if n_rows <= 0:
            raise RuntimeError("Prepared test split empty; cannot create submission")

        if task_name == "CodeGenerationAPPSPassAt5":
            rows = []
            default_code = self._default_prediction(metric, category)
            for _ in range(n_rows):
                rows.append({f"code{i}": default_code for i in range(1, 6)})
            pd.DataFrame(rows).to_csv(output_path, index=False)
            return str(output_path)

        if task_name == "QuestionAnsweringDuoRCAccuracy":
            # DuoRC evaluator requires exact columns: answer, has_answer
            rows = [{"answer": "", "has_answer": False} for _ in range(n_rows)]
            pd.DataFrame(rows, columns=["answer", "has_answer"]).to_csv(output_path, index=False)
            return str(output_path)

        value = self._default_prediction(metric, category)
        if dim2 <= 1:
            pd.DataFrame([{scoring_col: value} for _ in range(n_rows)]).to_csv(output_path, index=False)
            return str(output_path)

        rows = []
        for _ in range(n_rows):
            row = {f"pred{i+1}": value for i in range(dim2)}
            rows.append(row)
        pd.DataFrame(rows).to_csv(output_path, index=False)
        return str(output_path)

    def _extract_eval_result(self, stdout: str) -> Dict[str, float]:
        m = re.search(r"--- EVALUATION RESULT ---\s*(\{[\s\S]*?\})", stdout, re.DOTALL)
        if not m:
            raise RuntimeError("Cannot parse evaluate.py output JSON")
        data = json.loads(m.group(1))
        if not isinstance(data, dict) or not data:
            raise RuntimeError("Invalid evaluate.py result payload")
        metric_name = next(iter(data.keys()))
        metric_value = float(data[metric_name])
        return {"metric_name": metric_name, "metric_value": metric_value}

    def _normalize_metric(self, raw_score: float, task: Dict[str, Any]) -> float:
        metadata = task.get("metadata") or {}
        info = task.get("logging_info") or {}
        lower = bool(metadata.get("metric_lower_is_better", False))

        try:
            worst = float(info.get("estimated_worst_score"))
            best = float(info.get("optimal_score"))
        except Exception:
            worst = 0.0
            best = 1.0 if not lower else 0.0

        if lower:
            denom = (worst - best)
            if abs(denom) < 1e-9:
                score = 1.0 / (1.0 + max(0.0, raw_score))
            else:
                score = (worst - raw_score) / denom
        else:
            denom = (best - worst)
            if abs(denom) < 1e-9:
                score = raw_score
            else:
                score = (raw_score - worst) / denom
        return max(0.0, min(1.0, float(score)))

    def _run_prepare_pipeline(self, task: Dict[str, Any], agent_log_dir: str, data_mount_dir: str) -> Dict[str, Any]:
        task_path = task.get("task_path")
        prepare_py = os.path.join(task_path, "prepare.py")
        if not os.path.exists(prepare_py):
            raise RuntimeError(f"prepare.py missing under {task_path}")

        args = [
            "--global-shared-data-dir",
            self.shared_data_dir,
            "--agent-data-mount-dir",
            data_mount_dir,
            "--agent-log-dir",
            agent_log_dir,
        ]

        timeout_prepare = int(float(os.getenv("SCIMAS_AIRS_PREPARE_TIMEOUT_S", "180")))

        rc, out, err = self._run_script(prepare_py, args, timeout_s=timeout_prepare)
        if rc != 0:
            raise RuntimeError(f"prepare.py failed: {err[-800:]}")
        return {"prepare_stdout_tail": out[-1000:], "prepare_stderr_tail": err[-1000:]}

    def _ensure_prepared_data_cache(self, task: Dict[str, Any], run_agent_log_dir: str) -> Dict[str, Any]:
        if self._solver_prepare_once and self._prepared_data_cache_dir:
            return {"cache_dir": self._prepared_data_cache_dir, "prepare_stdout_tail": ""}

        base = Path(self._task_workspace())
        cache_data_dir = base / "_prepared_agent_data"
        cache_log_dir = base / "_prepared_agent_log"
        if cache_data_dir.exists():
            shutil.rmtree(cache_data_dir, ignore_errors=True)
        if cache_log_dir.exists():
            shutil.rmtree(cache_log_dir, ignore_errors=True)
        cache_data_dir.mkdir(parents=True, exist_ok=True)
        cache_log_dir.mkdir(parents=True, exist_ok=True)

        prepare_result = self._run_prepare_pipeline(
            task=task,
            agent_log_dir=str(run_agent_log_dir),
            data_mount_dir=str(cache_data_dir),
        )
        self._prepared_data_cache_dir = str(cache_data_dir)
        return {
            "cache_dir": self._prepared_data_cache_dir,
            "prepare_stdout_tail": str(prepare_result.get("prepare_stdout_tail") or ""),
        }

    def _materialize_data_mount_from_cache(self, cache_dir: str, target_dir: str) -> None:
        src = Path(cache_dir)
        dst = Path(target_dir)
        if not src.exists():
            raise RuntimeError(f"Prepared cache missing: {cache_dir}")
        if dst.exists():
            shutil.rmtree(dst, ignore_errors=True)
        shutil.copytree(src, dst)

    def _run_eval_pipeline(self, task: Dict[str, Any], agent_log_dir: str, data_mount_dir: str) -> Dict[str, Any]:
        task_path = task.get("task_path")
        eval_prepare_py = os.path.join(task_path, "evaluate_prepare.py")
        evaluate_py = os.path.join(task_path, "evaluate.py")

        if not os.path.exists(eval_prepare_py) or not os.path.exists(evaluate_py):
            raise RuntimeError(f"Evaluation scripts missing under {task_path}")

        args = [
            "--global-shared-data-dir",
            self.shared_data_dir,
            "--agent-data-mount-dir",
            data_mount_dir,
            "--agent-log-dir",
            agent_log_dir,
        ]

        timeout_prepare = int(float(os.getenv("SCIMAS_AIRS_PREPARE_TIMEOUT_S", "180")))
        timeout_eval = int(float(os.getenv("SCIMAS_AIRS_EVAL_TIMEOUT_S", "180")))

        rc, out2, err2 = self._run_script(eval_prepare_py, args, timeout_s=timeout_prepare)
        if rc != 0:
            raise RuntimeError(f"evaluate_prepare.py failed: {err2[-800:]}")

        with tempfile.TemporaryDirectory(prefix="airs_eval_") as td:
            data_link = os.path.join(td, "data")
            if os.path.lexists(data_link):
                os.remove(data_link)
            os.symlink(data_mount_dir, data_link)

            rc, out3, err3 = self._run_script(
                evaluate_py,
                ["--submission-file", "./data/submission.csv"],
                cwd=td,
                timeout_s=timeout_eval,
            )
            if rc != 0:
                raise RuntimeError(f"evaluate.py failed: {err3[-800:]}")

        parsed = self._extract_eval_result(out3)
        raw_score = float(parsed["metric_value"])
        score_norm = self._normalize_metric(raw_score=raw_score, task=task)
        return {
            "metric_name": parsed["metric_name"],
            "raw_score": raw_score,
            "score_norm": score_norm,
            "stdout_tail": out3[-1000:],
        }

    async def run_experiment(
        self,
        intervention: Optional[Dict[str, float]] = None,
        n_samples: int = 50,
        config: Optional[Dict[str, Any]] = None,
        agent_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        del intervention, n_samples  # Unused in AIRS workflow; kept for compatibility.
        task = self._current_task
        if not task:
            await self.reset_episode()
            task = self._current_task
        if not task:
            raise RuntimeError("No AIRS task selected")

        run_id = self._next_run_id()
        run_dir = Path(self._task_workspace()) / run_id
        data_mount = run_dir / "agent_data"
        agent_log = run_dir / "agent_log"
        data_mount.mkdir(parents=True, exist_ok=True)
        agent_log.mkdir(parents=True, exist_ok=True)

        started = datetime.utcnow()
        strategy = str((config or {}).get("strategy") or "default_baseline")
        ok = True
        metric_name = ""
        raw_score = 0.0
        score_norm = 0.0
        submission_path = ""
        dev_score = None
        dev_score_norm = None
        model_path = ""
        solver_log_path = ""
        solver_mode = "rule_template"
        error = None
        stdout_tail = ""
        prepare_stdout_tail = ""
        fallback_reason = ""

        try:
            prepare_cache = self._ensure_prepared_data_cache(task=task, run_agent_log_dir=str(agent_log))
            prepare_stdout_tail = str(prepare_cache.get("prepare_stdout_tail") or "")
            self._materialize_data_mount_from_cache(str(prepare_cache.get("cache_dir") or ""), str(data_mount))

            solver_result = None
            solver_error = None
            if self._solver_enabled:
                try:
                    solver_result = self._run_solver_submission(
                        task=task,
                        data_mount_dir=str(data_mount),
                        agent_log_dir=str(agent_log),
                        config=config or {},
                    )
                except Exception as e:
                    solver_error = str(e)
                    fallback_reason = solver_error
                    logger.info(f"AIRS solver fallback on {task.get('task_name')}/{run_id}: {solver_error}")

            if isinstance(solver_result, dict) and solver_result.get("submission_path"):
                submission_path = str(solver_result.get("submission_path"))
                model_path = str(solver_result.get("model_path") or "")
                solver_log_path = str(solver_result.get("solver_log_path") or "")
                dev_score = solver_result.get("dev_score")
                dev_score_norm = solver_result.get("dev_score_norm")
                solver_mode = str(solver_result.get("solver_mode") or "iterative_solver")
            else:
                if not self._solver_fallback_template:
                    raise RuntimeError(f"solver_failed_no_fallback:{fallback_reason or 'unknown'}")
                submission_path = self._write_submission(
                    task=task,
                    data_mount_dir=str(data_mount),
                    agent_log_dir=str(agent_log),
                    strategy=strategy,
                )
                solver_mode = "rule_template_fallback"

            eval_result = self._run_eval_pipeline(task=task, agent_log_dir=str(agent_log), data_mount_dir=str(data_mount))
            metric_name = eval_result["metric_name"]
            raw_score = float(eval_result["raw_score"])
            score_norm = float(eval_result["score_norm"])
            stdout_tail = str(eval_result.get("stdout_tail") or "")
        except Exception as e:
            ok = False
            error = str(e)
            logger.warning(f"AIRS run_experiment failed on {task.get('task_name')}/{run_id}: {e}")

        ended = datetime.utcnow()
        elapsed_s = max(0.0, (ended - started).total_seconds())
        run_record = {
            "run_id": run_id,
            "ts": started.isoformat() + "Z",
            "task_name": task.get("task_name"),
            "agent_id": agent_id,
            "strategy": strategy,
            "ok": ok,
            "submission_path": submission_path,
            "metric_name": metric_name,
            "raw_score": raw_score,
            "score_norm": score_norm,
            "dev_score": dev_score,
            "dev_score_norm": dev_score_norm,
            "elapsed_s": elapsed_s,
            "cost": min(1.0, elapsed_s / max(1.0, float(os.getenv("SCIMAS_AIRS_COST_TIME_BUDGET_S", "240")))),
            "stdout_tail": stdout_tail,
            "prepare_stdout_tail": prepare_stdout_tail,
            "model_path": model_path,
            "solver_log_path": solver_log_path,
            "solver_mode": solver_mode,
            "fallback_reason": fallback_reason,
            "error": error,
        }
        self._score_cache.append(run_record)
        return run_record

    async def evaluate_submission(self, submission_path: str) -> Dict[str, Any]:
        task = self._current_task
        if not task:
            return {"ok": False, "reason": "no_active_task"}
        if not submission_path or not os.path.exists(submission_path):
            return {"ok": False, "reason": "submission_not_found", "submission_path": submission_path}

        run_id = self._next_run_id()
        run_dir = Path(self._task_workspace()) / f"eval_{run_id}"
        data_mount = run_dir / "agent_data"
        agent_log = run_dir / "agent_log"
        data_mount.mkdir(parents=True, exist_ok=True)
        agent_log.mkdir(parents=True, exist_ok=True)

        shutil.copy2(submission_path, agent_log / "submission.csv")

        try:
            eval_result = self._run_eval_pipeline(task=task, agent_log_dir=str(agent_log), data_mount_dir=str(data_mount))
            return {
                "ok": True,
                "run_id": run_id,
                "task_name": task.get("task_name"),
                "submission_path": str(agent_log / "submission.csv"),
                "metric_name": eval_result["metric_name"],
                "raw_score": float(eval_result["raw_score"]),
                "score_norm": float(eval_result["score_norm"]),
                "stdout_tail": eval_result.get("stdout_tail"),
            }
        except Exception as e:
            return {"ok": False, "reason": str(e), "run_id": run_id}

    async def submit_paper(self, paper: Dict[str, Any]) -> Dict[str, Any]:
        self._paper_seq += 1
        paper_id = f"P{self._episode_id:03d}-{self._paper_seq:04d}"
        p = dict(paper or {})
        p["paper_id"] = paper_id
        self._paper_bank[paper_id] = {
            "paper_id": paper_id,
            "episode_id": self._episode_id,
            "paper": p,
        }
        self._paper_replications.setdefault(paper_id, [])
        return {"paper_id": paper_id, "episode_id": self._episode_id}

    async def get_paper(self, paper_id: str) -> Optional[Dict[str, Any]]:
        stored = self._paper_bank.get(paper_id)
        if not stored:
            return None
        return stored.get("paper")

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
        artifacts = paper.get("artifacts") or {}
        claimed = paper.get("claimed_results") or {}
        submission_path = artifacts.get("submission_path")
        if not submission_path:
            return {"ok": False, "reason": "paper_missing_submission", "paper_id": paper_id}

        eval_res = await self.evaluate_submission(submission_path=submission_path)
        if not bool(eval_res.get("ok")):
            return {"ok": False, "reason": f"replication_eval_failed:{eval_res.get('reason')}", "paper_id": paper_id}

        claimed_norm = float(claimed.get("score_norm", 0.0) or 0.0)
        repl_norm = float(eval_res.get("score_norm", 0.0) or 0.0)
        delta = abs(repl_norm - claimed_norm)
        support_ratio = max(0.0, min(1.0, 1.0 - (delta / max(1e-6, self._replication_delta_tol))))

        support = {
            "covered_edges": 1,
            "supported_edges": int(1 if support_ratio >= 0.5 else 0),
            "support_ratio": support_ratio,
            "delta_norm": delta,
            "threshold": self._replication_delta_tol,
            "mode": "score_consistency",
            "replicated_score_norm": repl_norm,
            "claimed_score_norm": claimed_norm,
        }
        record = {
            "ts": datetime.utcnow().isoformat() + "Z",
            "episode_id": self._episode_id,
            "paper_id": paper_id,
            "agent_id": agent_id,
            "source": source,
            "replication": replication or {},
            "support": support,
            "support_main": support,
            "support_holdout": None,
            "evaluate_result": eval_res,
        }
        self._paper_replications.setdefault(paper_id, []).append(record)
        return {
            "ok": True,
            "paper_id": paper_id,
            "support": support,
            "support_main": support,
            "support_holdout": None,
            "holdout_enabled": False,
            "replication_count": len(self._paper_replications.get(paper_id) or []),
        }

    async def get_replications(self, paper_id: str) -> Dict[str, Any]:
        records = list(self._paper_replications.get(paper_id) or [])
        return {"paper_id": paper_id, "records": records, "count": len(records)}

    async def evaluate_paper(self, paper: Dict[str, Any], paper_id: Optional[str] = None) -> Dict[str, Any]:
        def _clamp01(v: float) -> float:
            return max(0.0, min(1.0, float(v)))

        claimed = paper.get("claimed_results") or {}
        exp_count = int(paper.get("exp_count") or 0)
        citations = paper.get("citations") or []
        evidence_map = paper.get("evidence_map") or {}
        observation_refs = paper.get("observation_refs") or []

        score_norm = float(claimed.get("score_norm", 0.0) or 0.0)
        raw_score = float(claimed.get("raw_score", 0.0) or 0.0)
        metric_name = str(claimed.get("metric_name") or "")

        # Compatibility fields: map benchmark quality to graph-like quality.
        graph_score = _clamp01(score_norm)
        precision = graph_score
        recall = graph_score
        f1 = graph_score

        citation_strength = min(1.0, len(citations) / 4.0)
        obs_strength = min(1.0, len(observation_refs) / 3.0)
        evidence_score = _clamp01(0.6 * citation_strength + 0.4 * obs_strength)
        evidence_coverage_score = _clamp01(min(1.0, len(evidence_map) / 2.0))

        paper_id = paper_id or paper.get("paper_id")
        replication_records = list(self._paper_replications.get(paper_id) or []) if paper_id else []
        replication_verified = bool(replication_records)
        replication_support_scores = [
            float(((rec or {}).get("support") or {}).get("support_ratio", 0.0) or 0.0) for rec in replication_records
        ]
        replication_support_score = (
            sum(replication_support_scores) / len(replication_support_scores) if replication_support_scores else 0.0
        )
        replication_empirical_ok = replication_verified and replication_support_score >= 0.5
        replication_structural_ok = graph_score >= self._paper_gate_graph
        replication_ok = replication_empirical_ok if replication_verified else replication_structural_ok

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
        replication_readiness = (
            _clamp01(replication_support_score / max(1e-6, self._paper_gate_replication))
            if replication_verified
            else (0.25 if replication_structural_ok else 0.0)
        )
        readiness_components = [graph_readiness, evidence_readiness]
        if self._paper_require_replication:
            readiness_components.append(replication_readiness)
        readiness_score = sum(readiness_components) / max(1, len(readiness_components))

        rep_signal = replication_support_score if replication_verified else (0.2 if replication_structural_ok else 0.0)
        rep_signal = _clamp01(rep_signal)

        cost_penalty = self.lambda_cost * (exp_count / max(1, self.budget))
        quality_score = (
            self._fit_w_graph * graph_score
            + self._fit_w_evidence * evidence_score
            + self._fit_w_replication * rep_signal
        )
        fitness = quality_score + (self._fit_w_readiness * readiness_score) - cost_penalty

        if replication_verified and not replication_empirical_ok:
            fitness -= self._fit_failed_replication_penalty
        elif not replication_verified:
            fitness -= self._fit_unverified_penalty
        fitness = max(-1.0, min(1.5, fitness))

        return {
            "task_name": (self._current_task or {}).get("task_name"),
            "metric_name": metric_name,
            "raw_score": raw_score,
            "score_norm": score_norm,
            "graph_score": graph_score,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "evidence_score": evidence_score,
            "evidence_coverage_score": evidence_coverage_score,
            "evidence_edges_supported": len(evidence_map),
            "evidence_edges_total": max(1, len(evidence_map)),
            "evidence_details": {
                "citations_count": len(citations),
                "observation_refs_count": len(observation_refs),
            },
            "replication_ok": replication_ok,
            "replication_structural_ok": replication_structural_ok,
            "replication_verified": replication_verified,
            "replication_empirical_ok": replication_empirical_ok,
            "replication_support_score": replication_support_score,
            "replication_main_support_score": replication_support_score,
            "replication_holdout_support_score": 0.0,
            "replication_holdout_required": False,
            "replication_holdout_available": False,
            "replication_holdout_pass": True,
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
                "name": "airs_v1_gate_plus_smooth_score",
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

    async def score_hypothesis(self, hypothesis: List[str]) -> Dict[str, float]:
        # AIRS world does not use causal-graph hypothesis scoring; keep compatibility.
        h_len = len(hypothesis or [])
        p = max(0.0, min(1.0, h_len / 5.0))
        return {"precision": p, "recall": p, "f1": p}

    async def evaluate_hypothesis(self, hypothesis: List[str], exp_count: int) -> Dict[str, Any]:
        score = await self.score_hypothesis(hypothesis)
        f1 = score.get("f1", 0.0)
        cost_penalty = self.lambda_cost * (int(exp_count or 0) / max(1, self.budget))
        return {
            "f1": f1,
            "precision": score.get("precision", 0.0),
            "recall": score.get("recall", 0.0),
            "exp_count": exp_count,
            "budget": self.budget,
            "fitness": float(f1 - cost_penalty),
        }
