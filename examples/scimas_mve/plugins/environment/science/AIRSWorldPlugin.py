import asyncio
import csv
import hashlib
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

try:
    from .code_runtime import DevEvaluator, DockerExecutor, SandboxExecutor, WorkspaceManager
except Exception:  # pragma: no cover - fallback for direct module execution
    from examples.scimas_mve.plugins.environment.science.code_runtime import (  # type: ignore
        DevEvaluator,
        DockerExecutor,
        SandboxExecutor,
        WorkspaceManager,
    )

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
        workspace_root: str = "logs/runs/airs_workspace",
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
        task_lease_ttl_experiment: Optional[int] = None,
        experiment_max_active_leases: int = 2,
        hard_budget_gate: bool = True,
        max_experiment_runs: Optional[int] = None,
        max_episode_steps: int = 0,
        max_episode_cost: float = 0.0,
        experiment_min_completed_read: int = 1,
        experiment_min_completed_profile_data: int = 1,
        experiment_min_completed_retrieve_literature: int = 1,
        method_card_topk: int = 3,
        profile_data_timeout_s: int = 45,
        profile_sample_rows: int = 2000,
        profile_max_columns: int = 64,
        prepare_data_timeout_s: int = 240,
        profile_min_completed_prepare_data: int = 1,
        profile_use_cache_direct: bool = True,
        profile_timeseries_max_rows: int = 1200,
        code_agent_enable: bool = True,
        code_run_timeout_s: int = 180,
        code_cpu_limit_s: int = 120,
        code_memory_mb: int = 2048,
        code_forbid_network: bool = True,
        code_executor_backend: str = "docker",
        code_executor_fallback_soft: bool = True,
        code_docker_image: str = "python:3.11-slim",
        code_docker_bin: str = "docker",
        code_docker_cpus: float = 2.0,
        code_docker_gpus: str = "",
        code_docker_keepalive: bool = False,
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
        default_exp_ttl = max(6, self._task_lease_ttl * 5)
        self._task_lease_ttl_experiment = int(
            max(
                self._task_lease_ttl,
                int(
                    os.getenv(
                        "SCIMAS_TASK_LEASE_TTL_EXPERIMENT",
                        str(task_lease_ttl_experiment if task_lease_ttl_experiment is not None else default_exp_ttl),
                    )
                ),
            )
        )
        self._experiment_max_active_leases = int(
            max(1, int(os.getenv("SCIMAS_EXPERIMENT_MAX_ACTIVE_LEASES", str(experiment_max_active_leases))))
        )
        self._hard_budget_gate = bool(hard_budget_gate)
        try:
            configured_max_runs = int(max_experiment_runs) if max_experiment_runs is not None else self.budget
        except Exception:
            configured_max_runs = self.budget
        self._max_experiment_runs = int(max(1, configured_max_runs))
        self._max_episode_steps = int(max(0, int(max_episode_steps)))
        self._max_episode_cost = float(max(0.0, float(max_episode_cost)))
        self._code_agent_enable = bool(code_agent_enable)
        self._code_run_timeout_s = int(max(5, int(code_run_timeout_s)))
        self._code_cpu_limit_s = int(max(1, int(code_cpu_limit_s)))
        self._code_memory_mb = int(max(256, int(code_memory_mb)))
        self._code_forbid_network = bool(code_forbid_network)
        backend = str(code_executor_backend or "docker").strip().lower()
        self._code_executor_backend = backend if backend in {"docker", "soft"} else "docker"
        self._code_executor_fallback_soft = bool(code_executor_fallback_soft)
        self._code_docker_image = str(code_docker_image or "").strip()
        self._code_docker_bin = str(code_docker_bin or "docker").strip()
        self._code_docker_cpus = max(0.1, float(code_docker_cpus))
        self._code_docker_gpus = str(code_docker_gpus or "").strip()
        self._code_docker_keepalive = bool(code_docker_keepalive)
        self._log_mode = (os.getenv("SCIMAS_LOG_MODE", "compact") or "compact").strip().lower()
        self._taskboard_log_events = self._resolve_taskboard_log_events()
        self._runtime_monitor = os.getenv(
            "SCIMAS_RUNTIME_MONITOR",
            "1" if self._log_mode in {"compact", "minimal"} else "0",
        ).lower() not in {"0", "false", "no"}
        self._runtime_heartbeat_s = float(max(2.0, float(os.getenv("SCIMAS_RUNTIME_HEARTBEAT_S", "10"))))

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
        self._prepared_cache_persist_enable = os.getenv("SCIMAS_AIRS_PREPARED_CACHE_ENABLE", "1").lower() not in {
            "0",
            "false",
            "no",
        }

        self._write_min_completed_read = int(os.getenv("SCIMAS_WRITE_MIN_COMPLETED_READ", "1"))
        self._write_min_completed_experiment = int(os.getenv("SCIMAS_WRITE_MIN_COMPLETED_EXPERIMENT", "1"))
        self._write_min_completed_hypothesize = int(os.getenv("SCIMAS_WRITE_MIN_COMPLETED_HYPOTHESIZE", "1"))
        self._experiment_min_completed_read = int(
            max(0, int(os.getenv("SCIMAS_EXPERIMENT_MIN_COMPLETED_READ", str(experiment_min_completed_read))))
        )
        self._experiment_min_completed_profile_data = int(max(0, int(experiment_min_completed_profile_data)))
        self._experiment_min_completed_retrieve_literature = int(max(0, int(experiment_min_completed_retrieve_literature)))
        self._method_card_topk = int(max(1, int(method_card_topk)))
        self._profile_data_timeout_s = int(max(5, int(os.getenv("SCIMAS_PROFILE_DATA_TIMEOUT_S", str(profile_data_timeout_s)))))
        self._profile_sample_rows = int(max(64, int(os.getenv("SCIMAS_PROFILE_SAMPLE_ROWS", str(profile_sample_rows)))))
        self._profile_max_columns = int(max(8, int(os.getenv("SCIMAS_PROFILE_MAX_COLUMNS", str(profile_max_columns)))))
        self._prepare_data_timeout_s = int(max(30, int(os.getenv("SCIMAS_PREPARE_DATA_TIMEOUT_S", str(prepare_data_timeout_s)))))
        self._profile_min_completed_prepare_data = int(
            max(0, int(os.getenv("SCIMAS_PROFILE_MIN_COMPLETED_PREPARE_DATA", str(profile_min_completed_prepare_data))))
        )
        self._profile_use_cache_direct = os.getenv(
            "SCIMAS_PROFILE_USE_CACHE_DIRECT",
            "1" if bool(profile_use_cache_direct) else "0",
        ).lower() not in {"0", "false", "no"}
        self._profile_timeseries_max_rows = int(
            max(128, int(os.getenv("SCIMAS_PROFILE_TIMESERIES_MAX_ROWS", str(profile_timeseries_max_rows))))
        )
        self._profile_data_lock = asyncio.Lock()
        self._prepare_data_lock = asyncio.Lock()

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
        prepared_cache_dir_cfg = str(
            os.getenv("SCIMAS_AIRS_PREPARED_CACHE_DIR", "data/airs_prepared_cache") or "data/airs_prepared_cache"
        ).strip()
        self._prepared_cache_persist_root = self._resolve_path(
            prepared_cache_dir_cfg,
            project_root=self._project_root,
        )
        self._all_agent_ids = self._load_agent_ids()

        self._task_log_path = os.path.join(self._project_root, "logs", "app", "environment", "taskboard.jsonl")
        self._eval_failure_log_path = os.path.join(self._project_root, "logs", "app", "environment", "eval_failures.jsonl")
        self._eval_cache_log_path = os.path.join(self._project_root, "logs", "app", "environment", "eval_cache.jsonl")
        self._eval_cache_enable = os.getenv("SCIMAS_EVAL_CACHE_ENABLE", "1").lower() not in {"0", "false", "no"}
        self._eval_cache: Dict[str, Dict[str, Any]] = {}

        self._tasks_catalog: List[Dict[str, Any]] = []
        self._current_task: Optional[Dict[str, Any]] = None
        self._current_cards: List[Dict[str, Any]] = []
        self._current_workspace: Optional[str] = None
        self._prepared_data_cache_dir: Optional[str] = None

        self._task_board: Dict[str, Dict[str, Any]] = {}
        self._active_workers: set[str] = set()
        self._taskboard_event_counts: Dict[str, int] = {"create": 0, "claim": 0, "complete": 0, "release": 0}
        self._taskboard_release_reason_counts: Dict[str, int] = {}
        self._task_priority = {
            "read": 1,
            "prepare_data": 2,
            "profile_data": 2,
            "retrieve_literature": 2,
            "hypothesize": 3,
            "experiment": 4,
            "write": 5,
            "review": 6,
            "replicate": 7,
            "verify_strength": 8,
            "verify_issue": 9,
        }

        self._agent_views: Dict[str, List[int]] = {}
        self._agent_hint_accuracy: Dict[str, float] = {}
        self._score_cache: List[Dict[str, Any]] = []
        self._paper_bank: Dict[str, Dict[str, Any]] = {}
        self._paper_replications: Dict[str, List[Dict[str, Any]]] = {}
        self._data_card_cache: Optional[Dict[str, Any]] = None
        self._method_card_cache: Optional[Dict[str, Any]] = None

        self._load_tasks_catalog()
        self._init_taskboard()

    async def init(self) -> None:
        os.makedirs(os.path.dirname(self._task_log_path), exist_ok=True)
        os.makedirs(os.path.dirname(self._eval_failure_log_path), exist_ok=True)
        os.makedirs(os.path.dirname(self._eval_cache_log_path), exist_ok=True)
        os.makedirs(self.workspace_root, exist_ok=True)
        if self._prepared_cache_persist_enable:
            os.makedirs(self._prepared_cache_persist_root, exist_ok=True)
        self._load_eval_cache_sync()
        logger.info(
            f"AIRSWorldPlugin initialized: tasks_root={self.tasks_root}, shared_data_dir={self.shared_data_dir}, "
            f"catalog_size={len(self._tasks_catalog)}"
        )

    async def save_to_db(self) -> None:
        """No-op persistence hook for framework compatibility.

        AIRS world state is persisted via JSONL artifacts under logs/workspace, so
        environment-level DB persistence is intentionally skipped here.
        """
        return None

    async def load_from_db(self) -> None:
        """No-op load hook; episode state is reconstructed from configs/logs."""
        return None

    def _monitor(self, message: str) -> None:
        if not self._runtime_monitor:
            return
        level = str(os.getenv("SCIMAS_MONITOR_LOG_LEVEL", "INFO")).strip().upper()
        if level == "DEBUG":
            logger.debug(f"[MONITOR] {message}")
        elif level == "WARNING":
            logger.warning(f"[MONITOR] {message}")
        elif level == "ERROR":
            logger.error(f"[MONITOR] {message}")
        else:
            logger.info(f"[MONITOR] {message}")

    def _append_jsonl_sync(self, path: str, record: Dict[str, Any]) -> None:
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.warning(f"Failed to append jsonl {path}: {e}")

    def _classify_eval_error(self, text: str) -> str:
        raw = str(text or "").lower()
        if "module not found" in raw or "modulenotfounderror" in raw:
            return "missing_dependency"
        if "filenotfounderror" in raw or "no such file or directory" in raw:
            return "missing_file"
        if "syntaxerror" in raw:
            return "syntax_error"
        if "permission denied" in raw:
            return "permission_error"
        if "killed" in raw or "oom" in raw:
            return "resource_killed"
        if "submission" in raw and ("format" in raw or "schema" in raw):
            return "submission_format_error"
        return "runtime_error"

    def _load_eval_cache_sync(self) -> None:
        self._eval_cache = {}
        if not self._eval_cache_enable:
            return
        path = Path(self._eval_cache_log_path)
        if not path.exists():
            return
        try:
            with path.open("r", encoding="utf-8") as f:
                for raw in f:
                    line = raw.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    key = str(obj.get("cache_key") or "").strip()
                    payload = obj.get("payload")
                    if key and isinstance(payload, dict):
                        self._eval_cache[key] = payload
        except Exception as e:
            logger.warning(f"Failed to load eval cache {path}: {e}")

    def _append_eval_cache_sync(self, cache_key: str, payload: Dict[str, Any]) -> None:
        if not self._eval_cache_enable:
            return
        safe_payload = dict(payload or {})
        self._eval_cache[str(cache_key)] = safe_payload
        self._append_jsonl_sync(
            self._eval_cache_log_path,
            {
                "ts": datetime.utcnow().isoformat() + "Z",
                "episode_id": int(self._episode_id),
                "cache_key": str(cache_key),
                "payload": safe_payload,
            },
        )

    def _file_sha256(self, path: str) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            while True:
                chunk = f.read(1024 * 1024)
                if not chunk:
                    break
                h.update(chunk)
        return h.hexdigest()

    def _eval_cache_key(self, task: Dict[str, Any], submission_hash: str) -> str:
        task_name = str((task or {}).get("task_name") or "")
        return f"{task_name}|{submission_hash}"

    def _expected_submission_rows(self) -> Optional[int]:
        try:
            from datasets import load_from_disk
        except Exception:
            return None
        if not self._prepared_data_cache_dir:
            return None
        test_path = Path(self._prepared_data_cache_dir) / "test"
        if not test_path.exists():
            return None
        try:
            dset = load_from_disk(str(test_path))
            return int(len(dset))
        except Exception:
            return None

    def _required_submission_columns(self, task: Dict[str, Any]) -> List[str]:
        info = (task or {}).get("logging_info") or {}
        raw = info.get("scoring_column")
        cols: List[str] = []
        if isinstance(raw, list):
            for item in raw:
                name = str(item or "").strip()
                if name and name not in cols:
                    cols.append(name)
        else:
            name = str(raw or "").strip()
            if name:
                cols.append(name)
        if not cols:
            cols = ["prediction"]
        return cols

    def _preflight_submission_schema(self, task: Dict[str, Any], submission_path: str) -> Dict[str, Any]:
        if not submission_path or not os.path.exists(submission_path):
            return {
                "ok": False,
                "error_code": "submission_not_found",
                "message": f"submission not found: {submission_path}",
            }
        required_cols = self._required_submission_columns(task)
        rows = 0
        null_rows = 0
        try:
            with open(submission_path, "r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                fieldnames = list(reader.fieldnames or [])
                missing = [c for c in required_cols if c not in fieldnames]
                if missing:
                    return {
                        "ok": False,
                        "error_code": "missing_columns",
                        "message": f"submission missing required columns: {missing}",
                        "required_columns": required_cols,
                        "present_columns": fieldnames,
                    }
                for row in reader:
                    rows += 1
                    for col in required_cols:
                        val = row.get(col)
                        if val is None or str(val).strip() == "":
                            null_rows += 1
                            break
            if rows <= 0:
                return {
                    "ok": False,
                    "error_code": "empty_submission",
                    "message": "submission contains no data rows",
                    "required_columns": required_cols,
                }
            expected_rows = self._expected_submission_rows()
            if isinstance(expected_rows, int) and expected_rows > 0 and rows != expected_rows:
                return {
                    "ok": False,
                    "error_code": "row_count_mismatch",
                    "message": f"submission row count mismatch: got={rows}, expected={expected_rows}",
                    "required_columns": required_cols,
                    "row_count": rows,
                    "expected_rows": expected_rows,
                }
            if null_rows > 0:
                return {
                    "ok": False,
                    "error_code": "null_values_in_required_columns",
                    "message": f"submission has null/empty values in required columns: {null_rows} rows",
                    "required_columns": required_cols,
                    "row_count": rows,
                    "null_rows": null_rows,
                }
            return {
                "ok": True,
                "required_columns": required_cols,
                "row_count": rows,
                "null_rows": 0,
                "expected_rows": expected_rows,
            }
        except Exception as e:
            return {
                "ok": False,
                "error_code": "submission_parse_error",
                "message": f"failed to parse submission: {e}",
                "required_columns": required_cols,
            }

    def _evaluate_dependency_precheck(self, task: Dict[str, Any]) -> Dict[str, Any]:
        metric = str(((task or {}).get("logging_info") or {}).get("metric") or "")
        category = str(((task or {}).get("logging_info") or {}).get("category") or "")
        task_name = str((task or {}).get("task_name") or "")
        imports = ["import pandas", "import numpy"]
        if self._is_timeseries_submission_task(task=task, metric=metric, category=category) or "timeseries" in task_name.lower():
            imports.append("import sktime")
        cmd = [self._python_cmd, "-c", "; ".join(imports)]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=12)
        except Exception as e:
            return {"ok": False, "error_code": "precheck_runtime_error", "message": str(e), "cmd": cmd}
        if int(proc.returncode) != 0:
            err = str(proc.stderr or proc.stdout or "").strip()
            return {
                "ok": False,
                "error_code": "missing_dependency",
                "message": err[-600:],
                "cmd": cmd,
                "returncode": int(proc.returncode),
            }
        return {"ok": True, "cmd": cmd}

    def _resolve_path(self, path: str, project_root: str) -> str:
        p = Path(path)
        if p.is_absolute():
            return str(p)
        # Keep relative paths anchored to the MVE project root to avoid
        # accidentally writing artifacts outside examples/scimas_mve.
        return str((Path(project_root) / p).resolve())

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
            task_item = {
                "task_name": task_dir.name,
                "task_path": str(task_dir),
                "metadata": metadata,
                "logging_info": logging_info,
                "dataset_rel": dataset_rel,
                "metric": logging_info.get("metric"),
                "metric_lower_is_better": bool(metadata.get("metric_lower_is_better", False)),
                "project_description_path": str(project_desc) if project_desc.exists() else None,
            }
            support_hint = self._task_solver_support_hint(task_item)
            task_item["solver_supported"] = bool(support_hint.get("supported", False))
            task_item["solver_support_reason"] = str(support_hint.get("reason") or "")
            catalog.append(task_item)

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

    def _task_solver_support_hint(self, task: Dict[str, Any]) -> Dict[str, Any]:
        info = task.get("logging_info") or {}
        metric = str(info.get("metric") or "").lower()
        category = str(info.get("category") or "").lower()
        unsupported_metric_tokens = ("pass@", "mrr", "rouge", "exactmatch")
        if any(tok in metric for tok in unsupported_metric_tokens):
            return {"supported": False, "reason": f"metric_not_supported:{metric}"}
        if "code" in category or "question answering" in category:
            return {"supported": False, "reason": f"category_not_supported:{category}"}
        return {"supported": True, "reason": "ok"}

    def _resource_usage(self, current_tick: Optional[int] = None) -> Dict[str, Any]:
        logical_tick = self._normalize_tick(current_tick)
        experiment_runs = int(len(self._score_cache))
        cost_used = float(sum(float((r or {}).get("cost", 0.0) or 0.0) for r in self._score_cache))
        return {
            "logical_tick": logical_tick,
            "experiment_runs": experiment_runs,
            "cost_used": cost_used,
        }

    def _hard_stop_reason(self, current_tick: Optional[int] = None) -> Optional[str]:
        if not self._hard_budget_gate:
            return None
        usage = self._resource_usage(current_tick=current_tick)
        logical_tick = int(usage.get("logical_tick", 0) or 0)
        experiment_runs = int(usage.get("experiment_runs", 0) or 0)
        cost_used = float(usage.get("cost_used", 0.0) or 0.0)

        if self._max_episode_steps > 0 and logical_tick >= self._max_episode_steps:
            return f"max_steps_exceeded:{logical_tick}>={self._max_episode_steps}"
        if self._max_experiment_runs > 0 and experiment_runs >= self._max_experiment_runs:
            return f"max_experiments_exceeded:{experiment_runs}>={self._max_experiment_runs}"
        if self._max_episode_cost > 0.0 and cost_used >= self._max_episode_cost:
            return f"max_cost_exceeded:{cost_used:.4f}>={self._max_episode_cost:.4f}"
        return None

    def _pick_task(self) -> Dict[str, Any]:
        if self.task_sampling in {"fixed_supported_list", "supported_list"}:
            cands = [t for t in self._tasks_catalog if bool(t.get("solver_supported", False))]
            if self.fixed_tasks:
                name_set = set(self.fixed_tasks)
                cands = [t for t in cands if t.get("task_name") in name_set]
            if not cands:
                raise RuntimeError(
                    "task_sampling=fixed_supported_list but no solver-supported task is available. "
                    "Check tasks_root/raw datasets/fixed_tasks."
                )
            return self._rng.choice(cands)
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
        self._taskboard_release_reason_counts = {}

        self._current_task = self._pick_task()
        self._current_cards = self._build_cards_for_task(self._current_task)
        self._agent_views = {}
        self._agent_hint_accuracy = {}
        self._score_cache = []
        self._paper_bank = {}
        self._paper_replications = {}
        self._data_card_cache = None
        self._method_card_cache = None
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
        if event == "release":
            reason = str((meta or {}).get("reason") or "unknown")[:160]
            self._taskboard_release_reason_counts[reason] = int(self._taskboard_release_reason_counts.get(reason, 0) or 0) + 1
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
            "lease_ttl": self._lease_ttl_for_task_type(task_type),
            "last_heartbeat_tick": None,
            "heartbeat_count": 0,
            "started_tick": None,
            "last_phase": "",
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

    def _lease_ttl_for_task_type(self, task_type: str) -> int:
        task_type = str(task_type or "")
        if task_type == "experiment":
            return int(max(self._task_lease_ttl, self._task_lease_ttl_experiment))
        return int(self._task_lease_ttl)

    def _touch_task_lease(self, task: Dict[str, Any], now_tick: int, *, phase: Optional[str] = None) -> None:
        now_tick = int(now_tick)
        task["claimed_tick"] = now_tick
        task["last_heartbeat_tick"] = now_tick
        task["heartbeat_count"] = int(task.get("heartbeat_count", 0) or 0) + 1
        if phase:
            task["last_phase"] = str(phase)[:80]

    async def _expire_task_leases(self, now_tick: Optional[int] = None) -> int:
        now_tick = self._normalize_tick(now_tick)
        expired = 0
        for task in self._task_board.values():
            if task.get("status") not in {"claimed", "running"}:
                continue
            claimed_tick = int(task.get("claimed_tick", now_tick) or now_tick)
            ttl = int(task.get("lease_ttl", self._lease_ttl_for_task_type(str(task.get("task_type") or ""))) or self._task_lease_ttl)
            if (now_tick - claimed_tick) < ttl:
                continue
            holder = task.get("claimed_by")
            task["status"] = "open"
            task["claimed_by"] = None
            task["claimed_tick"] = None
            task["last_phase"] = "expired"
            self._append_taskboard_log(
                "release",
                task,
                {
                    "agent_id": holder,
                    "reason": "lease_expired",
                    "expired_at_tick": now_tick,
                    "ttl": ttl,
                },
            )
            expired += 1
        return expired

    def _bootstrap_taskboard(self) -> None:
        n_agents = self._estimate_agent_count()
        n_workers = len(self._active_workers) if self._active_workers else min(n_agents, self._active_worker_count)
        n_workers = max(1, n_workers)
        read_count = max(2, min(6, n_workers // 2 + 1))
        prepare_count = 1
        profile_count = max(1, min(3, n_workers // 3 + 1))
        method_count = max(1, min(3, n_workers // 3 + 1))
        experiment_count = max(2, min(self.budget, n_workers))
        hypothesize_count = max(2, min(5, n_workers // 2 + 1))
        write_count = max(1, min(3, n_workers // 3 + 1))

        for _ in range(read_count):
            self._create_task_internal("read", payload={"topic": "task_requirements"})
        for _ in range(prepare_count):
            self._create_task_internal("prepare_data", payload={})
        for _ in range(profile_count):
            self._create_task_internal("profile_data", payload={})
        for _ in range(method_count):
            self._create_task_internal("retrieve_literature", payload={"topic": "task_baselines"})
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
        if task_type == "experiment":
            completed = self._completed_task_success_counts()
            if completed.get("read", 0) < self._experiment_min_completed_read:
                blockers.append(f"need_completed_read>={self._experiment_min_completed_read}")
            if completed.get("profile_data", 0) < self._experiment_min_completed_profile_data:
                blockers.append(f"need_completed_profile_data>={self._experiment_min_completed_profile_data}")
            if completed.get("retrieve_literature", 0) < self._experiment_min_completed_retrieve_literature:
                blockers.append(f"need_completed_retrieve_literature>={self._experiment_min_completed_retrieve_literature}")
        if task_type == "profile_data":
            completed = self._completed_task_success_counts()
            if completed.get("prepare_data", 0) < self._profile_min_completed_prepare_data:
                blockers.append(f"need_completed_prepare_data>={self._profile_min_completed_prepare_data}")
        if task_type == "write":
            completed = self._completed_task_success_counts()
            if completed.get("read", 0) < self._write_min_completed_read:
                blockers.append(f"need_completed_read>={self._write_min_completed_read}")
            if completed.get("experiment", 0) < self._write_min_completed_experiment:
                blockers.append(f"need_completed_experiment>={self._write_min_completed_experiment}")
            if completed.get("hypothesize", 0) < self._write_min_completed_hypothesize:
                blockers.append(f"need_completed_hypothesize>={self._write_min_completed_hypothesize}")
        return blockers

    def _completed_task_success_counts(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for task in self._task_board.values():
            if task.get("status") != "completed":
                continue
            result = task.get("result") or {}
            action_data = (result.get("action_data") or {}) if isinstance(result, dict) else {}
            ok = bool(action_data.get("ok", True))
            if not ok:
                continue
            t = str(task.get("task_type"))
            counts[t] = counts.get(t, 0) + 1
        return counts

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
        stop_reason = self._hard_stop_reason()
        if stop_reason and str(task_type or "") == "experiment":
            return {"ok": False, "reason": f"hard_stop:{stop_reason}"}
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
        summary = {"open": 0, "claimed": 0, "running": 0, "completed": 0}
        for task in self._task_board.values():
            st = task.get("status", "open")
            summary[st] = summary.get(st, 0) + 1
        return {"tasks": tasks, "summary": summary}

    async def task_get(self, task_id: str, current_tick: Optional[int] = None) -> Dict[str, Any]:
        await self._expire_task_leases(now_tick=current_tick)
        task = self._task_board.get(str(task_id))
        if not task:
            return {"ok": False, "reason": "task_not_found"}
        return {"ok": True, "task": self._decorate_task_view(task)}

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
            if t.get("status") in {"claimed", "running"} and t.get("claimed_by") == agent_id
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
        stop_reason = self._hard_stop_reason(current_tick=now_tick)
        if stop_reason:
            task_type_for_stop = str(task.get("task_type") or "")
            allow_after_stop = {"write", "review", "replicate", "verify_strength", "verify_issue"}
            if task_type_for_stop not in allow_after_stop:
                return {
                    "ok": False,
                    "reason": f"hard_stop:{stop_reason}",
                    "task": self._decorate_task_view(task),
                }

        task_type = str(task.get("task_type") or "")
        payload = task.get("payload") or {}
        if task_type != "experiment":
            has_exp_focus = any(
                str(t.get("task_type") or "") == "experiment"
                for t in my_claimed
            )
            if has_exp_focus:
                return {
                    "ok": False,
                    "reason": "focus_locked_by_experiment",
                    "task": self._decorate_task_view(task),
                }
        if task_type == "experiment":
            active_experiment_leases = sum(
                1
                for t in self._task_board.values()
                if str(t.get("task_type") or "") == "experiment" and t.get("status") in {"claimed", "running"}
            )
            if active_experiment_leases >= self._experiment_max_active_leases:
                return {
                    "ok": False,
                    "reason": "experiment_capacity_reached",
                    "active_experiment_leases": active_experiment_leases,
                    "experiment_max_active_leases": self._experiment_max_active_leases,
                    "task": self._decorate_task_view(task),
                }
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
        task["lease_ttl"] = self._lease_ttl_for_task_type(task_type)
        task["started_tick"] = None
        task["heartbeat_count"] = 0
        task["last_phase"] = "claimed"
        self._touch_task_lease(task, now_tick, phase="claimed")
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
        if task.get("status") not in {"claimed", "running"}:
            return {"ok": False, "reason": f"task_not_claimed:{task.get('status')}", "task": task}
        if task.get("claimed_by") and task.get("claimed_by") != agent_id:
            return {"ok": False, "reason": "task_claimed_by_other", "task": task}
        task["status"] = "open"
        task["claimed_by"] = None
        task["claimed_tick"] = None
        task["last_phase"] = "released"
        self._append_taskboard_log("release", task, {"agent_id": agent_id, "reason": reason or ""})
        return {"ok": True, "task": task}

    async def task_start(
        self,
        task_id: str,
        agent_id: str,
        current_tick: Optional[int] = None,
        phase: Optional[str] = None,
    ) -> Dict[str, Any]:
        now_tick = self._normalize_tick(current_tick)
        await self._expire_task_leases(now_tick=now_tick)
        task = self._task_board.get(task_id)
        if not task:
            return {"ok": False, "reason": "task_not_found"}
        if task.get("claimed_by") and task.get("claimed_by") != agent_id:
            return {"ok": False, "reason": "task_claimed_by_other", "task": self._decorate_task_view(task)}
        if task.get("status") not in {"claimed", "running"}:
            return {"ok": False, "reason": f"task_not_claimed:{task.get('status')}", "task": self._decorate_task_view(task)}
        if task.get("started_tick") is None:
            task["started_tick"] = now_tick
        task["status"] = "running"
        self._touch_task_lease(task, now_tick, phase=phase or "running")
        return {"ok": True, "task": self._decorate_task_view(task)}

    async def task_heartbeat(
        self,
        task_id: str,
        agent_id: str,
        current_tick: Optional[int] = None,
        phase: Optional[str] = None,
    ) -> Dict[str, Any]:
        now_tick = self._normalize_tick(current_tick)
        await self._expire_task_leases(now_tick=now_tick)
        task = self._task_board.get(task_id)
        if not task:
            return {"ok": False, "reason": "task_not_found"}
        if task.get("claimed_by") and task.get("claimed_by") != agent_id:
            return {"ok": False, "reason": "task_claimed_by_other", "task": self._decorate_task_view(task)}
        if task.get("status") not in {"claimed", "running"}:
            return {"ok": False, "reason": f"task_not_claimed:{task.get('status')}", "task": self._decorate_task_view(task)}
        if task.get("started_tick") is None:
            task["started_tick"] = now_tick
        task["status"] = "running"
        self._touch_task_lease(task, now_tick, phase=phase or "heartbeat")
        return {"ok": True, "task": self._decorate_task_view(task)}

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
        if task.get("status") not in {"claimed", "running"}:
            return {"ok": False, "reason": f"task_not_claimed:{task.get('status')}", "task": task}
        if task.get("claimed_by") and task.get("claimed_by") != agent_id:
            return {"ok": False, "reason": "task_claimed_by_other", "task": task}
        task["status"] = "completed"
        task["completed_by"] = agent_id
        task["claimed_tick"] = None
        task["last_phase"] = "completed"
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
        usage = self._resource_usage(current_tick=current_tick)
        stop_reason = self._hard_stop_reason(current_tick=current_tick)
        completed = self._completed_task_counts()
        summary = {"open": 0, "claimed": 0, "running": 0, "completed": 0}
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
            "task_lease_ttl_experiment": self._task_lease_ttl_experiment,
            "experiment_max_active_leases": self._experiment_max_active_leases,
            "forbid_self_review": self._forbid_self_review,
            "forbid_self_replicate": self._forbid_self_replicate,
            "solver_enabled": self._solver_enabled,
            "solver_prepare_once": self._solver_prepare_once,
            "code_agent_enable": self._code_agent_enable,
            "code_run_timeout_s": self._code_run_timeout_s,
            "code_cpu_limit_s": self._code_cpu_limit_s,
            "code_memory_mb": self._code_memory_mb,
            "code_forbid_network": self._code_forbid_network,
            "code_executor_backend": self._code_executor_backend,
            "code_executor_fallback_soft": self._code_executor_fallback_soft,
            "code_docker_image": self._code_docker_image,
            "code_docker_bin": self._code_docker_bin,
            "code_docker_cpus": self._code_docker_cpus,
            "code_docker_gpus": self._code_docker_gpus,
            "code_docker_keepalive": self._code_docker_keepalive,
            "task_sampling": self.task_sampling,
            "solver_supported_task_count": int(sum(1 for t in self._tasks_catalog if bool(t.get("solver_supported", False)))),
            "taskboard_event_counts": dict(self._taskboard_event_counts),
            "hard_budget_gate": self._hard_budget_gate,
            "max_experiment_runs": self._max_experiment_runs,
            "max_episode_steps": self._max_episode_steps,
            "max_episode_cost": self._max_episode_cost,
            "data_card_ready": bool(self._data_card_cache),
            "method_card_ready": bool(self._method_card_cache),
            "completed_task_counts": completed,
            "pre_experiment_requirements": {
                "prepare_data": self._profile_min_completed_prepare_data,
                "read": self._experiment_min_completed_read,
                "profile_data": self._experiment_min_completed_profile_data,
                "retrieve_literature": self._experiment_min_completed_retrieve_literature,
            },
            "resource_usage": usage,
            "hard_stop_reason": stop_reason,
            # Kept for backward compatibility with existing planner interfaces.
            "variables": [],
            "target": "score",
        }

    async def get_shared_artifacts(self, include_cards: bool = True, max_refs: int = 8) -> Dict[str, Any]:
        """Expose episode-level shared prerequisites for agent-side hydration."""
        task = self._current_task
        if not task:
            await self.reset_episode()
            task = self._current_task
        if not task:
            return {"ok": False, "reason": "no_active_task"}

        refs: List[Dict[str, Any]] = []
        for card in (self._current_cards or [])[: max(1, int(max_refs))]:
            refs.append(
                {
                    "citation_id": card.get("citation_id"),
                    "title": card.get("title"),
                    "kind": card.get("kind"),
                    "snippet": self._truncate(card.get("text"), 180),
                }
            )

        data_card = dict(self._data_card_cache) if isinstance(self._data_card_cache, dict) else None
        method_card = dict(self._method_card_cache) if isinstance(self._method_card_cache, dict) else None
        if not bool(include_cards):
            data_card = None
            method_card = None

        return {
            "ok": True,
            "episode_id": self._episode_id,
            "task_name": task.get("task_name"),
            "prepare_data_ready": bool(self._prepared_cache_ready()),
            "data_card_ready": bool(self._data_card_cache),
            "method_card_ready": bool(self._method_card_cache),
            "data_card": data_card,
            "method_card": method_card,
            "notes_template": {
                "topic": "shared_task_context",
                "source": "shared_artifacts",
                "hints": [f"[{r.get('citation_id')}] {r.get('title')}" for r in refs if r.get("citation_id")],
                "cards": refs,
            },
        }

    async def get_taskboard_metrics(self) -> Dict[str, Any]:
        summary = {"open": 0, "claimed": 0, "running": 0, "completed": 0}
        for task in self._task_board.values():
            st = str(task.get("status") or "open")
            summary[st] = int(summary.get(st, 0) or 0) + 1
        active_experiment_leases = int(
            sum(
                1
                for t in self._task_board.values()
                if str(t.get("task_type") or "") == "experiment" and t.get("status") in {"claimed", "running"}
            )
        )
        release_reasons = dict(self._taskboard_release_reason_counts or {})
        top_release_reason = ""
        if release_reasons:
            top_release_reason = max(release_reasons.items(), key=lambda kv: int(kv[1] or 0))[0]
        return {
            "episode_id": self._episode_id,
            "task_name": (self._current_task or {}).get("task_name"),
            "event_counts": dict(self._taskboard_event_counts),
            "summary": summary,
            "release_reason_counts": release_reasons,
            "top_release_reason": top_release_reason,
            "active_worker_count": len(self._active_workers) if self._active_workers else self._active_worker_count,
            "max_claims_per_agent": self._max_claims_per_agent,
            "task_lease_ttl": self._task_lease_ttl,
            "task_lease_ttl_experiment": self._task_lease_ttl_experiment,
            "experiment_max_active_leases": self._experiment_max_active_leases,
            "active_experiment_leases": active_experiment_leases,
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

    def _json_safe_scalar(self, value: Any) -> Any:
        if isinstance(value, (bool, int)):
            return value
        if isinstance(value, float):
            if value != value or value in {float("inf"), float("-inf")}:
                return None
            return float(value)
        if value is None:
            return None
        text = str(value)
        return text[:160]

    def _truncate(self, text: Any, limit: int = 180) -> str:
        value = str(text or "")
        if len(value) <= max(8, int(limit)):
            return value
        return value[: max(5, int(limit) - 3)] + "..."

    def _is_array_like_cell(self, value: Any) -> bool:
        try:
            import numpy as np

            if isinstance(value, np.ndarray):
                return True
        except Exception:
            pass
        return isinstance(value, (list, tuple)) and not isinstance(value, (str, bytes))

    def _is_timeseries_task(self, task: Dict[str, Any]) -> bool:
        info = task.get("logging_info") or {}
        category = str(info.get("category") or "").lower()
        task_name = str(task.get("task_name") or "").lower()
        metric = str(info.get("metric") or "").lower()
        return ("time series" in category) or ("forecast" in task_name) or ("mase" in metric)

    def _summarize_array_series(self, series, *, max_samples: int = 1024) -> Dict[str, Any]:
        import numpy as np
        import pandas as pd

        ss = series.dropna()
        if ss.empty:
            return {"kind": "array", "available": False, "reason": "empty"}
        if len(ss) > max_samples:
            ss = ss.sample(n=max_samples, random_state=int(self.seed or 42))

        shapes: List[Any] = []
        dtypes: set[str] = set()
        lengths: List[int] = []
        for value in ss.tolist():
            if isinstance(value, np.ndarray):
                shape = tuple(int(x) for x in value.shape)
                shapes.append(shape)
                dtypes.add(str(value.dtype))
                lengths.append(int(value.size))
            elif isinstance(value, (list, tuple)):
                try:
                    arr = np.asarray(value)
                    shape = tuple(int(x) for x in arr.shape)
                    shapes.append(shape if shape else ("list", int(len(value))))
                    dtypes.add(str(arr.dtype))
                    lengths.append(int(arr.size if arr.size else len(value)))
                except Exception:
                    shapes.append(("list", int(len(value))))
                    dtypes.add("python_object")
                    lengths.append(int(len(value)))
            else:
                dtypes.add(type(value).__name__)
                shapes.append(("scalar",))
                lengths.append(1)

        shape_counts = pd.Series(shapes).value_counts().head(8).to_dict()
        len_stats = {
            "min": int(min(lengths)) if lengths else 0,
            "p50": int(np.median(lengths)) if lengths else 0,
            "p95": int(np.quantile(lengths, 0.95)) if lengths else 0,
            "max": int(max(lengths)) if lengths else 0,
        }
        return {
            "kind": "array",
            "available": True,
            "dtype_set": sorted(list(dtypes))[:8],
            "shape_top": {str(k): int(v) for k, v in shape_counts.items()},
            "len_stats": len_stats,
            "sample_count": int(len(lengths)),
        }

    def _summarize_columns(self, df: Any, max_cols: int = 16) -> List[Dict[str, Any]]:
        if df is None or getattr(df, "empty", True):
            return []
        out: List[Dict[str, Any]] = []
        row_count = int(len(df))
        for col in list(df.columns)[: max(1, int(max_cols))]:
            series = df[col]
            non_null = int(series.notna().sum())
            missing_ratio = float(max(0.0, min(1.0, 1.0 - (non_null / max(1, row_count)))))
            try:
                uniq = int(series.nunique(dropna=True))
            except Exception:
                uniq = 0
            samples: List[Any] = []
            try:
                for value in series.head(40).tolist():
                    safe_val = self._json_safe_scalar(value)
                    if safe_val is None:
                        continue
                    if safe_val in samples:
                        continue
                    samples.append(safe_val)
                    if len(samples) >= 3:
                        break
            except Exception:
                samples = []
            out.append(
                {
                    "name": str(col),
                    "dtype": str(getattr(series, "dtype", "unknown")),
                    "missing_ratio": round(missing_ratio, 4),
                    "unique": uniq,
                    "samples": samples,
                }
            )
        return out

    def _sample_rows(self, df: Any, max_rows: int = 3, max_cols: int = 8) -> List[Dict[str, Any]]:
        if df is None or getattr(df, "empty", True):
            return []
        cols = list(df.columns)[: max(1, int(max_cols))]
        rows: List[Dict[str, Any]] = []
        try:
            for _, row in df[cols].head(max(1, int(max_rows))).iterrows():
                item: Dict[str, Any] = {}
                for col in cols:
                    item[str(col)] = self._json_safe_scalar(row.get(col))
                rows.append(item)
        except Exception:
            return []
        return rows

    def _is_regression_metric(self, metric_name: str) -> bool:
        metric = str(metric_name or "").lower()
        return any(tok in metric for tok in ("mae", "mase", "meanabsoluteerror", "rmse", "mse", "spearman", "r2"))

    def _feature_numeric_series(self, series):
        import pandas as pd

        if series is None:
            return None, "unavailable"
        try:
            numeric = pd.to_numeric(series, errors="coerce")
            numeric_ratio = float(numeric.notna().sum()) / max(1, int(len(numeric)))
            if numeric_ratio >= 0.6:
                return numeric, "numeric"
        except Exception:
            pass
        try:
            return series.astype(str).str.len().astype(float), "text_length"
        except Exception:
            return None, "unavailable"

    def _paired_feature_target_arrays(self, feature_series, target_series, *, target_is_regression: bool):
        import numpy as np
        import pandas as pd

        x_series, x_source = self._feature_numeric_series(feature_series)
        if x_series is None:
            return np.asarray([], dtype=float), np.asarray([], dtype=float), "unavailable"
        if target_is_regression:
            try:
                y_series = pd.to_numeric(target_series, errors="coerce")
            except Exception:
                return np.asarray([], dtype=float), np.asarray([], dtype=float), x_source
            mask = x_series.notna() & y_series.notna()
            x = x_series[mask].astype(float).to_numpy()
            y = y_series[mask].astype(float).to_numpy()
        else:
            y_codes, _ = pd.factorize(target_series, sort=False)
            y_series = pd.Series(y_codes, index=target_series.index if hasattr(target_series, "index") else None)
            mask = x_series.notna() & (y_series >= 0)
            x = x_series[mask].astype(float).to_numpy()
            y = y_series[mask].astype(float).to_numpy()
        if x.size == 0 or y.size == 0:
            return np.asarray([], dtype=float), np.asarray([], dtype=float), x_source
        finite = np.isfinite(x) & np.isfinite(y)
        return x[finite], y[finite], x_source

    def _compute_mutual_information(self, x, y, *, regression: bool) -> Optional[float]:
        import numpy as np

        if x is None or y is None:
            return None
        if len(x) < 30 or len(y) < 30 or len(x) != len(y):
            return None
        if float(np.std(x)) <= 1e-12:
            return None
        if regression and float(np.std(y)) <= 1e-12:
            return None
        try:
            if regression:
                from sklearn.feature_selection import mutual_info_regression

                val = mutual_info_regression(x.reshape(-1, 1), y, random_state=int(self.seed or 42), n_neighbors=3)
            else:
                from sklearn.feature_selection import mutual_info_classif

                val = mutual_info_classif(x.reshape(-1, 1), y.astype(int), random_state=int(self.seed or 42), n_neighbors=3)
            if val is None or len(val) <= 0:
                return None
            score = float(val[0])
            if score != score or score in {float("inf"), float("-inf")}:
                return None
            return max(0.0, score)
        except Exception:
            return None

    def _compute_psi(self, expected, actual, bins: int = 10) -> Optional[float]:
        import numpy as np

        if expected is None or actual is None or len(expected) < 30 or len(actual) < 30:
            return None
        eps = 1e-6
        expected = np.asarray(expected, dtype=float)
        actual = np.asarray(actual, dtype=float)
        expected = expected[np.isfinite(expected)]
        actual = actual[np.isfinite(actual)]
        if expected.size < 30 or actual.size < 30:
            return None
        try:
            q = np.linspace(0.0, 1.0, num=max(3, int(bins) + 1))
            edges = np.quantile(expected, q)
            edges = np.unique(edges)
            if edges.size < 3:
                mu = float(np.mean(expected))
                sigma = float(np.std(expected))
                sigma = sigma if sigma > 1e-9 else 1.0
                edges = np.array([mu - 3 * sigma, mu - sigma, mu, mu + sigma, mu + 3 * sigma], dtype=float)
            edges = np.unique(edges)
            if edges.size < 3:
                return None
            edges[0] = -np.inf
            edges[-1] = np.inf
            e_counts, _ = np.histogram(expected, bins=edges)
            a_counts, _ = np.histogram(actual, bins=edges)
            e_pct = (e_counts.astype(float) + eps) / max(eps, float(e_counts.sum()) + eps * len(e_counts))
            a_pct = (a_counts.astype(float) + eps) / max(eps, float(a_counts.sum()) + eps * len(a_counts))
            psi = float(np.sum((e_pct - a_pct) * np.log(e_pct / a_pct)))
            if psi != psi or psi in {float("inf"), float("-inf")}:
                return None
            return max(0.0, psi)
        except Exception:
            return None

    def _compute_ks_stat(self, expected, actual) -> Optional[float]:
        import numpy as np

        if expected is None or actual is None or len(expected) < 30 or len(actual) < 30:
            return None
        exp = np.asarray(expected, dtype=float)
        act = np.asarray(actual, dtype=float)
        exp = exp[np.isfinite(exp)]
        act = act[np.isfinite(act)]
        if exp.size < 30 or act.size < 30:
            return None
        try:
            exp_sorted = np.sort(exp)
            act_sorted = np.sort(act)
            all_values = np.sort(np.concatenate([exp_sorted, act_sorted]))
            cdf_exp = np.searchsorted(exp_sorted, all_values, side="right") / float(exp_sorted.size)
            cdf_act = np.searchsorted(act_sorted, all_values, side="right") / float(act_sorted.size)
            ks = float(np.max(np.abs(cdf_exp - cdf_act)))
            return max(0.0, min(1.0, ks))
        except Exception:
            return None

    def _diagnose_information_dynamics(self, train_df, target_col: str, metric_name: str) -> Dict[str, Any]:
        import numpy as np
        import pandas as pd

        if train_df is None or getattr(train_df, "empty", True) or not target_col or target_col not in train_df.columns:
            return {}

        target_series = train_df[target_col]
        feature_cols = [c for c in train_df.columns if c != target_col][:24]
        target_is_regression = self._is_regression_metric(metric_name)
        if not target_is_regression:
            try:
                y_num = pd.to_numeric(target_series, errors="coerce")
                numeric_ratio = float(y_num.notna().sum()) / max(1, int(len(y_num)))
                uniq = int(target_series.nunique(dropna=True))
                target_is_regression = bool(numeric_ratio >= 0.9 and uniq > 30)
            except Exception:
                target_is_regression = False

        associations: List[Dict[str, Any]] = []
        for col in feature_cols:
            x, y, x_source = self._paired_feature_target_arrays(
                train_df[col],
                target_series,
                target_is_regression=target_is_regression,
            )
            if len(x) < 30 or len(y) < 30:
                continue
            corr = None
            if target_is_regression:
                x_std = float(np.std(x))
                y_std = float(np.std(y))
                if x_std > 1e-12 and y_std > 1e-12:
                    try:
                        corr = float(np.corrcoef(x, y)[0, 1])
                    except Exception:
                        corr = None
            mi = self._compute_mutual_information(x, y, regression=target_is_regression)
            associations.append(
                {
                    "feature": str(col),
                    "feature_source": x_source,
                    "sample_size": int(len(x)),
                    "pearson_corr": self._json_safe_scalar(corr),
                    "abs_corr": self._json_safe_scalar(abs(float(corr)) if corr is not None else None),
                    "mutual_info": self._json_safe_scalar(mi),
                }
            )

        def _assoc_rank(item: Dict[str, Any]) -> Tuple[float, float]:
            ac = float(item.get("abs_corr") or 0.0)
            mi = float(item.get("mutual_info") or 0.0)
            return (ac, mi)

        associations = sorted(associations, key=_assoc_rank, reverse=True)
        leakage_suspects = [
            {
                "feature": item.get("feature"),
                "abs_corr": item.get("abs_corr"),
                "mutual_info": item.get("mutual_info"),
            }
            for item in associations
            if float(item.get("abs_corr") or 0.0) >= 0.95
        ][:8]

        collinearity_pairs: List[Dict[str, Any]] = []
        numeric_df = pd.DataFrame(index=train_df.index)
        for col in feature_cols[:20]:
            converted, _ = self._feature_numeric_series(train_df[col])
            if converted is None:
                continue
            notna_ratio = float(converted.notna().sum()) / max(1, int(len(converted)))
            if notna_ratio < 0.6:
                continue
            numeric_df[str(col)] = converted.astype(float)
        if numeric_df.shape[1] >= 2:
            numeric_df = numeric_df.dropna(axis=0, how="any")
            if numeric_df.shape[0] >= 30:
                corr = numeric_df.corr().abs()
                cols = list(corr.columns)
                for i in range(len(cols)):
                    for j in range(i + 1, len(cols)):
                        score = float(corr.iloc[i, j])
                        if score >= 0.9:
                            collinearity_pairs.append({"feature_a": cols[i], "feature_b": cols[j], "abs_corr": round(score, 4)})
        collinearity_pairs = sorted(collinearity_pairs, key=lambda x: float(x.get("abs_corr") or 0.0), reverse=True)[:12]
        return {
            "feature_target_association": associations[:16],
            "leakage_suspects": leakage_suspects,
            "multicollinearity_pairs": collinearity_pairs,
            "summary": {
                "features_analyzed": int(len(feature_cols)),
                "association_count": int(len(associations)),
                "high_association_count": int(sum(1 for x in associations if float(x.get("abs_corr") or 0.0) >= 0.5)),
                "leakage_suspect_count": int(len(leakage_suspects)),
                "high_collinearity_pair_count": int(len(collinearity_pairs)),
            },
        }

    def _diagnose_distribution_stability(self, train_df, test_df, target_col: str) -> Dict[str, Any]:
        if (
            train_df is None
            or test_df is None
            or getattr(train_df, "empty", True)
            or getattr(test_df, "empty", True)
        ):
            return {}
        common_cols = [c for c in train_df.columns if c in set(test_df.columns) and c != target_col][:24]
        shifts: List[Dict[str, Any]] = []
        for col in common_cols:
            train_num, source_train = self._feature_numeric_series(train_df[col])
            test_num, source_test = self._feature_numeric_series(test_df[col])
            if train_num is None or test_num is None:
                continue
            train_arr = train_num.astype(float).to_numpy()
            test_arr = test_num.astype(float).to_numpy()
            psi = self._compute_psi(train_arr, test_arr, bins=10)
            ks = self._compute_ks_stat(train_arr, test_arr)
            if psi is None and ks is None:
                continue
            shifts.append(
                {
                    "feature": str(col),
                    "feature_source": source_train if source_train != "unavailable" else source_test,
                    "psi": self._json_safe_scalar(round(float(psi), 4) if psi is not None else None),
                    "ks_stat": self._json_safe_scalar(round(float(ks), 4) if ks is not None else None),
                }
            )
        shifts = sorted(shifts, key=lambda x: (float(x.get("psi") or 0.0), float(x.get("ks_stat") or 0.0)), reverse=True)
        severe = [
            x
            for x in shifts
            if (float(x.get("psi") or 0.0) > 0.25) or (float(x.get("ks_stat") or 0.0) > 0.2)
        ]
        return {
            "train_test_shift": shifts[:16],
            "severe_shift_features": severe[:10],
            "summary": {
                "features_compared": int(len(common_cols)),
                "shift_features_count": int(len(shifts)),
                "severe_shift_count": int(len(severe)),
                "max_psi": self._json_safe_scalar(max([float(x.get("psi") or 0.0) for x in shifts] or [0.0])),
                "max_ks_stat": self._json_safe_scalar(max([float(x.get("ks_stat") or 0.0) for x in shifts] or [0.0])),
            },
        }

    def _diagnose_quality(self, train_df, target_col: str) -> Dict[str, Any]:
        import pandas as pd

        if train_df is None or getattr(train_df, "empty", True):
            return {}
        feature_cols = [c for c in train_df.columns if c != target_col][:24]
        quality_rows: List[Dict[str, Any]] = []
        for col in feature_cols:
            series = train_df[col]
            try:
                numeric = pd.to_numeric(series, errors="coerce").dropna()
            except Exception:
                continue
            if numeric.shape[0] < 30:
                continue
            try:
                skew = float(numeric.skew())
                kurt = float(numeric.kurt())
                q1 = float(numeric.quantile(0.25))
                q3 = float(numeric.quantile(0.75))
                iqr = max(1e-9, q3 - q1)
                lower = q1 - 1.5 * iqr
                upper = q3 + 1.5 * iqr
                iqr_out = float(((numeric < lower) | (numeric > upper)).mean())
                std = float(numeric.std())
                if std > 1e-12:
                    sigma_out = float(((numeric - float(numeric.mean())).abs() > 3.0 * std).mean())
                else:
                    sigma_out = 0.0
                quality_rows.append(
                    {
                        "feature": str(col),
                        "skewness": self._json_safe_scalar(round(skew, 4)),
                        "kurtosis": self._json_safe_scalar(round(kurt, 4)),
                        "iqr_outlier_ratio": self._json_safe_scalar(round(iqr_out, 4)),
                        "sigma_outlier_ratio": self._json_safe_scalar(round(sigma_out, 4)),
                    }
                )
            except Exception:
                continue
        high_skew = [x for x in quality_rows if abs(float(x.get("skewness") or 0.0)) > 3.0]
        high_outlier = [x for x in quality_rows if float(x.get("iqr_outlier_ratio") or 0.0) > 0.1]
        return {
            "numeric_distribution": quality_rows[:16],
            "high_skew_features": high_skew[:10],
            "high_outlier_features": high_outlier[:10],
            "summary": {
                "numeric_features_analyzed": int(len(quality_rows)),
                "high_skew_feature_count": int(len(high_skew)),
                "high_outlier_feature_count": int(len(high_outlier)),
            },
        }

    def _build_task_priors(self, task: Dict[str, Any]) -> Dict[str, Any]:
        info = task.get("logging_info") or {}
        task_name = str(task.get("task_name") or "").lower()
        category = str(info.get("category") or "").lower()
        dataset = str(info.get("dataset") or "").lower()
        priors: List[Dict[str, Any]] = []
        if ("qm9" in task_name) or ("qm9" in dataset) or ("molecular" in category):
            priors.append(
                {
                    "domain": "molecular_property_prediction",
                    "physics_constants": [
                        {"name": "electronegativity_pauling", "elements": ["H", "C", "N", "O", "F"]},
                        {"name": "covalent_radius_pm", "elements": ["H", "C", "N", "O", "F"]},
                        {"name": "atomic_mass_u", "elements": ["H", "C", "N", "O", "F"]},
                    ],
                    "recommended_features": [
                        "atom-type composition statistics",
                        "pairwise distance histogram moments",
                        "charge-related aggregated features",
                    ],
                    "unit_checks": ["target unit consistency across train/dev/test", "magnitude sanity checks before regression"],
                }
            )
        if ("time series" in category) or ("forecast" in task_name):
            priors.append(
                {
                    "domain": "time_series_forecasting",
                    "recommended_protocol": [
                        "avoid random split; preserve temporal order",
                        "use lag/window features and seasonal diagnostics",
                        "monitor autocorrelation to choose split strategy",
                    ],
                }
            )
        if ("text" in category) or ("similarity" in task_name):
            priors.append(
                {
                    "domain": "text_modeling",
                    "recommended_protocol": [
                        "check sentence length drift and tokenization mismatch",
                        "start from tfidf-linear baseline before heavy models",
                    ],
                }
            )
        return {
            "task_name": task.get("task_name"),
            "dataset": info.get("dataset"),
            "category": info.get("category"),
            "priors": priors,
        }

    def _compute_naive_baseline(self, task: Dict[str, Any], train_df, val_df, target_col: str, metric_name: str) -> Dict[str, Any]:
        import numpy as np
        import pandas as pd

        if (
            train_df is None
            or val_df is None
            or getattr(train_df, "empty", True)
            or getattr(val_df, "empty", True)
            or not target_col
            or target_col not in train_df.columns
            or target_col not in val_df.columns
        ):
            return {"available": False, "reason": "target_or_split_missing"}

        metric_name = str(metric_name or "")
        is_regression = self._is_regression_metric(metric_name)
        candidates: List[Dict[str, Any]] = []

        if is_regression:
            y_train = pd.to_numeric(train_df[target_col], errors="coerce").dropna()
            y_val = pd.to_numeric(val_df[target_col], errors="coerce").dropna()
            if y_train.shape[0] < 30 or y_val.shape[0] < 30:
                return {"available": False, "reason": "insufficient_numeric_target"}
            const_candidates = {
                "mean": float(y_train.mean()),
                "median": float(y_train.median()),
            }
            for name, value in const_candidates.items():
                pred = np.full(y_val.shape[0], float(value), dtype=float)
                raw = self._compute_dev_metric(metric_name, y_val.to_numpy(dtype=float), pred)
                if raw is None:
                    continue
                norm = self._normalize_metric(float(raw), task)
                candidates.append(
                    {
                        "name": f"constant_{name}",
                        "raw_score": self._json_safe_scalar(float(raw)),
                        "score_norm": self._json_safe_scalar(float(norm)),
                    }
                )
            rng = np.random.default_rng(int(self.seed or 42))
            rand_pred = rng.choice(y_train.to_numpy(dtype=float), size=int(y_val.shape[0]), replace=True)
            raw_rand = self._compute_dev_metric(metric_name, y_val.to_numpy(dtype=float), rand_pred)
            if raw_rand is not None:
                norm_rand = self._normalize_metric(float(raw_rand), task)
                candidates.append(
                    {
                        "name": "random_from_train_target",
                        "raw_score": self._json_safe_scalar(float(raw_rand)),
                        "score_norm": self._json_safe_scalar(float(norm_rand)),
                    }
                )
        else:
            y_train = train_df[target_col].dropna()
            y_val = val_df[target_col].dropna()
            if y_train.shape[0] < 30 or y_val.shape[0] < 30:
                return {"available": False, "reason": "insufficient_class_target"}
            majority = y_train.value_counts(dropna=True).idxmax()
            pred_major = [majority for _ in range(int(y_val.shape[0]))]
            raw_major = self._compute_dev_metric(metric_name, list(y_val), pred_major)
            if raw_major is not None:
                candidates.append(
                    {
                        "name": "majority_class",
                        "raw_score": self._json_safe_scalar(float(raw_major)),
                        "score_norm": self._json_safe_scalar(float(self._normalize_metric(float(raw_major), task))),
                    }
                )
            rng = np.random.default_rng(int(self.seed or 42))
            y_train_values = list(y_train.values)
            pred_rand = [y_train_values[int(x)] for x in rng.integers(0, len(y_train_values), size=int(y_val.shape[0]))]
            raw_rand = self._compute_dev_metric(metric_name, list(y_val), pred_rand)
            if raw_rand is not None:
                candidates.append(
                    {
                        "name": "random_label_from_train",
                        "raw_score": self._json_safe_scalar(float(raw_rand)),
                        "score_norm": self._json_safe_scalar(float(self._normalize_metric(float(raw_rand), task))),
                    }
                )

        if not candidates:
            return {"available": False, "reason": "baseline_metric_unavailable"}

        best = sorted(candidates, key=lambda x: float(x.get("score_norm") or 0.0), reverse=True)[0]
        return {
            "available": True,
            "metric": metric_name,
            "candidates": candidates[:4],
            "best": best,
        }

    def _build_data_card(
        self,
        *,
        task: Dict[str, Any],
        data_mount_dir: str,
        focus_cols: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        info = task.get("logging_info") or {}
        metric_name = str(info.get("metric") or "")
        is_timeseries = self._is_timeseries_task(task)
        row_budget = int(self._profile_sample_rows)
        if is_timeseries:
            row_budget = int(max(128, min(row_budget, self._profile_timeseries_max_rows)))
        focus_set = {str(c) for c in (focus_cols or []) if str(c).strip()}
        preferred_cols = list(focus_set) + [str(info.get("scoring_column") or "").strip(), "label", "labels", "target"]
        train_df = self._load_prepared_split_df(
            data_mount_dir=data_mount_dir,
            split_name="train",
            max_rows=row_budget,
            max_cols=self._profile_max_columns,
            preferred_cols=preferred_cols,
        )
        val_df = self._load_prepared_split_df(
            data_mount_dir=data_mount_dir,
            split_name="validation",
            max_rows=row_budget,
            max_cols=self._profile_max_columns,
            preferred_cols=preferred_cols,
        )
        if val_df is None:
            val_df = self._load_prepared_split_df(
                data_mount_dir=data_mount_dir,
                split_name="val",
                max_rows=row_budget,
                max_cols=self._profile_max_columns,
                preferred_cols=preferred_cols,
            )
        if val_df is None:
            val_df = self._load_prepared_split_df(
                data_mount_dir=data_mount_dir,
                split_name="dev",
                max_rows=row_budget,
                max_cols=self._profile_max_columns,
                preferred_cols=preferred_cols,
            )
        test_df = self._load_prepared_split_df(
            data_mount_dir=data_mount_dir,
            split_name="test",
            max_rows=row_budget,
            max_cols=self._profile_max_columns,
            preferred_cols=preferred_cols,
        )

        target_col = ""
        if train_df is not None and not train_df.empty:
            target_col = self._infer_target_column(task=task, train_df=train_df)

        schema_df = train_df if train_df is not None else val_df
        if schema_df is not None and focus_set:
            keep_cols = [c for c in schema_df.columns if str(c) in focus_set]
            if keep_cols:
                schema_df = schema_df[keep_cols]

        train_rows_total = self._prepared_split_row_count(data_mount_dir=data_mount_dir, split_names=["train"])
        dev_rows_total = self._prepared_split_row_count(data_mount_dir=data_mount_dir, split_names=["validation", "val", "dev"])
        test_rows_total = self._prepared_split_row_count(data_mount_dir=data_mount_dir, split_names=["test"])
        split_stats = {
            "train_rows": int(train_rows_total),
            "dev_rows": int(dev_rows_total),
            "test_rows": int(test_rows_total),
            "train_cols": int(len(train_df.columns)) if train_df is not None else 0,
        }
        sampled_rows = {
            "train_rows_loaded": int(len(train_df)) if train_df is not None else 0,
            "dev_rows_loaded": int(len(val_df)) if val_df is not None else 0,
            "test_rows_loaded": int(len(test_df)) if test_df is not None else 0,
            "max_rows_per_split": int(self._profile_sample_rows),
            "max_columns": int(self._profile_max_columns),
        }

        label_profile: Dict[str, Any] = {}
        if train_df is not None and target_col and target_col in train_df.columns:
            series = train_df[target_col].dropna()
            if not series.empty:
                label_profile["target_column"] = target_col
                # Graph-style targets may contain ndarray/list cells; avoid hash-based operations.
                try:
                    first_non_na = series.iloc[0]
                except Exception:
                    first_non_na = None
                if self._is_array_like_cell(first_non_na):
                    label_profile["target_kind"] = "array"
                    label_profile["target_structure"] = self._summarize_array_series(series, max_samples=1024)
                else:
                    try:
                        uniq = int(series.nunique(dropna=True))
                    except Exception:
                        uniq = 0
                    label_profile["unique_labels"] = uniq
                    if uniq <= 30 and uniq > 0:
                        try:
                            counts = series.value_counts(dropna=True).head(10)
                            total = float(max(1, int(series.shape[0])))
                            label_profile["top_label_distribution"] = [
                                {
                                    "label": self._json_safe_scalar(idx),
                                    "count": int(cnt),
                                    "ratio": round(float(cnt) / total, 4),
                                }
                                for idx, cnt in counts.items()
                            ]
                        except Exception:
                            label_profile["top_label_distribution"] = []
                    else:
                        try:
                            num_series = series.astype(float)
                            label_profile["target_stats"] = {
                                "mean": self._json_safe_scalar(num_series.mean()),
                                "std": self._json_safe_scalar(num_series.std()),
                                "min": self._json_safe_scalar(num_series.min()),
                                "max": self._json_safe_scalar(num_series.max()),
                            }
                        except Exception:
                            label_profile["target_stats"] = {}

        schema = self._summarize_columns(schema_df, max_cols=16)
        risks: List[str] = []
        if split_stats["train_rows"] <= 0:
            risks.append("train_split_missing_or_empty")
        if split_stats["dev_rows"] <= 0:
            risks.append("dev_split_missing_or_empty")
        if target_col == "":
            risks.append("target_column_not_inferred")
        if label_profile.get("unique_labels", 0) and int(label_profile.get("unique_labels", 0)) > 150:
            risks.append("high_label_cardinality")
        if split_stats["train_rows"] > sampled_rows["train_rows_loaded"]:
            risks.append("train_profile_sampled")
        if split_stats["dev_rows"] > sampled_rows["dev_rows_loaded"]:
            risks.append("dev_profile_sampled")
        if split_stats["test_rows"] > sampled_rows["test_rows_loaded"]:
            risks.append("test_profile_sampled")

        if is_timeseries:
            information_dynamics = {}
            distribution_stability = self._diagnose_distribution_stability(
                train_df=train_df,
                test_df=test_df,
                target_col=target_col,
            )
            quality_diagnostics = {
                "summary": {
                    "mode": "timeseries_fast_profile",
                    "max_rows_per_split": int(row_budget),
                    "note": "fast-path enabled to keep profile_data within time budget",
                }
            }
        else:
            information_dynamics = self._diagnose_information_dynamics(
                train_df=train_df,
                target_col=target_col,
                metric_name=metric_name,
            )
            distribution_stability = self._diagnose_distribution_stability(
                train_df=train_df,
                test_df=test_df,
                target_col=target_col,
            )
            quality_diagnostics = self._diagnose_quality(train_df=train_df, target_col=target_col)
        task_priors = self._build_task_priors(task=task)
        naive_baseline = self._compute_naive_baseline(
            task=task,
            train_df=train_df,
            val_df=val_df,
            target_col=target_col,
            metric_name=metric_name,
        )

        if int(((information_dynamics.get("summary") or {}).get("leakage_suspect_count") or 0) > 0):
            risks.append("possible_label_leakage_or_shortcut_feature")
        if int(((information_dynamics.get("summary") or {}).get("high_collinearity_pair_count") or 0) > 0):
            risks.append("high_multicollinearity")
        if int(((distribution_stability.get("summary") or {}).get("severe_shift_count") or 0) > 0):
            risks.append("covariate_shift_detected")
        if int(((quality_diagnostics.get("summary") or {}).get("high_skew_feature_count") or 0) > 0):
            risks.append("high_skew_numeric_features")
        if int(((quality_diagnostics.get("summary") or {}).get("high_outlier_feature_count") or 0) > 0):
            risks.append("high_outlier_ratio_features")

        return {
            "ok": True,
            "degraded": False,
            "reason": "",
            "card_type": "data_card",
            "task_name": task.get("task_name"),
            "metric": metric_name,
            "dataset": info.get("dataset"),
            "focus_columns": sorted(focus_set)[:16],
            "split_stats": split_stats,
            "sampled_rows": sampled_rows,
            "target_column": target_col,
            "label_profile": label_profile,
            "schema": schema,
            "sample_rows": self._sample_rows(schema_df, max_rows=3, max_cols=8),
            "information_dynamics": information_dynamics,
            "distribution_stability": distribution_stability,
            "quality_diagnostics": quality_diagnostics,
            "task_priors": task_priors,
            "naive_baseline": naive_baseline,
            "risk_flags": sorted(set(risks)),
        }

    def _build_degraded_data_card(
        self,
        *,
        task: Dict[str, Any],
        reason: str,
        focus_cols: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        info = task.get("logging_info") or {}
        focus_set = sorted({str(c) for c in (focus_cols or []) if str(c).strip()})[:16]
        return {
            "ok": True,
            "degraded": True,
            "reason": str(reason or "")[:200],
            "card_type": "data_card",
            "task_name": task.get("task_name"),
            "metric": str(info.get("metric") or ""),
            "dataset": info.get("dataset"),
            "focus_columns": focus_set,
            "split_stats": {"train_rows": 0, "dev_rows": 0, "test_rows": 0, "train_cols": 0},
            "sampled_rows": {
                "train_rows_loaded": 0,
                "dev_rows_loaded": 0,
                "test_rows_loaded": 0,
                "max_rows_per_split": int(self._profile_sample_rows),
                "max_columns": int(self._profile_max_columns),
            },
            "target_column": "",
            "label_profile": {},
            "schema": [],
            "sample_rows": [],
            "information_dynamics": {},
            "distribution_stability": {},
            "quality_diagnostics": {},
            "task_priors": self._build_task_priors(task=task),
            "naive_baseline": {"available": False, "reason": "degraded_data_card"},
            "risk_flags": ["profile_data_degraded", str(reason or "profile_data_degraded")[:200]],
        }

    def _prepared_split_row_count(self, data_mount_dir: str, split_names: List[str]) -> int:
        from datasets import load_from_disk

        for name in split_names:
            split_path = Path(data_mount_dir) / str(name)
            if not split_path.exists():
                continue
            try:
                return int(len(load_from_disk(str(split_path))))
            except Exception:
                continue
        return 0

    def _method_templates_for_task(self, *, metric: str, category: str, research_problem: str) -> List[Dict[str, Any]]:
        metric_l = metric.lower()
        category_l = category.lower()
        problem_l = research_problem.lower()

        if "mrr" in metric_l or "retrieval" in category_l:
            return [
                {
                    "name": "bm25_or_sparse_retrieval_baseline",
                    "use_when": "text-to-code or text retrieval with ranking targets",
                    "key_steps": ["strong lexical baseline", "top-k ranking output", "MRR-aligned validation"],
                    "pitfalls": ["query/corpus id mismatch", "rankings format invalid"],
                },
                {
                    "name": "dual_encoder_with_hard_negatives",
                    "use_when": "enough training pairs and embedding workflow available",
                    "key_steps": ["contrastive loss", "in-batch negatives", "ANN retrieval stage"],
                    "pitfalls": ["insufficient negatives", "evaluation leakage across splits"],
                },
            ]
        if "pass@" in metric_l or ("code" in category_l and "generation" in problem_l):
            return [
                {
                    "name": "prompted_codegen_baseline",
                    "use_when": "code generation with strict output schema",
                    "key_steps": ["deterministic prompt template", "multi-sample candidates", "syntax sanity checks"],
                    "pitfalls": ["invalid CSV schema", "runtime-unsafe snippets"],
                },
                {
                    "name": "self_debug_codegen_loop",
                    "use_when": "testcase feedback can be used on dev split",
                    "key_steps": ["capture traceback", "patch function-level logic", "re-run unit checks"],
                    "pitfalls": ["overfitting tiny dev subset", "ignoring pass@k diversity"],
                },
            ]
        if "mae" in metric_l or "mase" in metric_l or "spearman" in metric_l or "regression" in category_l:
            return [
                {
                    "name": "ridge_or_linear_regression_baseline",
                    "use_when": "tabular/text regression with limited budget",
                    "key_steps": ["feature normalization", "regularization sweep", "error distribution analysis"],
                    "pitfalls": ["target leakage features", "metric direction confusion"],
                },
                {
                    "name": "gradient_boosting_regressor",
                    "use_when": "nonlinear tabular patterns dominate",
                    "key_steps": ["basic feature engineering", "early stopping", "fold-level stability checks"],
                    "pitfalls": ["overfitting small data", "insufficient calibration"],
                },
            ]
        return [
            {
                "name": "tfidf_linear_classifier",
                "use_when": "text-heavy classification with constrained runtime",
                "key_steps": ["ngram sweep", "class-weight handling", "format validity checks"],
                "pitfalls": ["label imbalance ignored", "tokenization mismatch train/test"],
            },
            {
                "name": "tree_or_boosting_classifier",
                "use_when": "mixed tabular features and nonlinear boundaries",
                "key_steps": ["categorical handling", "regularized depth", "threshold calibration"],
                "pitfalls": ["leaky engineered features", "unstable seed sensitivity"],
            },
        ]

    def _profile_data_pipeline(
        self,
        *,
        task: Dict[str, Any],
        data_mount_dir: str,
        agent_log_dir: str,
        focus_cols: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        cache_dir = self._prepared_cache_dir()
        if not cache_dir:
            raise RuntimeError("prepare_cache_not_ready")
        # Fast path for profile_data: avoid heavy full-dir copy and read prepared cache directly.
        if self._profile_use_cache_direct:
            return self._build_data_card(task=task, data_mount_dir=cache_dir, focus_cols=focus_cols)
        self._materialize_data_mount_from_cache(cache_dir, data_mount_dir)
        return self._build_data_card(task=task, data_mount_dir=data_mount_dir, focus_cols=focus_cols)

    def _prepared_cache_dir(self) -> str:
        cache_dir = str(self._prepared_data_cache_dir or "")
        if not cache_dir:
            return ""
        path = Path(cache_dir)
        if not path.exists():
            return ""
        return cache_dir

    def _prepared_cache_ready(self) -> bool:
        cache_dir = self._prepared_cache_dir()
        if not cache_dir:
            return False
        root = Path(cache_dir)
        for split in ("train", "validation", "val", "dev", "test"):
            if (root / split).exists():
                return True
        return any(root.iterdir()) if root.exists() else False

    def _prepared_cache_ready_at(self, cache_dir: Path) -> bool:
        if not cache_dir.exists():
            return False
        for split in ("train", "validation", "val", "dev", "test"):
            if (cache_dir / split).exists():
                return True
        try:
            return any(cache_dir.iterdir())
        except Exception:
            return False

    def _raw_dataset_path_for_task(self, task: Dict[str, Any]) -> Path:
        dataset_rel = str((task or {}).get("dataset_rel") or "").strip()
        if dataset_rel:
            return Path(self.shared_data_dir) / dataset_rel
        dataset_name = str(((task or {}).get("logging_info") or {}).get("dataset") or "").strip()
        if dataset_name:
            return Path(self.shared_data_dir) / dataset_name
        return Path(self.shared_data_dir)

    def _raw_dataset_fingerprint_for_task(self, task: Dict[str, Any]) -> str:
        path = self._raw_dataset_path_for_task(task)
        if not path.exists():
            return "missing"
        try:
            st = path.stat()
            head = [f"{path}", f"{int(st.st_mtime)}", f"{int(st.st_size)}"]
            if path.is_dir():
                entries = sorted(path.iterdir(), key=lambda p: p.name)[:48]
                for item in entries:
                    try:
                        ist = item.stat()
                        head.append(f"{item.name}:{int(ist.st_mtime)}:{int(ist.st_size)}")
                    except Exception:
                        head.append(f"{item.name}:na")
            return hashlib.sha1("|".join(head).encode("utf-8")).hexdigest()
        except Exception:
            return f"stat_error:{path.name}"

    def _prepare_script_hash_for_task(self, task: Dict[str, Any]) -> str:
        prepare_path = Path(str((task or {}).get("task_path") or "")) / "prepare.py"
        if not prepare_path.exists():
            return "missing_prepare"
        try:
            return self._file_sha256(str(prepare_path))
        except Exception:
            return "prepare_hash_error"

    def _prepared_persistent_cache_key(self, task: Dict[str, Any]) -> str:
        task_name = str((task or {}).get("task_name") or "").strip() or "unknown_task"
        dataset_rel = str((task or {}).get("dataset_rel") or "").strip()
        metric = str(((task or {}).get("logging_info") or {}).get("metric") or "").strip()
        raw_fp = self._raw_dataset_fingerprint_for_task(task)
        prep_hash = self._prepare_script_hash_for_task(task)
        key_blob = "|".join(
            [
                "v1",
                task_name,
                dataset_rel,
                metric,
                raw_fp,
                prep_hash,
            ]
        )
        return hashlib.sha1(key_blob.encode("utf-8")).hexdigest()

    def _prepared_persistent_cache_path(self, task: Dict[str, Any]) -> Path:
        return Path(self._prepared_cache_persist_root) / self._prepared_persistent_cache_key(task)

    def _try_load_prepared_cache_from_persistent(self, task: Dict[str, Any]) -> Optional[str]:
        if not self._prepared_cache_persist_enable:
            return None
        target = self._prepared_persistent_cache_path(task)
        if not self._prepared_cache_ready_at(target):
            return None
        self._prepared_data_cache_dir = str(target)
        return str(target)

    def _publish_prepared_cache_to_persistent(self, *, task: Dict[str, Any], local_cache_dir: str) -> Optional[str]:
        if not self._prepared_cache_persist_enable:
            return None
        src = Path(str(local_cache_dir or ""))
        if not self._prepared_cache_ready_at(src):
            return None

        target = self._prepared_persistent_cache_path(task)
        if self._prepared_cache_ready_at(target):
            return str(target)

        root = Path(self._prepared_cache_persist_root)
        root.mkdir(parents=True, exist_ok=True)
        staging = root / f"{target.name}.__staging__.{uuid.uuid4().hex[:8]}"
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
        staging.mkdir(parents=True, exist_ok=True)

        try:
            for child in src.iterdir():
                src_child = src / child.name
                dst_child = staging / child.name
                if src_child.is_dir():
                    shutil.copytree(src_child, dst_child, dirs_exist_ok=True)
                elif src_child.exists():
                    shutil.copy2(src_child, dst_child)
            meta = {
                "created_at": datetime.utcnow().isoformat() + "Z",
                "task_name": str((task or {}).get("task_name") or ""),
                "dataset_rel": str((task or {}).get("dataset_rel") or ""),
                "metric": str(((task or {}).get("logging_info") or {}).get("metric") or ""),
                "prepare_hash": self._prepare_script_hash_for_task(task),
                "raw_fingerprint": self._raw_dataset_fingerprint_for_task(task),
                "cache_key": target.name,
            }
            with open(staging / "_cache_meta.json", "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)
                f.write("\n")

            try:
                os.replace(str(staging), str(target))
            except Exception:
                # Multi-process race fallback: if another writer published first, reuse it.
                if not self._prepared_cache_ready_at(target):
                    raise
            return str(target)
        finally:
            if staging.exists():
                shutil.rmtree(staging, ignore_errors=True)

    def _load_data_card_from_disk(self, profile_dir: Path) -> Optional[Dict[str, Any]]:
        card_path = profile_dir / "data_card.json"
        if not card_path.exists():
            return None
        try:
            card = json.loads(card_path.read_text(encoding="utf-8"))
            if isinstance(card, dict):
                return card
        except Exception:
            return None
        return None

    async def profile_data(
        self,
        agent_id: Optional[str] = None,
        focus_cols: Optional[List[str]] = None,
        refresh: bool = False,
    ) -> Dict[str, Any]:
        task = self._current_task
        if not task:
            await self.reset_episode()
            task = self._current_task
        if not task:
            return {"ok": False, "reason": "no_active_task"}

        profile_dir = Path(self._task_workspace()) / "_analysis" / "profile_data"
        data_mount = profile_dir / "agent_data"
        agent_log = profile_dir / "agent_log"
        profile_dir.mkdir(parents=True, exist_ok=True)
        data_mount.mkdir(parents=True, exist_ok=True)
        agent_log.mkdir(parents=True, exist_ok=True)
        focus_list = focus_cols if isinstance(focus_cols, list) else None

        if self._data_card_cache and not refresh:
            card = dict(self._data_card_cache)
            card["cache_hit"] = True
            card["agent_id"] = agent_id
            return card
        if not refresh:
            disk_card = self._load_data_card_from_disk(profile_dir)
            if isinstance(disk_card, dict):
                disk_card = dict(disk_card)
                disk_card["cache_hit"] = True
                disk_card["agent_id"] = agent_id
                self._data_card_cache = dict(disk_card)
                return disk_card

        try:
            async with self._profile_data_lock:
                if self._data_card_cache and not refresh:
                    card = dict(self._data_card_cache)
                    card["cache_hit"] = True
                    card["agent_id"] = agent_id
                    return card
                if not refresh:
                    disk_card = self._load_data_card_from_disk(profile_dir)
                    if isinstance(disk_card, dict):
                        disk_card = dict(disk_card)
                        disk_card["cache_hit"] = True
                        disk_card["agent_id"] = agent_id
                        self._data_card_cache = dict(disk_card)
                        return disk_card

                try:
                    card = await asyncio.wait_for(
                        asyncio.to_thread(
                            self._profile_data_pipeline,
                            task=task,
                            data_mount_dir=str(data_mount),
                            agent_log_dir=str(agent_log),
                            focus_cols=focus_list,
                        ),
                        timeout=float(self._profile_data_timeout_s),
                    )
                except asyncio.TimeoutError:
                    logger.warning(
                        f"profile_data timed out after {self._profile_data_timeout_s}s for "
                        f"task={task.get('task_name')}; returning degraded data card"
                    )
                    card = self._build_degraded_data_card(
                        task=task,
                        reason=f"profile_data_timeout:{self._profile_data_timeout_s}s",
                        focus_cols=focus_list,
                    )
                except Exception as e:
                    msg = str(e)
                    if "prepare_cache_not_ready" in msg:
                        logger.info(
                            f"profile_data skipped until prepare_data is ready for task={task.get('task_name')}"
                        )
                        reason = "prepare_cache_not_ready"
                    else:
                        logger.warning(f"profile_data build failed for task={task.get('task_name')}: {e}")
                        reason = f"profile_data_build_failed:{e}"
                    card = self._build_degraded_data_card(
                        task=task,
                        reason=reason,
                        focus_cols=focus_list,
                    )

            card["agent_id"] = agent_id
            card["topic"] = "data_profile"
            (profile_dir / "data_card.json").write_text(
                json.dumps(card, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            self._data_card_cache = dict(card)
            return card
        except Exception as e:
            logger.warning(f"profile_data failed for task={task.get('task_name')}: {e}")
            card = self._build_degraded_data_card(
                task=task,
                reason=f"profile_data_failed:{e}",
                focus_cols=focus_list,
            )
            card["agent_id"] = agent_id
            card["topic"] = "data_profile"
            self._data_card_cache = dict(card)
            return card

    async def prepare_data(self, agent_id: Optional[str] = None, refresh: bool = False) -> Dict[str, Any]:
        task = self._current_task
        if not task:
            await self.reset_episode()
            task = self._current_task
        if not task:
            return {"ok": False, "reason": "no_active_task"}

        if self._prepared_cache_ready() and not refresh:
            cache_dir = self._prepared_cache_dir()
            return {
                "ok": True,
                "cached": True,
                "agent_id": agent_id,
                "cache_dir": cache_dir,
                "split_stats": {
                    "train_rows": self._prepared_split_row_count(cache_dir, ["train"]),
                    "dev_rows": self._prepared_split_row_count(cache_dir, ["validation", "val", "dev"]),
                    "test_rows": self._prepared_split_row_count(cache_dir, ["test"]),
                },
            }

        async with self._prepare_data_lock:
            if self._prepared_cache_ready() and not refresh:
                cache_dir = self._prepared_cache_dir()
                return {
                    "ok": True,
                    "cached": True,
                    "agent_id": agent_id,
                    "cache_dir": cache_dir,
                    "split_stats": {
                        "train_rows": self._prepared_split_row_count(cache_dir, ["train"]),
                        "dev_rows": self._prepared_split_row_count(cache_dir, ["validation", "val", "dev"]),
                        "test_rows": self._prepared_split_row_count(cache_dir, ["test"]),
                    },
                }
            try:
                run_log_dir = str(Path(self._task_workspace()) / "_prepared_agent_log")
                prepare_cache = await asyncio.wait_for(
                    asyncio.to_thread(self._ensure_prepared_data_cache, task=task, run_agent_log_dir=run_log_dir),
                    timeout=float(self._prepare_data_timeout_s),
                )
                cache_dir = str(prepare_cache.get("cache_dir") or "")
                return {
                    "ok": True,
                    "cached": False,
                    "agent_id": agent_id,
                    "cache_dir": cache_dir,
                    "prepare_stdout_tail": str(prepare_cache.get("prepare_stdout_tail") or ""),
                    "split_stats": {
                        "train_rows": self._prepared_split_row_count(cache_dir, ["train"]),
                        "dev_rows": self._prepared_split_row_count(cache_dir, ["validation", "val", "dev"]),
                        "test_rows": self._prepared_split_row_count(cache_dir, ["test"]),
                    },
                }
            except asyncio.TimeoutError:
                logger.warning(
                    f"prepare_data timed out after {self._prepare_data_timeout_s}s for task={task.get('task_name')}"
                )
                return {
                    "ok": False,
                    "reason": f"prepare_data_timeout:{self._prepare_data_timeout_s}s",
                    "agent_id": agent_id,
                }
            except Exception as e:
                logger.warning(f"prepare_data failed for task={task.get('task_name')}: {e}")
                return {"ok": False, "reason": f"prepare_data_failed:{e}", "agent_id": agent_id}

    async def retrieve_method_card(
        self,
        agent_id: Optional[str] = None,
        topic: Optional[str] = None,
        refresh: bool = False,
    ) -> Dict[str, Any]:
        if self._method_card_cache and not refresh:
            card = dict(self._method_card_cache)
            card["cache_hit"] = True
            card["agent_id"] = agent_id
            return card

        task = self._current_task
        if not task:
            await self.reset_episode()
            task = self._current_task
        if not task:
            return {"ok": False, "reason": "no_active_task"}

        info = task.get("logging_info") or {}
        metric = str(info.get("metric") or "")
        category = str(info.get("category") or "")
        research_problem = str(info.get("research_problem") or "")
        baselines = self._method_templates_for_task(metric=metric, category=category, research_problem=research_problem)

        evidence_refs = []
        for card in self._current_cards[:24]:
            title = str(card.get("title") or "").lower()
            text = str(card.get("text") or "")
            if ("baseline" in title) or ("method" in title) or ("metric" in title) or ("evaluation" in title):
                evidence_refs.append(
                    {
                        "citation_id": card.get("citation_id"),
                        "title": card.get("title"),
                        "snippet": self._truncate(text, 180),
                    }
                )
            if len(evidence_refs) >= self._method_card_topk:
                break
        if not evidence_refs:
            for card in self._current_cards[: self._method_card_topk]:
                evidence_refs.append(
                    {
                        "citation_id": card.get("citation_id"),
                        "title": card.get("title"),
                        "snippet": self._truncate(card.get("text"), 140),
                    }
                )

        method_card = {
            "ok": True,
            "card_type": "method_card",
            "task_name": task.get("task_name"),
            "topic": topic or "task_baselines",
            "metric": metric,
            "category": category,
            "research_problem": research_problem,
            "recommended_baselines": baselines[: self._method_card_topk],
            "evaluation_protocol": [
                "align dev optimization with official metric direction",
                "enforce submission schema checks each run",
                "track seed stability before promoting best run",
            ],
            "common_pitfalls": [
                "metric_mismatch_between_dev_and_official_eval",
                "format_invalid_submission",
                "insufficient_ablation_or_error_analysis",
            ],
            "task_evidence_refs": evidence_refs,
            "source": "local_method_library",
            "agent_id": agent_id,
        }
        method_dir = Path(self._task_workspace()) / "_analysis" / "method_card"
        method_dir.mkdir(parents=True, exist_ok=True)
        (method_dir / "method_card.json").write_text(
            json.dumps(method_card, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        self._method_card_cache = dict(method_card)
        return method_card

    async def publish_method_card(
        self,
        method_card: Optional[Dict[str, Any]] = None,
        agent_id: Optional[str] = None,
        refresh: bool = False,
    ) -> Dict[str, Any]:
        del refresh
        task = self._current_task
        if not task:
            await self.reset_episode()
            task = self._current_task
        if not task:
            return {"ok": False, "reason": "no_active_task", "agent_id": agent_id}
        if not isinstance(method_card, dict):
            return {"ok": False, "reason": "invalid_method_card", "agent_id": agent_id}

        card = dict(method_card)
        card["ok"] = bool(card.get("ok", True))
        card["card_type"] = "method_card"
        card.setdefault("version", "v2")
        card.setdefault("task_name", task.get("task_name"))
        card.setdefault("metric", str((task.get("logging_info") or {}).get("metric") or ""))
        card.setdefault("category", str((task.get("logging_info") or {}).get("category") or ""))
        card["agent_id"] = agent_id
        method_dir = Path(self._task_workspace()) / "_analysis" / "method_card"
        method_dir.mkdir(parents=True, exist_ok=True)
        (method_dir / "method_card.json").write_text(
            json.dumps(card, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        self._method_card_cache = dict(card)
        return {"ok": True, "method_card": card, "agent_id": agent_id}

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
        script_name = Path(script_path).name
        start_ts = datetime.utcnow()
        start_clock = datetime.now().timestamp()
        self._monitor(
            f"script_start name={script_name} timeout_s={timeout_s} cwd={cwd or os.getcwd()} "
            f"args={' '.join(args[:8])}"
        )

        stop_event = threading.Event()
        heartbeat_thread: Optional[threading.Thread] = None

        if self._runtime_monitor:
            def _heartbeat() -> None:
                while not stop_event.wait(self._runtime_heartbeat_s):
                    elapsed = datetime.now().timestamp() - start_clock
                    self._monitor(
                        f"script_running name={script_name} elapsed_s={elapsed:.1f} timeout_s={timeout_s}"
                    )

            heartbeat_thread = threading.Thread(target=_heartbeat, name=f"monitor-{script_name}", daemon=True)
            heartbeat_thread.start()

        try:
            proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout_s)
            elapsed = datetime.now().timestamp() - start_clock
            self._monitor(
                f"script_done name={script_name} rc={proc.returncode} elapsed_s={elapsed:.1f} "
                f"stdout_tail={len((proc.stdout or '')[-400:])} stderr_tail={len((proc.stderr or '')[-400:])}"
            )
            return proc.returncode, proc.stdout or "", proc.stderr or ""
        except subprocess.TimeoutExpired:
            elapsed = datetime.now().timestamp() - start_clock
            self._monitor(
                f"script_timeout name={script_name} elapsed_s={elapsed:.1f} timeout_s={timeout_s} "
                f"started_at={start_ts.isoformat()}Z"
            )
            raise
        finally:
            stop_event.set()
            if heartbeat_thread is not None:
                heartbeat_thread.join(timeout=0.2)

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

    def _resolve_scoring_column(self, info: Dict[str, Any], default: str = "prediction") -> str:
        raw = info.get("scoring_column")
        if isinstance(raw, list):
            for item in raw:
                name = str(item or "").strip()
                if name:
                    return name
            return default
        name = str(raw or "").strip()
        return name or default

    def _is_timeseries_submission_task(self, task: Dict[str, Any], metric: str, category: str) -> bool:
        task_name = str(task.get("task_name") or "").lower()
        metric_l = str(metric or "").lower()
        category_l = str(category or "").lower()
        return (
            "timeseries" in task_name
            or "time series" in category_l
            or "mase" in metric_l
        )

    def _build_timeseries_submission_rows(self, test_df, horizon: int) -> List[List[float]]:
        rows: List[List[float]] = []
        for _, row in test_df.iterrows():
            raw = row.get("target")
            seq: List[float] = []
            if isinstance(raw, list):
                seq = [float(x) for x in raw]
            elif isinstance(raw, tuple):
                seq = [float(x) for x in list(raw)]
            elif hasattr(raw, "tolist"):
                try:
                    seq = [float(x) for x in raw.tolist()]
                except Exception:
                    seq = []
            elif isinstance(raw, str):
                try:
                    parsed = json.loads(raw)
                    if isinstance(parsed, list):
                        seq = [float(x) for x in parsed]
                except Exception:
                    seq = []
            valid_hist = [x for x in seq if not (isinstance(x, float) and math.isnan(x))]
            base = valid_hist[-1] if valid_hist else 0.0
            pred = list(seq) + [float(base) for _ in range(max(1, int(horizon)))]
            rows.append(pred)
        return rows

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

    def _load_prepared_split_df(
        self,
        data_mount_dir: str,
        split_name: str,
        max_rows: Optional[int] = None,
        max_cols: Optional[int] = None,
        preferred_cols: Optional[List[str]] = None,
    ):
        from datasets import load_from_disk
        import pandas as pd

        split_path = Path(data_mount_dir) / split_name
        if not split_path.exists():
            return None
        dset = load_from_disk(str(split_path))
        col_limit = int(max_cols or 0)
        if col_limit > 0:
            col_names = list(getattr(dset, "column_names", []) or [])
            if len(col_names) > col_limit:
                preferred: List[str] = []
                for col in (preferred_cols or []):
                    c = str(col).strip()
                    if c and c in col_names and c not in preferred:
                        preferred.append(c)
                keep: List[str] = []
                keep.extend(preferred[:col_limit])
                if len(keep) < col_limit:
                    for col in col_names:
                        if col in keep:
                            continue
                        keep.append(col)
                        if len(keep) >= col_limit:
                            break
                drop = [c for c in col_names if c not in keep]
                if drop:
                    dset = dset.remove_columns(drop)
        row_limit = int(max_rows or 0)
        if row_limit > 0:
            try:
                total_rows = int(len(dset))
            except Exception:
                total_rows = 0
            if total_rows > row_limit:
                head_n = max(1, row_limit // 2)
                tail_n = max(0, row_limit - head_n)
                start_tail = max(head_n, total_rows - tail_n)
                sample_idx = list(range(head_n))
                if start_tail < total_rows:
                    sample_idx.extend(list(range(start_tail, total_rows)))
                dset = dset.select(sample_idx[:row_limit])
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

    def _safe_code_plan(self, config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        cfg = dict(config or {})
        raw = cfg.get("code_plan")
        if not isinstance(raw, dict):
            return {}
        files_raw = raw.get("files")
        files: List[Dict[str, str]] = []
        if isinstance(files_raw, list):
            for item in files_raw[:16]:
                if not isinstance(item, dict):
                    continue
                rel_path = str(item.get("path") or "").replace("\\", "/").strip()
                content = item.get("content")
                if not rel_path or not isinstance(content, str):
                    continue
                if rel_path.startswith("/") or rel_path.startswith("../") or "/../" in rel_path:
                    continue
                files.append({"path": rel_path[:220], "content": content[:120000]})
        run_cmd = str(raw.get("run_cmd") or "").strip()
        if not run_cmd:
            run_cmd = "python src/main.py --data-dir ./data --output-dir ./outputs --task-manifest ./.task_manifest.json"
        return {
            "run_cmd": run_cmd[:600],
            "files": files,
            "notes": str(raw.get("notes") or "")[:500],
        }

    def _run_code_submission(
        self,
        task: Dict[str, Any],
        data_mount_dir: str,
        agent_log_dir: str,
        config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        code_plan = self._safe_code_plan(config=config)
        if not code_plan:
            raise RuntimeError("code_plan_missing")
        if not bool(code_plan.get("files")):
            raise RuntimeError("code_plan_files_empty")

        workspace_dir = Path(agent_log_dir).parent / "workspace"
        manager = WorkspaceManager(str(workspace_dir))
        target_hint = ""
        train_df = self._load_prepared_split_df(data_mount_dir=data_mount_dir, split_name="train")
        if train_df is not None:
            target_hint = self._infer_target_column(task=task, train_df=train_df)
        use_docker = self._code_executor_backend == "docker"
        manager.bootstrap(
            task=task,
            data_mount_dir=data_mount_dir,
            target_column_hint=target_hint,
            prefer_symlink=(not use_docker),
        )
        manager.apply_files(code_plan.get("files") or [])
        pre_snap = manager.snapshot("before_run")
        base_run_cmd = str(code_plan.get("run_cmd") or "").strip()
        bridge_enabled = str(os.getenv("SCIMAS_CODE_DATA_BRIDGE_ENABLE", "1")).lower() not in {"0", "false", "no"}
        files_blob = "\n".join(
            str((item or {}).get("content") or "") for item in (code_plan.get("files") or []) if isinstance(item, dict)
        ).lower()
        csv_assumption_detected = any(
            tok in files_blob for tok in ("./data/train.csv", "./data/test.csv", "pd.read_csv(", "read_csv(")
        )
        use_data_bridge = bridge_enabled and csv_assumption_detected
        effective_run_cmd = (
            f"python src/_scimas_data_bridge.py && {base_run_cmd}" if use_data_bridge else base_run_cmd
        )

        fallback_used = False
        executor_name = "soft"
        if use_docker:
            executor_name = "docker"
            docker_executor = DockerExecutor(
                image=self._code_docker_image,
                timeout_s=self._code_run_timeout_s,
                memory_mb=self._code_memory_mb,
                cpus=self._code_docker_cpus,
                gpus=self._code_docker_gpus,
                docker_bin=self._code_docker_bin,
                forbid_network=self._code_forbid_network,
                keepalive=self._code_docker_keepalive,
            )
            run_result = docker_executor.run(
                command=effective_run_cmd,
                cwd=str(workspace_dir),
                env={"PYTHONUNBUFFERED": "1"},
            )
            if run_result.blocked and self._code_executor_fallback_soft:
                fallback_used = True
                executor_name = "soft_fallback"
                soft_executor = SandboxExecutor(
                    timeout_s=self._code_run_timeout_s,
                    cpu_limit_s=self._code_cpu_limit_s,
                    memory_mb=self._code_memory_mb,
                    forbid_network=self._code_forbid_network,
                )
                run_result = soft_executor.run(
                    command=effective_run_cmd,
                    cwd=str(workspace_dir),
                    env={"PYTHONUNBUFFERED": "1"},
                )
        else:
            executor_name = "soft"
            executor = SandboxExecutor(
                timeout_s=self._code_run_timeout_s,
                cpu_limit_s=self._code_cpu_limit_s,
                memory_mb=self._code_memory_mb,
                forbid_network=self._code_forbid_network,
            )
            run_result = executor.run(
                command=effective_run_cmd,
                cwd=str(workspace_dir),
                env={"PYTHONUNBUFFERED": "1"},
            )

        code_log_path = Path(agent_log_dir) / "code_run.json"
        code_log = {
            "task_name": task.get("task_name"),
            "executor_backend_config": self._code_executor_backend,
            "executor_used": executor_name,
            "executor_fallback_used": fallback_used,
            "csv_assumption_detected": csv_assumption_detected,
            "data_bridge_used": use_data_bridge,
            "effective_run_cmd": effective_run_cmd,
            "code_plan": code_plan,
            "run_result": run_result.to_dict(),
            "workspace_dir": str(workspace_dir),
            "snapshot_before_run": pre_snap,
        }
        with open(code_log_path, "w", encoding="utf-8") as f:
            json.dump(code_log, f, ensure_ascii=False, indent=2)
            f.write("\n")

        if run_result.exit_code != 0:
            raise RuntimeError(
                "code_execution_failed:"
                + (run_result.stderr or run_result.stdout or f"exit_code={run_result.exit_code}")[-1200:]
            )

        submission_candidates = [
            workspace_dir / "outputs" / "submission.csv",
            workspace_dir / "submission.csv",
        ]
        submission_src = None
        for cand in submission_candidates:
            if cand.exists():
                submission_src = cand
                break
        if submission_src is None:
            raise RuntimeError("submission_not_found_in_workspace")

        final_submission_path = Path(agent_log_dir) / "submission.csv"
        shutil.copy2(submission_src, final_submission_path)

        dev_eval = DevEvaluator(
            infer_target_column_fn=self._infer_target_column,
            compute_dev_metric_fn=self._compute_dev_metric,
            normalize_metric_fn=self._normalize_metric,
        ).evaluate(task=task, data_mount_dir=data_mount_dir, workspace_dir=str(workspace_dir))

        return {
            "submission_path": str(final_submission_path),
            "model_path": "",
            "solver_log_path": "",
            "dev_score": dev_eval.get("raw_score") if isinstance(dev_eval, dict) else None,
            "dev_score_norm": dev_eval.get("score_norm") if isinstance(dev_eval, dict) else None,
            "solver_mode": "code_agent",
            "stdout_tail": (run_result.stdout or "")[-1500:],
            "stderr_tail": (run_result.stderr or "")[-1500:],
            "exit_code": int(run_result.exit_code),
            "executor_used": executor_name,
            "executor_fallback_used": fallback_used,
            "code_workspace": str(workspace_dir),
            "code_log_path": str(code_log_path),
            "code_artifacts": list(run_result.artifacts),
            "dev_eval": dev_eval,
        }

    def _load_code_agent_failure_from_log(self, agent_log_dir: str) -> Dict[str, Any]:
        log_path = Path(agent_log_dir) / "code_run.json"
        if not log_path.exists():
            return {}
        try:
            payload = json.loads(log_path.read_text(encoding="utf-8"))
        except Exception:
            return {}
        run_result = payload.get("run_result")
        if not isinstance(run_result, dict):
            return {}
        return {
            "exit_code": run_result.get("exit_code"),
            "stdout_tail": str(run_result.get("stdout") or "")[-1500:],
            "stderr_tail": str(run_result.get("stderr") or "")[-1500:],
        }

    def _collect_split_columns(self, data_mount_dir: str, split_name: str) -> List[str]:
        cols: List[str] = []
        csv_path = Path(data_mount_dir) / f"{split_name}.csv"
        if csv_path.exists():
            try:
                with csv_path.open("r", encoding="utf-8", newline="") as f:
                    reader = csv.reader(f)
                    header = next(reader, [])
                cols = [str(c).strip() for c in header if str(c).strip()]
            except Exception:
                cols = []
        if cols:
            return cols[:256]

        split_dir = Path(data_mount_dir) / split_name
        if not split_dir.exists():
            return []
        try:
            from datasets import load_from_disk

            ds = load_from_disk(str(split_dir))
            cols = [str(c).strip() for c in (getattr(ds, "column_names", []) or []) if str(c).strip()]
        except Exception:
            cols = []
        return cols[:256]

    def _collect_data_columns(self, data_mount_dir: str) -> Dict[str, List[str]]:
        return {
            "train": self._collect_split_columns(data_mount_dir=data_mount_dir, split_name="train"),
            "test": self._collect_split_columns(data_mount_dir=data_mount_dir, split_name="test"),
        }

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
        category = str(info.get("category") or "")
        scoring_col = self._resolve_scoring_column(info, default="prediction")
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

        if self._is_timeseries_submission_task(task=task, metric=metric, category=category):
            horizon = int(max(1, int(os.getenv("SCIMAS_AIRS_TS_HORIZON", "59"))))
            rows = self._build_timeseries_submission_rows(test_df=test_df, horizon=horizon)
            pd.DataFrame([{scoring_col: json.dumps(r, ensure_ascii=False)} for r in rows]).to_csv(output_path, index=False)
            solver_log = {
                "solver_mode": "timeseries_naive_last",
                "scoring_col": scoring_col,
                "horizon": horizon,
                "test_rows": int(len(test_df)),
                "metric": metric,
            }
            with open(solver_log_path, "w", encoding="utf-8") as f:
                json.dump(solver_log, f, ensure_ascii=False, indent=2)
                f.write("\n")
            return {
                "submission_path": str(output_path),
                "model_path": "",
                "solver_log_path": str(solver_log_path),
                "dev_score": None,
                "dev_score_norm": None,
                "solver_spec": {"model_family": "naive_series"},
                "solver_mode": "iterative_solver",
            }

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
        scoring_col = self._resolve_scoring_column(info, default="prediction")
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

        if self._is_timeseries_submission_task(task=task, metric=metric, category=category):
            test_df = prepared.get("test")
            if test_df is None:
                raise RuntimeError("timeseries_submission_missing_test")
            try:
                test_df = test_df.to_pandas()
            except Exception:
                import pandas as pd
                test_df = pd.DataFrame(test_df[:])
            horizon = int(max(1, int(os.getenv("SCIMAS_AIRS_TS_HORIZON", "59"))))
            rows = self._build_timeseries_submission_rows(test_df=test_df, horizon=horizon)
            pd.DataFrame([{scoring_col: json.dumps(r, ensure_ascii=False)} for r in rows]).to_csv(output_path, index=False)
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
        del run_agent_log_dir
        if self._solver_prepare_once and self._prepared_cache_ready():
            return {"cache_dir": self._prepared_cache_dir(), "prepare_stdout_tail": "", "cache_scope": "episode"}
        if self._solver_prepare_once and self._prepared_data_cache_dir and (not Path(str(self._prepared_data_cache_dir)).exists()):
            self._prepared_data_cache_dir = None
        if self._solver_prepare_once:
            persistent_cache = self._try_load_prepared_cache_from_persistent(task)
            if persistent_cache:
                logger.info(
                    f"AIRS prepare cache hit (cross-run): task={str((task or {}).get('task_name') or '')} "
                    f"key={Path(persistent_cache).name}"
                )
                return {
                    "cache_dir": persistent_cache,
                    "prepare_stdout_tail": "",
                    "cache_scope": "cross_run_persistent",
                }

        base = Path(self._task_workspace())
        cache_data_dir = base / "_prepared_agent_data"
        cache_log_dir = base / "_prepared_agent_log"
        stage_data_dir = base / f"_prepared_agent_data.__staging__.{uuid.uuid4().hex[:8]}"
        stage_log_dir = base / f"_prepared_agent_log.__staging__.{uuid.uuid4().hex[:8]}"
        if stage_data_dir.exists():
            shutil.rmtree(stage_data_dir, ignore_errors=True)
        if stage_log_dir.exists():
            shutil.rmtree(stage_log_dir, ignore_errors=True)
        stage_data_dir.mkdir(parents=True, exist_ok=True)
        stage_log_dir.mkdir(parents=True, exist_ok=True)

        try:
            prepare_result = self._run_prepare_pipeline(
                task=task,
                agent_log_dir=str(stage_log_dir),
                data_mount_dir=str(stage_data_dir),
            )
            if cache_data_dir.exists():
                shutil.rmtree(cache_data_dir, ignore_errors=True)
            if cache_log_dir.exists():
                shutil.rmtree(cache_log_dir, ignore_errors=True)
            os.replace(str(stage_data_dir), str(cache_data_dir))
            os.replace(str(stage_log_dir), str(cache_log_dir))
        except Exception:
            if stage_data_dir.exists():
                shutil.rmtree(stage_data_dir, ignore_errors=True)
            if stage_log_dir.exists():
                shutil.rmtree(stage_log_dir, ignore_errors=True)
            raise
        final_cache_dir = str(cache_data_dir)
        if self._solver_prepare_once:
            persistent_cache = self._publish_prepared_cache_to_persistent(task=task, local_cache_dir=str(cache_data_dir))
            if persistent_cache:
                final_cache_dir = str(persistent_cache)
        self._prepared_data_cache_dir = final_cache_dir
        return {
            "cache_dir": self._prepared_data_cache_dir,
            "prepare_stdout_tail": str(prepare_result.get("prepare_stdout_tail") or ""),
            "cache_scope": "cross_run_persistent" if final_cache_dir != str(cache_data_dir) else "episode",
        }

    def _materialize_data_mount_from_cache(self, cache_dir: str, target_dir: str) -> None:
        src = Path(cache_dir)
        dst = Path(target_dir)
        if not src.exists():
            raise RuntimeError(f"Prepared cache missing: {cache_dir}")
        dst_parent = dst.parent
        dst_parent.mkdir(parents=True, exist_ok=True)
        staging = dst_parent / f"{dst.name}.__staging__.{uuid.uuid4().hex[:8]}"
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
        staging.mkdir(parents=True, exist_ok=True)
        try:
            # Idempotent materialization: copy each child independently and allow
            # re-entry without FileExistsError.
            for child in src.iterdir():
                src_child = src / child.name
                dst_child = staging / child.name
                if src_child.is_dir():
                    shutil.copytree(src_child, dst_child, dirs_exist_ok=True)
                elif src_child.exists():
                    shutil.copy2(src_child, dst_child)
            if dst.exists():
                shutil.rmtree(dst, ignore_errors=True)
            os.replace(str(staging), str(dst))
        except Exception as e:
            raise RuntimeError(f"materialize_cache_failed:{e}")
        finally:
            if staging.exists():
                shutil.rmtree(staging, ignore_errors=True)

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
        submission_path = os.path.join(data_mount_dir, "submission.csv")
        eval_meta = {
            "ts": datetime.utcnow().isoformat() + "Z",
            "episode_id": int(self._episode_id),
            "task_name": str(task.get("task_name") or ""),
            "task_metric": str((task.get("logging_info") or {}).get("metric") or ""),
            "agent_log_dir": str(agent_log_dir),
            "data_mount_dir": str(data_mount_dir),
            "submission_path": submission_path,
            "submission_exists": bool(os.path.exists(submission_path)),
        }

        rc, out2, err2 = self._run_script(eval_prepare_py, args, timeout_s=timeout_prepare)
        if rc != 0:
            self._append_jsonl_sync(
                self._eval_failure_log_path,
                {
                    **eval_meta,
                    "stage": "evaluate_prepare",
                    "rc": int(rc),
                    "error_type": self._classify_eval_error(err2),
                    "stdout_tail": str(out2 or "")[-1200:],
                    "stderr_tail": str(err2 or "")[-1600:],
                },
            )
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
                err_text = f"{out3}\n{err3}"
                self._append_jsonl_sync(
                    self._eval_failure_log_path,
                    {
                        **eval_meta,
                        "stage": "evaluate",
                        "rc": int(rc),
                        "error_type": self._classify_eval_error(err_text),
                        "stdout_tail": str(out3 or "")[-1200:],
                        "stderr_tail": str(err3 or "")[-1600:],
                    },
                )
                # Some environments miss sktime required by official evaluator.
                # Fallback to an internal MASE implementation for this specific task.
                if self._is_timeseries_submission_task(
                    task=task,
                    metric=str((task.get("logging_info") or {}).get("metric") or ""),
                    category=str((task.get("logging_info") or {}).get("category") or ""),
                ) and ("No module named 'sktime'" in err_text or 'No module named "sktime"' in err_text):
                    logger.warning("evaluate.py missing sktime; switching to internal timeseries fallback evaluator")
                    try:
                        return self._run_eval_pipeline_timeseries_fallback(data_mount_dir=data_mount_dir)
                    except Exception as fallback_exc:
                        logger.warning(f"timeseries fallback evaluator failed: {fallback_exc}")
                        # Never crash write on fallback path; return a finite penalty score.
                        penalty = float(1e6)
                        return {
                            "metric_name": str((task.get("logging_info") or {}).get("metric") or "MASE"),
                            "raw_score": penalty,
                            "score_norm": self._normalize_metric(raw_score=penalty, task=task),
                            "stdout_tail": f"fallback_eval_failed:{fallback_exc}",
                        }
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

    def _run_eval_pipeline_timeseries_fallback(self, data_mount_dir: str) -> Dict[str, Any]:
        import ast
        import numpy as np
        import pandas as pd
        from datasets import load_from_disk

        def _safe_literal_eval_with_nan(text: str) -> List[float]:
            fixed = str(text).replace("NaN", "None")
            obj = ast.literal_eval(fixed)
            if not isinstance(obj, list):
                raise ValueError("prediction is not a list")
            return [float(x) if x is not None else float("nan") for x in obj]

        def _mase(y_true: np.ndarray, y_pred: np.ndarray, y_train: np.ndarray) -> float:
            numer = float(np.mean(np.abs(y_true - y_pred)))
            train = np.asarray(y_train, dtype=float)
            if train.size <= 1:
                return float("inf")
            denom = float(np.mean(np.abs(np.diff(train))))
            if denom <= 1e-12:
                return float("inf")
            return numer / denom

        test_with_labels_path = Path(data_mount_dir) / "test_with_labels"
        submission_path = Path(data_mount_dir) / "submission.csv"
        if not test_with_labels_path.exists() or not submission_path.exists():
            penalty = float(1e6)
            task = self._current_task or {}
            return {
                "metric_name": "MASE",
                "raw_score": penalty,
                "score_norm": self._normalize_metric(raw_score=penalty, task=task),
                "stdout_tail": "fallback_eval_missing_inputs",
            }

        ds = load_from_disk(str(test_with_labels_path))
        labels = ds["label_target"]
        train_targets = ds["target"]
        submission_df = pd.read_csv(submission_path, header=0)
        preds = submission_df.values.squeeze()
        total_rows = min(int(getattr(preds, "shape", [0])[0] or 0), int(len(labels)))
        if total_rows <= 0:
            raise RuntimeError("timeseries_fallback_empty_submission")

        # Keep fallback cheap and deterministic; full 145k-row MASE can block one tick for minutes.
        max_rows = int(max(32, int(os.getenv("SCIMAS_TS_EVAL_FALLBACK_MAX_ROWS", "512"))))
        if total_rows <= max_rows:
            eval_indices = list(range(total_rows))
        else:
            step = float(total_rows - 1) / float(max_rows - 1)
            eval_indices = sorted({int(round(i * step)) for i in range(max_rows)})

        mases: List[float] = []
        bad_rows = 0
        for idx in eval_indices:
            pred_cell = preds[idx]
            label = labels[idx]
            train_target = train_targets[idx]
            try:
                pred = np.array(_safe_literal_eval_with_nan(pred_cell), dtype=float)
                label_arr = np.array(label, dtype=float)
                if pred.shape != label_arr.shape:
                    bad_rows += 1
                    continue

                train_arr = np.array(train_target, dtype=float)
                train_size = train_arr.shape[0]
                train_clean = train_arr[~np.isnan(train_arr)]
                pred_tail = pred[train_size:]
                label_tail = label_arr[train_size:]
                mask = ~np.isnan(label_tail)
                pred_tail = pred_tail[mask]
                label_tail = label_tail[mask]
                if label_tail.shape[0] == 0:
                    bad_rows += 1
                    continue
                mase_val = _mase(y_true=label_tail, y_pred=pred_tail, y_train=train_clean)
                if np.isfinite(mase_val):
                    mases.append(float(mase_val))
                else:
                    bad_rows += 1
            except Exception:
                bad_rows += 1
                continue

        # If everything failed, return a finite penalty score instead of raising,
        # so write-action can complete and the episode can progress.
        if mases:
            raw_score = float(np.mean(mases))
        else:
            raw_score = float(1e6)
        # Task is MASE (lower is better), normalize with the same helper.
        task = self._current_task or {}
        score_norm = self._normalize_metric(raw_score=raw_score, task=task)
        return {
            "metric_name": "MASE",
            "raw_score": raw_score,
            "score_norm": score_norm,
            "stdout_tail": (
                f"fallback_eval_used:internal_mase "
                f"rows={len(eval_indices)}/{total_rows} valid={len(mases)} bad={bad_rows}"
            ),
        }

    async def run_experiment(
        self,
        intervention: Optional[Dict[str, float]] = None,
        n_samples: int = 50,
        config: Optional[Dict[str, Any]] = None,
        agent_id: Optional[str] = None,
        current_tick: Optional[int] = None,
    ) -> Dict[str, Any]:
        del intervention, n_samples  # Unused in AIRS workflow; kept for compatibility.
        now_tick = self._normalize_tick(current_tick)
        task = self._current_task
        if not task:
            await self.reset_episode()
            task = self._current_task
        if not task:
            raise RuntimeError("No AIRS task selected")

        run_id = self._next_run_id()
        stop_reason = self._hard_stop_reason(current_tick=now_tick)
        if stop_reason:
            return {
                "run_id": run_id,
                "ts": datetime.utcnow().isoformat() + "Z",
                "task_name": task.get("task_name"),
                "agent_id": agent_id,
                "strategy": str((config or {}).get("strategy") or "default_baseline"),
                "ok": False,
                "submission_path": "",
                "metric_name": str((task.get("logging_info") or {}).get("metric") or ""),
                "raw_score": 0.0,
                "score_norm": 0.0,
                "dev_score": None,
                "dev_score_norm": None,
                "elapsed_s": 0.0,
                "cost": 0.0,
                "stdout_tail": "",
                "prepare_stdout_tail": "",
                "model_path": "",
                "solver_log_path": "",
                "solver_mode": "hard_stop",
                "fallback_reason": "",
                "error": f"hard_stop:{stop_reason}",
                "eval_split": "dev",
                "failure_stage": "prepare",
                "exit_code": None,
                "code_agent_attempted": False,
                "code_agent_ok": False,
                "code_agent_error": "",
                "code_agent_exit_code": None,
                "code_agent_stdout_tail": "",
                "code_agent_stderr_tail": "",
                "fallback_solver_used": False,
                "fallback_solver_ok": False,
                "data_columns": {"train": [], "test": []},
                "execution_path": "solver_only",
                "executor_diag": {
                    "backend": self._code_executor_backend,
                    "timeout_hit": False,
                    "memory_hit": False,
                    "killed": False,
                    "image": self._code_docker_image,
                    "command": "",
                },
            }
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
        stderr_tail = ""
        prepare_stdout_tail = ""
        fallback_reason = ""
        code_workspace = ""
        code_log_path = ""
        code_artifacts: List[str] = []
        dev_eval = None
        executor_used = ""
        executor_fallback_used = False
        failure_stage = "unknown"
        exit_code = None
        code_agent_attempted = False
        code_agent_ok = False
        code_agent_error = ""
        code_agent_exit_code = None
        code_agent_stdout_tail = ""
        code_agent_stderr_tail = ""
        fallback_solver_used = False
        fallback_solver_ok = False
        data_columns: Dict[str, List[str]] = {"train": [], "test": []}
        executor_diag: Dict[str, Any] = {
            "backend": self._code_executor_backend,
            "timeout_hit": False,
            "memory_hit": False,
            "killed": False,
            "image": self._code_docker_image,
            "command": "",
        }

        try:
            if self._prepared_cache_ready():
                prepare_cache = {"cache_dir": self._prepared_cache_dir(), "prepare_stdout_tail": ""}
            else:
                async with self._prepare_data_lock:
                    if self._prepared_cache_ready():
                        prepare_cache = {"cache_dir": self._prepared_cache_dir(), "prepare_stdout_tail": ""}
                    else:
                        prepare_cache = await asyncio.wait_for(
                            asyncio.to_thread(self._ensure_prepared_data_cache, task=task, run_agent_log_dir=str(agent_log)),
                            timeout=float(self._prepare_data_timeout_s),
                        )
            prepare_stdout_tail = str(prepare_cache.get("prepare_stdout_tail") or "")
            self._materialize_data_mount_from_cache(str(prepare_cache.get("cache_dir") or ""), str(data_mount))
            data_columns = self._collect_data_columns(data_mount_dir=str(data_mount))

            solver_result = None
            code_error = None
            if self._code_agent_enable:
                code_agent_attempted = True
                try:
                    code_result = self._run_code_submission(
                        task=task,
                        data_mount_dir=str(data_mount),
                        agent_log_dir=str(agent_log),
                        config=config or {},
                    )
                    if isinstance(code_result, dict) and code_result.get("submission_path"):
                        solver_result = code_result
                        code_agent_ok = True
                        code_agent_exit_code = int(code_result.get("exit_code", 0))
                        code_agent_stdout_tail = str(code_result.get("stdout_tail") or "")
                        code_agent_stderr_tail = str(code_result.get("stderr_tail") or "")
                except Exception as e:
                    code_error = str(e)
                    code_agent_ok = False
                    code_agent_error = code_error
                    code_failure = self._load_code_agent_failure_from_log(agent_log_dir=str(agent_log))
                    raw_exit = code_failure.get("exit_code")
                    if isinstance(raw_exit, int):
                        code_agent_exit_code = raw_exit
                    code_agent_stdout_tail = str(code_failure.get("stdout_tail") or "")[-1500:]
                    code_agent_stderr_tail = str(code_failure.get("stderr_tail") or "")[-1500:]
                    fallback_reason = f"code_agent:{code_error}"
                    failure_stage = "execute"
                    logger.info(f"AIRS code-agent fallback on {task.get('task_name')}/{run_id}: {code_error}")

            solver_error = None
            if solver_result is None and self._solver_enabled:
                fallback_solver_used = bool(code_agent_attempted and not code_agent_ok)
                try:
                    solver_result = self._run_solver_submission(
                        task=task,
                        data_mount_dir=str(data_mount),
                        agent_log_dir=str(agent_log),
                        config=config or {},
                    )
                    fallback_solver_ok = isinstance(solver_result, dict) and bool(solver_result.get("submission_path"))
                except Exception as e:
                    solver_error = str(e)
                    fallback_reason = solver_error
                    fallback_solver_ok = False
                    failure_stage = "execute"
                    logger.info(f"AIRS solver fallback on {task.get('task_name')}/{run_id}: {solver_error}")

            if isinstance(solver_result, dict) and solver_result.get("submission_path"):
                submission_path = str(solver_result.get("submission_path"))
                model_path = str(solver_result.get("model_path") or "")
                solver_log_path = str(solver_result.get("solver_log_path") or "")
                dev_score = solver_result.get("dev_score")
                dev_score_norm = solver_result.get("dev_score_norm")
                solver_mode = str(solver_result.get("solver_mode") or "iterative_solver")
                stdout_tail = str(solver_result.get("stdout_tail") or "")
                stderr_tail = str(solver_result.get("stderr_tail") or "")
                code_workspace = str(solver_result.get("code_workspace") or "")
                code_log_path = str(solver_result.get("code_log_path") or "")
                artifacts_val = solver_result.get("code_artifacts")
                if isinstance(artifacts_val, list):
                    code_artifacts = [str(x) for x in artifacts_val[:64]]
                dev_eval = solver_result.get("dev_eval")
                executor_used = str(solver_result.get("executor_used") or "")
                executor_fallback_used = bool(solver_result.get("executor_fallback_used", False))
                exit_code = solver_result.get("exit_code")
                failure_stage = str(solver_result.get("failure_stage") or "execute")
                diag_raw = solver_result.get("executor_diag")
                if isinstance(diag_raw, dict):
                    executor_diag.update(diag_raw)
                cmd_raw = solver_result.get("run_cmd")
                if isinstance(cmd_raw, str) and cmd_raw.strip():
                    executor_diag["command"] = cmd_raw.strip()[:300]
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
                failure_stage = "format"
                fallback_solver_used = bool(code_agent_attempted and not code_agent_ok)
                fallback_solver_ok = False

            metric_name = str((task.get("logging_info") or {}).get("metric") or "")
            if isinstance(dev_score, (int, float)):
                raw_score = float(dev_score)
                if isinstance(dev_score_norm, (int, float)):
                    score_norm = float(dev_score_norm)
                else:
                    score_norm = float(self._normalize_metric(raw_score=raw_score, task=task))
            elif isinstance(dev_score_norm, (int, float)):
                score_norm = float(dev_score_norm)
                raw_score = float(dev_score_norm)
        except Exception as e:
            ok = False
            error = str(e)
            if failure_stage == "unknown":
                failure_stage = "execute"
            logger.warning(f"AIRS run_experiment failed on {task.get('task_name')}/{run_id}: {e}")

        ended = datetime.utcnow()
        elapsed_s = max(0.0, (ended - started).total_seconds())
        merged_err = (
            f"{error or ''}\n{stderr_tail or ''}\n{stdout_tail or ''}\n"
            f"{code_agent_error or ''}\n{code_agent_stderr_tail or ''}\n{code_agent_stdout_tail or ''}"
        ).lower()
        if "timed out" in merged_err or "timeout" in merged_err:
            executor_diag["timeout_hit"] = True
        if "out of memory" in merged_err or "oom" in merged_err:
            executor_diag["memory_hit"] = True
        if "killed" in merged_err:
            executor_diag["killed"] = True
        if code_agent_attempted and code_agent_ok:
            execution_path = "code_agent_only"
        elif code_agent_attempted:
            execution_path = "code_agent_fail_fallback_solver"
        else:
            execution_path = "solver_only"
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
            "stderr_tail": stderr_tail,
            "prepare_stdout_tail": prepare_stdout_tail,
            "model_path": model_path,
            "solver_log_path": solver_log_path,
            "solver_mode": solver_mode,
            "code_workspace": code_workspace,
            "code_log_path": code_log_path,
            "code_artifacts": code_artifacts,
            "dev_eval": dev_eval,
            "executor_used": executor_used,
            "executor_fallback_used": executor_fallback_used,
            "fallback_reason": fallback_reason,
            "error": error,
            "eval_split": "dev",
            "failure_stage": failure_stage,
            "exit_code": exit_code if isinstance(exit_code, int) else None,
            "code_agent_attempted": bool(code_agent_attempted),
            "code_agent_ok": bool(code_agent_ok),
            "code_agent_error": str(code_agent_error or ""),
            "code_agent_exit_code": code_agent_exit_code if isinstance(code_agent_exit_code, int) else None,
            "code_agent_stdout_tail": str(code_agent_stdout_tail or ""),
            "code_agent_stderr_tail": str(code_agent_stderr_tail or ""),
            "fallback_solver_used": bool(fallback_solver_used),
            "fallback_solver_ok": bool(fallback_solver_ok),
            "data_columns": data_columns if isinstance(data_columns, dict) else {"train": [], "test": []},
            "execution_path": execution_path,
            "executor_diag": executor_diag,
        }
        self._score_cache.append(run_record)
        return run_record

    async def evaluate_submission(self, submission_path: str) -> Dict[str, Any]:
        task = self._current_task
        if not task:
            return {"ok": False, "reason": "no_active_task"}
        run_id = self._next_run_id()
        if not submission_path or not os.path.exists(submission_path):
            return {
                "ok": False,
                "reason": "submission_not_found",
                "submission_path": submission_path,
                "run_id": run_id,
                "error_type": "format_error",
                "failure_stage": "preflight_schema",
            }

        preflight = self._preflight_submission_schema(task=task, submission_path=submission_path)
        if not bool(preflight.get("ok")):
            payload = {
                "ok": False,
                "reason": f"format_error:{preflight.get('error_code')}",
                "submission_path": submission_path,
                "run_id": run_id,
                "error_type": "format_error",
                "failure_stage": "preflight_schema",
                "preflight": preflight,
                "cache_hit": False,
            }
            try:
                sub_hash = self._file_sha256(submission_path)
                cache_key = self._eval_cache_key(task=task, submission_hash=sub_hash)
                payload["submission_hash"] = sub_hash
                payload["cache_key"] = cache_key
                self._append_eval_cache_sync(cache_key=cache_key, payload=payload)
            except Exception:
                pass
            return payload

        submission_hash = self._file_sha256(submission_path)
        cache_key = self._eval_cache_key(task=task, submission_hash=submission_hash)
        if self._eval_cache_enable and cache_key in self._eval_cache:
            cached = dict(self._eval_cache.get(cache_key) or {})
            cached["run_id"] = run_id
            cached["cache_hit"] = True
            cached["cache_key"] = cache_key
            cached["submission_hash"] = submission_hash
            return cached

        dep_check = self._evaluate_dependency_precheck(task=task)
        if not bool(dep_check.get("ok")):
            payload = {
                "ok": False,
                "reason": f"system_error:{dep_check.get('error_code')}",
                "submission_path": submission_path,
                "run_id": run_id,
                "error_type": "system_error",
                "failure_stage": "environment_precheck",
                "dependency_check": dep_check,
                "cache_hit": False,
                "cache_key": cache_key,
                "submission_hash": submission_hash,
            }
            self._append_eval_cache_sync(cache_key=cache_key, payload=payload)
            return payload

        run_dir = Path(self._task_workspace()) / f"eval_{run_id}"
        data_mount = run_dir / "agent_data"
        agent_log = run_dir / "agent_log"
        data_mount.mkdir(parents=True, exist_ok=True)
        agent_log.mkdir(parents=True, exist_ok=True)

        shutil.copy2(submission_path, agent_log / "submission.csv")

        try:
            eval_result = self._run_eval_pipeline(task=task, agent_log_dir=str(agent_log), data_mount_dir=str(data_mount))
            payload = {
                "ok": True,
                "run_id": run_id,
                "task_name": task.get("task_name"),
                "submission_path": str(agent_log / "submission.csv"),
                "metric_name": eval_result["metric_name"],
                "raw_score": float(eval_result["raw_score"]),
                "score_norm": float(eval_result["score_norm"]),
                "stdout_tail": eval_result.get("stdout_tail"),
                "cache_hit": False,
                "cache_key": cache_key,
                "submission_hash": submission_hash,
                "preflight": preflight,
            }
            self._append_eval_cache_sync(cache_key=cache_key, payload=payload)
            return payload
        except Exception as e:
            payload = {
                "ok": False,
                "reason": str(e),
                "run_id": run_id,
                "error_type": self._classify_eval_error(str(e)),
                "failure_stage": "evaluate_submission",
                "cache_hit": False,
                "cache_key": cache_key,
                "submission_hash": submission_hash,
                "preflight": preflight,
            }
            self._append_eval_cache_sync(cache_key=cache_key, payload=payload)
            return payload

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
