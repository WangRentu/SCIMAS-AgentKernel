import json
import os
import shutil
import subprocess
import time
import csv
import importlib.util
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

try:
    import resource  # type: ignore
except Exception:  # pragma: no cover
    resource = None


@dataclass
class RunResult:
    command: str
    exit_code: int
    stdout: str
    stderr: str
    duration_s: float
    timed_out: bool
    blocked: bool
    artifacts: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "command": self.command,
            "exit_code": self.exit_code,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "duration_s": self.duration_s,
            "timed_out": self.timed_out,
            "blocked": self.blocked,
            "artifacts": list(self.artifacts),
        }


class WorkspaceManager:
    """Manage per-run code workspace with safe file operations and snapshots."""

    def __init__(self, workspace_dir: str):
        self.root = Path(workspace_dir).resolve()

    def _safe_rel_path(self, rel_path: str) -> Path:
        rel = str(rel_path or "").replace("\\", "/").strip()
        if not rel:
            raise ValueError("empty_path")
        if rel.startswith("/") or rel.startswith("../") or "/../" in rel:
            raise ValueError(f"unsafe_path:{rel}")
        target = (self.root / rel).resolve()
        if not str(target).startswith(str(self.root)):
            raise ValueError(f"path_escape:{rel}")
        return target

    def bootstrap(
        self,
        task: Dict[str, Any],
        data_mount_dir: str,
        target_column_hint: str = "",
        prefer_symlink: bool = True,
    ) -> Dict[str, Any]:
        self.root.mkdir(parents=True, exist_ok=True)
        for name in ("src", "outputs", "logs", "snapshots"):
            (self.root / name).mkdir(parents=True, exist_ok=True)

        source = Path(data_mount_dir).resolve()
        split_info = self._collect_data_split_info(source)
        manifest = {
            "task_name": task.get("task_name"),
            "task_path": task.get("task_path"),
            "metric": (task.get("logging_info") or {}).get("metric"),
            "category": (task.get("logging_info") or {}).get("category"),
            "dataset": (task.get("logging_info") or {}).get("dataset"),
            "scoring_column": (task.get("logging_info") or {}).get("scoring_column", "prediction"),
            "target_column_hint": target_column_hint or "",
            "available_splits": split_info.get("available_splits") or [],
            "prepare_split_manifest": split_info.get("prepare_split_manifest") or {},
            "dev_split_name": split_info.get("dev_split_name") or "",
            "dev_split_from_train": bool(split_info.get("dev_split_from_train", False)),
        }
        (self.root / ".task_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        data_link = self.root / "data"
        if data_link.exists() or data_link.is_symlink():
            if data_link.is_dir() and not data_link.is_symlink():
                shutil.rmtree(data_link, ignore_errors=True)
            else:
                data_link.unlink(missing_ok=True)
        if prefer_symlink:
            try:
                os.symlink(str(source), str(data_link))
            except Exception:
                shutil.copytree(source, data_link)
        else:
            shutil.copytree(source, data_link)
        self._materialize_csv_compat_views(data_link)

        starter = self.root / "src" / "main.py"
        if not starter.exists():
            starter.write_text(_default_starter_script(), encoding="utf-8")
        bridge = self.root / "src" / "_scimas_data_bridge.py"
        if not bridge.exists():
            bridge.write_text(_data_bridge_script(), encoding="utf-8")
        return {"workspace_dir": str(self.root), "manifest_path": str(self.root / ".task_manifest.json")}

    def _collect_data_split_info(self, data_root: Path) -> Dict[str, Any]:
        available_splits: List[str] = []
        for split in ("train", "validation", "val", "dev", "test", "test_with_labels"):
            split_path = data_root / split
            csv_path = data_root / f"{split}.csv"
            if split_path.exists() or csv_path.exists():
                available_splits.append(split)

        split_manifest: Dict[str, Any] = {}
        manifest_path = data_root / "_prepare_split_manifest.json"
        if manifest_path.exists():
            try:
                loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    split_manifest = loaded
            except Exception:
                split_manifest = {}

        dev_split_name = str(split_manifest.get("dev_split_name") or "")
        if not dev_split_name:
            for cand in ("validation", "val", "dev"):
                if cand in available_splits:
                    dev_split_name = cand
                    break
        return {
            "available_splits": available_splits,
            "prepare_split_manifest": split_manifest,
            "dev_split_name": dev_split_name,
            "dev_split_from_train": bool(split_manifest.get("created_from_train", False)),
        }

    def _materialize_csv_compat_views(self, data_root: Path) -> None:
        """
        Best-effort compatibility bridge for code agents that assume CSV files.
        AIRS prepare.py often writes HuggingFace datasets via `save_to_disk` under
        ./data/train and ./data/test. This exports train.csv/test.csv if missing.
        """
        for split in ("train", "test", "validation", "val"):
            split_dir = data_root / split
            csv_path = data_root / f"{split}.csv"
            if csv_path.exists() or not split_dir.exists() or not split_dir.is_dir():
                continue
            marker_state = split_dir / "state.json"
            marker_info = split_dir / "dataset_info.json"
            if not marker_state.exists() or not marker_info.exists():
                continue
            try:
                self._export_hf_split_to_csv(split_dir=split_dir, csv_path=csv_path)
            except Exception:
                # Compatibility export is optional; keep workspace bootstrapping resilient.
                continue

    def _export_hf_split_to_csv(self, split_dir: Path, csv_path: Path) -> None:
        from datasets import load_from_disk

        ds = load_from_disk(str(split_dir))
        columns = list(getattr(ds, "column_names", []) or [])
        if not columns:
            return

        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            for row in ds:
                out: Dict[str, Any] = {}
                for col in columns:
                    value = row.get(col)
                    if isinstance(value, (dict, list, tuple)):
                        out[col] = json.dumps(value, ensure_ascii=False)
                    else:
                        out[col] = value
                writer.writerow(out)

    def apply_files(self, files: List[Dict[str, Any]]) -> List[str]:
        changed: List[str] = []
        for item in files:
            if not isinstance(item, dict):
                continue
            rel_path = str(item.get("path") or "").strip()
            content = item.get("content")
            if not rel_path or not isinstance(content, str):
                continue
            path = self._safe_rel_path(rel_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
            changed.append(str(path.relative_to(self.root)))
        return changed

    def snapshot(self, label: str) -> str:
        safe_label = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(label or "snapshot"))[:80]
        ts = int(time.time())
        snap_dir = self.root / "snapshots" / f"{ts}_{safe_label}"
        snap_dir.mkdir(parents=True, exist_ok=True)
        src_dir = self.root / "src"
        if src_dir.exists():
            shutil.copytree(src_dir, snap_dir / "src", dirs_exist_ok=True)
        return str(snap_dir)


class SandboxExecutor:
    """Soft sandbox executor with timeout/resource limits and command deny-list."""

    def __init__(
        self,
        timeout_s: int = 180,
        cpu_limit_s: int = 120,
        memory_mb: int = 2048,
        forbid_network: bool = True,
    ):
        self.timeout_s = int(max(5, timeout_s))
        self.cpu_limit_s = int(max(1, cpu_limit_s))
        self.memory_mb = int(max(256, memory_mb))
        self.forbid_network = bool(forbid_network)
        self._deny_tokens = (
            "curl ",
            "wget ",
            "pip install",
            "conda install",
            "apt ",
            "apt-get ",
            "git clone",
            "ssh ",
            "scp ",
            "nc ",
        )

    def _preexec(self):  # pragma: no cover - preexec path hard to test here
        if resource is None:
            return
        try:
            resource.setrlimit(resource.RLIMIT_CPU, (self.cpu_limit_s, self.cpu_limit_s + 1))
        except Exception:
            pass
        try:
            limit_bytes = int(self.memory_mb) * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_AS, (limit_bytes, limit_bytes))
        except Exception:
            pass
        try:
            resource.setrlimit(resource.RLIMIT_NOFILE, (256, 256))
        except Exception:
            pass

    def _collect_artifacts(self, cwd: str) -> List[str]:
        root = Path(cwd)
        if not root.exists():
            return []
        artifacts: List[str] = []
        allow_suffix = {".csv", ".json", ".txt", ".md", ".png", ".jpg", ".jpeg", ".pkl"}
        for p in root.rglob("*"):
            if len(artifacts) >= 64:
                break
            if not p.is_file():
                continue
            rel = str(p.relative_to(root))
            if rel.startswith("snapshots/"):
                continue
            if p.suffix.lower() in allow_suffix:
                artifacts.append(rel)
        return artifacts

    def run(self, command: str, cwd: str, env: Optional[Dict[str, str]] = None) -> RunResult:
        cmd = str(command or "").strip()
        if not cmd:
            return RunResult(
                command=cmd,
                exit_code=2,
                stdout="",
                stderr="empty_command",
                duration_s=0.0,
                timed_out=False,
                blocked=True,
                artifacts=[],
            )
        lowered = cmd.lower()
        if self.forbid_network and any(tok in lowered for tok in self._deny_tokens):
            return RunResult(
                command=cmd,
                exit_code=126,
                stdout="",
                stderr="blocked_command_by_sandbox_policy",
                duration_s=0.0,
                timed_out=False,
                blocked=True,
                artifacts=[],
            )

        merged_env = dict(os.environ)
        if env:
            merged_env.update({str(k): str(v) for k, v in env.items()})
        if self.forbid_network:
            merged_env["SCIMAS_NO_NETWORK"] = "1"

        start = time.time()
        try:
            proc = subprocess.run(
                ["bash", "-lc", cmd],
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=self.timeout_s,
                env=merged_env,
                preexec_fn=self._preexec if hasattr(os, "fork") else None,
            )
            duration = max(0.0, time.time() - start)
            return RunResult(
                command=cmd,
                exit_code=int(proc.returncode),
                stdout=proc.stdout or "",
                stderr=proc.stderr or "",
                duration_s=duration,
                timed_out=False,
                blocked=False,
                artifacts=self._collect_artifacts(cwd),
            )
        except subprocess.TimeoutExpired as e:
            duration = max(0.0, time.time() - start)
            return RunResult(
                command=cmd,
                exit_code=124,
                stdout=(e.stdout or ""),
                stderr=(e.stderr or "") + "\nTIMEOUT_EXPIRED",
                duration_s=duration,
                timed_out=True,
                blocked=False,
                artifacts=self._collect_artifacts(cwd),
            )


class DockerExecutor:
    """Harder isolation executor using Docker containers."""

    def __init__(
        self,
        *,
        image: str,
        timeout_s: int = 180,
        memory_mb: int = 2048,
        cpus: float = 2.0,
        gpus: str = "",
        docker_bin: str = "docker",
        forbid_network: bool = True,
        keepalive: bool = False,
    ):
        self.image = str(image or "").strip()
        self.timeout_s = int(max(5, timeout_s))
        self.memory_mb = int(max(256, memory_mb))
        self.cpus = max(0.1, float(cpus))
        self.gpus = str(gpus or "").strip()
        self.docker_bin = str(docker_bin or "docker").strip()
        self.forbid_network = bool(forbid_network)
        self.keepalive = bool(keepalive)

    def _docker_available(self) -> bool:
        if not self.image:
            return False
        if shutil.which(self.docker_bin) is None:
            return False
        return True

    def _collect_artifacts(self, cwd: str) -> List[str]:
        root = Path(cwd)
        if not root.exists():
            return []
        artifacts: List[str] = []
        allow_suffix = {".csv", ".json", ".txt", ".md", ".png", ".jpg", ".jpeg", ".pkl"}
        for p in root.rglob("*"):
            if len(artifacts) >= 64:
                break
            if not p.is_file():
                continue
            rel = str(p.relative_to(root))
            if rel.startswith("snapshots/"):
                continue
            if p.suffix.lower() in allow_suffix:
                artifacts.append(rel)
        return artifacts

    def _container_name(self, host_cwd: str) -> str:
        digest = hashlib.sha1(f"{self.image}|{host_cwd}".encode("utf-8")).hexdigest()[:12]
        return f"scimas_exec_{digest}"

    def _inspect_state(self, name: str) -> Optional[str]:
        proc = subprocess.run(
            [self.docker_bin, "inspect", "-f", "{{.State.Status}}", name],
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            return None
        return (proc.stdout or "").strip().lower()

    def _ensure_keepalive_container(self, host_cwd: str) -> Optional[str]:
        name = self._container_name(host_cwd)
        state = self._inspect_state(name)
        if state == "running":
            return name
        if state in {"created", "exited"}:
            start_proc = subprocess.run(
                [self.docker_bin, "start", name],
                capture_output=True,
                text=True,
            )
            if start_proc.returncode == 0:
                return name
            return None

        run_cmd = [
            self.docker_bin,
            "run",
            "-d",
            "--name",
            name,
            "-m",
            f"{self.memory_mb}m",
            "--cpus",
            f"{self.cpus:.2f}",
            "-v",
            f"{host_cwd}:/workspace",
            "-w",
            "/workspace",
        ]
        if self.gpus:
            run_cmd.extend(["--gpus", self.gpus])
        if self.forbid_network:
            run_cmd.extend(["--network", "none"])
        run_cmd.extend([self.image, "sh", "-lc", "tail -f /dev/null"])
        proc = subprocess.run(run_cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            return None
        return name

    def run(self, command: str, cwd: str, env: Optional[Dict[str, str]] = None) -> RunResult:
        cmd = str(command or "").strip()
        if not cmd:
            return RunResult(
                command=cmd,
                exit_code=2,
                stdout="",
                stderr="empty_command",
                duration_s=0.0,
                timed_out=False,
                blocked=True,
                artifacts=[],
            )
        if not self._docker_available():
            return RunResult(
                command=cmd,
                exit_code=127,
                stdout="",
                stderr="docker_unavailable_or_image_missing",
                duration_s=0.0,
                timed_out=False,
                blocked=True,
                artifacts=[],
            )

        host_cwd = str(Path(cwd).resolve())
        if self.keepalive:
            name = self._ensure_keepalive_container(host_cwd)
            if not name:
                return RunResult(
                    command=cmd,
                    exit_code=127,
                    stdout="",
                    stderr="docker_keepalive_container_unavailable",
                    duration_s=0.0,
                    timed_out=False,
                    blocked=True,
                    artifacts=[],
                )
            docker_cmd = [self.docker_bin, "exec"]
            if env:
                for key, value in env.items():
                    docker_cmd.extend(["-e", f"{str(key)}={str(value)}"])
            docker_cmd.extend(["-w", "/workspace", name, "sh", "-lc", cmd])
        else:
            docker_cmd = [
                self.docker_bin,
                "run",
                "--rm",
                "-m",
                f"{self.memory_mb}m",
                "--cpus",
                f"{self.cpus:.2f}",
                "-v",
                f"{host_cwd}:/workspace",
                "-w",
                "/workspace",
            ]
            if self.gpus:
                docker_cmd.extend(["--gpus", self.gpus])
            if self.forbid_network:
                docker_cmd.extend(["--network", "none"])
            if env:
                for key, value in env.items():
                    docker_cmd.extend(["-e", f"{str(key)}={str(value)}"])
            docker_cmd.extend([self.image, "sh", "-lc", cmd])

        start = time.time()
        try:
            proc = subprocess.run(
                docker_cmd,
                cwd=host_cwd,
                capture_output=True,
                text=True,
                timeout=self.timeout_s,
            )
            duration = max(0.0, time.time() - start)
            return RunResult(
                command=cmd,
                exit_code=int(proc.returncode),
                stdout=proc.stdout or "",
                stderr=proc.stderr or "",
                duration_s=duration,
                timed_out=False,
                blocked=False,
                artifacts=self._collect_artifacts(host_cwd),
            )
        except subprocess.TimeoutExpired as e:
            duration = max(0.0, time.time() - start)
            return RunResult(
                command=cmd,
                exit_code=124,
                stdout=(e.stdout or ""),
                stderr=(e.stderr or "") + "\nDOCKER_TIMEOUT_EXPIRED",
                duration_s=duration,
                timed_out=True,
                blocked=False,
                artifacts=self._collect_artifacts(host_cwd),
            )


class DevEvaluator:
    """Evaluate dev split predictions without exposing test labels."""

    def __init__(
        self,
        *,
        infer_target_column_fn: Callable[[Dict[str, Any], Any], str],
        compute_dev_metric_fn: Callable[[str, Any, Any], Optional[float]],
        normalize_metric_fn: Callable[[float, Dict[str, Any]], float],
    ):
        self._infer_target_column_fn = infer_target_column_fn
        self._compute_dev_metric_fn = compute_dev_metric_fn
        self._normalize_metric_fn = normalize_metric_fn

    def _load_df(self, path: Path):
        import pandas as pd

        suffix = path.suffix.lower()
        if suffix == ".csv":
            return pd.read_csv(path)
        if suffix == ".parquet":
            return pd.read_parquet(path)
        if suffix in {".jsonl", ".json"}:
            return pd.read_json(path, lines=(suffix == ".jsonl"))
        return None

    def _find_split(self, data_dir: Path, names: List[str]):
        for name in names:
            split_dir = data_dir / name
            if split_dir.exists() and split_dir.is_dir():
                try:
                    from datasets import load_from_disk

                    ds = load_from_disk(str(split_dir))
                    try:
                        return ds.to_pandas()
                    except Exception:
                        import pandas as pd

                        return pd.DataFrame(ds[:])
                except Exception:
                    pass
            for ext in (".csv", ".parquet", ".jsonl", ".json"):
                path = data_dir / f"{name}{ext}"
                if path.exists():
                    return self._load_df(path)
        return None

    def _find_dev_predictions(self, workspace_dir: Path):
        metrics_path = workspace_dir / "outputs" / "dev_metrics.json"
        if metrics_path.exists():
            try:
                data = json.loads(metrics_path.read_text(encoding="utf-8"))
                return data, metrics_path
            except Exception:
                return None, metrics_path
        codegen_metrics_path = workspace_dir / "outputs" / "dev_metrics_codegen.json"
        if codegen_metrics_path.exists():
            try:
                data = json.loads(codegen_metrics_path.read_text(encoding="utf-8"))
                return data, codegen_metrics_path
            except Exception:
                return None, codegen_metrics_path

        candidates = [
            workspace_dir / "outputs" / "dev_predictions.csv",
            workspace_dir / "outputs" / "dev_predictions.parquet",
            workspace_dir / "dev_predictions.csv",
            workspace_dir / "dev_predictions.parquet",
        ]
        for path in candidates:
            if path.exists():
                return self._load_df(path), path
        return None, None

    def _evaluate_retrieval_dev(self, *, task: Dict[str, Any], data_dir: Path, workspace_dir: Path) -> Dict[str, Any]:
        import numpy as np
        import pandas as pd

        queries_dir = data_dir / "validation" / "queries_with_labels"
        if not queries_dir.exists():
            queries_dir = data_dir / "dev" / "queries_with_labels"
        if not queries_dir.exists():
            return {"ok": False, "reason": "retrieval_dev_queries_with_labels_missing"}

        try:
            from datasets import load_from_disk

            labels_ds = load_from_disk(str(queries_dir))
            labels_df = labels_ds.to_pandas()
        except Exception as e:
            return {"ok": False, "reason": f"retrieval_dev_labels_load_failed:{e}"}

        pred_paths = [
            workspace_dir / "outputs" / "dev_retrieval.csv",
            workspace_dir / "outputs" / "dev_predictions.csv",
            workspace_dir / "dev_retrieval.csv",
            workspace_dir / "dev_predictions.csv",
        ]
        pred_df = None
        pred_path = None
        for p in pred_paths:
            if p.exists():
                pred_df = pd.read_csv(p)
                pred_path = p
                break
        if pred_df is None:
            return {"ok": False, "reason": "retrieval_dev_predictions_not_found"}
        if "query" not in pred_df.columns or "rankings" not in pred_df.columns:
            return {"ok": False, "reason": "retrieval_dev_predictions_missing_query_or_rankings"}
        if "query" not in labels_df.columns or "id" not in labels_df.columns:
            return {"ok": False, "reason": "retrieval_dev_labels_missing_query_or_id"}

        label_map = {}
        for _, row in labels_df.iterrows():
            label_map[str(row["query"])] = row["id"]

        scores: List[float] = []
        matched = 0
        for _, row in pred_df.iterrows():
            q = str(row.get("query", ""))
            if q not in label_map:
                continue
            matched += 1
            gold = label_map[q]
            raw_rankings = row.get("rankings", "[]")
            rankings = []
            try:
                if isinstance(raw_rankings, str):
                    parsed = json.loads(raw_rankings)
                else:
                    parsed = raw_rankings
                if isinstance(parsed, list):
                    rankings = parsed
            except Exception:
                rankings = []
            score = 0.0
            for idx, cand in enumerate(rankings):
                if str(cand) == str(gold):
                    score = 1.0 / float(idx + 1)
                    break
            scores.append(score)

        if matched <= 0:
            return {"ok": False, "reason": "retrieval_dev_no_matched_queries"}
        raw_score = float(np.mean(scores)) if scores else 0.0
        score_norm = self._normalize_metric_fn(raw_score, task)
        return {
            "ok": True,
            "raw_score": raw_score,
            "score_norm": float(score_norm),
            "rows": int(matched),
            "pred_path": str(pred_path) if pred_path else "",
            "source": "retrieval_dev_predictions",
        }

    def _evaluate_codegen_dev(self, *, task: Dict[str, Any], data_dir: Path, workspace_dir: Path) -> Dict[str, Any]:
        import pandas as pd

        pred_paths = [
            workspace_dir / "outputs" / "dev_codegen.csv",
            workspace_dir / "outputs" / "dev_predictions.csv",
            workspace_dir / "dev_codegen.csv",
            workspace_dir / "dev_predictions.csv",
        ]
        pred_df = None
        pred_path = None
        for p in pred_paths:
            if p.exists():
                pred_df = pd.read_csv(p)
                pred_path = p
                break
        if pred_df is None:
            return {"ok": False, "reason": "codegen_dev_predictions_not_found"}

        required_cols = [f"code{i}" for i in range(1, 6)]
        if any(col not in pred_df.columns for col in required_cols):
            return {"ok": False, "reason": "codegen_dev_predictions_missing_code1_to_code5"}

        train_dir = data_dir / "train"
        if not train_dir.exists():
            return {"ok": False, "reason": "codegen_train_split_not_found"}
        try:
            from datasets import load_from_disk

            train_ds = load_from_disk(str(train_dir))
        except Exception as e:
            return {"ok": False, "reason": f"codegen_train_load_failed:{e}"}

        n = min(int(pred_df.shape[0]), int(len(train_ds)))
        if n <= 0:
            return {"ok": False, "reason": "codegen_dev_empty_predictions_or_train"}

        dev_cap = int(max(1, min(64, int(os.getenv("SCIMAS_CODEGEN_DEV_MAX_SAMPLES", "24")))))
        n = min(n, dev_cap)

        task_path = str(task.get("task_path") or "")
        utils_path = Path(task_path) / "utils.py"
        if not utils_path.exists():
            return {"ok": False, "reason": "codegen_utils_missing"}
        try:
            spec = importlib.util.spec_from_file_location("airs_codegen_utils", str(utils_path))
            if spec is None or spec.loader is None:
                raise RuntimeError("import_spec_failed")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            eval_fn = getattr(module, "evaluate_all_testcases", None)
            if eval_fn is None:
                raise RuntimeError("evaluate_all_testcases_missing")
        except Exception as e:
            return {"ok": False, "reason": f"codegen_utils_import_failed:{e}"}

        submissions = pred_df[required_cols].iloc[:n].values.tolist()
        testcases = [train_ds[i] for i in range(n)]
        try:
            max_workers = int(max(1, int(os.getenv("SCIMAS_CODEGEN_DEV_MAX_WORKERS", "2"))))
            raw_score = float(eval_fn(submissions, testcases, verbose=False, max_workers=max_workers))
        except Exception as e:
            return {"ok": False, "reason": f"codegen_dev_eval_failed:{e}"}

        score_norm = self._normalize_metric_fn(raw_score, task)
        return {
            "ok": True,
            "raw_score": raw_score,
            "score_norm": float(score_norm),
            "rows": int(n),
            "pred_path": str(pred_path) if pred_path else "",
            "source": "codegen_dev_predictions",
        }

    def evaluate(self, *, task: Dict[str, Any], data_mount_dir: str, workspace_dir: str) -> Dict[str, Any]:
        data_dir = Path(data_mount_dir)
        ws = Path(workspace_dir)
        info = task.get("logging_info") or {}
        metric_name = str(info.get("metric") or "").lower()
        category = str(info.get("category") or "").lower()

        if "mrr" in metric_name or "retrieval" in category:
            retrieval_eval = self._evaluate_retrieval_dev(task=task, data_dir=data_dir, workspace_dir=ws)
            if bool(retrieval_eval.get("ok")):
                return retrieval_eval
        if "pass@" in metric_name or ("code" in category and "generation" in str(info.get("research_problem") or "").lower()):
            codegen_eval = self._evaluate_codegen_dev(task=task, data_dir=data_dir, workspace_dir=ws)
            if bool(codegen_eval.get("ok")):
                return codegen_eval

        dev_df = self._find_split(data_dir, ["validation", "val", "dev"])
        if dev_df is None:
            return {"ok": False, "reason": "dev_split_not_found"}

        pred_obj, pred_path = self._find_dev_predictions(ws)
        if pred_obj is None:
            return {"ok": False, "reason": "dev_predictions_not_found"}
        if isinstance(pred_obj, dict):
            raw_score = pred_obj.get("raw_score")
            if isinstance(raw_score, (int, float)):
                score_norm = self._normalize_metric_fn(float(raw_score), task)
                return {
                    "ok": True,
                    "raw_score": float(raw_score),
                    "score_norm": float(score_norm),
                    "rows": int(dev_df.shape[0]),
                    "pred_path": str(pred_path) if pred_path else "",
                    "source": "dev_metrics_json",
                }
            return {"ok": False, "reason": "dev_metrics_json_missing_raw_score"}

        target_col = self._infer_target_column_fn(task, dev_df)
        if not target_col or target_col not in pred_obj.columns and target_col not in dev_df.columns:
            # target inference failed.
            return {"ok": False, "reason": "target_column_not_found"}
        if target_col not in dev_df.columns:
            return {"ok": False, "reason": "target_not_in_dev_split"}

        scoring_col = str(((task.get("logging_info") or {}).get("scoring_column")) or "prediction")
        pred_col = ""
        for cand in (scoring_col, "prediction", "pred", "label"):
            if cand in pred_obj.columns:
                pred_col = cand
                break
        if not pred_col and len(pred_obj.columns) == 1:
            pred_col = str(pred_obj.columns[0])
        if not pred_col:
            return {"ok": False, "reason": "prediction_column_not_found"}

        n = min(int(dev_df.shape[0]), int(pred_obj.shape[0]))
        if n <= 0:
            return {"ok": False, "reason": "empty_dev_or_prediction"}
        y_true = dev_df[target_col].iloc[:n].tolist()
        y_pred = pred_obj[pred_col].iloc[:n].tolist()

        metric_name = str((task.get("logging_info") or {}).get("metric") or "")
        raw_score = self._compute_dev_metric_fn(metric_name, y_true, y_pred)
        if raw_score is None:
            return {"ok": False, "reason": f"metric_not_supported_for_dev:{metric_name}"}
        score_norm = self._normalize_metric_fn(float(raw_score), task)
        return {
            "ok": True,
            "raw_score": float(raw_score),
            "score_norm": float(score_norm),
            "rows": int(n),
            "pred_col": pred_col,
            "pred_path": str(pred_path) if pred_path else "",
            "source": "dev_predictions",
        }


def _default_starter_script() -> str:
    return """import argparse
import json
from pathlib import Path

import pandas as pd


def _load_table(base: Path, stem: str):
    for ext in (".csv", ".parquet", ".jsonl", ".json"):
        path = base / f"{stem}{ext}"
        if not path.exists():
            continue
        if ext == ".csv":
            return pd.read_csv(path)
        if ext == ".parquet":
            return pd.read_parquet(path)
        return pd.read_json(path, lines=(ext == ".jsonl"))
    return None


def _load_ds(path: Path):
    if not path.exists():
        return None
    try:
        from datasets import load_from_disk

        ds = load_from_disk(str(path))
        return ds
    except Exception:
        return None


def _write_retrieval_baseline(data_dir: Path, out_dir: Path):
    test_queries = _load_ds(data_dir / "test" / "queries")
    search_corpus = _load_ds(data_dir / "train" / "search_corpus")
    if test_queries is None or search_corpus is None:
        raise RuntimeError("retrieval split missing")
    query_list = list(test_queries["query"])
    corpus_ids = list(search_corpus["id"])
    if not corpus_ids:
        raise RuntimeError("empty retrieval corpus ids")

    def _rank(q):
        n = len(corpus_ids)
        start = abs(hash(str(q))) % n
        return [int(corpus_ids[(start + i) % n]) for i in range(min(50, n))]

    test_rows = [{"query": str(q), "rankings": json.dumps(_rank(q), ensure_ascii=False)} for q in query_list]
    pd.DataFrame(test_rows).to_csv(out_dir / "submission.csv", index=False)

    dev_queries = _load_ds(data_dir / "validation" / "queries_with_labels") or _load_ds(data_dir / "dev" / "queries_with_labels")
    if dev_queries is not None:
        dev_rows = [{"query": str(q), "rankings": json.dumps(_rank(q), ensure_ascii=False)} for q in list(dev_queries["query"])]
        pd.DataFrame(dev_rows).to_csv(out_dir / "dev_retrieval.csv", index=False)


def _write_codegen_baseline(data_dir: Path, out_dir: Path):
    test_df = _load_table(data_dir, "test")
    train_ds = _load_ds(data_dir / "train")
    if test_df is None or train_ds is None:
        raise RuntimeError("codegen split missing")
    code_stub = "def solve():\\n    return\\n"
    row = {f"code{i}": code_stub for i in range(1, 6)}
    pd.DataFrame([row for _ in range(len(test_df))]).to_csv(out_dir / "submission.csv", index=False)
    dev_n = min(24, len(train_ds))
    if dev_n > 0:
        pd.DataFrame([row for _ in range(dev_n)]).to_csv(out_dir / "dev_codegen.csv", index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--output-dir", type=str, default="./outputs")
    parser.add_argument("--task-manifest", type=str, default="./.task_manifest.json")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {}
    if Path(args.task_manifest).exists():
        manifest = json.loads(Path(args.task_manifest).read_text(encoding="utf-8"))

    metric = str(manifest.get("metric") or "").lower()
    category = str(manifest.get("category") or "").lower()
    if "mrr" in metric or "retrieval" in str(manifest.get("task_name") or "").lower():
        _write_retrieval_baseline(data_dir, out_dir)
        print("starter retrieval run completed")
        return
    if "pass@" in metric or ("code" in category and "generation" in str(manifest.get("task_name") or "").lower()):
        _write_codegen_baseline(data_dir, out_dir)
        print("starter codegen run completed")
        return

    train_df = _load_table(data_dir, "train")
    test_df = _load_table(data_dir, "test")
    dev_df = _load_table(data_dir, "validation") or _load_table(data_dir, "val") or _load_table(data_dir, "dev")
    if train_df is None or test_df is None:
        raise RuntimeError("missing train/test split")

    scoring_col = str(manifest.get("scoring_column") or "prediction")
    target = str(manifest.get("target_column_hint") or "")
    if not target or target not in train_df.columns:
        for col in ("label", "target", "answer", "y", "relatedness_score"):
            if col in train_df.columns:
                target = col
                break
    if not target:
        target = str(train_df.columns[-1])

    y = train_df[target]
    if y.dtype.kind in ("i", "f"):
        guess = float(y.mean())
    else:
        guess = y.mode().iloc[0] if not y.mode().empty else ""

    pd.DataFrame([{scoring_col: guess} for _ in range(len(test_df))]).to_csv(out_dir / "submission.csv", index=False)
    if dev_df is not None:
        pd.DataFrame([{scoring_col: guess} for _ in range(len(dev_df))]).to_csv(out_dir / "dev_predictions.csv", index=False)

    print("starter run completed")


if __name__ == "__main__":
    main()
"""


def _data_bridge_script() -> str:
    return """from pathlib import Path
import csv
import json


def _export_split(split_dir: Path, csv_path: Path):
    try:
        from datasets import load_from_disk
    except Exception:
        return
    if csv_path.exists():
        return
    if not (split_dir / "state.json").exists():
        return
    if not (split_dir / "dataset_info.json").exists():
        return
    try:
        ds = load_from_disk(str(split_dir))
    except Exception:
        return
    cols = list(getattr(ds, "column_names", []) or [])
    if not cols:
        return
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for row in ds:
            out = {}
            for c in cols:
                v = row.get(c)
                if isinstance(v, (dict, list, tuple)):
                    out[c] = json.dumps(v, ensure_ascii=False)
                else:
                    out[c] = v
            writer.writerow(out)


def main():
    data_dir = Path("./data")
    if not data_dir.exists():
        return
    for split in ("train", "test", "validation", "val", "dev"):
        _export_split(data_dir / split, data_dir / f"{split}.csv")


if __name__ == "__main__":
    main()
"""
