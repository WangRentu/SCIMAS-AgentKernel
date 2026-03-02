from __future__ import annotations

import json
import os
import re
from collections import Counter, defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional


_STAGE_ORDER = [
    "prepare",
    "profile",
    "literature",
    "read",
    "hypothesize",
    "experiment",
    "review",
    "write",
    "replicate",
    "other",
]


def _read_json(path: str) -> Optional[Dict[str, Any]]:
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        return []
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows


def _latest_by_reset(rows: List[Dict[str, Any]], key: str) -> List[Dict[str, Any]]:
    if not rows:
        return []
    start_idx = 0
    prev_num = None
    for idx, row in enumerate(rows):
        try:
            cur_num = int(row.get(key))
        except Exception:
            cur_num = None
        if prev_num is not None and cur_num is not None and cur_num < prev_num:
            start_idx = idx
        prev_num = cur_num
    return rows[start_idx:]


def _to_int(v: Any, default: int = 0) -> int:
    try:
        return int(v)
    except Exception:
        return default


def _to_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


def _to_dt(ts: str) -> Optional[datetime]:
    if not ts:
        return None
    try:
        if ts.endswith("Z"):
            ts = ts[:-1] + "+00:00"
        return datetime.fromisoformat(ts)
    except Exception:
        return None


def _extract_error_signature(stderr_text: str) -> str:
    text = str(stderr_text or "").strip()
    if not text:
        return ""
    lines = [x.strip() for x in text.splitlines() if x.strip()]
    if not lines:
        return ""
    tail = lines[-1]
    # Keep concise while still actionable (e.g., ValueError: xxx).
    return tail[:220]


def _extract_dev_score(stdout_text: str, stderr_text: str) -> Optional[float]:
    merged = f"{stdout_text}\n{stderr_text}"
    patterns = [
        r"\bdev[_\s-]?score\b\s*[:=]\s*(-?\d+(?:\.\d+)?)",
        r"\b(score|metric|mase|accuracy|f1)\b\s*[:=]\s*(-?\d+(?:\.\d+)?)",
    ]
    for pat in patterns:
        m = re.search(pat, merged, flags=re.IGNORECASE)
        if not m:
            continue
        g = m.group(1 if len(m.groups()) == 1 else 2)
        try:
            return float(g)
        except Exception:
            continue
    return None


def _stage_for_task_type(task_type: str) -> str:
    t = str(task_type or "").strip().lower()
    if t in {"prepare_data"}:
        return "prepare"
    if t in {"profile_data"}:
        return "profile"
    if t in {"retrieve_literature"}:
        return "literature"
    if t in {"read"}:
        return "read"
    if t in {"hypothesize"}:
        return "hypothesize"
    if t in {"experiment"}:
        return "experiment"
    if t in {"review", "verify_issue", "verify_strength"}:
        return "review"
    if t in {"write"}:
        return "write"
    if t in {"replicate"}:
        return "replicate"
    return "other"


def _extract_run_id(obj: Any) -> Optional[str]:
    if isinstance(obj, dict):
        rid = obj.get("run_id")
        if isinstance(rid, str) and rid.strip():
            return rid.strip()
        for v in obj.values():
            out = _extract_run_id(v)
            if out:
                return out
    elif isinstance(obj, list):
        for item in obj:
            out = _extract_run_id(item)
            if out:
                return out
    return None


def _parse_episode_task_name(episode_dir_name: str) -> Dict[str, Any]:
    m = re.match(r"episode_(\d+)__(.+)$", episode_dir_name)
    if not m:
        return {"episode_id": 0, "task_name": episode_dir_name}
    return {"episode_id": int(m.group(1)), "task_name": m.group(2)}


def _collect_taskboard(base_dir: str) -> Dict[str, Any]:
    tb_path = os.path.join(base_dir, "logs", "app", "environment", "taskboard.jsonl")
    events = _latest_by_reset(_read_jsonl(tb_path), "episode_id")

    task_latest: Dict[str, Dict[str, Any]] = {}
    release_reasons = Counter()
    agent_stats: Dict[str, Dict[str, Any]] = defaultdict(
        lambda: {"claimed": 0, "completed": 0, "released": 0, "active": 0, "errors": 0}
    )
    timeline: List[Dict[str, Any]] = []
    phase_counts: Dict[str, Counter] = defaultdict(Counter)
    claim_start_durations: List[int] = []
    start_complete_durations: List[int] = []
    claim_complete_seconds: List[float] = []
    claim_ts_by_task: Dict[str, str] = {}
    artifacts_index: Dict[str, Dict[str, Any]] = {}

    for idx, row in enumerate(events):
        event = str(row.get("event") or "")
        ts = str(row.get("ts") or "")
        episode_id = _to_int(row.get("episode_id"), 0)
        task = row.get("task") or {}
        task_id = str(task.get("task_id") or "")
        task_type = str(task.get("task_type") or "")
        owner = str(task.get("claimed_by") or (row.get("meta") or {}).get("agent_id") or "")

        if not task_id:
            continue

        if event == "release":
            reason = str((row.get("meta") or {}).get("reason") or "unknown")
            release_reasons[reason] += 1
            if owner:
                agent_stats[owner]["released"] += 1
                if reason.startswith("inner_action_failed") or "timeout" in reason or "error" in reason:
                    agent_stats[owner]["errors"] += 1
        elif event == "claim":
            if owner:
                agent_stats[owner]["claimed"] += 1
                agent_stats[owner]["active"] += 1
        elif event == "complete":
            if owner:
                agent_stats[owner]["completed"] += 1
                agent_stats[owner]["active"] = max(0, agent_stats[owner]["active"] - 1)

        run_id = _extract_run_id((task.get("result") or {}))
        started_tick = task.get("started_tick")
        claimed_tick = task.get("claimed_tick")
        last_heartbeat_tick = task.get("last_heartbeat_tick")
        claim_to_start_ticks = None
        start_to_complete_ticks = None
        if claimed_tick is not None and started_tick is not None:
            claim_to_start_ticks = _to_int(started_tick, -1) - _to_int(claimed_tick, -1)
            if claim_to_start_ticks >= 0:
                claim_start_durations.append(claim_to_start_ticks)
        if started_tick is not None and last_heartbeat_tick is not None:
            start_to_complete_ticks = _to_int(last_heartbeat_tick, -1) - _to_int(started_tick, -1)
            if start_to_complete_ticks >= 0:
                start_complete_durations.append(start_to_complete_ticks)

        stage = _stage_for_task_type(task_type)
        phase_counts[stage][event] += 1

        if event == "claim":
            claim_ts_by_task[task_id] = ts
        if event == "complete":
            t0 = _to_dt(claim_ts_by_task.get(task_id, ""))
            t1 = _to_dt(ts)
            if t0 and t1:
                dt_s = (t1 - t0).total_seconds()
                if dt_s >= 0:
                    claim_complete_seconds.append(dt_s)

            action_data = ((task.get("result") or {}).get("action_data") or {})
            card = action_data.get("data_card")
            if isinstance(card, dict) and card:
                artifacts_index[f"{task_id}::data_card"] = {
                    "artifact_type": "data_card",
                    "task_id": task_id,
                    "task_type": task_type,
                    "episode_id": episode_id,
                    "owner": owner,
                    "run_id": run_id or "",
                    "ts": ts,
                    "payload": card,
                }
            method = action_data.get("method_card")
            if isinstance(method, dict) and method:
                artifacts_index[f"{task_id}::method_card"] = {
                    "artifact_type": "method_card",
                    "task_id": task_id,
                    "task_type": task_type,
                    "episode_id": episode_id,
                    "owner": owner,
                    "run_id": run_id or "",
                    "ts": ts,
                    "payload": method,
                }
            evidence = action_data.get("evidence_cards")
            if isinstance(evidence, list) and evidence:
                artifacts_index[f"{task_id}::evidence_cards"] = {
                    "artifact_type": "evidence_cards",
                    "task_id": task_id,
                    "task_type": task_type,
                    "episode_id": episode_id,
                    "owner": owner,
                    "run_id": run_id or "",
                    "ts": ts,
                    "payload": evidence[:30],
                }
            if task_type == "write":
                artifacts_index[f"{task_id}::paper"] = {
                    "artifact_type": "paper",
                    "task_id": task_id,
                    "task_type": task_type,
                    "episode_id": episode_id,
                    "owner": owner,
                    "run_id": run_id or "",
                    "ts": ts,
                    "payload": action_data if isinstance(action_data, dict) else {},
                }
            if task_type == "replicate":
                artifacts_index[f"{task_id}::replication_report"] = {
                    "artifact_type": "replication_report",
                    "task_id": task_id,
                    "task_type": task_type,
                    "episode_id": episode_id,
                    "owner": owner,
                    "run_id": run_id or "",
                    "ts": ts,
                    "payload": action_data if isinstance(action_data, dict) else {},
                }

        timeline.append(
            {
                "id": f"EV{idx:06d}",
                "ts": ts,
                "episode_id": episode_id,
                "event": event,
                "task_id": task_id,
                "task_type": task_type,
                "stage": stage,
                "owner": owner,
                "status": str(task.get("status") or ""),
                "lease_ttl": _to_int(task.get("lease_ttl"), 0),
                "heartbeat": _to_int(task.get("heartbeat_count"), 0),
                "claimed_tick": claimed_tick,
                "started_tick": started_tick,
                "last_heartbeat_tick": last_heartbeat_tick,
                "claim_to_start_ticks": claim_to_start_ticks,
                "start_to_complete_ticks": start_to_complete_ticks,
                "release_reason": str((row.get("meta") or {}).get("reason") or ""),
                "run_id": run_id,
                "result": task.get("result") or {},
            }
        )

        task_latest[task_id] = {
            "task_id": task_id,
            "episode_id": episode_id,
            "task_type": task_type,
            "stage": stage,
            "state": str(task.get("status") or ""),
            "owner": owner,
            "lease_ttl": _to_int(task.get("lease_ttl"), 0),
            "heartbeat": _to_int(task.get("heartbeat_count"), 0),
            "claimed_tick": claimed_tick,
            "started_tick": started_tick,
            "last_heartbeat_tick": last_heartbeat_tick,
            "claim_to_start_ticks": claim_to_start_ticks,
            "start_to_complete_ticks": start_to_complete_ticks,
            "depends_on": list(task.get("depends_on") or []),
            "last_update": ts,
            "last_event": event,
            "release_reason": str((row.get("meta") or {}).get("reason") or ""),
            "run_id": run_id,
            "result": task.get("result"),
        }

    task_rows = sorted(task_latest.values(), key=lambda x: (x.get("episode_id", 0), x.get("task_id", "")))
    timeline = sorted(timeline, key=lambda x: x.get("ts", ""))

    # agent status heuristics
    agents: List[Dict[str, Any]] = []
    for aid, s in sorted(agent_stats.items(), key=lambda kv: kv[0]):
        if s["active"] > 0:
            status = "running"
        elif s["errors"] > 0 and s["completed"] == 0:
            status = "error"
        elif s["completed"] > 0:
            status = "idle"
        else:
            status = "idle"
        agents.append(
            {
                "agent_id": aid,
                "status": status,
                "claimed": s["claimed"],
                "completed": s["completed"],
                "released": s["released"],
                "errors": s["errors"],
            }
        )

    event_counts = Counter(x.get("event") for x in timeline)
    top_release_reason = release_reasons.most_common(1)[0][0] if release_reasons else "-"
    timing = {
        "avg_claim_to_start_ticks": (sum(claim_start_durations) / len(claim_start_durations)) if claim_start_durations else 0.0,
        "avg_start_to_complete_ticks": (sum(start_complete_durations) / len(start_complete_durations)) if start_complete_durations else 0.0,
        "avg_claim_to_complete_s": (sum(claim_complete_seconds) / len(claim_complete_seconds)) if claim_complete_seconds else 0.0,
        "samples_claim_to_start": len(claim_start_durations),
        "samples_start_to_complete": len(start_complete_durations),
        "samples_claim_to_complete": len(claim_complete_seconds),
    }

    return {
        "events": timeline,
        "task_snapshot": task_rows,
        "agents": agents,
        "event_counts": dict(event_counts),
        "release_reasons": dict(release_reasons),
        "top_release_reason": top_release_reason,
        "phase_counts": {k: dict(v) for k, v in phase_counts.items()},
        "timing": timing,
        "artifacts_index": list(artifacts_index.values()),
    }


def _collect_workspace_runs(base_dir: str, max_runs: int = 200) -> Dict[str, Any]:
    runs_root = os.path.join(base_dir, "logs", "runs", "airs_workspace")
    rows: List[Dict[str, Any]] = []

    if not os.path.exists(runs_root):
        return {"runs": [], "artifacts": {}}

    episode_dirs = sorted([x for x in os.listdir(runs_root) if x.startswith("episode_")])
    for ep_dir in episode_dirs:
        ep_info = _parse_episode_task_name(ep_dir)
        ep_path = os.path.join(runs_root, ep_dir)

        data_card_path = os.path.join(ep_path, "_analysis", "profile_data", "data_card.json")
        method_card_path = os.path.join(ep_path, "_analysis", "method_card", "method_card.json")
        data_card = _read_json(data_card_path)
        method_card = _read_json(method_card_path)

        run_dirs = sorted([x for x in os.listdir(ep_path) if x.startswith("RUN")])
        for run_name in run_dirs:
            agent_log = os.path.join(ep_path, run_name, "agent_log")
            code_run_path = os.path.join(agent_log, "code_run.json")
            solver_run_path = os.path.join(agent_log, "solver_run.json")
            submission_path = os.path.join(agent_log, "submission.csv")

            code_log = _read_json(code_run_path)
            solver_log = _read_json(solver_run_path)
            if not code_log and not solver_log and not os.path.exists(submission_path):
                continue

            code_plan = (code_log or {}).get("code_plan") or {}
            files = []
            for item in list(code_plan.get("files") or [])[:8]:
                if not isinstance(item, dict):
                    continue
                files.append(
                    {
                        "path": str(item.get("path") or ""),
                        "content": str(item.get("content") or "")[:120000],
                    }
                )

            run_result = (code_log or {}).get("run_result") or {}
            stdout_text = str(run_result.get("stdout") or "")
            stderr_text = str(run_result.get("stderr") or "")
            dev_score = _extract_dev_score(stdout_text, stderr_text)
            error_signature = _extract_error_signature(stderr_text)
            rows.append(
                {
                    "episode_id": ep_info.get("episode_id", 0),
                    "task_name": (code_log or {}).get("task_name") or ep_info.get("task_name", ""),
                    "run_id": run_name,
                    "run_key": f"{ep_info.get('episode_id', 0)}::{run_name}",
                    "executor": (code_log or {}).get("executor_used") or "",
                    "exit_code": run_result.get("exit_code"),
                    "duration_s": _to_float(run_result.get("duration_s"), 0.0),
                    "timed_out": bool(run_result.get("timed_out", False)),
                    "stderr": stderr_text[-12000:],
                    "stdout": stdout_text[-12000:],
                    "error_signature": error_signature,
                    "dev_score": dev_score,
                    "command": str(run_result.get("command") or ""),
                    "artifacts": list(run_result.get("artifacts") or []),
                    "code_plan": {
                        "run_cmd": str(code_plan.get("run_cmd") or ""),
                        "notes": str(code_plan.get("notes") or ""),
                        "files": files,
                    },
                    "submission_exists": os.path.exists(submission_path),
                    "submission_path": submission_path if os.path.exists(submission_path) else "",
                    "data_card": data_card,
                    "method_card": method_card,
                    "code_log_path": code_run_path if code_log else "",
                    "solver_log_path": solver_run_path if solver_log else "",
                    "workspace_dir": str((code_log or {}).get("workspace_dir") or ""),
                    "snapshot_before_run": str((code_log or {}).get("snapshot_before_run") or ""),
                }
            )

    rows = sorted(rows, key=lambda x: (x.get("episode_id", 0), x.get("run_id", "")))
    if len(rows) > max_runs:
        rows = rows[-max_runs:]

    artifacts = {
        r["run_key"]: {
            "submission_path": r.get("submission_path") or "",
            "code_log_path": r.get("code_log_path") or "",
            "solver_log_path": r.get("solver_log_path") or "",
            "workspace_dir": r.get("workspace_dir") or "",
            "snapshot_before_run": r.get("snapshot_before_run") or "",
            "artifact_count": len(r.get("artifacts") or []),
        }
        for r in rows
    }

    return {"runs": rows, "artifacts": artifacts}


def _collect_team_metrics(base_dir: str) -> Dict[str, Any]:
    sim_dir = os.path.join(base_dir, "logs", "app", "simulation")
    team_rows = _latest_by_reset(_read_jsonl(os.path.join(sim_dir, "team_metrics.jsonl")), "episode_index")
    chain_rows = _latest_by_reset(_read_jsonl(os.path.join(sim_dir, "research_chain_metrics.jsonl")), "episode_index")

    chain_by_ep = {(_to_int(r.get("episode_index"), 0) + 1): r for r in chain_rows}
    episodes: List[Dict[str, Any]] = []
    for row in team_rows:
        ep = _to_int(row.get("episode_index"), 0) + 1
        chain = chain_by_ep.get(ep, {})
        episodes.append(
            {
                "episode": ep,
                "team_fitness": _to_float(row.get("team_fitness"), 0.0),
                "collaboration_ratio": _to_float(row.get("collaboration_ratio"), 0.0),
                "publishable_rate": _to_float(row.get("publishable_rate"), 0.0),
                "replication_pass_rate": _to_float(row.get("replication_pass_rate"), 0.0),
                "task_claim": _to_int(row.get("taskboard_claim_events"), _to_int(chain.get("taskboard_claim_events"), 0)),
                "task_complete": _to_int(
                    row.get("taskboard_complete_events"), _to_int(chain.get("taskboard_complete_events"), 0)
                ),
                "task_release": _to_int(
                    row.get("taskboard_release_events"), _to_int(chain.get("taskboard_release_events"), 0)
                ),
                "release_per_complete": _to_float(
                    row.get("taskboard_release_per_complete"), _to_float(chain.get("taskboard_release_per_complete"), 0.0)
                ),
                "complete_per_claim": _to_float(
                    row.get("taskboard_complete_per_claim"), _to_float(chain.get("taskboard_complete_per_claim"), 0.0)
                ),
                "action_experiment": _to_int(chain.get("action_experiment"), 0),
                "action_write": _to_int(chain.get("action_write"), 0),
                "action_review": _to_int(chain.get("action_review"), 0),
                "action_replicate": _to_int(chain.get("action_replicate"), 0),
                "task_top_release_reason": str(
                    row.get("taskboard_top_release_reason") or chain.get("taskboard_top_release_reason") or ""
                ),
            }
        )

    summary = {
        "episodes_count": len(episodes),
        "final_team_fitness": episodes[-1]["team_fitness"] if episodes else 0.0,
        "final_publishable_rate": episodes[-1]["publishable_rate"] if episodes else 0.0,
        "final_replication_pass": episodes[-1]["replication_pass_rate"] if episodes else 0.0,
        "final_complete_per_claim": episodes[-1]["complete_per_claim"] if episodes else 0.0,
        "final_release_per_complete": episodes[-1]["release_per_complete"] if episodes else 0.0,
    }
    return {"episodes": episodes, "summary": summary}


def _collect_papers(base_dir: str, max_rows: int = 200) -> List[Dict[str, Any]]:
    papers_path = os.path.join(base_dir, "logs", "app", "research", "papers.jsonl")
    rows = _latest_by_reset(_read_jsonl(papers_path), "episode_id")
    out: List[Dict[str, Any]] = []
    for r in rows[-max_rows:]:
        metrics = r.get("metrics") or {}
        out.append(
            {
                "ts": str(r.get("ts") or ""),
                "episode_id": _to_int(r.get("episode_id"), 0),
                "paper_id": str(r.get("paper_id") or ""),
                "agent_id": str(r.get("agent_id") or ""),
                "source": str(r.get("source") or ""),
                "fitness": _to_float(metrics.get("fitness"), 0.0),
                "f1": _to_float(metrics.get("f1"), 0.0),
                "replication_ok": bool(metrics.get("replication_ok", False)),
                "replication_verified": bool(metrics.get("replication_verified", False)),
                "publishable": bool(metrics.get("publishable", False)),
            }
        )
    return out


def _collect_evidence_cards(base_dir: str, max_rows: int = 300) -> List[Dict[str, Any]]:
    path = os.path.join(base_dir, "logs", "app", "research", "evidence_cards.jsonl")
    rows = _latest_by_reset(_read_jsonl(path), "episode_id")
    out: List[Dict[str, Any]] = []
    for r in rows[-max_rows:]:
        out.append(
            {
                "ts": str(r.get("ts") or ""),
                "episode_id": _to_int(r.get("episode_id"), 0),
                "evidence_id": str(r.get("evidence_id") or ""),
                "agent_id": str(r.get("agent_id") or ""),
                "source": str(r.get("source") or ""),
                "task_id": str(r.get("task_id") or ""),
                "run_id": str(r.get("run_id") or ""),
                "kind": str(r.get("kind") or ""),
                "content": r.get("content"),
            }
        )
    return out


def _collect_action_trace(base_dir: str, max_rows: int = 600) -> List[Dict[str, Any]]:
    path = os.path.join(base_dir, "logs", "app", "action", "trace.jsonl")
    rows = _latest_by_reset(_read_jsonl(path), "episode_id")
    out: List[Dict[str, Any]] = []
    for r in rows[-max_rows:]:
        out.append(
            {
                "ts": str(r.get("ts") or ""),
                "episode_id": _to_int(r.get("episode_id"), 0),
                "agent_id": str(r.get("agent_id") or ""),
                "action": str(r.get("action") or ""),
                "status": str(r.get("status") or ""),
                "task_id": str(r.get("task_id") or ""),
                "run_id": str(r.get("run_id") or ""),
                "reason": str((r.get("meta") or {}).get("reason") or ""),
                "summary": str(r.get("summary") or ""),
            }
        )
    return out


def _build_dashboard_payload(base_dir: str) -> Dict[str, Any]:
    tb = _collect_taskboard(base_dir)
    tm = _collect_team_metrics(base_dir)
    runs = _collect_workspace_runs(base_dir)
    papers = _collect_papers(base_dir)
    evidence_cards = _collect_evidence_cards(base_dir)
    action_trace = _collect_action_trace(base_dir)

    state_counts = Counter([str(x.get("state") or "unknown") for x in tb.get("task_snapshot") or []])
    release_reasons = tb.get("release_reasons") or {}
    alert_items: List[Dict[str, Any]] = []
    lease_expired = _to_int(release_reasons.get("lease_expired"), 0)
    if lease_expired > 0:
        alert_items.append({"level": "warn", "key": "lease_expired", "value": lease_expired})
    inner_failed = sum(
        int(v or 0) for k, v in release_reasons.items() if "inner_action_failed" in str(k)
    )
    if inner_failed > 0:
        alert_items.append({"level": "error", "key": "inner_action_failed", "value": inner_failed})
    run_fail = sum(1 for r in (runs.get("runs") or []) if r.get("exit_code") not in (None, 0))
    if run_fail > 0:
        alert_items.append({"level": "error", "key": "code_run_failed", "value": run_fail})
    timed_out = sum(1 for r in (runs.get("runs") or []) if bool(r.get("timed_out")))
    if timed_out > 0:
        alert_items.append({"level": "warn", "key": "code_run_timeout", "value": timed_out})
    alert_count = len(alert_items)

    health = "green"
    if alert_count >= 2:
        health = "red"
    elif alert_count == 1:
        health = "yellow"

    episode_ids = {int(_to_int(x.get("episode_id"), 0)) for x in (tb.get("events") or [])}
    episode_ids = {x for x in episode_ids if x > 0}
    episodes_count_fallback = max(episode_ids) if episode_ids else 0
    episodes_count = int(tm.get("summary", {}).get("episodes_count", 0) or 0) or episodes_count_fallback
    try:
        live_refresh_seconds = max(2.0, float(os.getenv("SCIMAS_DASHBOARD_BROWSER_REFRESH_S", "5")))
    except Exception:
        live_refresh_seconds = 5.0

    meta = {
        "health": health,
        "alert_count": alert_count,
        "alerts": alert_items,
        "state_counts": dict(state_counts),
        "top_release_reason": tb.get("top_release_reason", "-"),
        "episodes_count": episodes_count,
        "log_mode": str(os.getenv("SCIMAS_LOG_MODE", "compact")),
        "live_refresh_seconds": live_refresh_seconds,
    }

    return {
        "meta": meta,
        "team": tm,
        "taskboard": tb,
        "runs": runs,
        "papers": papers,
        "evidence_cards": evidence_cards,
        "action_trace": action_trace,
    }


def build_dashboard_payload(base_dir: str) -> Dict[str, Any]:
    """Public payload builder for separated frontend-backend dashboard servers."""
    return _build_dashboard_payload(base_dir=base_dir)


def _render_html(payload: Dict[str, Any]) -> str:
    payload_json = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    template = """<!doctype html>
<html lang=\"zh-CN\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>SCIMAS ResearchOps Console</title>
  <style>
    :root {{
      --bg: #0b1220;
      --bg2: #101a2c;
      --panel: #131f35;
      --panel2: #172641;
      --line: #253858;
      --text: #e7edf7;
      --muted: #95a4be;
      --accent: #4cc9f0;
      --ok: #20bf6b;
      --warn: #f6c453;
      --err: #ff6b6b;
      --chip: #1a2b47;
    }}
    * {{ box-sizing: border-box; }}
    html, body {{ margin: 0; height: 100%; background: radial-gradient(circle at 0% 0%, #132540 0%, transparent 35%), radial-gradient(circle at 100% 100%, #10243a 0%, transparent 32%), var(--bg); color: var(--text); font: 14px/1.45 "IBM Plex Sans", "Noto Sans SC", "Microsoft YaHei", sans-serif; }}
    .app {{ display: grid; grid-template-rows: 56px 1fr; height: 100%; }}
    .header {{ display: flex; align-items: center; justify-content: space-between; padding: 0 16px; border-bottom: 1px solid var(--line); background: rgba(9,15,28,.65); backdrop-filter: blur(6px); position: sticky; top: 0; z-index: 10; }}
    .title {{ font-weight: 650; letter-spacing: .2px; }}
    .header-left, .header-right {{ display: flex; align-items: center; gap: 10px; }}
    .badge {{ border: 1px solid var(--line); background: var(--chip); padding: 4px 10px; border-radius: 999px; color: var(--muted); font-size: 12px; }}
    .health-dot {{ width: 10px; height: 10px; border-radius: 99px; display: inline-block; margin-right: 6px; }}
    .green {{ background: var(--ok); }} .yellow {{ background: var(--warn); }} .red {{ background: var(--err); }}

    .body {{ display: grid; grid-template-columns: 300px 1fr 460px; min-height: 0; transition: grid-template-columns .2s ease; }}
    .body.inspector-collapsed {{ grid-template-columns: 300px 1fr 0px; }}
    .sidebar, .main, .inspector {{ min-height: 0; overflow: hidden; }}
    .sidebar {{ border-right: 1px solid var(--line); background: linear-gradient(180deg, rgba(15,24,42,.98), rgba(14,21,36,.95)); padding: 12px; display: flex; flex-direction: column; gap: 12px; }}
    .main {{ padding: 12px; overflow: auto; display: grid; grid-template-rows: minmax(280px, 44%) minmax(280px, 56%); gap: 12px; }}
    .inspector {{ border-left: 1px solid var(--line); background: linear-gradient(180deg, rgba(18,30,50,.97), rgba(14,24,41,.95)); padding: 12px; display: grid; grid-template-rows: auto auto 1fr; gap: 10px; }}
    .body.inspector-collapsed .inspector {{ display: none; }}

    .panel {{ border: 1px solid var(--line); border-radius: 12px; background: linear-gradient(180deg, rgba(21,34,58,.98), rgba(17,29,49,.95)); overflow: hidden; }}
    .panel-head {{ display: flex; justify-content: space-between; align-items: center; padding: 10px 12px; border-bottom: 1px solid var(--line); background: rgba(10,18,33,.45); }}
    .panel-title {{ font-weight: 620; }}
    .panel-body {{ padding: 10px 12px; }}

    .filters input[type=text], .filters select {{ width: 100%; background: #0f1b30; border: 1px solid #243958; color: var(--text); border-radius: 8px; padding: 8px 10px; }}
    .chip-row {{ display: flex; flex-wrap: wrap; gap: 8px; }}
    .chip {{ border: 1px solid var(--line); background: var(--chip); color: var(--muted); border-radius: 999px; padding: 3px 8px; font-size: 12px; }}
    .agent-list {{ max-height: 260px; overflow: auto; }}
    .agent-item {{ display: flex; justify-content: space-between; padding: 7px 8px; border-radius: 8px; cursor: pointer; }}
    .agent-item:hover, .agent-item.active {{ background: #1b2f4f; }}

    .timeline-wrap {{ display: grid; grid-template-columns: repeat(10, minmax(150px,1fr)); gap: 8px; overflow: auto; padding-bottom: 6px; }}
    .lane {{ border: 1px solid var(--line); border-radius: 10px; background: rgba(12,21,37,.55); min-height: 220px; }}
    .lane-head {{ padding: 8px; font-weight: 620; color: #b9cae3; border-bottom: 1px solid var(--line); text-transform: capitalize; }}
    .lane-body {{ padding: 8px; display: flex; flex-direction: column; gap: 8px; }}
    .event-card {{ border: 1px solid #2a456c; border-left: 3px solid #4e7fb7; border-radius: 8px; padding: 7px; background: #132440; cursor: pointer; }}
    .event-card.success {{ border-left-color: var(--ok); }}
    .event-card.fail {{ border-left-color: var(--err); }}
    .event-card.release {{ border-left-color: var(--warn); }}
    .event-card:hover {{ filter: brightness(1.08); }}
    .event-meta {{ color: var(--muted); font-size: 12px; }}

    .stats-row {{ display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 8px; }}
    .stat-pill {{ border: 1px solid var(--line); background: #14253f; border-radius: 999px; padding: 4px 10px; font-size: 12px; color: #bdd0eb; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 12.5px; }}
    th, td {{ padding: 7px 8px; border-bottom: 1px solid #223655; text-align: left; white-space: nowrap; }}
    th {{ color: #9fb2ce; background: #11213a; position: sticky; top: 0; z-index: 2; }}
    tr:hover td {{ background: #1a2e4d; }}
    tr.selected td {{ background: #213a5f !important; }}
    .table-wrap {{ max-height: 100%; overflow: auto; border: 1px solid var(--line); border-radius: 10px; }}

    .tabs {{ display: flex; gap: 6px; flex-wrap: wrap; }}
    .tab-btn {{ border: 1px solid #2a4266; background: #152945; color: #c8d8ef; border-radius: 8px; padding: 5px 10px; cursor: pointer; font-size: 12px; }}
    .tab-btn.active {{ background: #1f3a62; border-color: #3e6da9; color: #fff; }}
    .action-btn {{ border: 1px solid var(--line); background: #162b48; color: #d7e7fb; border-radius: 8px; padding: 4px 10px; cursor: pointer; }}
    .action-btn:hover {{ filter: brightness(1.1); }}
    .alert-list {{ list-style: none; padding: 0; margin: 0; display: flex; flex-direction: column; gap: 6px; }}
    .alert-item {{ border: 1px solid #324b70; border-radius: 8px; padding: 6px 8px; font-size: 12px; }}
    .alert-item.warn {{ border-color: #8a6a2c; background: rgba(246, 196, 83, 0.08); }}
    .alert-item.error {{ border-color: #7f3434; background: rgba(255, 107, 107, 0.08); }}
    .inspector-content {{ border: 1px solid var(--line); border-radius: 10px; background: #0f1c31; overflow: auto; padding: 10px; }}
    .kv {{ display: grid; grid-template-columns: 120px 1fr; gap: 6px 10px; font-size: 12.5px; }}
    .k {{ color: var(--muted); }}
    .mono {{ font-family: "JetBrains Mono", "Consolas", monospace; font-size: 12px; }}
    pre {{ margin: 0; white-space: pre-wrap; word-break: break-word; background: #0a1528; border: 1px solid #213452; border-radius: 8px; padding: 8px; max-height: 280px; overflow: auto; }}
    .file-tabs {{ display: flex; gap: 6px; flex-wrap: wrap; margin-bottom: 8px; }}
    .file-tab {{ border: 1px solid #2a4266; border-radius: 8px; background: #132947; color: #c9d8f0; font-size: 12px; padding: 3px 8px; cursor: pointer; }}
    .file-tab.active {{ background: #214170; }}

    @media (max-width: 1440px) {{
      .body {{ grid-template-columns: 270px 1fr 420px; }}
      .timeline-wrap {{ grid-template-columns: repeat(10, minmax(130px,1fr)); }}
    }}
  </style>
</head>
<body>
  <script id=\"scimas-payload\" type=\"application/json\">__PAYLOAD_JSON__</script>
  <div class=\"app\">
    <div class=\"header\">
      <div class=\"header-left\">
        <div class=\"title\">SCIMAS ResearchOps Console</div>
        <span class=\"badge\"><span id=\"health-dot\" class=\"health-dot\"></span><span id=\"health-text\">health</span></span>
        <span class=\"badge\" id=\"alert-chip\">alerts: 0</span>
        <span class=\"badge\" id=\"top-release-chip\">top release: -</span>
        <button id=\"snapshot-btn\" class=\"action-btn\" type=\"button\">Snapshot</button>
        <button id=\"export-btn\" class=\"action-btn\" type=\"button\">Export JSON</button>
      </div>
      <div class=\"header-right\">
        <input id=\"run-search\" class=\"badge\" style=\"min-width:190px; background:#0f1d33; color:#d9e8fb;\" placeholder=\"Run/Episode Search\" />
        <select id=\"episode-select\" class=\"badge\"></select>
        <select id=\"mode-select\" class=\"badge\"><option value=\"live\">Live</option><option value=\"replay\">Replay</option></select>
        <input id=\"replay-slider\" type=\"range\" min=\"0\" max=\"100\" value=\"100\" style=\"width:160px; display:none;\" />
        <span class=\"badge\" id=\"tick-chip\">tick: -</span>
        <button id=\"inspector-toggle\" class=\"action-btn\" type=\"button\">Hide Inspector</button>
      </div>
    </div>

    <div class=\"body\" id=\"layout-body\">
      <aside class=\"sidebar\">
        <section class=\"panel\">
          <div class=\"panel-head\"><div class=\"panel-title\">Filters</div></div>
          <div class=\"panel-body filters\">
            <input id=\"search-input\" type=\"text\" placeholder=\"search task/agent/error...\" />
            <div style=\"height:8px\"></div>
            <div class=\"chip-row\" id=\"type-filters\"></div>
            <div style=\"height:8px\"></div>
            <div class=\"chip-row\" id=\"error-filters\"></div>
          </div>
        </section>

        <section class=\"panel\">
          <div class=\"panel-head\"><div class=\"panel-title\">Agents</div><div class=\"event-meta\" id=\"agent-count\"></div></div>
          <div class=\"panel-body agent-list\" id=\"agent-list\"></div>
        </section>

        <section class=\"panel\">
          <div class=\"panel-head\"><div class=\"panel-title\">Alert Center</div><div class=\"event-meta\" id=\"alert-count-mini\"></div></div>
          <div class=\"panel-body\">
            <ul class=\"alert-list\" id=\"alert-list\"></ul>
          </div>
        </section>
      </aside>

      <main class=\"main\">
        <section class=\"panel\" style=\"min-height:0;\">
          <div class=\"panel-head\">
            <div class=\"panel-title\">Process Timeline</div>
            <div class=\"event-meta\">read → hypothesize → experiment → review → write → replicate</div>
          </div>
          <div class=\"panel-body\" style=\"min-height:0; height:100%;\">
            <div class=\"timeline-wrap\" id=\"timeline\"></div>
          </div>
        </section>

        <section class=\"panel\" style=\"min-height:0;\">
          <div class=\"panel-head\"><div class=\"panel-title\">Taskboard Live</div><div class=\"event-meta\" id=\"task-summary\"></div></div>
          <div class=\"panel-body\" style=\"min-height:0; height:100%;\">
            <div class=\"stats-row\" id=\"task-pills\"></div>
            <div class=\"table-wrap\" style=\"height: calc(100% - 38px);\">
              <table>
                <thead>
                  <tr><th>Task ID</th><th>Type</th><th>State</th><th>Owner</th><th>Lease/HB</th><th>Deps</th><th>C→S</th><th>S→C</th><th>Last Update</th><th>Release Reason</th><th>Artifacts</th><th>Run</th></tr>
                </thead>
                <tbody id=\"task-tbody\"></tbody>
              </table>
            </div>
          </div>
        </section>
      </main>

      <aside class=\"inspector\">
        <section class=\"panel\">
          <div class=\"panel-head\"><div class=\"panel-title\">Inspector</div><div class=\"event-meta\" id=\"inspector-hint\">select event/task/agent</div></div>
          <div class=\"panel-body\"><div class=\"kv\" id=\"inspector-summary\"></div></div>
        </section>
        <section class=\"tabs\" id=\"inspector-tabs\"></section>
        <section class=\"inspector-content\" id=\"inspector-content\"></section>
      </aside>
    </div>
  </div>

  <script>
    const payload = JSON.parse(document.getElementById('scimas-payload').textContent || '{}');
    const state = {
      episode: 0,
      mode: 'live',
      search: '',
      runSearch: '',
      typeFilter: new Set(),
      errFilter: new Set(),
      agent: '',
      selected: null,
      tab: 'overview',
      filePath: '',
      replayPercent: 100,
      inspectorCollapsed: false,
      snapshots: [],
    };

    const stageOrder = ['prepare','profile','literature','read','hypothesize','experiment','review','write','replicate','other'];
    const events = (payload.taskboard && payload.taskboard.events) || [];
    const tasks = (payload.taskboard && payload.taskboard.task_snapshot) || [];
    const agents = (payload.taskboard && payload.taskboard.agents) || [];
    const runs = (payload.runs && payload.runs.runs) || [];
    const papers = payload.papers || [];
    const evidenceCards = payload.evidence_cards || [];
    const actionTrace = payload.action_trace || [];
    const alertItems = (payload.meta && payload.meta.alerts) || [];
    const runMap = Object.fromEntries(runs.map(r => [r.run_id, r]));

    function esc(s) {
      return String(s == null ? '' : s)
        .replaceAll('&', '&amp;')
        .replaceAll('<', '&lt;')
        .replaceAll('>', '&gt;');
    }

    function toLocal(ts) {
      if (!ts) return '-';
      try { return new Date(ts).toLocaleString(); } catch { return String(ts); }
    }

    function getReplayTimeCutoff() {
      if (state.mode !== 'replay') return null;
      const tsList = events.map(e => Date.parse(e.ts)).filter(v => Number.isFinite(v)).sort((a, b) => a - b);
      if (!tsList.length) return null;
      const idx = Math.max(0, Math.min(tsList.length - 1, Math.floor((state.replayPercent / 100) * (tsList.length - 1))));
      return tsList[idx];
    }

    function artifactBadges(taskRow) {
      const bits = [];
      if (taskRow.run_id) bits.push('run');
      const resultText = JSON.stringify(taskRow.result || {}).toLowerCase();
      if (resultText.includes('data_card')) bits.push('data');
      if (resultText.includes('method_card')) bits.push('method');
      if (resultText.includes('evidence')) bits.push('evidence');
      if (taskRow.task_type === 'write') bits.push('paper');
      if (taskRow.task_type === 'replicate') bits.push('repl');
      return bits.join('|') || '-';
    }

    function formatTick(v) {
      if (v == null || v === '' || Number(v) < 0) return '-';
      return String(v);
    }

    function setHeader() {
      const meta = payload.meta || {};
      const health = meta.health || 'green';
      document.getElementById('health-dot').className = `health-dot ${health}`;
      document.getElementById('health-text').textContent = `health: ${health}`;
      document.getElementById('alert-chip').textContent = `alerts: ${meta.alert_count || 0}`;
      document.getElementById('top-release-chip').textContent = `top release: ${meta.top_release_reason || '-'}`;
      const episodesCount = (payload.team && payload.team.summary && payload.team.summary.episodes_count) || 0;
      const timing = (payload.taskboard && payload.taskboard.timing) || {};
      const avgCs = Number(timing.avg_claim_to_start_ticks || 0).toFixed(1);
      const avgSc = Number(timing.avg_start_to_complete_ticks || 0).toFixed(1);
      document.getElementById('tick-chip').textContent = `episodes: ${episodesCount} | C→S:${avgCs} S→C:${avgSc}`;
      document.getElementById('alert-count-mini').textContent = `${meta.alert_count || 0} active`;
      const alertList = document.getElementById('alert-list');
      alertList.innerHTML = (alertItems.length ? alertItems : [{ level: 'warn', key: 'none', value: 0 }]).map(a =>
        `<li class=\"alert-item ${esc(a.level || 'warn')}\"><b>${esc(a.key)}</b> <span class=\"event-meta\">${esc(a.value)}</span></li>`
      ).join('');
    }

    function setupEpisodeSelect() {
      const select = document.getElementById('episode-select');
      const eps = new Set();
      events.forEach(e => eps.add(Number(e.episode_id || 0)));
      runs.forEach(r => eps.add(Number(r.episode_id || 0)));
      const sorted = [...eps].filter(x => x > 0).sort((a, b) => a - b);
      select.innerHTML = `<option value=\"0\">All Episodes</option>` + sorted.map(ep => `<option value=\"${ep}\">Episode ${ep}</option>`).join('');
      select.value = '0';
      select.onchange = () => { state.episode = Number(select.value || 0); renderAll(); };
      const modeSel = document.getElementById('mode-select');
      const replaySlider = document.getElementById('replay-slider');
      modeSel.onchange = (e) => {
        state.mode = e.target.value;
        replaySlider.style.display = state.mode === 'replay' ? 'inline-block' : 'none';
        renderAll();
      };
      replaySlider.oninput = () => { state.replayPercent = Number(replaySlider.value || 100); renderAll(); };

      const runSearch = document.getElementById('run-search');
      runSearch.oninput = () => { state.runSearch = runSearch.value.trim().toLowerCase(); renderAll(); };

      document.getElementById('inspector-toggle').onclick = () => {
        state.inspectorCollapsed = !state.inspectorCollapsed;
        const body = document.getElementById('layout-body');
        const btn = document.getElementById('inspector-toggle');
        body.classList.toggle('inspector-collapsed', state.inspectorCollapsed);
        btn.textContent = state.inspectorCollapsed ? 'Show Inspector' : 'Hide Inspector';
      };

      document.getElementById('snapshot-btn').onclick = () => {
        const snap = {
          ts: new Date().toISOString(),
          mode: state.mode,
          episode: state.episode,
          selected: state.selected,
          taskboard_timing: (payload.taskboard && payload.taskboard.timing) || {},
          top_release_reason: (payload.meta && payload.meta.top_release_reason) || '',
          alerts: alertItems,
        };
        state.snapshots.push(snap);
        const blob = new Blob([JSON.stringify(snap, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `scimas_snapshot_ep${state.episode || 'all'}_${Date.now()}.json`;
        a.click();
        URL.revokeObjectURL(url);
      };

      document.getElementById('export-btn').onclick = () => {
        const exportData = {
          meta: payload.meta || {},
          team: payload.team || {},
          taskboard: payload.taskboard || {},
          runs: payload.runs || {},
          papers,
          evidence_cards: evidenceCards,
          action_trace: actionTrace,
          snapshots: state.snapshots,
        };
        const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `scimas_console_export_${Date.now()}.json`;
        a.click();
        URL.revokeObjectURL(url);
      };
    }

    function setupAutoRefresh() {
      const secs = Number((payload.meta && payload.meta.live_refresh_seconds) || 0);
      if (!(secs > 0)) return;
      const intervalMs = Math.max(2000, Math.floor(secs * 1000));
      const timer = window.setInterval(() => {
        if (state.mode !== 'live') return;
        if (document.hidden) return;
        window.location.reload();
      }, intervalMs);
      window.addEventListener('beforeunload', () => window.clearInterval(timer));
    }

    function setupFilters() {
      const typeValues = [...new Set(tasks.map(t => String(t.task_type || '')))].filter(Boolean).sort();
      const typeWrap = document.getElementById('type-filters');
      typeWrap.innerHTML = typeValues.map(t => `<button class=\"chip\" data-type=\"${esc(t)}\">${esc(t)}</button>`).join('');
      typeWrap.querySelectorAll('button').forEach(btn => {
        btn.onclick = () => {
          const t = btn.dataset.type;
          if (state.typeFilter.has(t)) state.typeFilter.delete(t); else state.typeFilter.add(t);
          btn.style.borderColor = state.typeFilter.has(t) ? '#4cc9f0' : 'var(--line)';
          btn.style.color = state.typeFilter.has(t) ? '#dff6ff' : 'var(--muted)';
          renderAll();
        };
      });

      const errValues = ['lease_expired','inner_action_failed','timeout','oom','json_parse'];
      const errWrap = document.getElementById('error-filters');
      errWrap.innerHTML = errValues.map(t => `<button class=\"chip\" data-err=\"${esc(t)}\">${esc(t)}</button>`).join('');
      errWrap.querySelectorAll('button').forEach(btn => {
        btn.onclick = () => {
          const t = btn.dataset.err;
          if (state.errFilter.has(t)) state.errFilter.delete(t); else state.errFilter.add(t);
          btn.style.borderColor = state.errFilter.has(t) ? '#f6c453' : 'var(--line)';
          btn.style.color = state.errFilter.has(t) ? '#fff6d9' : 'var(--muted)';
          renderAll();
        };
      });

      const search = document.getElementById('search-input');
      search.oninput = () => { state.search = search.value.trim().toLowerCase(); renderAll(); };
    }

    function matchFilters(obj) {
      const ep = Number(obj.episode_id || 0);
      if (state.episode > 0 && ep !== state.episode) return false;
      const replayCut = getReplayTimeCutoff();
      if (replayCut != null) {
        const tsVal = Date.parse(obj.ts || obj.last_update || '');
        if (Number.isFinite(tsVal) && tsVal > replayCut) return false;
      }
      if (state.agent && String(obj.owner || obj.agent_id || '').toLowerCase() !== state.agent.toLowerCase()) return false;
      if (state.typeFilter.size > 0) {
        const t = String(obj.task_type || '');
        if (!state.typeFilter.has(t)) return false;
      }
      const text = JSON.stringify(obj).toLowerCase();
      if (state.search && !text.includes(state.search)) return false;
      if (state.runSearch) {
        const runText = `${obj.run_id || ''} ${obj.task_id || ''} ${obj.owner || ''}`.toLowerCase();
        if (!runText.includes(state.runSearch)) return false;
      }
      if (state.errFilter.size > 0) {
        let ok = false;
        for (const e of state.errFilter) {
          if (text.includes(e)) { ok = true; break; }
        }
        if (!ok) return false;
      }
      return true;
    }

    function renderAgents() {
      const box = document.getElementById('agent-list');
      const rows = agents.filter(a => (state.episode <= 0) || true);
      document.getElementById('agent-count').textContent = `${rows.length} agents`;
      box.innerHTML = rows.map(a => {
        const active = state.agent === a.agent_id ? 'active' : '';
        const dot = a.status === 'running' ? 'green' : (a.status === 'error' ? 'red' : 'yellow');
        return `<div class=\"agent-item ${active}\" data-agent=\"${esc(a.agent_id)}\"><div><span class=\"health-dot ${dot}\"></span>${esc(a.agent_id)}</div><div class=\"event-meta\">${a.claimed}/${a.completed}/${a.released}</div></div>`;
      }).join('');
      box.querySelectorAll('.agent-item').forEach(el => {
        el.onclick = () => {
          const aid = el.dataset.agent || '';
          state.agent = (state.agent === aid) ? '' : aid;
          renderAll();
        };
      });
    }

    function renderTimeline() {
      const wrap = document.getElementById('timeline');
      const lanes = Object.fromEntries(stageOrder.map(s => [s, []]));
      const filtered = events.filter(matchFilters);
      filtered.forEach(ev => {
        const stage = stageOrder.includes(ev.stage) ? ev.stage : 'other';
        lanes[stage].push(ev);
      });
      for (const k of stageOrder) lanes[k] = lanes[k].slice(-20);

      wrap.innerHTML = stageOrder.map(stage => {
        const items = lanes[stage] || [];
        return `<section class=\"lane\">
          <div class=\"lane-head\">${stage} <span class=\"event-meta\">${items.length}</span></div>
          <div class=\"lane-body\">${items.map(ev => {
            const cls = ev.event === 'complete' ? 'success' : (ev.event === 'release' ? 'release fail' : (ev.event === 'claim' ? 'running' : ''));
            const c2s = ev.claim_to_start_ticks != null ? `C→S:${ev.claim_to_start_ticks}` : '';
            const s2c = ev.start_to_complete_ticks != null ? `S→C:${ev.start_to_complete_ticks}` : '';
            return `<article class=\"event-card ${cls}\" data-ev=\"${ev.id}\">
              <div><b>${esc(ev.task_type || ev.stage)}</b></div>
              <div class=\"event-meta\">${esc(ev.task_id)} · ${esc(ev.owner || '-')}</div>
              <div class=\"event-meta\">${esc(ev.event)} · ${toLocal(ev.ts)}</div>
              <div class=\"event-meta\">${esc(c2s)} ${esc(s2c)} ${esc(ev.run_id || '')}</div>
            </article>`;
          }).join('')}</div>
        </section>`;
      }).join('');

      wrap.querySelectorAll('.event-card').forEach(el => {
        el.onclick = () => {
          const ev = events.find(x => x.id === el.dataset.ev);
          state.selected = { type: 'event', value: ev };
          state.tab = 'overview';
          renderInspector();
        };
      });
    }

    function renderTaskboard() {
      const filtered = tasks.filter(matchFilters);
      const counts = new CounterPolyfill(filtered.map(x => String(x.state || 'unknown')));
      const c2sVals = filtered.map(x => Number(x.claim_to_start_ticks)).filter(v => Number.isFinite(v) && v >= 0);
      const s2cVals = filtered.map(x => Number(x.start_to_complete_ticks)).filter(v => Number.isFinite(v) && v >= 0);
      const avg = (arr) => arr.length ? (arr.reduce((a, b) => a + b, 0) / arr.length).toFixed(2) : '-';

      const pills = [
        ['open', counts.get('open')],
        ['claimed', counts.get('claimed')],
        ['running', counts.get('running')],
        ['completed', counts.get('completed')],
        ['released', counts.get('released')],
        ['avg C→S', avg(c2sVals)],
        ['avg S→C', avg(s2cVals)],
      ];
      document.getElementById('task-pills').innerHTML = pills.map(([k, v]) => `<span class=\"stat-pill\">${k}: ${v || 0}</span>`).join('');
      document.getElementById('task-summary').textContent = `rows: ${filtered.length}`;

      const tbody = document.getElementById('task-tbody');
      tbody.innerHTML = filtered.map(t => {
        const selected = state.selected && state.selected.type === 'task' && state.selected.value.task_id === t.task_id ? 'selected' : '';
        const deps = (t.depends_on || []).slice(0, 2).join('→') || '-';
        const rr = t.release_reason || '-';
        return `<tr class=\"${selected}\" data-task=\"${esc(t.task_id)}\">` +
          `<td>${esc(t.task_id)}</td><td>${esc(t.task_type || '')}</td><td>${esc(t.state || '')}</td>` +
          `<td>${esc(t.owner || '-')}</td><td>${esc(String(t.lease_ttl || 0))}/${esc(String(t.heartbeat || 0))}</td>` +
          `<td>${esc(deps)}</td><td>${esc(formatTick(t.claim_to_start_ticks))}</td><td>${esc(formatTick(t.start_to_complete_ticks))}</td>` +
          `<td>${esc(toLocal(t.last_update))}</td><td>${esc(rr)}</td><td>${esc(artifactBadges(t))}</td><td>${esc(t.run_id || '-')}</td></tr>`;
      }).join('');

      tbody.querySelectorAll('tr').forEach(el => {
        el.onclick = () => {
          const t = tasks.find(x => x.task_id === el.dataset.task);
          state.selected = { type: 'task', value: t };
          state.tab = 'overview';
          renderInspector();
          renderTaskboard();
        };
      });
    }

    function findRunBySelected(sel) {
      if (!sel) return null;
      if (sel.type === 'run') return sel.value;
      const runId = sel.value && sel.value.run_id;
      if (!runId) return null;
      return runs.find(r => r.run_id === runId && ((state.episode <= 0) || Number(r.episode_id) === state.episode)) || runs.find(r => r.run_id === runId);
    }

    function renderInspector() {
      const sel = state.selected;
      const summary = document.getElementById('inspector-summary');
      const hint = document.getElementById('inspector-hint');
      const tabs = document.getElementById('inspector-tabs');
      const content = document.getElementById('inspector-content');

      if (!sel) {
        hint.textContent = 'select event/task/agent';
        summary.innerHTML = '<div class=\"k\">status</div><div>No selection</div>';
        tabs.innerHTML = '';
        content.innerHTML = '<div class=\"event-meta\">点击时间线或任务行查看细节。</div>';
        return;
      }

      const run = findRunBySelected(sel);
      const v = sel.value || {};
      hint.textContent = `${sel.type} · ${v.task_id || v.run_id || v.id || '-'}`;
      summary.innerHTML = `
        <div class=\"k\">episode</div><div>${esc(v.episode_id || (run && run.episode_id) || '-')}</div>
        <div class=\"k\">task</div><div>${esc(v.task_type || (run && run.task_name) || '-')}</div>
        <div class=\"k\">owner</div><div>${esc(v.owner || '-')}</div>
        <div class=\"k\">state</div><div>${esc(v.state || v.event || '-')}</div>
        <div class=\"k\">run_id</div><div class=\"mono\">${esc((run && run.run_id) || v.run_id || '-')}</div>
      `;

      const tabNames = ['overview', 'code', 'console', 'artifacts', 'reasoning', 'paper', 'compare'];
      tabs.innerHTML = tabNames.map(name => `<button class=\"tab-btn ${state.tab===name?'active':''}\" data-tab=\"${name}\">${name}</button>`).join('');
      tabs.querySelectorAll('.tab-btn').forEach(btn => {
        btn.onclick = () => { state.tab = btn.dataset.tab; renderInspector(); };
      });

      if (state.tab === 'overview') {
        const linkedRuns = runs.filter(r => Number(r.episode_id) === Number(v.episode_id || 0)).slice(-8);
        const phase = (payload.taskboard && payload.taskboard.phase_counts && payload.taskboard.phase_counts[v.stage || v.task_type || '']) || {};
        content.innerHTML = `
          <div class=\"event-meta\">Linked runs in episode: ${linkedRuns.length}</div>
          <div style=\"height:8px\"></div>
          <div class=\"mono\">${linkedRuns.map(r => `${r.run_id} | exit=${r.exit_code} | dev=${r.dev_score ?? '-'} | ${r.duration_s.toFixed(2)}s`).join('<br/>') || '-'}</div>
          <div style=\"height:8px\"></div>
          <div class=\"event-meta\">Phase Counts</div>
          <pre class=\"mono\">${esc(JSON.stringify(phase, null, 2))}</pre>
        `;
      } else if (state.tab === 'code') {
        if (!run) { content.innerHTML = '<div class=\"event-meta\">No linked run.</div>'; return; }
        const files = (run.code_plan && run.code_plan.files) || [];
        if (files.length === 0) {
          content.innerHTML = '<div class=\"event-meta\">No code plan files.</div>'; return;
        }
        if (!state.filePath || !files.some(f => f.path === state.filePath)) state.filePath = files[0].path;
        content.innerHTML = `
          <div class=\"event-meta\">run_cmd: <span class=\"mono\">${esc((run.code_plan && run.code_plan.run_cmd) || '-')}</span></div>
          <div style=\"height:8px\"></div>
          <div class=\"file-tabs\">${files.map(f => `<button class=\"file-tab ${state.filePath===f.path?'active':''}\" data-file=\"${esc(f.path)}\">${esc(f.path)}</button>`).join('')}</div>
          <pre class=\"mono\">${esc((files.find(f => f.path === state.filePath) || {}).content || '')}</pre>
        `;
        content.querySelectorAll('.file-tab').forEach(btn => {
          btn.onclick = () => { state.filePath = btn.dataset.file; renderInspector(); };
        });
      } else if (state.tab === 'console') {
        if (!run) { content.innerHTML = '<div class=\"event-meta\">No linked run.</div>'; return; }
        const errRows = runs
          .filter(r => Number(r.episode_id) === Number(run.episode_id))
          .map(r => r.error_signature || '')
          .filter(Boolean);
        const grouped = {};
        errRows.forEach(e => grouped[e] = (grouped[e] || 0) + 1);
        const topErr = Object.entries(grouped).sort((a, b) => b[1] - a[1]).slice(0, 8);
        content.innerHTML = `
          <div class=\"kv\">
            <div class=\"k\">exit_code</div><div>${esc(run.exit_code)}</div>
            <div class=\"k\">duration_s</div><div>${esc(Number(run.duration_s || 0).toFixed(3))}</div>
            <div class=\"k\">timed_out</div><div>${esc(run.timed_out)}</div>
            <div class=\"k\">error_signature</div><div>${esc(run.error_signature || '-')}</div>
          </div>
          <div style=\"height:8px\"></div>
          <div class=\"event-meta\">error clusters (episode)</div>
          <pre class=\"mono\">${esc(JSON.stringify(topErr, null, 2))}</pre>
          <div style=\"height:8px\"></div>
          <div class=\"event-meta\">stderr</div>
          <pre class=\"mono\">${esc(run.stderr || '')}</pre>
          <div style=\"height:8px\"></div>
          <div class=\"event-meta\">stdout</div>
          <pre class=\"mono\">${esc(run.stdout || '')}</pre>
        `;
      } else if (state.tab === 'artifacts') {
        if (!run) { content.innerHTML = '<div class=\"event-meta\">No linked run.</div>'; return; }
        const dc = run.data_card || {};
        const mc = run.method_card || {};
        const evidenceRows = evidenceCards
          .filter(e => String(e.run_id || '') === String(run.run_id || '') || Number(e.episode_id || 0) === Number(run.episode_id || 0))
          .slice(-20);
        const linkedPapers = papers.filter(p => Number(p.episode_id || 0) === Number(run.episode_id || 0)).slice(-10);
        const artifactLines = (run.artifacts || []).map(a => `<li class=\"mono\">${esc(a)}</li>`).join('') || '<li>-</li>';
        content.innerHTML = `
          <div class=\"kv\">
            <div class=\"k\">submission</div><div class=\"mono\">${esc(run.submission_path || '-')}</div>
            <div class=\"k\">code_log</div><div class=\"mono\">${esc(run.code_log_path || '-')}</div>
            <div class=\"k\">solver_log</div><div class=\"mono\">${esc(run.solver_log_path || '-')}</div>
            <div class=\"k\">workspace</div><div class=\"mono\">${esc(run.workspace_dir || '-')}</div>
            <div class=\"k\">snapshot</div><div class=\"mono\">${esc(run.snapshot_before_run || '-')}</div>
          </div>
          <div style=\"height:8px\"></div>
          <div class=\"event-meta\">run artifacts</div><ul>${artifactLines}</ul>
          <div class=\"event-meta\">data card</div>
          <pre class=\"mono\">${esc(JSON.stringify({
            task_name: dc.task_name, degraded: dc.degraded, split_stats: dc.split_stats, risk_flags: dc.risk_flags
          }, null, 2))}</pre>
          <div class=\"event-meta\">method card</div>
          <pre class=\"mono\">${esc(JSON.stringify({
            task_name: mc.task_name, metric: mc.metric, category: mc.category, baselines: (mc.recommended_baselines || []).map(x => x.name)
          }, null, 2))}</pre>
          <div class=\"event-meta\">lineage: run → evidence → paper</div>
          <pre class=\"mono\">${esc(JSON.stringify({
            run_id: run.run_id,
            evidence_refs: evidenceRows.map(e => e.evidence_id || e.task_id || e.kind),
            paper_refs: linkedPapers.map(p => p.paper_id),
          }, null, 2))}</pre>
        `;
      } else if (state.tab === 'reasoning') {
        const traces = actionTrace
          .filter(t => (state.episode <= 0 || Number(t.episode_id) === state.episode))
          .filter(t => !v.owner || String(t.agent_id || '') === String(v.owner || ''))
          .slice(-40);
        content.innerHTML = `
          <div class=\"event-meta\">structured trace (non-CoT)</div>
          <pre class=\"mono\">${esc(JSON.stringify((v.result || v), null, 2))}</pre>
          <div style=\"height:8px\"></div>
          <div class=\"event-meta\">recent action trace</div>
          <pre class=\"mono\">${esc(JSON.stringify(traces, null, 2))}</pre>
        `;
      } else if (state.tab === 'paper') {
        const ep = Number(v.episode_id || 0);
        const rows = papers.filter(p => ep <= 0 || Number(p.episode_id) === ep).slice(-20);
        const evRows = evidenceCards.filter(e => ep <= 0 || Number(e.episode_id) === ep).slice(-20);
        content.innerHTML = rows.length ? rows.map(p => `
          <div class=\"panel\" style=\"margin-bottom:8px;\">
            <div class=\"panel-body\">
              <div><b>${esc(p.paper_id || '-')}</b> <span class=\"event-meta\">${esc(p.source || '')}</span></div>
              <div class=\"kv\" style=\"margin-top:6px;\">
                <div class=\"k\">agent</div><div>${esc(p.agent_id || '-')}</div>
                <div class=\"k\">fitness</div><div>${esc(p.fitness)}</div>
                <div class=\"k\">f1</div><div>${esc(p.f1)}</div>
                <div class=\"k\">publishable</div><div>${esc(p.publishable)}</div>
                <div class=\"k\">replication_ok</div><div>${esc(p.replication_ok)}</div>
              </div>
            </div>
          </div>
        `).join('') : '<div class=\"event-meta\">No paper records.</div>';
        content.innerHTML += `<div style=\"height:8px\"></div><div class=\"event-meta\">evidence refs</div><pre class=\"mono\">${esc(JSON.stringify(evRows, null, 2))}</pre>`;
      } else if (state.tab === 'compare') {
        const byEp = {};
        runs.filter(r => state.episode <= 0 || Number(r.episode_id) === state.episode).forEach(r => {
          const epk = String(r.episode_id || 0);
          if (!byEp[epk]) byEp[epk] = { runs: 0, fail: 0, timeout: 0, avg_dur: 0, dev_scores: [] };
          byEp[epk].runs += 1;
          byEp[epk].avg_dur += Number(r.duration_s || 0);
          if (r.exit_code !== null && r.exit_code !== 0) byEp[epk].fail += 1;
          if (r.timed_out) byEp[epk].timeout += 1;
          if (r.dev_score != null) byEp[epk].dev_scores.push(Number(r.dev_score));
        });
        Object.values(byEp).forEach(vv => {
          vv.avg_dur = vv.runs ? vv.avg_dur / vv.runs : 0;
          vv.dev_avg = vv.dev_scores.length ? vv.dev_scores.reduce((a, b) => a + b, 0) / vv.dev_scores.length : null;
        });
        content.innerHTML = `
          <div class=\"event-meta\">episode compare (runs/errors/dev score)</div>
          <pre class=\"mono\">${esc(JSON.stringify(byEp, null, 2))}</pre>
        `;
      }
    }

    class CounterPolyfill {
      constructor(items) { this.map = new Map(); for (const x of items || []) this.map.set(x, (this.map.get(x) || 0) + 1); }
      get(k) { return this.map.get(k) || 0; }
    }

    function renderAll() {
      renderAgents();
      renderTimeline();
      renderTaskboard();
      renderInspector();
    }

    function bootstrap() {
      setHeader();
      setupEpisodeSelect();
      setupFilters();
      setupAutoRefresh();
      renderAll();
    }

    bootstrap();
  </script>
</body>
</html>
"""
    return template.replace("__PAYLOAD_JSON__", payload_json)


def generate_trend_dashboard(base_dir: str, out_dir: str) -> Dict[str, Any]:
    out_path = os.path.join(out_dir, "trend_dashboard.html")
    os.makedirs(out_dir, exist_ok=True)
    payload = _build_dashboard_payload(base_dir=base_dir)
    html_text = _render_html(payload)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html_text)
    return {
        "path": out_path,
        "status": "ok",
        "episodes": int((payload.get("meta") or {}).get("episodes_count", 0)),
        "meta": payload.get("meta") or {},
        "task_rows": len(((payload.get("taskboard") or {}).get("task_snapshot") or [])),
        "run_rows": len(((payload.get("runs") or {}).get("runs") or [])),
    }
