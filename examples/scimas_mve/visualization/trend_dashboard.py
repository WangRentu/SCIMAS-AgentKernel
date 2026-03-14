from __future__ import annotations

import hashlib
import json
import os
import re
import csv
import math
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


def _sha1_id(prefix: str, payload: Dict[str, Any]) -> str:
    raw = json.dumps(payload or {}, ensure_ascii=False, sort_keys=True, default=str)
    return f"{prefix}:{hashlib.sha1(raw.encode('utf-8')).hexdigest()[:16]}"


def _extract_text(v: Any) -> str:
    if isinstance(v, dict):
        if isinstance(v.get("text"), str):
            return str(v.get("text") or "")
        return json.dumps(v, ensure_ascii=False)
    return str(v or "")


def _clip(v: Any, n: int = 300) -> str:
    s = _extract_text(v)
    return s[:n]


def _estimate_tokens_from_chars(chars: int, chars_per_token: float = 4.0) -> int:
    if chars <= 0:
        return 0
    cpt = float(chars_per_token) if chars_per_token and chars_per_token > 0 else 4.0
    return max(1, int(round(float(chars) / cpt)))


def _llm_token_costs(input_tokens: int, output_tokens: int) -> Dict[str, float]:
    try:
        input_cny_per_mtok = float(os.getenv("SCIMAS_LLM_INPUT_CNY_PER_MTOKEN", "0.6"))
    except Exception:
        input_cny_per_mtok = 0.6
    try:
        output_cny_per_mtok = float(os.getenv("SCIMAS_LLM_OUTPUT_CNY_PER_MTOKEN", "1.2"))
    except Exception:
        output_cny_per_mtok = 1.2

    input_cost_cny = (float(max(0, input_tokens)) / 1_000_000.0) * input_cny_per_mtok
    output_cost_cny = (float(max(0, output_tokens)) / 1_000_000.0) * output_cny_per_mtok
    return {
        "input_cny_per_mtoken": float(input_cny_per_mtok),
        "output_cny_per_mtoken": float(output_cny_per_mtok),
        "input_cost_cny": float(input_cost_cny),
        "output_cost_cny": float(output_cost_cny),
        "total_cost_cny": float(input_cost_cny + output_cost_cny),
    }


def _normalize_llm_io_rows(rows: List[Dict[str, Any]], max_rows: int = 500) -> Dict[str, List[Dict[str, Any]]]:
    rows = _latest_by_reset(rows, "episode_id")
    summary: List[Dict[str, Any]] = []
    detail: List[Dict[str, Any]] = []
    try:
        chars_per_token = float(os.getenv("SCIMAS_LLM_CHARS_PER_TOKEN_EST", "4.0"))
    except Exception:
        chars_per_token = 4.0
    for rec in rows[-max_rows:]:
        meta = rec.get("meta") if isinstance(rec.get("meta"), dict) else {}
        inp = rec.get("input") if isinstance(rec.get("input"), dict) else {}
        out = rec.get("output") if isinstance(rec.get("output"), dict) else {}
        prompt_obj = inp.get("prompt")
        prompt_text = _extract_text(prompt_obj)
        raw_obj = out.get("raw_response")
        raw_text = _extract_text(raw_obj)
        parsed = out.get("parsed_json")
        prompt_tokens_est = _estimate_tokens_from_chars(len(prompt_text), chars_per_token=chars_per_token)
        completion_tokens_est = _estimate_tokens_from_chars(len(raw_text), chars_per_token=chars_per_token)
        total_tokens_est = int(prompt_tokens_est + completion_tokens_est)
        record = {
            "id": _sha1_id("llm", {"meta": meta, "prompt": prompt_text[:512], "raw": raw_text[:512]}),
            "ts": str(meta.get("ts") or ""),
            "tick": _to_int(meta.get("tick"), 0),
            "episode_id": _to_int(meta.get("episode_id"), 0),
            "task_name": str(meta.get("task_name") or ""),
            "agent_id": str(meta.get("agent_id") or ""),
            "action": str(meta.get("action") or ""),
            "kind": str(meta.get("kind") or ""),
            "ok_status": bool(out.get("ok", False)),
            "reason": str(out.get("reason") or ""),
            "prompt_chars": len(prompt_text),
            "response_chars": len(raw_text),
            "prompt_tokens_est": prompt_tokens_est,
            "completion_tokens_est": completion_tokens_est,
            "total_tokens_est": total_tokens_est,
            "prompt_preview": _clip(prompt_text, 260),
            "response_preview": _clip(raw_text, 260),
            "has_parsed_json": isinstance(parsed, (dict, list)),
            "prompt": prompt_text,
            "raw_response": raw_text,
            "parsed_json": parsed if isinstance(parsed, (dict, list)) else {},
            "record": rec,
        }
        summary.append(
            {
                "id": record["id"],
                "ts": record["ts"],
                "tick": record["tick"],
                "episode_id": record["episode_id"],
                "task_name": record["task_name"],
                "agent_id": record["agent_id"],
                "action": record["action"],
                "ok_status": record["ok_status"],
                "reason": record["reason"],
                "prompt_chars": record["prompt_chars"],
                "response_chars": record["response_chars"],
                "prompt_tokens_est": record["prompt_tokens_est"],
                "completion_tokens_est": record["completion_tokens_est"],
                "total_tokens_est": record["total_tokens_est"],
                "prompt_preview": record["prompt_preview"],
                "response_preview": record["response_preview"],
            }
        )
        detail.append(record)
    summary = sorted(summary, key=lambda x: x.get("ts", ""), reverse=True)
    detail = sorted(detail, key=lambda x: x.get("ts", ""), reverse=True)
    return {"rows_summary": summary, "rows_detail": detail}


def _collect_llm_usage_summary(llm_rows_summary: List[Dict[str, Any]]) -> Dict[str, Any]:
    rows = list(llm_rows_summary or [])
    prompt_tokens_total = 0
    completion_tokens_total = 0
    for row in rows:
        prompt_tokens_total += _to_int(row.get("prompt_tokens_est"), 0)
        completion_tokens_total += _to_int(row.get("completion_tokens_est"), 0)
    total_tokens = int(prompt_tokens_total + completion_tokens_total)
    costs = _llm_token_costs(prompt_tokens_total, completion_tokens_total)
    return {
        "sample_rows": len(rows),
        "input_tokens_est": int(prompt_tokens_total),
        "output_tokens_est": int(completion_tokens_total),
        "total_tokens_est": total_tokens,
        "pricing": {
            "input_cny_per_mtoken": costs["input_cny_per_mtoken"],
            "output_cny_per_mtoken": costs["output_cny_per_mtoken"],
        },
        "cost_cny": {
            "input": costs["input_cost_cny"],
            "output": costs["output_cost_cny"],
            "total": costs["total_cost_cny"],
        },
        "token_estimation": {
            "method": "chars_div_chars_per_token",
            "chars_per_token": float(os.getenv("SCIMAS_LLM_CHARS_PER_TOKEN_EST", "4.0")),
        },
    }


def _normalize_rag_io_rows(rows: List[Dict[str, Any]], max_rows: int = 500) -> Dict[str, List[Dict[str, Any]]]:
    rows = _latest_by_reset(rows, "episode_id")
    summary: List[Dict[str, Any]] = []
    detail: List[Dict[str, Any]] = []
    for rec in rows[-max_rows:]:
        meta = rec.get("meta") if isinstance(rec.get("meta"), dict) else {}
        inp = rec.get("input") if isinstance(rec.get("input"), dict) else {}
        out = rec.get("output") if isinstance(rec.get("output"), dict) else {}
        selected_rows = out.get("selected_rows") if isinstance(out.get("selected_rows"), list) else []
        record = {
            "id": _sha1_id("rag", {"meta": meta, "in": inp, "out": {"status": out.get("status"), "rows": len(selected_rows)}}),
            "ts": str(meta.get("ts") or ""),
            "episode_id": _to_int(meta.get("episode_id"), 0),
            "task_name": str(meta.get("task_name") or ""),
            "agent_id": str(meta.get("agent_id") or ""),
            "action": str(meta.get("action") or ""),
            "operation": str(meta.get("operation") or ""),
            "run_id": str(meta.get("run_id") or ""),
            "paper_id": str(meta.get("paper_id") or ""),
            "status": str(out.get("status") or ""),
            "fallback_reason": str(out.get("fallback_reason") or ""),
            "result_count": _to_int(out.get("result_count"), 0),
            "selected_count": _to_int(out.get("selected_count"), 0),
            "retrieval_mode": str(out.get("retrieval_mode") or inp.get("retrieve_mode") or ""),
            "query_text": _extract_text(inp.get("query_text")),
            "collections": inp.get("collections") if isinstance(inp.get("collections"), list) else [],
            "quotas": inp.get("quotas") if isinstance(inp.get("quotas"), dict) else {},
            "selected_rows": selected_rows[:40],
            "refs": out.get("refs") if isinstance(out.get("refs"), list) else [],
            "context": _extract_text(out.get("context")),
            "record": rec,
        }
        summary.append(
            {
                "id": record["id"],
                "ts": record["ts"],
                "episode_id": record["episode_id"],
                "task_name": record["task_name"],
                "agent_id": record["agent_id"],
                "action": record["action"],
                "operation": record["operation"],
                "status": record["status"],
                "fallback_reason": record["fallback_reason"],
                "selected_count": record["selected_count"],
                "retrieval_mode": record["retrieval_mode"],
                "query_preview": _clip(record["query_text"], 220),
            }
        )
        detail.append(record)
    summary = sorted(summary, key=lambda x: x.get("ts", ""), reverse=True)
    detail = sorted(detail, key=lambda x: x.get("ts", ""), reverse=True)
    return {"rows_summary": summary, "rows_detail": detail}


def _normalize_retrieve_pipeline_rows(rows: List[Dict[str, Any]], max_rows: int = 500) -> Dict[str, List[Dict[str, Any]]]:
    rows = _latest_by_reset(rows, "episode_id")
    summary: List[Dict[str, Any]] = []
    detail: List[Dict[str, Any]] = []
    for rec in rows[-max_rows:]:
        payload = rec.get("payload") if isinstance(rec.get("payload"), dict) else {}
        row = {
            "id": _sha1_id("retrieve_pipeline", {"ts": rec.get("ts"), "agent_id": rec.get("agent_id"), "phase": rec.get("phase"), "payload": payload}),
            "ts": str(rec.get("ts") or ""),
            "episode_id": _to_int(rec.get("episode_id"), 0),
            "task_name": str(rec.get("task_name") or ""),
            "agent_id": str(rec.get("agent_id") or ""),
            "phase": str(rec.get("phase") or ""),
            "refresh": bool(rec.get("refresh", False)),
            "payload": payload,
            "record": rec,
        }
        summary.append(
            {
                "id": row["id"],
                "ts": row["ts"],
                "episode_id": row["episode_id"],
                "task_name": row["task_name"],
                "agent_id": row["agent_id"],
                "phase": row["phase"],
                "ok": bool((payload or {}).get("ok", True)),
                "source": str((payload or {}).get("source") or ""),
                "reward": (payload or {}).get("reward"),
            }
        )
        detail.append(row)
    summary = sorted(summary, key=lambda x: x.get("ts", ""), reverse=True)
    detail = sorted(detail, key=lambda x: x.get("ts", ""), reverse=True)
    return {"rows_summary": summary, "rows_detail": detail}


def _normalize_retrieve_guardrail_rows(rows: List[Dict[str, Any]], max_rows: int = 500) -> Dict[str, List[Dict[str, Any]]]:
    rows = _latest_by_reset(rows, "episode_id")
    summary: List[Dict[str, Any]] = []
    detail: List[Dict[str, Any]] = []
    for rec in rows[-max_rows:]:
        quality = rec.get("quality") if isinstance(rec.get("quality"), dict) else {}
        row = {
            "id": _sha1_id("retrieve_guardrail", {"ts": rec.get("ts"), "agent_id": rec.get("agent_id"), "quality": quality}),
            "ts": str(rec.get("ts") or ""),
            "episode_id": _to_int(rec.get("episode_id"), 0),
            "task_name": str(rec.get("task_name") or ""),
            "agent_id": str(rec.get("agent_id") or ""),
            "source": str(rec.get("source") or ""),
            "quality": quality,
            "record": rec,
        }
        summary.append(
            {
                "id": row["id"],
                "ts": row["ts"],
                "episode_id": row["episode_id"],
                "task_name": row["task_name"],
                "agent_id": row["agent_id"],
                "source": row["source"],
                "level": str(quality.get("level") or ""),
                "degraded": bool(quality.get("degraded", False)),
                "citation_coverage": _to_float(quality.get("citation_coverage"), 0.0),
                "executable_minimum": bool(quality.get("executable_minimum", False)),
            }
        )
        detail.append(row)
    summary = sorted(summary, key=lambda x: x.get("ts", ""), reverse=True)
    detail = sorted(detail, key=lambda x: x.get("ts", ""), reverse=True)
    return {"rows_summary": summary, "rows_detail": detail}


def _normalize_retrieve_evidence_rows(rows: List[Dict[str, Any]], max_rows: int = 500) -> Dict[str, List[Dict[str, Any]]]:
    rows = _latest_by_reset(rows, "episode_id")
    summary: List[Dict[str, Any]] = []
    detail: List[Dict[str, Any]] = []
    for rec in rows[-max_rows:]:
        evidence = rec.get("evidence") if isinstance(rec.get("evidence"), list) else []
        row = {
            "id": _sha1_id("retrieve_evidence", {"ts": rec.get("ts"), "agent_id": rec.get("agent_id"), "count": len(evidence)}),
            "ts": str(rec.get("ts") or ""),
            "episode_id": _to_int(rec.get("episode_id"), 0),
            "task_name": str(rec.get("task_name") or ""),
            "agent_id": str(rec.get("agent_id") or ""),
            "rag_status": str(rec.get("rag_status") or ""),
            "evidence_count": _to_int(rec.get("evidence_count"), len(evidence)),
            "evidence": evidence[:60],
            "record": rec,
        }
        summary.append(
            {
                "id": row["id"],
                "ts": row["ts"],
                "episode_id": row["episode_id"],
                "task_name": row["task_name"],
                "agent_id": row["agent_id"],
                "rag_status": row["rag_status"],
                "evidence_count": row["evidence_count"],
            }
        )
        detail.append(row)
    summary = sorted(summary, key=lambda x: x.get("ts", ""), reverse=True)
    detail = sorted(detail, key=lambda x: x.get("ts", ""), reverse=True)
    return {"rows_summary": summary, "rows_detail": detail}


def _collect_llm_io(base_dir: str, max_rows: int = 500) -> Dict[str, List[Dict[str, Any]]]:
    path = os.path.join(base_dir, "logs", "app", "audit", "llm_io.jsonl")
    return _normalize_llm_io_rows(_read_jsonl(path), max_rows=max_rows)


def _collect_rag_io(base_dir: str, max_rows: int = 500) -> Dict[str, List[Dict[str, Any]]]:
    path = os.path.join(base_dir, "logs", "app", "audit", "rag_io.jsonl")
    return _normalize_rag_io_rows(_read_jsonl(path), max_rows=max_rows)


def _collect_retrieve_pipeline(base_dir: str, max_rows: int = 500) -> Dict[str, List[Dict[str, Any]]]:
    path = os.path.join(base_dir, "logs", "app", "action", "retrieve_pipeline.jsonl")
    return _normalize_retrieve_pipeline_rows(_read_jsonl(path), max_rows=max_rows)


def _collect_retrieve_guardrail(base_dir: str, max_rows: int = 500) -> Dict[str, List[Dict[str, Any]]]:
    path = os.path.join(base_dir, "logs", "app", "action", "retrieve_guardrail.jsonl")
    return _normalize_retrieve_guardrail_rows(_read_jsonl(path), max_rows=max_rows)


def _collect_retrieve_evidence(base_dir: str, max_rows: int = 500) -> Dict[str, List[Dict[str, Any]]]:
    path = os.path.join(base_dir, "logs", "app", "action", "retrieve_evidence.jsonl")
    return _normalize_retrieve_evidence_rows(_read_jsonl(path), max_rows=max_rows)


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


def _to_optional_float(v: Any) -> Optional[float]:
    try:
        if v is None or isinstance(v, bool):
            return None
        return float(v)
    except Exception:
        return None


def _to_optional_finite_float(v: Any) -> Optional[float]:
    n = _to_optional_float(v)
    if n is None:
        return None
    if not math.isfinite(n):
        return None
    return n


def _infer_metric_lower_is_better(metric_name: str) -> Optional[bool]:
    metric = str(metric_name or "").strip().lower()
    if not metric:
        return None
    if re.search(r"(mase|mae|mse|rmse|mape|smape|loss|error|nll|cross.?entropy|perplexity|wer|cer|distance|latency)", metric):
        return True
    if re.search(r"(acc|accuracy|auc|f1|precision|recall|r2|ndcg|map|bleu|rouge|pass@|hit@)", metric):
        return False
    return None


def _read_submission_preview(path: str) -> Dict[str, Any]:
    preview = {"exists": False, "columns": [], "has_rows": False}
    if not os.path.exists(path):
        return preview
    preview["exists"] = True
    try:
        with open(path, "r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            preview["columns"] = [str(x).strip() for x in (next(reader, []) or []) if str(x).strip()]
            preview["has_rows"] = next(reader, None) is not None
    except Exception:
        return preview
    return preview


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
            workspace_dir = os.path.join(ep_path, run_name, "workspace")
            code_run_path = os.path.join(agent_log, "code_run.json")
            solver_run_path = os.path.join(agent_log, "solver_run.json")
            submission_path = os.path.join(agent_log, "submission.csv")
            if not os.path.exists(submission_path):
                submission_path = os.path.join(workspace_dir, "outputs", "submission.csv")
            dev_metrics_path = os.path.join(workspace_dir, "outputs", "dev_metrics.json")
            dev_predictions_path = os.path.join(workspace_dir, "outputs", "dev_predictions.csv")
            manifest_path = os.path.join(workspace_dir, ".task_manifest.json")

            code_log = _read_json(code_run_path)
            solver_log = _read_json(solver_run_path)
            dev_metrics = _read_json(dev_metrics_path)
            task_manifest = _read_json(manifest_path)
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
            raw_score = _to_optional_finite_float((dev_metrics or {}).get("raw_score"))
            parsed_dev_score = _extract_dev_score(stdout_text, stderr_text)
            dev_score = raw_score if raw_score is not None else _to_optional_finite_float(parsed_dev_score)
            metric_name = str(
                (dev_metrics or {}).get("metric_name")
                or (task_manifest or {}).get("metric")
                or (method_card or {}).get("metric")
                or ""
            )
            metric_lower_is_better = _infer_metric_lower_is_better(metric_name)
            selection_score = None
            if dev_score is not None and metric_lower_is_better is not None:
                selection_score = -float(dev_score) if metric_lower_is_better else float(dev_score)
            submission_preview = _read_submission_preview(submission_path)
            scoring_column = (task_manifest or {}).get("scoring_column")
            if isinstance(scoring_column, list):
                required_submission_columns = [str(x).strip() for x in scoring_column if str(x).strip()]
            elif isinstance(scoring_column, str) and scoring_column.strip():
                required_submission_columns = [scoring_column.strip()]
            else:
                required_submission_columns = []
            preflight_ok = bool(submission_preview.get("exists") and submission_preview.get("has_rows"))
            if required_submission_columns:
                preflight_ok = preflight_ok and submission_preview.get("columns") == required_submission_columns
            exit_code = run_result.get("exit_code")
            exec_ok = bool(exit_code in (None, 0) and os.path.exists(submission_path))
            has_dev_proxy = bool(raw_score is not None or os.path.exists(dev_predictions_path) or dev_score is not None)
            scientific_ok = bool(exec_ok and has_dev_proxy)
            scientific_reason_parts: List[str] = []
            if not exec_ok:
                scientific_reason_parts.append("exec_failed")
            if not has_dev_proxy:
                scientific_reason_parts.append("missing_dev_proxy")
            scientific_reason = "ok" if scientific_ok else "|".join(scientific_reason_parts)
            publish_ready = bool(scientific_ok and preflight_ok)
            publish_reason_parts: List[str] = list(scientific_reason_parts)
            if scientific_ok and not preflight_ok:
                publish_reason_parts.append("submission_preflight_failed")
            publish_reason = "ok" if publish_ready else "|".join(publish_reason_parts)
            error_signature = _extract_error_signature(stderr_text)
            rows.append(
                {
                    "episode_id": ep_info.get("episode_id", 0),
                    "task_name": (code_log or {}).get("task_name") or ep_info.get("task_name", ""),
                    "run_id": run_name,
                    "run_key": f"{ep_info.get('episode_id', 0)}::{run_name}",
                    "executor": (code_log or {}).get("executor_used") or "",
                    "data_bridge_used": bool((code_log or {}).get("data_bridge_used", False)),
                    "effective_run_cmd": str((code_log or {}).get("effective_run_cmd") or ""),
                    "exit_code": run_result.get("exit_code"),
                    "duration_s": _to_float(run_result.get("duration_s"), 0.0),
                    "timed_out": bool(run_result.get("timed_out", False)),
                    "stderr": stderr_text[-12000:],
                    "stdout": stdout_text[-12000:],
                    "error_signature": error_signature,
                    "dev_score": dev_score,
                    "raw_score": dev_score,
                    "metric_name": metric_name,
                    "metric_lower_is_better": metric_lower_is_better,
                    "selection_score": selection_score,
                    "exec_ok": exec_ok,
                    "has_dev_proxy": has_dev_proxy,
                    "preflight_ok": preflight_ok,
                    "scientific_ok": scientific_ok,
                    "scientific_reason": scientific_reason,
                    "publish_ready": publish_ready,
                    "publish_reason": publish_reason,
                    # Backward-compatible aliases for older frontend code paths.
                    "evidence_ok": publish_ready,
                    "evidence_reason": publish_reason,
                    "required_submission_columns": required_submission_columns,
                    "submission_columns": list(submission_preview.get("columns") or []),
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
                    "workspace_dir": str((code_log or {}).get("workspace_dir") or workspace_dir),
                    "snapshot_before_run": str((code_log or {}).get("snapshot_before_run") or ""),
                    "dev_metrics_path": dev_metrics_path if dev_metrics else "",
                    "dev_predictions_path": dev_predictions_path if os.path.exists(dev_predictions_path) else "",
                    "task_manifest_path": manifest_path if task_manifest else "",
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


def _collect_code_loop_logs(base_dir: str, max_rows: int = 300) -> List[Dict[str, Any]]:
    path = os.path.join(base_dir, "logs", "app", "action", "code_loop.jsonl")
    rows = _latest_by_reset(_read_jsonl(path), "episode_id")
    out: List[Dict[str, Any]] = []
    for r in rows[-max_rows:]:
        out.append(
            {
                "ts": str(r.get("ts") or ""),
                "tick": _to_int(r.get("tick"), 0),
                "episode_id": _to_int(r.get("episode_id"), 0),
                "task_name": str(r.get("task_name") or ""),
                "agent_id": str(r.get("agent_id") or ""),
                "attempt_count": _to_int(r.get("attempt_count"), 0),
                "best_dev_score_norm": r.get("best_dev_score_norm"),
                "attempts": list(r.get("attempts") or []),
            }
        )
    return out


def _collect_precondition_gates(base_dir: str, max_rows: int = 400) -> List[Dict[str, Any]]:
    path = os.path.join(base_dir, "logs", "app", "action", "precondition_gate.jsonl")
    rows = _latest_by_reset(_read_jsonl(path), "episode_id")
    out: List[Dict[str, Any]] = []
    for r in rows[-max_rows:]:
        out.append(
            {
                "ts": str(r.get("ts") or ""),
                "tick": _to_int(r.get("tick"), 0),
                "episode_id": _to_int(r.get("episode_id"), 0),
                "task_name": str(r.get("task_name") or ""),
                "agent_id": str(r.get("agent_id") or ""),
                "action": str(r.get("action") or ""),
                "phase": str(r.get("phase") or ""),
                "failures": list(r.get("failures") or []),
                "summary": r.get("summary") if isinstance(r.get("summary"), dict) else {},
            }
        )
    return out


def _collect_eval_failures(base_dir: str, max_rows: int = 500) -> List[Dict[str, Any]]:
    path = os.path.join(base_dir, "logs", "app", "environment", "eval_failures.jsonl")
    rows = _latest_by_reset(_read_jsonl(path), "episode_id")
    out: List[Dict[str, Any]] = []
    for r in rows[-max_rows:]:
        out.append(
            {
                "ts": str(r.get("ts") or ""),
                "episode_id": _to_int(r.get("episode_id"), 0),
                "task_name": str(r.get("task_name") or ""),
                "task_metric": str(r.get("task_metric") or ""),
                "stage": str(r.get("stage") or ""),
                "rc": _to_int(r.get("rc"), 0),
                "error_type": str(r.get("error_type") or ""),
                "submission_exists": bool(r.get("submission_exists", False)),
                "submission_path": str(r.get("submission_path") or ""),
                "stderr_tail": str(r.get("stderr_tail") or "")[-1600:],
                "stdout_tail": str(r.get("stdout_tail") or "")[-1200:],
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
    code_loops = _collect_code_loop_logs(base_dir)
    precondition_gates = _collect_precondition_gates(base_dir)
    eval_failures = _collect_eval_failures(base_dir)
    llm_io = _collect_llm_io(base_dir, max_rows=800)
    llm_usage_summary = _collect_llm_usage_summary((llm_io or {}).get("rows_summary") or [])
    rag_io = _collect_rag_io(base_dir, max_rows=800)
    retrieve_pipeline = _collect_retrieve_pipeline(base_dir, max_rows=800)
    retrieve_guardrail = _collect_retrieve_guardrail(base_dir, max_rows=800)
    retrieve_evidence = _collect_retrieve_evidence(base_dir, max_rows=800)

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
        "code_loops": code_loops,
        "precondition_gates": precondition_gates,
        "eval_failures": eval_failures,
        "audit": {
            "llm_usage_summary": llm_usage_summary,
            "llm_io_summary": (llm_io or {}).get("rows_summary", [])[:120],
            "rag_io_summary": (rag_io or {}).get("rows_summary", [])[:120],
            "retrieve_pipeline_summary": (retrieve_pipeline or {}).get("rows_summary", [])[:120],
            "retrieve_guardrail_summary": (retrieve_guardrail or {}).get("rows_summary", [])[:120],
            "retrieve_evidence_summary": (retrieve_evidence or {}).get("rows_summary", [])[:120],
        },
    }


def build_dashboard_payload(base_dir: str) -> Dict[str, Any]:
    """Public payload builder for separated frontend-backend dashboard servers."""
    return _build_dashboard_payload(base_dir=base_dir)


def normalize_llm_io_rows(rows: List[Dict[str, Any]], max_rows: int = 500) -> Dict[str, List[Dict[str, Any]]]:
    return _normalize_llm_io_rows(rows, max_rows=max_rows)


def normalize_rag_io_rows(rows: List[Dict[str, Any]], max_rows: int = 500) -> Dict[str, List[Dict[str, Any]]]:
    return _normalize_rag_io_rows(rows, max_rows=max_rows)


def normalize_retrieve_pipeline_rows(rows: List[Dict[str, Any]], max_rows: int = 500) -> Dict[str, List[Dict[str, Any]]]:
    return _normalize_retrieve_pipeline_rows(rows, max_rows=max_rows)


def normalize_retrieve_guardrail_rows(rows: List[Dict[str, Any]], max_rows: int = 500) -> Dict[str, List[Dict[str, Any]]]:
    return _normalize_retrieve_guardrail_rows(rows, max_rows=max_rows)


def normalize_retrieve_evidence_rows(rows: List[Dict[str, Any]], max_rows: int = 500) -> Dict[str, List[Dict[str, Any]]]:
    return _normalize_retrieve_evidence_rows(rows, max_rows=max_rows)


def generate_trend_dashboard(base_dir: str, out_dir: str) -> Dict[str, Any]:
    out_path = os.path.join(out_dir, "trend_dashboard.html")
    os.makedirs(out_dir, exist_ok=True)
    payload = _build_dashboard_payload(base_dir=base_dir)
    # In a true separated architecture, we might just save the JSON
    with open(os.path.join(out_dir, "dashboard_data.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    return {
        "path": out_path,
        "status": "ok",
        "episodes": int((payload.get("meta") or {}).get("episodes_count", 0)),
        "meta": payload.get("meta") or {},
        "task_rows": len(((payload.get("taskboard") or {}).get("task_snapshot") or [])),
        "run_rows": len(((payload.get("runs") or {}).get("runs") or [])),
    }
