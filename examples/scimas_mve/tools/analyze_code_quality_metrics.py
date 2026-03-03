#!/usr/bin/env python3
import argparse
import json
import os
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple


def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    if not os.path.exists(path):
        return records
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if isinstance(obj, dict):
                records.append(obj)
    return records


def _as_dt(ts: Any) -> datetime:
    if not isinstance(ts, str):
        return datetime.min
    try:
        if ts.endswith("Z"):
            ts = ts[:-1] + "+00:00"
        return datetime.fromisoformat(ts)
    except Exception:
        return datetime.min


def _group_key(rec: Dict[str, Any]) -> Tuple[Any, Any, Any]:
    return (rec.get("episode_id"), rec.get("task_name"), rec.get("agent_id"))


def _failed_diag(rec: Dict[str, Any]) -> bool:
    d = rec.get("diagnosis") if isinstance(rec.get("diagnosis"), dict) else {}
    err_cls = str(d.get("error_class") or "")
    return bool(err_cls and err_cls != "none")


def _error_codes(rec: Dict[str, Any]) -> List[str]:
    d = rec.get("diagnosis") if isinstance(rec.get("diagnosis"), dict) else {}
    raw = d.get("error_codes")
    if isinstance(raw, list):
        return [str(x) for x in raw if str(x)]
    return []


def compute_duplicate_error_repeat_rate(diagnosis_records: List[Dict[str, Any]]) -> Dict[str, Any]:
    grouped: Dict[Tuple[Any, Any, Any], List[Dict[str, Any]]] = defaultdict(list)
    for r in diagnosis_records:
        grouped[_group_key(r)].append(r)
    adjacent_pairs = 0
    repeats = 0
    for _, recs in grouped.items():
        recs.sort(key=lambda x: (_as_dt(x.get("ts")), int(x.get("tick") or 0)))
        failed = [r for r in recs if _failed_diag(r)]
        for i in range(1, len(failed)):
            adjacent_pairs += 1
            prev_codes = set(_error_codes(failed[i - 1]))
            curr_codes = set(_error_codes(failed[i]))
            if prev_codes & curr_codes:
                repeats += 1
    rate = (repeats / adjacent_pairs) if adjacent_pairs > 0 else 0.0
    return {
        "adjacent_failure_pairs": adjacent_pairs,
        "repeated_pairs": repeats,
        "rate": rate,
    }


def compute_repair_recovery_rate(diagnosis_records: List[Dict[str, Any]]) -> Dict[str, Any]:
    grouped: Dict[Tuple[Any, Any, Any], List[Dict[str, Any]]] = defaultdict(list)
    for r in diagnosis_records:
        grouped[_group_key(r)].append(r)
    total_failed_repairs = 0
    recovered = 0
    horizon = 3
    for _, recs in grouped.items():
        recs.sort(key=lambda x: (_as_dt(x.get("ts")), int(x.get("tick") or 0)))
        for i, rec in enumerate(recs):
            if str(rec.get("decision") or "") != "repair":
                continue
            if not _failed_diag(rec):
                continue
            total_failed_repairs += 1
            window = recs[i + 1 : i + 1 + horizon]
            success_found = any(
                isinstance(w.get("diagnosis"), dict) and str((w.get("diagnosis") or {}).get("error_class") or "") == "none"
                for w in window
            )
            if success_found:
                recovered += 1
    rate = (recovered / total_failed_repairs) if total_failed_repairs > 0 else 0.0
    return {
        "failed_repair_attempts": total_failed_repairs,
        "recovered_within_next_3": recovered,
        "rate": rate,
    }


def compute_stagnation_before_optimize(code_loop_records: List[Dict[str, Any]]) -> Dict[str, Any]:
    sessions = 0
    total_invalid_before_opt = 0
    for rec in code_loop_records:
        attempts = rec.get("attempts")
        if not isinstance(attempts, list) or not attempts:
            continue
        first_opt_idx = None
        for i, att in enumerate(attempts):
            if str((att or {}).get("phase") or "") == "optimize":
                first_opt_idx = i
                break
        if first_opt_idx is None:
            continue
        sessions += 1
        invalid = 0
        for att in attempts[:first_opt_idx]:
            ok = bool((att or {}).get("ok", False))
            if not ok:
                invalid += 1
        total_invalid_before_opt += invalid
    avg = (total_invalid_before_opt / sessions) if sessions > 0 else 0.0
    return {
        "sessions_with_optimize": sessions,
        "total_invalid_before_optimize": total_invalid_before_opt,
        "avg_invalid_before_optimize": avg,
    }


def compute_format_failure_share(diagnosis_records: List[Dict[str, Any]]) -> Dict[str, Any]:
    total_failed = 0
    format_failed = 0
    for rec in diagnosis_records:
        if not _failed_diag(rec):
            continue
        total_failed += 1
        diag = rec.get("diagnosis") if isinstance(rec.get("diagnosis"), dict) else {}
        if str(diag.get("error_class") or "") == "format":
            format_failed += 1
    share = (format_failed / total_failed) if total_failed > 0 else 0.0
    return {
        "total_failed_diagnoses": total_failed,
        "format_failed": format_failed,
        "share": share,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze code-loop quality metrics from SCIMAS logs.")
    parser.add_argument(
        "--project-root",
        default="examples/scimas_mve",
        help="Project root containing logs/app/action.",
    )
    parser.add_argument(
        "--diagnosis-log",
        default="",
        help="Optional explicit path for code_diagnosis.jsonl",
    )
    parser.add_argument(
        "--code-loop-log",
        default="",
        help="Optional explicit path for code_loop.jsonl",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Output JSON path (default: <project-root>/logs/app/simulation/code_quality_metrics.json)",
    )
    args = parser.parse_args()

    project_root = os.path.abspath(args.project_root)
    diagnosis_log = args.diagnosis_log or os.path.join(project_root, "logs", "app", "action", "code_diagnosis.jsonl")
    code_loop_log = args.code_loop_log or os.path.join(project_root, "logs", "app", "action", "code_loop.jsonl")
    output_path = args.output or os.path.join(project_root, "logs", "app", "simulation", "code_quality_metrics.json")

    diagnosis_records = _read_jsonl(diagnosis_log)
    code_loop_records = _read_jsonl(code_loop_log)

    metrics = {
        "meta": {
            "project_root": project_root,
            "diagnosis_log": diagnosis_log,
            "code_loop_log": code_loop_log,
            "diagnosis_records": len(diagnosis_records),
            "code_loop_records": len(code_loop_records),
            "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        },
        "duplicate_error_repeat_rate": compute_duplicate_error_repeat_rate(diagnosis_records),
        "repair_recovery_rate": compute_repair_recovery_rate(diagnosis_records),
        "stagnation_before_optimize": compute_stagnation_before_optimize(code_loop_records),
        "format_failure_share": compute_format_failure_share(diagnosis_records),
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(f"[OK] wrote metrics -> {output_path}")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
