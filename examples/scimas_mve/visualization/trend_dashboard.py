from __future__ import annotations

import html
import json
import os
from collections import defaultdict
from typing import Any, Dict, List, Tuple


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
    prev = rows[0].get(key)
    try:
        prev_num = int(prev)
    except Exception:
        prev_num = None
    for idx in range(1, len(rows)):
        cur = rows[idx].get(key)
        try:
            cur_num = int(cur)
        except Exception:
            cur_num = None
        if prev_num is not None and cur_num is not None and cur_num < prev_num:
            start_idx = idx
        prev_num = cur_num
    return rows[start_idx:]


def _latest_team_metrics(path: str) -> List[Dict[str, Any]]:
    rows = _read_jsonl(path)
    latest = _latest_by_reset(rows, "episode_index")
    return sorted(latest, key=lambda r: int(r.get("episode_index", 0) or 0))


def _tail_evolution_for_run(path: str, episodes_count: int) -> List[Dict[str, Any]]:
    rows = _read_jsonl(path)
    if not rows or episodes_count <= 1:
        return []
    return rows[-max(0, episodes_count - 1) :]


def _latest_taskboard(path: str) -> List[Dict[str, Any]]:
    return _latest_by_reset(_read_jsonl(path), "episode_id")


def _latest_papers(path: str) -> List[Dict[str, Any]]:
    return _latest_by_reset(_read_jsonl(path), "episode_id")


def _latest_chain_metrics(path: str) -> List[Dict[str, Any]]:
    rows = _read_jsonl(path)
    latest = _latest_by_reset(rows, "episode_index")
    return sorted(latest, key=lambda r: int(r.get("episode_index", 0) or 0))


def _to_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


def _to_int(v: Any, default: int = 0) -> int:
    try:
        return int(v)
    except Exception:
        return default


def _sparkline_svg(
    values: List[float],
    width: int = 420,
    height: int = 120,
    color: str = "#1f77b4",
    fill: str | None = None,
) -> str:
    if not values:
        return f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}"></svg>'
    lo = min(values)
    hi = max(values)
    if abs(hi - lo) < 1e-12:
        hi = lo + 1.0
    n = max(1, len(values) - 1)
    pts: List[str] = []
    for i, v in enumerate(values):
        x = (i / n) * (width - 1)
        y = (1.0 - (v - lo) / (hi - lo)) * (height - 18) + 8
        pts.append(f"{x:.1f},{y:.1f}")
    polyline = " ".join(pts)
    baseline = height - 10
    area = ""
    if fill:
        area = (
            f'<polygon points="0,{baseline} {polyline} {width - 1},{baseline}" '
            f'fill="{fill}" opacity="0.15"></polygon>'
        )
    return (
        f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-label="chart">'
        f'<line x1="0" y1="{baseline}" x2="{width}" y2="{baseline}" stroke="#d9dde3" stroke-width="1"/>'
        f"{area}"
        f'<polyline points="{polyline}" fill="none" stroke="{color}" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"/>'
        f"</svg>"
    )


def _badge(ok: bool, label_true: str = "OK", label_false: str = "NO") -> str:
    cls = "ok" if ok else "bad"
    label = label_true if ok else label_false
    return f'<span class="badge {cls}">{html.escape(label)}</span>'


def _aggregate_taskboard(rows: List[Dict[str, Any]], episodes_count: int) -> Dict[int, Dict[str, Any]]:
    stats: Dict[int, Dict[str, Any]] = {
        ep: {"create": 0, "claim": 0, "complete": 0, "release": 0, "release_reasons": defaultdict(int)}
        for ep in range(1, episodes_count + 1)
    }
    for row in rows:
        ep = _to_int(row.get("episode_id"), -1)
        if ep < 1 or ep > episodes_count:
            continue
        event = str(row.get("event") or "")
        if event in stats[ep]:
            stats[ep][event] += 1
        if event == "release":
            meta = row.get("meta") or {}
            reason = str(meta.get("reason") or "unknown")
            stats[ep]["release_reasons"][reason] += 1
    for ep in stats:
        rel = int(stats[ep]["release"] or 0)
        comp = int(stats[ep]["complete"] or 0)
        stats[ep]["release_complete_ratio"] = (rel / comp) if comp else (1.0 if rel else 0.0)
        stats[ep]["top_release_reason"] = ""
        if stats[ep]["release_reasons"]:
            stats[ep]["top_release_reason"] = max(
                stats[ep]["release_reasons"].items(), key=lambda kv: kv[1]
            )[0]
        stats[ep]["release_reasons"] = dict(stats[ep]["release_reasons"])
    return stats


def _aggregate_papers(rows: List[Dict[str, Any]], episodes_count: int) -> Dict[int, Dict[str, Any]]:
    stats: Dict[int, Dict[str, Any]] = {
        ep: {
            "writes": 0,
            "reviews": 0,
            "avg_write_fitness": 0.0,
            "avg_write_f1": 0.0,
            "replication_verified_rate": 0.0,
            "replication_ok_rate": 0.0,
            "holdout_pass_rate": 0.0,
        }
        for ep in range(1, episodes_count + 1)
    }
    write_fit_vals: Dict[int, List[float]] = defaultdict(list)
    write_f1_vals: Dict[int, List[float]] = defaultdict(list)
    rep_verified: Dict[int, List[int]] = defaultdict(list)
    rep_ok: Dict[int, List[int]] = defaultdict(list)
    holdout_pass: Dict[int, List[int]] = defaultdict(list)

    for row in rows:
        ep = _to_int(row.get("episode_id"), -1)
        if ep < 1 or ep > episodes_count:
            continue
        source = str(row.get("source") or "")
        metrics = row.get("metrics") or {}
        if source == "write":
            stats[ep]["writes"] += 1
            if "fitness" in metrics:
                write_fit_vals[ep].append(_to_float(metrics.get("fitness")))
            if "f1" in metrics:
                write_f1_vals[ep].append(_to_float(metrics.get("f1")))
        elif source == "review":
            stats[ep]["reviews"] += 1
        if metrics:
            rep_verified[ep].append(1 if bool(metrics.get("replication_verified", False)) else 0)
            rep_ok[ep].append(1 if bool(metrics.get("replication_ok", False)) else 0)
            if metrics.get("replication_holdout_pass") is not None:
                holdout_pass[ep].append(1 if bool(metrics.get("replication_holdout_pass", False)) else 0)

    for ep in range(1, episodes_count + 1):
        wf = write_fit_vals.get(ep, [])
        w1 = write_f1_vals.get(ep, [])
        rv = rep_verified.get(ep, [])
        ro = rep_ok.get(ep, [])
        hp = holdout_pass.get(ep, [])
        stats[ep]["avg_write_fitness"] = sum(wf) / len(wf) if wf else 0.0
        stats[ep]["avg_write_f1"] = sum(w1) / len(w1) if w1 else 0.0
        stats[ep]["replication_verified_rate"] = sum(rv) / len(rv) if rv else 0.0
        stats[ep]["replication_ok_rate"] = sum(ro) / len(ro) if ro else 0.0
        stats[ep]["holdout_pass_rate"] = sum(hp) / len(hp) if hp else 0.0
    return stats


def _diagnosis(episode_rows: List[Dict[str, Any]]) -> List[str]:
    if not episode_rows:
        return ["没有可用 episode 指标。"]
    fit = [r["team_fitness"] for r in episode_rows]
    collab = [r["collaboration_ratio"] for r in episode_rows]
    release_ratio = [
        _to_float(r.get("task_release_complete_ratio"), _to_float(r.get("task_release_per_complete")))
        for r in episode_rows
    ]
    writes = [r["writes"] for r in episode_rows]

    notes: List[str] = []
    if fit[-1] < fit[0]:
        notes.append("团队总分 `team_fitness` 首尾下降，当前演化未收敛。")
    elif fit[-1] > fit[0]:
        notes.append("团队总分 `team_fitness` 首尾上升，存在正向演化信号。")
    else:
        notes.append("团队总分 `team_fitness` 基本持平。")
    if collab[-1] < collab[0]:
        notes.append("协作占比 `collaboration_ratio` 首尾下降，协作未随演化增强。")
    elif collab[-1] > collab[0]:
        notes.append("协作占比 `collaboration_ratio` 首尾上升。")
    if max(release_ratio) > 0.4:
        notes.append("任务释放/完成比偏高，说明严格依赖或写作前置条件与当前策略不匹配。")
    if sum(writes) == 0:
        notes.append("没有成功写作记录，系统被流程门槛完全卡住。")
    elif writes[-1] < max(writes):
        notes.append("后期写作数量低于峰值，可能出现任务空转或前置条件失败。")
    return notes


def _series_card(title: str, values: List[float], color: str, unit: str = "") -> str:
    if values:
        first = values[0]
        last = values[-1]
        best = max(values)
        delta = last - first
    else:
        first = last = best = delta = 0.0
    return f"""
    <section class="panel">
      <div class="panel-head">
        <h3>{html.escape(title)}</h3>
        <div class="mini-stats">
          <span>first: {first:.4f}{unit}</span>
          <span>last: {last:.4f}{unit}</span>
          <span>best: {best:.4f}{unit}</span>
          <span class="{'pos' if delta >= 0 else 'neg'}">delta: {delta:+.4f}{unit}</span>
        </div>
      </div>
      {_sparkline_svg(values, color=color, fill=color)}
    </section>
    """


def _build_dashboard_payload(base_dir: str) -> Dict[str, Any]:
    sim_dir = os.path.join(base_dir, "logs", "app", "simulation")
    env_dir = os.path.join(base_dir, "logs", "app", "environment")

    team_rows = _latest_team_metrics(os.path.join(sim_dir, "team_metrics.jsonl"))
    chain_rows = _latest_chain_metrics(os.path.join(sim_dir, "research_chain_metrics.jsonl"))
    episodes_count = len(team_rows)
    evo_rows = _tail_evolution_for_run(os.path.join(sim_dir, "evolution.jsonl"), episodes_count)
    taskboard_rows = _latest_taskboard(os.path.join(env_dir, "taskboard.jsonl"))

    task_stats = _aggregate_taskboard(taskboard_rows, episodes_count)
    chain_by_ep = {int(r.get("episode_index", 0) or 0) + 1: r for r in chain_rows}

    episode_rows: List[Dict[str, Any]] = []
    for idx, row in enumerate(team_rows, start=1):
        ep = idx
        task = task_stats.get(ep, {})
        chain = chain_by_ep.get(ep, {})
        evo = evo_rows[idx - 1] if idx - 1 < len(evo_rows) else {}
        donor = ((evo or {}).get("top_donors") or [{}])[0] if evo else {}
        task_claim = _to_int(row.get("taskboard_claim_events"), _to_int(chain.get("taskboard_claim_events"), _to_int(task.get("claim"))))
        task_complete = _to_int(
            row.get("taskboard_complete_events"),
            _to_int(chain.get("taskboard_complete_events"), _to_int(task.get("complete"))),
        )
        task_release = _to_int(
            row.get("taskboard_release_events"),
            _to_int(chain.get("taskboard_release_events"), _to_int(task.get("release"))),
        )
        complete_per_claim = _to_float(
            row.get("taskboard_complete_per_claim"),
            _to_float(chain.get("taskboard_complete_per_claim"), (float(task_complete) / float(task_claim)) if task_claim else 0.0),
        )
        release_per_complete = _to_float(
            row.get("taskboard_release_per_complete"),
            _to_float(chain.get("taskboard_release_per_complete"), _to_float(task.get("release_complete_ratio"))),
        )
        writes = _to_int(chain.get("action_write"), 0)
        reviews = _to_int(chain.get("action_review"), 0)
        episode_rows.append(
            {
                "episode": ep,
                "team_fitness": _to_float(row.get("team_fitness")),
                "team_graph": _to_float(row.get("team_graph")),
                "team_evidence": _to_float(row.get("team_evidence")),
                "team_evidence_coverage": _to_float(row.get("team_evidence_coverage")),
                "team_cost": _to_float(row.get("team_cost")),
                "collaboration_ratio": _to_float(row.get("collaboration_ratio")),
                "replication_all_pass": bool(row.get("replication_all_pass", False)),
                "replication_pass_rate": _to_float(row.get("replication_pass_rate")),
                "replication_verified_rate": _to_float(row.get("replication_verified_rate")),
                "publishable_rate": _to_float(row.get("publishable_rate")),
                "preprint_ready_rate": _to_float(row.get("preprint_ready_rate")),
                "team_readiness": _to_float(row.get("team_readiness")),
                "team_replication_support": _to_float(row.get("team_replication_support")),
                "team_contribution_credit": _to_float(row.get("team_contribution_credit")),
                "share_evidence": _to_int(row.get("team_share_sent_evidence")),
                "share_observation": _to_int(row.get("team_share_sent_observation")),
                "task_create": _to_int(task.get("create")),
                "task_claim": task_claim,
                "task_complete": task_complete,
                "task_release": task_release,
                "task_complete_per_claim": complete_per_claim,
                "task_release_per_complete": release_per_complete,
                # backward compatibility for old diagnosis/card keys
                "task_release_complete_ratio": release_per_complete,
                "task_top_release_reason": str(task.get("top_release_reason") or ""),
                "writes": writes,
                "reviews": reviews,
                "avg_write_fitness": _to_float(chain.get("action_write"), float(writes)),
                "avg_write_f1": 0.0,
                "paper_replication_verified_rate": _to_float(row.get("replication_verified_rate")),
                "paper_replication_ok_rate": _to_float(row.get("replication_pass_rate")),
                "paper_holdout_pass_rate": 0.0,
                "research_steps_total": _to_int(chain.get("research_steps_total"), 0),
                "research_steps_per_claim": _to_float(chain.get("research_steps_per_claim"), 0.0),
                "action_read": _to_int(chain.get("action_read"), 0),
                "action_hypothesize": _to_int(chain.get("action_hypothesize"), 0),
                "action_experiment": _to_int(chain.get("action_experiment"), 0),
                "action_write": _to_int(chain.get("action_write"), 0),
                "action_review": _to_int(chain.get("action_review"), 0),
                "action_replicate": _to_int(chain.get("action_replicate"), 0),
                "active_worker_count": _to_int(chain.get("active_worker_count"), _to_int(taskboard_rows[0].get("meta", {}).get("active_worker_count"), 0) if taskboard_rows else 0),
                "top_donor": donor.get("agent_id"),
                "top_donor_selection_score": _to_float(donor.get("selection_score")) if donor else None,
            }
        )

    return {
        "episodes": episode_rows,
        "meta": {
            "episodes_count": episodes_count,
            "team_metrics_rows": len(team_rows),
            "chain_metrics_rows": len(chain_rows),
            "evolution_rows": len(evo_rows),
            "taskboard_rows_latest_run": len(taskboard_rows),
            "log_mode": (os.getenv("SCIMAS_LOG_MODE", "compact") or "compact"),
        },
    }


def _render_html(payload: Dict[str, Any]) -> str:
    episodes = payload.get("episodes") or []
    meta = payload.get("meta") or {}
    if not episodes:
        return """<!doctype html><html><head><meta charset="utf-8"><title>SCIMAS Dashboard</title></head>
<body><h2>SCIMAS Evolution Dashboard</h2><p>No episode metrics found yet.</p></body></html>"""

    fit = [r["team_fitness"] for r in episodes]
    graph = [r["team_graph"] for r in episodes]
    evidence = [r["team_evidence"] for r in episodes]
    collab = [r["collaboration_ratio"] for r in episodes]
    team_ready = [r["team_readiness"] for r in episodes]
    publishable = [r["publishable_rate"] for r in episodes]
    rep_pass = [r["replication_pass_rate"] for r in episodes]
    rep_verified = [r["replication_verified_rate"] for r in episodes]
    claim_events = [float(r["task_claim"]) for r in episodes]
    complete_per_claim = [r["task_complete_per_claim"] for r in episodes]
    release_per_complete = [r["task_release_per_complete"] for r in episodes]
    research_steps_total = [float(r["research_steps_total"]) for r in episodes]
    research_steps_per_claim = [r["research_steps_per_claim"] for r in episodes]
    write_actions = [float(r["action_write"]) for r in episodes]
    experiment_actions = [float(r["action_experiment"]) for r in episodes]

    final_row = episodes[-1]
    best_fit_ep = max(episodes, key=lambda r: r["team_fitness"])
    best_flow_ep = max(episodes, key=lambda r: r["task_complete_per_claim"])
    worst_flow_ep = max(episodes, key=lambda r: r["task_release_per_complete"])

    flow_ok = final_row["task_complete_per_claim"] >= 0.60 and final_row["task_release_per_complete"] <= 0.35
    replication_ok = final_row["replication_pass_rate"] >= 0.30
    publishable_ok = final_row["publishable_rate"] >= 0.15
    rc_alert_threshold = float(os.getenv("SCIMAS_DASHBOARD_RC_ALERT", "0.4"))
    cc_warn_threshold = float(os.getenv("SCIMAS_DASHBOARD_CC_WARN", "0.5"))

    rows_html = []
    rc_alert_eps: List[int] = []
    cc_warn_eps: List[int] = []
    for r in episodes:
        ep = int(r["episode"])
        rc_alert = float(r["task_release_per_complete"]) > rc_alert_threshold
        cc_warn = float(r["task_complete_per_claim"]) < cc_warn_threshold
        row_cls = "row-alert" if (rc_alert or cc_warn) else ""
        rc_cls = "cell-danger" if rc_alert else ""
        cc_cls = "cell-warn" if cc_warn else ""
        if rc_alert:
            rc_alert_eps.append(ep)
        if cc_warn:
            cc_warn_eps.append(ep)
        rows_html.append(
            f'<tr class="{row_cls}">'
            f"<td>{r['episode']}</td>"
            f"<td>{r['team_fitness']:.4f}</td>"
            f"<td>{r['task_claim']}/{r['task_complete']}/{r['task_release']}</td>"
            f'<td class="{cc_cls}">{r["task_complete_per_claim"]:.2f}</td>'
            f'<td class="{rc_cls}">{r["task_release_per_complete"]:.2f}</td>'
            f"<td>{r['research_steps_total']}</td>"
            f"<td>{r['research_steps_per_claim']:.2f}</td>"
            f"<td>{r['action_read']}/{r['action_hypothesize']}/{r['action_experiment']}/{r['action_write']}</td>"
            f"<td>{r['collaboration_ratio']:.3f}</td>"
            f"<td>{r['publishable_rate']:.2f}</td>"
            f"<td>{r['replication_pass_rate']:.2f}</td>"
            f"<td>{_badge(r['replication_all_pass'])}</td>"
            f"<td>{html.escape(str(r.get('top_donor') or '-'))}</td>"
            "</tr>"
        )

    cards = [
        ("Episodes", str(meta.get("episodes_count", len(episodes))), f"log_mode={html.escape(str(meta.get('log_mode')))}"),
        ("Final Team Fitness", f"{final_row['team_fitness']:.4f}", f"best: EP{best_fit_ep['episode']}={best_fit_ep['team_fitness']:.4f}"),
        ("Task Flow (C/C)", f"{final_row['task_complete_per_claim']:.2f}", f"best: EP{best_flow_ep['episode']}={best_flow_ep['task_complete_per_claim']:.2f}"),
        ("Task Flow (R/C)", f"{final_row['task_release_per_complete']:.2f}", f"worst: EP{worst_flow_ep['episode']}={worst_flow_ep['task_release_per_complete']:.2f}"),
        ("Research Steps", str(final_row["research_steps_total"]), f"steps/claim={final_row['research_steps_per_claim']:.2f}"),
        ("Action Mix (R/H/E/W)", f"{final_row['action_read']}/{final_row['action_hypothesize']}/{final_row['action_experiment']}/{final_row['action_write']}", "科研动作主链"),
        ("Publishable Rate", f"{final_row['publishable_rate']:.2f}", f"team_readiness={final_row['team_readiness']:.2f}"),
        ("Replication Pass", f"{final_row['replication_pass_rate']:.2f}", f"verified={final_row['replication_verified_rate']:.2f}"),
    ]
    cards_html = "".join(
        f'<div class="card"><div class="label">{html.escape(k)}</div><div class="value">{html.escape(v)}</div><div class="hint">{html.escape(h)}</div></div>'
        for k, v, h in cards
    )

    metric_defs = [
        ("task_claim/task_complete/task_release", "任务板三元事件：认领、完成、释放。用于判断是否空转。"),
        ("task_complete_per_claim", "完成/认领。越高说明任务被实质推进。"),
        ("task_release_per_complete", "释放/完成。越低越好，高值常见于依赖不满足或写作失败回滚。"),
        ("research_steps_total", "科研动作总数（read/hypothesize/experiment/write/review/replicate）。越高说明不止在抢任务。"),
        ("research_steps_per_claim", "科研动作总数/任务认领数。越高说明每次认领更有产出。"),
        ("Action Mix (R/H/E/W)", "读任务、假设、实验、写作四类关键动作计数。用于看链路是否断在某个环节。"),
        ("team_fitness", "团队总体目标分（质量+复现+成本）。"),
        ("publishable_rate", "可发表比例（通过门槛的论文占比）。"),
        ("replication_pass_rate", "复现通过比例。"),
        ("collaboration_ratio", "共享动作占比（share_evidence/share_observation）。"),
    ]
    metric_defs_html = "".join(
        f'<tr><td><b>{html.escape(name)}</b></td><td>{html.escape(desc)}</td></tr>' for name, desc in metric_defs
    )

    diagnosis = _diagnosis(episodes)
    if rc_alert_eps:
        diagnosis.append(
            f"R/C 高风险回合（>{rc_alert_threshold:.2f}）：EP {', '.join(str(x) for x in rc_alert_eps)}。"
        )
    if cc_warn_eps:
        diagnosis.append(
            f"C/C 低效回合（<{cc_warn_threshold:.2f}）：EP {', '.join(str(x) for x in cc_warn_eps)}。"
        )
    diagnosis_html = "".join(f"<li>{html.escape(x)}</li>" for x in diagnosis)

    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>SCIMAS Evolution Dashboard</title>
  <style>
    :root {{
      --bg: #f4f8fb;
      --panel: #ffffff;
      --ink: #17202a;
      --muted: #607087;
      --line: #e6eaf0;
      --accent: #0a6a95;
      --danger: #b42318;
      --good: #067647;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background:
        radial-gradient(circle at 0% 0%, #d9f2ee, transparent 38%),
        radial-gradient(circle at 100% 100%, #d8ecf8, transparent 34%),
        var(--bg);
      color: var(--ink);
      font: 14px/1.5 "IBM Plex Sans", "Source Han Sans SC", "Noto Sans SC", "Microsoft YaHei", sans-serif;
    }}
    .wrap {{ max-width: 1320px; margin: 0 auto; padding: 22px; }}
    .hero {{
      background: linear-gradient(130deg, #0e3b5f, #0a6a95);
      color: #fff;
      border-radius: 14px;
      padding: 18px 20px 16px;
      margin-bottom: 16px;
      box-shadow: 0 10px 28px rgba(16,59,92,.18);
    }}
    .hero h1 {{ margin: 0 0 6px; font-size: 22px; }}
    .hero p {{ margin: 0; color: rgba(255,255,255,.88); }}
    .chip-row {{ margin-top: 10px; display: flex; flex-wrap: wrap; gap: 8px; }}
    .chip {{
      border-radius: 999px;
      padding: 4px 10px;
      background: rgba(255,255,255,.14);
      color: #fff;
      font-size: 12px;
      border: 1px solid rgba(255,255,255,.28);
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 12px;
      margin-bottom: 14px;
    }}
    .card {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 12px 14px;
      min-height: 92px;
    }}
    .label {{ color: var(--muted); font-size: 12px; }}
    .value {{ font-size: 24px; font-weight: 700; margin-top: 4px; }}
    .hint {{ color: var(--muted); margin-top: 4px; }}
    .stack {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 12px;
      margin-bottom: 12px;
    }}
    .stack.cols3 {{
      grid-template-columns: repeat(3, 1fr);
    }}
    .panel {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 12px;
    }}
    .panel h2, .panel h3 {{ margin: 0; font-size: 16px; }}
    .panel-head {{ display: flex; justify-content: space-between; gap: 8px; align-items: baseline; margin-bottom: 8px; }}
    .mini-stats {{ display: flex; gap: 10px; color: var(--muted); font-size: 12px; flex-wrap: wrap; justify-content: flex-end; }}
    .mini-stats .pos {{ color: var(--good); }}
    .mini-stats .neg {{ color: var(--danger); }}
    .mono {{ font-family: "JetBrains Mono", "Consolas", "SFMono-Regular", monospace; }}
    table {{
      width: 100%;
      border-collapse: collapse;
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 12px;
      overflow: hidden;
    }}
    th, td {{
      border-bottom: 1px solid var(--line);
      padding: 8px 10px;
      text-align: left;
      vertical-align: middle;
      white-space: nowrap;
    }}
    th {{ background: #f8fafc; color: var(--muted); font-weight: 600; }}
    tr:last-child td {{ border-bottom: none; }}
    tr.row-alert td {{
      background: #fffdfa;
    }}
    td.cell-warn {{
      background: #fff7d6 !important;
      color: #8a5a00;
      font-weight: 700;
    }}
    td.cell-danger {{
      background: #ffe4e2 !important;
      color: #9f1239;
      font-weight: 700;
    }}
    .legend {{
      display: inline-flex;
      align-items: center;
      gap: 6px;
      font-size: 12px;
      border-radius: 999px;
      padding: 2px 8px;
      border: 1px solid var(--line);
      background: #fafcff;
      color: var(--muted);
    }}
    .dot {{
      width: 10px;
      height: 10px;
      border-radius: 99px;
      display: inline-block;
      border: 1px solid rgba(0,0,0,.08);
    }}
    .dot.warn {{ background: #fff2b2; }}
    .dot.danger {{ background: #fecaca; }}
    .badge {{
      display: inline-block;
      font-size: 11px;
      padding: 2px 7px;
      border-radius: 999px;
      border: 1px solid;
    }}
    .badge.ok {{ color: var(--good); border-color: #9ae6b4; background: #ecfdf3; }}
    .badge.bad {{ color: var(--danger); border-color: #f7b7b2; background: #fff1f1; }}
    .footer {{
      color: var(--muted);
      margin-top: 10px;
      font-size: 12px;
    }}
    @media (max-width: 980px) {{
      .grid, .stack {{ grid-template-columns: 1fr; }}
      .stack.cols3 {{ grid-template-columns: 1fr; }}
      .panel-head {{ flex-direction: column; align-items: flex-start; }}
      .mini-stats {{ justify-content: flex-start; }}
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <section class="hero">
      <h1>SCIMAS 演化运行仪表盘</h1>
      <p>核心聚焦：任务流是否有效推进（claim→complete→release）以及科研主链动作是否真实发生并转化为质量提升。</p>
      <div class="chip-row">
        <span class="chip">流程健康: {_badge(flow_ok, "正常", "待修复")}</span>
        <span class="chip">复现通过: {_badge(replication_ok, "达标", "偏低")}</span>
        <span class="chip">可发表率: {_badge(publishable_ok, "达标", "偏低")}</span>
      </div>
    </section>

    <section class="grid">{cards_html}</section>

    <section class="stack cols3">
      {_series_card("Task Claim Events", claim_events, "#1d4ed8")}
      {_series_card("Task Complete/Claim", complete_per_claim, "#0f766e")}
      {_series_card("Task Release/Complete", release_per_complete, "#b42318")}
      {_series_card("Research Steps Total", research_steps_total, "#0ea5e9")}
      {_series_card("Research Steps/Claim", research_steps_per_claim, "#10b981")}
      {_series_card("Write Actions", write_actions, "#7c3aed")}
    </section>

    <section class="stack">
      {_series_card("Team Fitness", fit, "#0f766e")}
      {_series_card("Collaboration Ratio", collab, "#1d4ed8")}
      {_series_card("Experiment Actions", experiment_actions, "#2563eb")}
      {_series_card("Team Graph Score", graph, "#0ea5e9")}
      {_series_card("Team Evidence Score", evidence, "#f59e0b")}
      {_series_card("Team Readiness", team_ready, "#10b981")}
      {_series_card("Publishable Rate", publishable, "#14b8a6")}
      {_series_card("Replication Pass Rate", rep_pass, "#ef4444")}
      {_series_card("Replication Verified Rate", rep_verified, "#dc2626")}
    </section>

    <section class="panel" style="margin-bottom:12px;">
      <div class="panel-head">
        <h2>指标说明（当前面板）</h2>
        <div class="mini-stats">
          <span>高优先级先看任务流 + steps/claim</span>
          <span class="mono">release/complete 越低越好</span>
        </div>
      </div>
      <table>
        <thead>
          <tr><th>指标</th><th>含义解释</th></tr>
        </thead>
        <tbody>{metric_defs_html}</tbody>
      </table>
    </section>

    <section class="panel" style="margin-bottom:12px;">
      <div class="panel-head">
        <h2>Episode 明细</h2>
        <div class="mini-stats">
          <span class="mono">C/C = complete/claim</span>
          <span class="mono">R/C = release/complete</span>
          <span class="legend"><i class="dot warn"></i>C/C &lt; {cc_warn_threshold:.2f}</span>
          <span class="legend"><i class="dot danger"></i>R/C &gt; {rc_alert_threshold:.2f}</span>
        </div>
      </div>
      <table>
        <thead>
          <tr>
            <th>EP</th>
            <th>team_fit</th>
            <th>claim/complete/release</th>
            <th>C/C</th>
            <th>R/C</th>
            <th>steps</th>
            <th>steps/claim</th>
            <th>R/H/E/W</th>
            <th>collab</th>
            <th>publishable_rate</th>
            <th>rep_pass_rate</th>
            <th>team replication</th>
            <th>top donor</th>
          </tr>
        </thead>
        <tbody>
          {''.join(rows_html)}
        </tbody>
      </table>
    </section>

    <section class="panel" style="margin-bottom:12px;">
      <h2>诊断建议</h2>
      <ul>{diagnosis_html}</ul>
    </section>

    <section class="panel" style="margin-bottom:12px;">
      <h2>如何阅读（建议顺序）</h2>
      <ol>
        <li>先看 `Task Complete/Claim` 和 `Task Release/Complete`。这两个决定系统是否在空转。</li>
        <li>再看 `Research Steps/Claim` 与 `Action Mix`。判断 claim 后有没有真的进入科研动作链。</li>
        <li>最后看 `Team Fitness / Publishable / Replication Pass`，确认流程优化是否转化为科研质量。</li>
      </ol>
    </section>

    <div class="footer">
      数据来源：`team_metrics.jsonl` / `research_chain_metrics.jsonl` / `evolution.jsonl` / `taskboard.jsonl`（当前 run 自动截取最新一段）。
    </div>
  </div>
</body>
</html>"""


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
        "episodes": len(payload.get("episodes") or []),
        "meta": payload.get("meta") or {},
    }
