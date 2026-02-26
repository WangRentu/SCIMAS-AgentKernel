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
    release_ratio = [r["task_release_complete_ratio"] for r in episode_rows]
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
    research_dir = os.path.join(base_dir, "logs", "app", "research")

    team_rows = _latest_team_metrics(os.path.join(sim_dir, "team_metrics.jsonl"))
    episodes_count = len(team_rows)
    evo_rows = _tail_evolution_for_run(os.path.join(sim_dir, "evolution.jsonl"), episodes_count)
    taskboard_rows = _latest_taskboard(os.path.join(env_dir, "taskboard.jsonl"))
    paper_rows = _latest_papers(os.path.join(research_dir, "papers.jsonl"))

    task_stats = _aggregate_taskboard(taskboard_rows, episodes_count)
    paper_stats = _aggregate_papers(paper_rows, episodes_count)

    episode_rows: List[Dict[str, Any]] = []
    for idx, row in enumerate(team_rows, start=1):
        ep = idx
        task = task_stats.get(ep, {})
        paper = paper_stats.get(ep, {})
        evo = evo_rows[idx - 1] if idx - 1 < len(evo_rows) else {}
        donor = ((evo or {}).get("top_donors") or [{}])[0] if evo else {}
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
                "task_claim": _to_int(task.get("claim")),
                "task_complete": _to_int(task.get("complete")),
                "task_release": _to_int(task.get("release")),
                "task_release_complete_ratio": _to_float(task.get("release_complete_ratio")),
                "task_top_release_reason": str(task.get("top_release_reason") or ""),
                "writes": _to_int(paper.get("writes")),
                "reviews": _to_int(paper.get("reviews")),
                "avg_write_fitness": _to_float(paper.get("avg_write_fitness")),
                "avg_write_f1": _to_float(paper.get("avg_write_f1")),
                "paper_replication_verified_rate": _to_float(paper.get("replication_verified_rate")),
                "paper_replication_ok_rate": _to_float(paper.get("replication_ok_rate")),
                "paper_holdout_pass_rate": _to_float(paper.get("holdout_pass_rate")),
                "top_donor": donor.get("agent_id"),
                "top_donor_selection_score": _to_float(donor.get("selection_score")) if donor else None,
            }
        )

    return {
        "episodes": episode_rows,
        "meta": {
            "episodes_count": episodes_count,
            "team_metrics_rows": len(team_rows),
            "evolution_rows": len(evo_rows),
            "taskboard_rows_latest_run": len(taskboard_rows),
            "paper_rows_latest_run": len(paper_rows),
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
    evidence_cov = [r["team_evidence_coverage"] for r in episodes]
    collab = [r["collaboration_ratio"] for r in episodes]
    team_ready = [r["team_readiness"] for r in episodes]
    publishable = [r["publishable_rate"] for r in episodes]
    rep_pass = [r["replication_pass_rate"] for r in episodes]
    rep_verified = [r["replication_verified_rate"] for r in episodes]
    rep_support = [r["team_replication_support"] for r in episodes]
    releases = [r["task_release_complete_ratio"] for r in episodes]
    writes = [float(r["writes"]) for r in episodes]
    write_fit = [r["avg_write_fitness"] for r in episodes]

    final_row = episodes[-1]
    best_fit_ep = max(episodes, key=lambda r: r["team_fitness"])
    worst_release_ep = max(episodes, key=lambda r: r["task_release_complete_ratio"])

    rows_html = []
    for r in episodes:
        rows_html.append(
            "<tr>"
            f"<td>{r['episode']}</td>"
            f"<td>{r['team_fitness']:.4f}</td>"
            f"<td>{r['team_graph']:.4f}</td>"
            f"<td>{r['team_evidence']:.4f}</td>"
            f"<td>{r['collaboration_ratio']:.4f}</td>"
            f"<td>{r['task_claim']}/{r['task_complete']}/{r['task_release']}</td>"
            f"<td>{r['task_release_complete_ratio']:.2f}</td>"
            f"<td>{r['writes']}</td>"
            f"<td>{r['avg_write_fitness']:.3f}</td>"
            f"<td>{r['publishable_rate']:.2f}</td>"
            f"<td>{r['replication_pass_rate']:.2f}</td>"
            f"<td>{r['paper_holdout_pass_rate']:.2f}</td>"
            f"<td>{_badge(r['replication_all_pass'])}</td>"
            f"<td>{html.escape(str(r.get('top_donor') or '-'))}</td>"
            "</tr>"
        )

    cards = [
        ("Episodes", str(meta.get("episodes_count", len(episodes))), "运行轮次"),
        ("Final Team Fitness", f"{final_row['team_fitness']:.4f}", f"最佳 EP{best_fit_ep['episode']}={best_fit_ep['team_fitness']:.4f}"),
        ("Final Collaboration", f"{final_row['collaboration_ratio']:.4f}", f"share(E/O)={final_row['share_evidence']}/{final_row['share_observation']}"),
        ("Release/Complete", f"{final_row['task_release_complete_ratio']:.2f}", f"最差 EP{worst_release_ep['episode']}={worst_release_ep['task_release_complete_ratio']:.2f}"),
        ("Writes (Final EP)", str(final_row["writes"]), f"avg write fitness={final_row['avg_write_fitness']:.3f}"),
        ("Team Readiness", f"{final_row['team_readiness']:.2f}", f"publishable_rate={final_row['publishable_rate']:.2f}"),
        ("Replication Pass Rate", f"{final_row['replication_pass_rate']:.2f}", f"verified_rate={final_row['replication_verified_rate']:.2f}"),
        ("Paper Holdout Pass", f"{final_row['paper_holdout_pass_rate']:.2f}", f"log_mode={html.escape(str(meta.get('log_mode')))}"),
    ]
    cards_html = "".join(
        f'<div class="card"><div class="label">{html.escape(k)}</div><div class="value">{html.escape(v)}</div><div class="hint">{html.escape(h)}</div></div>'
        for k, v, h in cards
    )

    metric_defs = [
        ("Team Fitness", "团队综合效果分。综合结构质量、证据质量、复现率/可发表率、准备度和成本。越高越好。"),
        ("Collaboration Ratio", "协作动作占比（share_evidence/share_observation）。反映团队是否真的在共享信息。通常越高越好，但过高可能挤占科研动作。"),
        ("Task Release/Complete Ratio", "任务释放数 / 任务完成数。高说明任务空转、前置条件不满足或流程门槛过严。越低越好。"),
        ("Avg Write Fitness", "本集成功写作 paper 的平均质量分（论文级评估）。越高越好。"),
        ("Team Graph Score", "团队平均结构正确性（边预测接近真值的程度）。越高越好。"),
        ("Team Evidence Score", "团队平均证据质量（边级证据的质量加权分）。越高越好。"),
        ("Team Evidence Coverage", "团队平均证据覆盖率（有证据支持的 claimed edge 比例）。越高越好。"),
        ("Team Readiness", "科研准备度：离达标门槛（结构/证据/复现）还有多远的平滑分。越高越说明系统在接近可发表状态。"),
        ("Publishable Rate", "团队中通过论文门槛（结构+证据+复现）的比例。越高越好。"),
        ("Replication Pass Rate", "团队中 `replication_ok=true` 的比例。反映复现通过情况。越高越好。"),
        ("Replication Verified Rate", "团队中已有复现实证记录的比例。反映是否真的在做复现。越高越好。"),
        ("Team Replication Support", "团队论文复现实验支持度均值（support ratio）。越高越好。"),
        ("Paper Holdout Pass Rate", "按 `papers.jsonl` 统计的论文记录 holdout 通过率（paper/review 记录层）。越高越好。"),
    ]
    metric_defs_html = "".join(
        f'<tr><td><b>{html.escape(name)}</b></td><td>{html.escape(desc)}</td></tr>' for name, desc in metric_defs
    )

    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>SCIMAS Evolution Dashboard</title>
  <style>
    :root {{
      --bg: #f6f7fb;
      --panel: #ffffff;
      --ink: #17202a;
      --muted: #607087;
      --line: #e6eaf0;
      --accent: #0f766e;
      --danger: #b42318;
      --good: #067647;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: radial-gradient(circle at 0% 0%, #e7f7f4, transparent 40%), var(--bg);
      color: var(--ink);
      font: 14px/1.45 "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif;
    }}
    .wrap {{ max-width: 1240px; margin: 0 auto; padding: 22px; }}
    .hero {{
      background: linear-gradient(135deg, #103b5c, #0f766e);
      color: #fff;
      border-radius: 14px;
      padding: 18px 20px;
      margin-bottom: 16px;
      box-shadow: 0 10px 28px rgba(16,59,92,.18);
    }}
    .hero h1 {{ margin: 0 0 6px; font-size: 22px; }}
    .hero p {{ margin: 0; color: rgba(255,255,255,.88); }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
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
    .diag ul {{ margin: 8px 0 0; padding-left: 18px; }}
    .diag li {{ margin: 6px 0; }}
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
      .panel-head {{ flex-direction: column; align-items: flex-start; }}
      .mini-stats {{ justify-content: flex-start; }}
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <section class="hero">
      <h1>SCIMAS 演化运行仪表盘（精简版）</h1>
      <p>显示演化核心指标（团队质量、任务板效率、论文质量、复现与可发表率），并在页面内解释每个指标含义，便于直接判断机制是否有效。</p>
    </section>

    <section class="grid">{cards_html}</section>

    <section class="stack">
      {_series_card("Team Fitness", fit, "#0f766e")}
      {_series_card("Collaboration Ratio", collab, "#1d4ed8")}
      {_series_card("Task Release/Complete Ratio", releases, "#b42318")}
      {_series_card("Avg Write Fitness", write_fit, "#7c3aed")}
      {_series_card("Team Graph Score", graph, "#0ea5e9")}
      {_series_card("Team Evidence Score", evidence, "#f59e0b")}
      {_series_card("Team Evidence Coverage", evidence_cov, "#f97316")}
      {_series_card("Team Readiness", team_ready, "#10b981")}
      {_series_card("Publishable Rate", publishable, "#14b8a6")}
      {_series_card("Replication Pass Rate", rep_pass, "#ef4444")}
      {_series_card("Replication Verified Rate", rep_verified, "#dc2626")}
      {_series_card("Team Replication Support", rep_support, "#8b5cf6")}
    </section>

    <section class="panel" style="margin-bottom:12px;">
      <div class="panel-head">
        <h2>指标说明（含方向）</h2>
        <div class="mini-stats">
          <span>大多数指标越高越好</span>
          <span>`Task Release/Complete Ratio` 越低越好</span>
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
        <h2>Episode 明细（直观版）</h2>
        <div class="mini-stats">
          <span>claim/complete/release 用于观察任务板是否空转</span>
          <span>publishable/replication_pass 用于观察新评估协议下的科研达标率</span>
        </div>
      </div>
      <table>
        <thead>
          <tr>
            <th>EP</th>
            <th>team_fit</th>
            <th>graph</th>
            <th>evidence</th>
            <th>collab</th>
            <th>claim/complete/release</th>
            <th>release/complete</th>
            <th>writes</th>
            <th>avg_write_fit</th>
            <th>publishable_rate</th>
            <th>rep_pass_rate</th>
            <th>paper_holdout_pass</th>
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
      <h2>如何看这页（建议顺序）</h2>
      <ol>
        <li>先看顶部卡片：`Final Team Fitness`、`Final Collaboration`、`Release/Complete`。</li>
        <li>再看 `Team Fitness / Collaboration / Team Readiness` 曲线，判断演化方向和“接近门槛”的趋势是否一致。</li>
        <li>如果 `Release/Complete` 偏高，再看 `Avg Write Fitness` 是否同步下降，通常说明写作门槛过严或策略没适配。</li>
        <li>再看 `Publishable Rate / Replication Pass Rate / Replication Verified Rate`，判断科研流程是否真的形成闭环。</li>
        <li>最后看表格里的 `top donor` 和 `paper_holdout_pass`，判断进化选择与复现门槛是否真的发挥作用。</li>
      </ol>
    </section>

    <div class="footer">
      数据来源：`team_metrics.jsonl` / `evolution.jsonl` / `taskboard.jsonl` / `papers.jsonl`（当前 run 自动截取最新一段）。
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
