from __future__ import annotations

import json
import logging
import os
from collections import defaultdict
from typing import Any, Dict, List, Tuple

logger = logging.getLogger(__name__)


def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        return []
    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except Exception:
                continue
    return records


def _latest_run(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not records:
        return []
    start_idx = 0
    last_tick = int(records[0].get("tick", 0) or 0)
    for idx in range(1, len(records)):
        tick = int(records[idx].get("tick", 0) or 0)
        if tick == 0 and last_tick > 0:
            start_idx = idx
        last_tick = tick
    return records[start_idx:]


def _avg_by_tick(values_by_tick: Dict[int, List[float]]) -> Tuple[List[int], List[float]]:
    ticks = sorted(values_by_tick.keys())
    means = []
    for tick in ticks:
        vals = values_by_tick[tick]
        means.append(sum(vals) / len(vals) if vals else 0.0)
    return ticks, means


def _build_data(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    rewards_by_tick: Dict[int, List[float]] = defaultdict(list)
    write_f1_by_tick: Dict[int, List[float]] = defaultdict(list)
    write_fit_by_tick: Dict[int, List[float]] = defaultdict(list)
    write_points_by_agent: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    exp_count_by_agent_tick: Dict[str, Dict[int, int]] = defaultdict(dict)

    for rec in records:
        tick = int(rec.get("tick", 0) or 0)
        aid = str(rec.get("agent_id", "unknown"))
        action = rec.get("action")
        reward = float(rec.get("reward", 0.0) or 0.0)
        rewards_by_tick[tick].append(reward)

        exp_count = int(rec.get("exp_count", 0) or 0)
        prev = exp_count_by_agent_tick[aid].get(tick, 0)
        exp_count_by_agent_tick[aid][tick] = max(prev, exp_count)

        if action != "write":
            continue
        detail = rec.get("detail") or {}
        metrics = detail.get("metrics") or {}
        f1 = metrics.get("f1")
        fit = metrics.get("fitness")
        if f1 is not None:
            write_f1_by_tick[tick].append(float(f1))
        if fit is not None:
            write_fit_by_tick[tick].append(float(fit))
        write_points_by_agent[aid].append(
            {
                "tick": tick,
                "f1": float(f1) if f1 is not None else None,
                "fitness": float(fit) if fit is not None else None,
            }
        )

    reward_ticks, reward_mean = _avg_by_tick(rewards_by_tick)
    wf1_ticks, wf1_mean = _avg_by_tick(write_f1_by_tick)
    wfit_ticks, wfit_mean = _avg_by_tick(write_fit_by_tick)

    final_fitness: Dict[str, float] = {}
    for aid, pts in write_points_by_agent.items():
        vals = [p["fitness"] for p in pts if p["fitness"] is not None]
        final_fitness[aid] = vals[-1] if vals else float("-inf")

    top_agents = sorted(
        write_points_by_agent.keys(),
        key=lambda a: (final_fitness.get(a, float("-inf")), len(write_points_by_agent[a])),
        reverse=True,
    )
    top_agents = top_agents[:8]

    exp_series: Dict[str, Dict[str, List[float]]] = {}
    for aid in top_agents:
        by_tick = exp_count_by_agent_tick.get(aid, {})
        ticks = sorted(by_tick.keys())
        running = 0
        ys: List[int] = []
        last_val = 0
        for tick in ticks:
            val = by_tick[tick]
            if val >= last_val:
                running = val
                last_val = val
            ys.append(running)
        exp_series[aid] = {"x": ticks, "y": ys}

    return {
        "reward_ticks": reward_ticks,
        "reward_mean": reward_mean,
        "wf1_ticks": wf1_ticks,
        "wf1_mean": wf1_mean,
        "wfit_ticks": wfit_ticks,
        "wfit_mean": wfit_mean,
        "write_points_by_agent": write_points_by_agent,
        "top_agents": top_agents,
        "exp_series": exp_series,
        "num_records": len(records),
    }


def _write_fallback_html(path: str, payload: Dict[str, Any]) -> None:
    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Trend Dashboard</title>
</head>
<body>
  <h2>Trend Dashboard (fallback)</h2>
  <p>Plotly not available. Raw series are shown below.</p>
  <pre>{json.dumps(payload, ensure_ascii=False, indent=2)}</pre>
</body>
</html>
"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)


def generate_trend_dashboard(base_dir: str, out_dir: str) -> Dict[str, Any]:
    trace_path = os.path.join(base_dir, "logs", "app", "action", "trace.jsonl")
    out_path = os.path.join(out_dir, "trend_dashboard.html")

    records = _read_jsonl(trace_path)
    records = _latest_run(records)
    payload = _build_data(records)

    if not payload["num_records"]:
        _write_fallback_html(out_path, {"message": "No trace records found.", "trace_path": trace_path})
        return {"path": out_path, "num_records": 0, "status": "empty"}

    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "Team Mean Reward by Tick",
                "Team Mean Write F1 by Tick",
                "Write Fitness Trend (Top Agents)",
                "Experiment Count Trend (Top Agents)",
            ),
        )

        fig.add_trace(
            go.Scatter(
                x=payload["reward_ticks"],
                y=payload["reward_mean"],
                mode="lines+markers",
                name="team_reward_mean",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=payload["wf1_ticks"],
                y=payload["wf1_mean"],
                mode="lines+markers",
                name="team_write_f1_mean",
            ),
            row=1,
            col=2,
        )

        for aid in payload["top_agents"]:
            pts = payload["write_points_by_agent"].get(aid, [])
            xs = [p["tick"] for p in pts if p["fitness"] is not None]
            ys = [p["fitness"] for p in pts if p["fitness"] is not None]
            if xs:
                fig.add_trace(
                    go.Scatter(x=xs, y=ys, mode="lines+markers", name=f"{aid}:fitness"),
                    row=2,
                    col=1,
                )

            exp = payload["exp_series"].get(aid, {})
            ex = exp.get("x") or []
            ey = exp.get("y") or []
            if ex:
                fig.add_trace(
                    go.Scatter(x=ex, y=ey, mode="lines", name=f"{aid}:exp_count"),
                    row=2,
                    col=2,
                )

        fig.update_layout(
            title="Multi-Agent Causal Discovery Trend Dashboard",
            legend=dict(orientation="h"),
            height=900,
            template="plotly_white",
        )
        fig.update_xaxes(title_text="tick")
        fig.update_yaxes(title_text="value")

        fig.write_html(out_path, include_plotlyjs=True, full_html=True)
        return {"path": out_path, "num_records": payload["num_records"], "status": "ok"}
    except Exception as e:
        logger.warning(f"Generate plotly dashboard failed, falling back to raw html: {e}")
        _write_fallback_html(out_path, payload)
        return {"path": out_path, "num_records": payload["num_records"], "status": "fallback", "error": str(e)}
