from __future__ import annotations

import argparse
import csv
import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List


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


def _latest_run(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not rows:
        return []
    start_idx = 0
    last_ep = int(rows[0].get("episode_index", 0) or 0)
    for idx in range(1, len(rows)):
        current_ep = int(rows[idx].get("episode_index", 0) or 0)
        if current_ep == 0 and last_ep > 0:
            start_idx = idx
        last_ep = current_ep
    latest = rows[start_idx:]
    return sorted(latest, key=lambda r: int(r.get("episode_index", 0) or 0))


def _slope(values: List[float]) -> float:
    n = len(values)
    if n < 2:
        return 0.0
    x_mean = (n - 1) / 2.0
    y_mean = sum(values) / n
    numerator = 0.0
    denominator = 0.0
    for i, value in enumerate(values):
        dx = i - x_mean
        numerator += dx * (value - y_mean)
        denominator += dx * dx
    if denominator <= 0:
        return 0.0
    return numerator / denominator


def _is_monotonic_non_decreasing(values: List[float], eps: float = 1e-9) -> bool:
    if not values:
        return False
    for i in range(1, len(values)):
        if values[i] + eps < values[i - 1]:
            return False
    return True


def _write_plot_html(path: str, episodes: List[int], team_fitness: List[float], collaboration: List[float]) -> None:
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        fig = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=("Team Fitness by Episode", "Collaboration Ratio by Episode"),
        )
        fig.add_trace(
            go.Scatter(x=episodes, y=team_fitness, mode="lines+markers", name="team_fitness"),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(x=episodes, y=collaboration, mode="lines+markers", name="collaboration_ratio"),
            row=1,
            col=2,
        )
        fig.update_layout(height=520, template="plotly_white", title="Episode-Level Evolution Check")
        fig.update_xaxes(title_text="episode_index")
        fig.update_yaxes(title_text="value")
        fig.write_html(path, include_plotlyjs=True, full_html=True)
    except Exception:
        fallback = {
            "episodes": episodes,
            "team_fitness": team_fitness,
            "collaboration_ratio": collaboration,
        }
        with open(path, "w", encoding="utf-8") as f:
            f.write("<html><body><h3>Episode-Level Evolution Check (fallback)</h3><pre>")
            f.write(json.dumps(fallback, ensure_ascii=False, indent=2))
            f.write("</pre></body></html>")


def analyze(project_root: str) -> Dict[str, Any]:
    sim_dir = os.path.join(project_root, "logs", "app", "simulation")
    os.makedirs(sim_dir, exist_ok=True)

    metrics_path = os.path.join(sim_dir, "team_metrics.jsonl")
    metrics_rows = _latest_run(_read_jsonl(metrics_path))

    if not metrics_rows:
        summary = {
            "ts": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "ok": False,
            "reason": "no_team_metrics",
            "metrics_path": metrics_path,
        }
        with open(os.path.join(sim_dir, "analysis_summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
            f.write("\n")
        return summary

    episodes = [int(row.get("episode_index", idx) or idx) for idx, row in enumerate(metrics_rows)]
    team_fitness = [float(row.get("team_fitness", 0.0) or 0.0) for row in metrics_rows]
    collaboration = [float(row.get("collaboration_ratio", 0.0) or 0.0) for row in metrics_rows]

    fit_first = team_fitness[0]
    fit_last = team_fitness[-1]
    fit_slope = _slope(team_fitness)
    fit_monotonic = _is_monotonic_non_decreasing(team_fitness)

    collab_first = collaboration[0]
    collab_last = collaboration[-1]
    collab_slope = _slope(collaboration)
    collab_monotonic = _is_monotonic_non_decreasing(collaboration)

    fit_up = fit_last >= fit_first
    collab_up = collab_last >= collab_first

    summary = {
        "ts": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "ok": True,
        "episodes_analyzed": len(metrics_rows),
        "fitness": {
            "first": fit_first,
            "last": fit_last,
            "delta": fit_last - fit_first,
            "slope": fit_slope,
            "monotonic_non_decreasing": fit_monotonic,
            "upward": fit_up,
        },
        "collaboration": {
            "first": collab_first,
            "last": collab_last,
            "delta": collab_last - collab_first,
            "slope": collab_slope,
            "monotonic_non_decreasing": collab_monotonic,
            "upward": collab_up,
        },
        "acceptance": {
            "team_fitness_upward": fit_up,
            "collaboration_ratio_upward": collab_up,
            "both_upward": fit_up and collab_up,
        },
    }

    summary_json_path = os.path.join(sim_dir, "analysis_summary.json")
    summary_txt_path = os.path.join(sim_dir, "analysis_summary.txt")
    timeseries_csv_path = os.path.join(sim_dir, "analysis_timeseries.csv")
    trend_html_path = os.path.join(sim_dir, "analysis_trend.html")

    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
        f.write("\n")

    with open(summary_txt_path, "w", encoding="utf-8") as f:
        f.write("Episode Evolution Analysis\n")
        f.write(f"Episodes analyzed: {len(metrics_rows)}\n")
        f.write(f"Team fitness first/last: {fit_first:.6f} -> {fit_last:.6f}\n")
        f.write(f"Team fitness delta/slope: {fit_last - fit_first:.6f} / {fit_slope:.6f}\n")
        f.write(f"Team fitness monotonic non-decreasing: {fit_monotonic}\n")
        f.write(f"Collaboration first/last: {collab_first:.6f} -> {collab_last:.6f}\n")
        f.write(f"Collaboration delta/slope: {collab_last - collab_first:.6f} / {collab_slope:.6f}\n")
        f.write(f"Collaboration monotonic non-decreasing: {collab_monotonic}\n")
        f.write(f"Acceptance both upward: {fit_up and collab_up}\n")

    with open(timeseries_csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "episode_index",
                "team_fitness",
                "team_graph",
                "team_evidence",
                "team_cost",
                "replication_all_pass",
                "collaboration_ratio",
            ]
        )
        for row in metrics_rows:
            writer.writerow(
                [
                    int(row.get("episode_index", 0) or 0),
                    float(row.get("team_fitness", 0.0) or 0.0),
                    float(row.get("team_graph", 0.0) or 0.0),
                    float(row.get("team_evidence", 0.0) or 0.0),
                    float(row.get("team_cost", 0.0) or 0.0),
                    bool(row.get("replication_all_pass", False)),
                    float(row.get("collaboration_ratio", 0.0) or 0.0),
                ]
            )

    _write_plot_html(trend_html_path, episodes, team_fitness, collaboration)
    summary["artifacts"] = {
        "summary_json": summary_json_path,
        "summary_txt": summary_txt_path,
        "timeseries_csv": timeseries_csv_path,
        "trend_html": trend_html_path,
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze multi-episode team fitness and collaboration trends.")
    parser.add_argument(
        "--project-root",
        default=os.getenv("MAS_PROJECT_ABS_PATH", os.path.join("examples", "scimas_mvp")),
        help="Path to scimas_mvp project root.",
    )
    args = parser.parse_args()

    project_root = os.path.abspath(args.project_root)
    summary = analyze(project_root)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
