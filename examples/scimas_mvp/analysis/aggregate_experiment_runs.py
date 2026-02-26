from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, List


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _read_json(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


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


def _mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _std(values: List[float]) -> float:
    if len(values) <= 1:
        return 0.0
    m = _mean(values)
    return math.sqrt(sum((v - m) ** 2 for v in values) / (len(values) - 1))


def _ci95(values: List[float]) -> float:
    if not values:
        return 0.0
    return 1.96 * _std(values) / math.sqrt(len(values))


def _summarize_run(run_dir: str) -> Dict[str, Any]:
    sim_dir = os.path.join(run_dir, "simulation")
    manifest = _read_json(os.path.join(sim_dir, "run_manifest.json"))
    analysis = _read_json(os.path.join(sim_dir, "analysis_summary.json"))
    metrics = _read_jsonl(os.path.join(sim_dir, "team_metrics.jsonl"))
    final_team = metrics[-1] if metrics else {}
    return {
        "run_dir": run_dir,
        "variant_name": manifest.get("variant_name") or "unknown",
        "experiment_name": manifest.get("experiment_name"),
        "world_seed": manifest.get("world_seed"),
        "git_commit": manifest.get("git_commit"),
        "episodes_analyzed": analysis.get("episodes_analyzed"),
        "fitness_first": ((analysis.get("fitness") or {}).get("first")),
        "fitness_last": ((analysis.get("fitness") or {}).get("last")),
        "fitness_delta": ((analysis.get("fitness") or {}).get("delta")),
        "fitness_upward": bool((analysis.get("fitness") or {}).get("upward", False)),
        "collab_first": ((analysis.get("collaboration") or {}).get("first")),
        "collab_last": ((analysis.get("collaboration") or {}).get("last")),
        "collab_delta": ((analysis.get("collaboration") or {}).get("delta")),
        "collab_upward": bool((analysis.get("collaboration") or {}).get("upward", False)),
        "both_upward": bool((analysis.get("acceptance") or {}).get("both_upward", False)),
        "final_team_fitness": final_team.get("team_fitness"),
        "final_team_graph": final_team.get("team_graph"),
        "final_team_evidence": final_team.get("team_evidence"),
        "final_team_cost": final_team.get("team_cost"),
        "final_collaboration_ratio": final_team.get("collaboration_ratio"),
        "final_team_contribution_credit": final_team.get("team_contribution_credit"),
        "final_team_share_sent_evidence": final_team.get("team_share_sent_evidence"),
        "final_team_share_sent_observation": final_team.get("team_share_sent_observation"),
        "final_replication_all_pass": bool(final_team.get("replication_all_pass", False)),
    }


def _aggregate_variant(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    def vals(key: str) -> List[float]:
        out: List[float] = []
        for r in rows:
            v = r.get(key)
            if isinstance(v, (int, float)):
                out.append(float(v))
        return out

    fitness_last = vals("fitness_last")
    fitness_delta = vals("fitness_delta")
    collab_last = vals("collab_last")
    collab_delta = vals("collab_delta")
    final_team_fitness = vals("final_team_fitness")
    final_collab = vals("final_collaboration_ratio")
    final_contrib = vals("final_team_contribution_credit")
    final_share_e = vals("final_team_share_sent_evidence")
    final_share_o = vals("final_team_share_sent_observation")

    return {
        "num_runs": len(rows),
        "seeds": sorted([str(r.get("world_seed")) for r in rows]),
        "rates": {
            "fitness_upward_rate": sum(1 for r in rows if r.get("fitness_upward")) / max(1, len(rows)),
            "collab_upward_rate": sum(1 for r in rows if r.get("collab_upward")) / max(1, len(rows)),
            "both_upward_rate": sum(1 for r in rows if r.get("both_upward")) / max(1, len(rows)),
            "final_replication_all_pass_rate": sum(1 for r in rows if r.get("final_replication_all_pass")) / max(1, len(rows)),
        },
        "fitness_last": {"mean": _mean(fitness_last), "std": _std(fitness_last), "ci95": _ci95(fitness_last)},
        "fitness_delta": {"mean": _mean(fitness_delta), "std": _std(fitness_delta), "ci95": _ci95(fitness_delta)},
        "collab_last": {"mean": _mean(collab_last), "std": _std(collab_last), "ci95": _ci95(collab_last)},
        "collab_delta": {"mean": _mean(collab_delta), "std": _std(collab_delta), "ci95": _ci95(collab_delta)},
        "final_team_fitness": {
            "mean": _mean(final_team_fitness),
            "std": _std(final_team_fitness),
            "ci95": _ci95(final_team_fitness),
        },
        "final_collaboration_ratio": {"mean": _mean(final_collab), "std": _std(final_collab), "ci95": _ci95(final_collab)},
        "final_team_contribution_credit": {
            "mean": _mean(final_contrib),
            "std": _std(final_contrib),
            "ci95": _ci95(final_contrib),
        },
        "final_team_share_sent_evidence": {"mean": _mean(final_share_e), "std": _std(final_share_e), "ci95": _ci95(final_share_e)},
        "final_team_share_sent_observation": {
            "mean": _mean(final_share_o),
            "std": _std(final_share_o),
            "ci95": _ci95(final_share_o),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate archived scimas_mvp experiment runs.")
    parser.add_argument("--experiment-name", required=True, help="Folder name under examples/scimas_mvp/logs/experiments/")
    args = parser.parse_args()

    exp_root = os.path.join(PROJECT_ROOT, "logs", "experiments", args.experiment_name)
    runs_root = os.path.join(exp_root, "runs")
    if not os.path.isdir(runs_root):
        raise SystemExit(f"Runs directory not found: {runs_root}")

    run_dirs = [os.path.join(runs_root, d) for d in sorted(os.listdir(runs_root)) if os.path.isdir(os.path.join(runs_root, d))]
    run_rows = [_summarize_run(d) for d in run_dirs]
    run_rows = [r for r in run_rows if r.get("variant_name")]
    by_variant: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in run_rows:
        by_variant[str(row["variant_name"])].append(row)

    aggregate = {
        "ts": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "experiment_name": args.experiment_name,
        "num_runs": len(run_rows),
        "variants": {variant: _aggregate_variant(rows) for variant, rows in sorted(by_variant.items())},
    }

    os.makedirs(exp_root, exist_ok=True)
    json_path = os.path.join(exp_root, "aggregate_summary.json")
    csv_path = os.path.join(exp_root, "aggregate_summary.csv")
    runs_csv_path = os.path.join(exp_root, "runs_table.csv")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(aggregate, f, ensure_ascii=False, indent=2)
        f.write("\n")

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "variant",
                "num_runs",
                "fitness_delta_mean",
                "fitness_delta_ci95",
                "collab_delta_mean",
                "collab_delta_ci95",
                "fitness_upward_rate",
                "collab_upward_rate",
                "both_upward_rate",
                "final_replication_all_pass_rate",
                "final_team_contribution_credit_mean",
            ]
        )
        for variant, stats in sorted(aggregate["variants"].items()):
            writer.writerow(
                [
                    variant,
                    stats["num_runs"],
                    stats["fitness_delta"]["mean"],
                    stats["fitness_delta"]["ci95"],
                    stats["collab_delta"]["mean"],
                    stats["collab_delta"]["ci95"],
                    stats["rates"]["fitness_upward_rate"],
                    stats["rates"]["collab_upward_rate"],
                    stats["rates"]["both_upward_rate"],
                    stats["rates"]["final_replication_all_pass_rate"],
                    stats["final_team_contribution_credit"]["mean"],
                ]
            )

    if run_rows:
        fieldnames = sorted({k for row in run_rows for k in row.keys()})
        with open(runs_csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(run_rows)

    print(json.dumps(aggregate, ensure_ascii=False, indent=2))
    print(f"Wrote: {json_path}")
    print(f"Wrote: {csv_path}")
    print(f"Wrote: {runs_csv_path}")


if __name__ == "__main__":
    main()
