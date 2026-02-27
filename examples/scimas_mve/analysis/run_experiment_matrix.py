from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime
from typing import Dict, List


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
REPO_ROOT = os.path.abspath(os.path.join(PROJECT_ROOT, "..", "..", ".."))


def _parse_seeds(raw: str) -> List[int]:
    seeds: List[int] = []
    for part in (raw or "").split(","):
        part = part.strip()
        if not part:
            continue
        seeds.append(int(part))
    return seeds


def _default_variants() -> Dict[str, Dict[str, str]]:
    return {
        "baseline_like": {
            "SCIMAS_EVOLVE_W_INDIV": "1.0",
            "SCIMAS_EVOLVE_W_CONTRIB": "0.0",
            "SCIMAS_EVOLVE_W_COLLAB": "0.0",
        },
        "aligned_v1": {
            "SCIMAS_EVOLVE_W_INDIV": "0.70",
            "SCIMAS_EVOLVE_W_CONTRIB": "0.20",
            "SCIMAS_EVOLVE_W_COLLAB": "0.10",
            "SCIMAS_TARGET_COLLAB_RATIO": "0.15",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run comparable multi-seed scimas_mve experiments and archive results.")
    parser.add_argument("--experiment-name", default=f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    parser.add_argument("--seeds", default="42,43,44", help="Comma-separated world seeds.")
    parser.add_argument(
        "--variants",
        default="baseline_like,aligned_v1",
        help="Comma-separated variant names. Built-ins: baseline_like, aligned_v1",
    )
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--extra-env", action="append", default=[], help="Extra KEY=VALUE env overrides.")
    parser.add_argument("--no-clean-sim-logs", action="store_true", help="Do not clean simulation logs before each run.")
    args = parser.parse_args()

    seeds = _parse_seeds(args.seeds)
    variants_map = _default_variants()
    chosen_variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    unknown = [v for v in chosen_variants if v not in variants_map]
    if unknown:
        raise SystemExit(f"Unknown variants: {unknown}. Available: {sorted(variants_map)}")

    archive_root = os.path.join(PROJECT_ROOT, "logs", "experiments", args.experiment_name, "runs")
    os.makedirs(archive_root, exist_ok=True)

    extra_env: Dict[str, str] = {}
    for item in args.extra_env:
        if "=" not in item:
            raise SystemExit(f"--extra-env must be KEY=VALUE, got {item}")
        k, v = item.split("=", 1)
        extra_env[k.strip()] = v

    total = len(seeds) * len(chosen_variants)
    idx = 0
    for variant in chosen_variants:
        for seed in seeds:
            idx += 1
            run_tag = f"{variant}_seed{seed}"
            env = os.environ.copy()
            env.update(variants_map[variant])
            env.update(extra_env)
            env["SCIMAS_WORLD_SEED"] = str(seed)
            env["SCIMAS_EXPERIMENT_NAME"] = args.experiment_name
            env["SCIMAS_VARIANT_NAME"] = variant
            env["SCIMAS_RUN_TAG"] = run_tag
            env["SCIMAS_EXPERIMENT_ARCHIVE_DIR"] = archive_root
            if not args.no_clean_sim_logs:
                env["SCIMAS_CLEAN_SIM_LOGS"] = "1"

            cmd = [args.python, "-m", "examples.scimas_mve.run_simulation"]
            print(f"[{idx}/{total}] Running variant={variant} seed={seed}")
            proc = subprocess.run(cmd, cwd=REPO_ROOT, env=env)
            if proc.returncode != 0:
                raise SystemExit(f"Run failed for variant={variant}, seed={seed}, code={proc.returncode}")

    print(f"Completed {total} runs. Archived under: {archive_root}")
    print("Next: python -m examples.scimas_mve.analysis.aggregate_experiment_runs --experiment-name", args.experiment_name)


if __name__ == "__main__":
    main()
