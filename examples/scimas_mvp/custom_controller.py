"""Example custom controller"""

from __future__ import annotations

import json
import os
import random
from datetime import datetime
from typing import Any, Dict, List

from agentkernel_standalone.mas.controller.controller import ControllerImpl
from agentkernel_standalone.toolkit.logger import get_logger
from examples.scimas_mvp.visualization.trend_dashboard import generate_trend_dashboard

logger = get_logger(__name__)


class CustomController(ControllerImpl):
    """Controller extension"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._evolve_rng = random.Random(42)
        
    async def update_agents_status(self) -> None:
        """Trigger each pod to refresh agent status within the environment.

        Returns:
            None
        """
        logger.info("Updating agent status...")

    def _fit_value(self, rec: Dict[str, Any]) -> float:
        fit = rec.get("fitness") or {}
        value = fit.get("fitness")
        return float(value) if value is not None else float("-inf")

    def _normalize_policy(self, policy: Dict[str, float]) -> Dict[str, float]:
        clipped = {k: max(1e-6, float(v)) for k, v in (policy or {}).items()}
        total = sum(clipped.values()) or 1.0
        return {k: v / total for k, v in clipped.items()}

    def _mutate_policy(self, policy: Dict[str, float], noise_scale: float = 0.03) -> Dict[str, float]:
        mutated = {}
        for action, prob in (policy or {}).items():
            noise = self._evolve_rng.uniform(-noise_scale, noise_scale)
            mutated[action] = max(1e-6, float(prob) + noise)
        return self._normalize_policy(mutated)

    async def evolve_population(self, top_ratio: float = 0.2, noise_scale: float = 0.03) -> Dict[str, Any]:
        agents = self.get_agent_ids()
        if len(agents) < 2:
            return {"ok": False, "reason": "not_enough_agents"}

        records: List[Dict[str, Any]] = []
        for aid in agents:
            fitness = await self.run_agent_method(aid, "state", "get_state", "last_fitness")
            policy = await self.run_agent_method(aid, "state", "get_state", "policy")
            records.append({"agent_id": aid, "fitness": fitness, "policy": policy or {}})
        ranked = sorted(records, key=self._fit_value, reverse=True)
        k = max(1, int(len(ranked) * top_ratio))
        donors = ranked[:k]
        receivers = list(reversed(ranked[-k:]))

        changes = []
        for idx, rec in enumerate(receivers):
            donor = donors[idx % len(donors)]
            donor_policy = self._normalize_policy(donor.get("policy") or {})
            new_policy = self._mutate_policy(donor_policy, noise_scale=noise_scale)
            await self.run_agent_method(rec["agent_id"], "state", "set_state", "policy", new_policy)
            changes.append(
                {
                    "receiver": rec["agent_id"],
                    "donor": donor["agent_id"],
                    "noise_scale": noise_scale,
                }
            )
        result = {
            "ts": datetime.utcnow().isoformat() + "Z",
            "ok": True,
            "top_k": k,
            "noise_scale": noise_scale,
            "changes": changes,
        }
        base = os.getenv("MAS_PROJECT_ABS_PATH", ".")
        evo_path = os.path.join(base, "logs", "app", "simulation", "evolution.jsonl")
        try:
            os.makedirs(os.path.dirname(evo_path), exist_ok=True)
            with open(evo_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.warning(f"Failed to persist evolution result: {e}")
        return result

    async def reset_episode_state(self) -> None:
        await self.run_environment("science", "reset_episode")
        for aid in self.get_agent_ids():
            await self.run_agent_method(aid, "state", "set_state", "exp_count", 0)
            await self.run_agent_method(aid, "state", "set_state", "observations", [])
            await self.run_agent_method(aid, "state", "set_state", "replications", [])
            await self.run_agent_method(aid, "state", "set_state", "shared_observations", [])
            await self.run_agent_method(aid, "state", "set_state", "hypothesis", [])
            await self.run_agent_method(aid, "state", "set_state", "notes", [])
            await self.run_agent_method(aid, "state", "set_state", "shared_notes", [])
            await self.run_agent_method(aid, "state", "set_state", "inbox_evidence", [])
            await self.run_agent_method(aid, "state", "set_state", "last_action", None)
            await self.run_agent_method(aid, "state", "set_state", "last_reward", 0.0)
            await self.run_agent_method(aid, "state", "set_state", "last_fitness", None)
            await self.run_agent_method(aid, "state", "set_state", "last_paper_id", None)
            await self.run_agent_method(aid, "state", "set_state", "current_task_id", None)

    async def finalize_episode(self, episode_index: int = 0) -> Dict[str, Any]:
        """Aggregate agent fitness and write a leaderboard for the episode."""
        base = os.getenv("MAS_PROJECT_ABS_PATH", ".")
        out_dir = os.path.join(base, "logs", "app", "simulation")
        os.makedirs(out_dir, exist_ok=True)
        # leaderboard_path = os.path.join(out_dir, "leaderboard.jsonl")
        leaderboard_json_path = os.path.join(out_dir, "leaderboard.json")
        leaderboard_csv_path = os.path.join(out_dir, "leaderboard.csv")
        team_metrics_path = os.path.join(out_dir, "team_metrics.jsonl")

        agents = self.get_agent_ids()
        records = []
        for aid in agents:
            fitness = await self.run_agent_method(aid, "state", "get_state", "last_fitness")
            policy = await self.run_agent_method(aid, "state", "get_state", "policy")
            exp_count = await self.run_agent_method(aid, "state", "get_state", "exp_count")
            hypothesis = await self.run_agent_method(aid, "state", "get_state", "hypothesis")
            paper_id = await self.run_agent_method(aid, "state", "get_state", "last_paper_id")
            record = {
                "agent_id": aid,
                "fitness": fitness,
                "exp_count": exp_count,
                "hypothesis": hypothesis,
                "policy": policy,
                "paper_id": paper_id,
            }
            records.append(record)

        records_sorted = sorted(records, key=self._fit_value, reverse=True)

        summary = {
            "ts": datetime.utcnow().isoformat() + "Z",
            "episode_leaderboard": records_sorted,
        }

        flat_rows = []
        for idx, rec in enumerate(records_sorted, start=1):
            fit = rec.get("fitness") or {}
            flat_rows.append(
                {
                    "rank": idx,
                    "agent_id": rec.get("agent_id"),
                    "paper_id": rec.get("paper_id"),
                    "f1": fit.get("f1"),
                    "graph_score": fit.get("graph_score"),
                    "evidence_score": fit.get("evidence_score"),
                    "replication_ok": fit.get("replication_ok"),
                    "fitness": fit.get("fitness"),
                    "exp_count": rec.get("exp_count"),
                    "hypothesis": rec.get("hypothesis"),
                }
            )

        graph_vals = [float((r.get("fitness") or {}).get("graph_score", 0.0) or 0.0) for r in records]
        evidence_vals = [float((r.get("fitness") or {}).get("evidence_score", 0.0) or 0.0) for r in records]
        replication_flags = [bool((r.get("fitness") or {}).get("replication_ok", False)) for r in records if r.get("fitness")]
        budget_vals = [int((r.get("fitness") or {}).get("budget", 10) or 10) for r in records if r.get("fitness")]
        mean_budget = sum(budget_vals) / len(budget_vals) if budget_vals else 10
        cost = sum(float(rec.get("exp_count") or 0) / max(1.0, float(mean_budget)) for rec in records) / max(1, len(records))
        team_graph = sum(graph_vals) / max(1, len(graph_vals))
        team_evidence = sum(evidence_vals) / max(1, len(evidence_vals))
        team_fitness = team_graph + 0.2 * team_evidence - cost
        replication_all_pass = all(replication_flags) if replication_flags else False
        replication_penalty = 0.3
        if not replication_all_pass:
            team_fitness *= replication_penalty

        world_spec = await self.run_environment("science", "get_world_spec")
        current_episode_id = world_spec.get("episode_id")
        trace_path = os.path.join(base, "logs", "app", "action", "trace.jsonl")
        share_count = 0
        total_actions = 0
        if os.path.exists(trace_path):
            try:
                with open(trace_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        rec = json.loads(line)
                        if current_episode_id is not None and rec.get("episode_id") != current_episode_id:
                            continue
                        action = rec.get("action")
                        if action:
                            total_actions += 1
                            if action in ("share_evidence", "share_observation"):
                                share_count += 1
            except Exception as e:
                logger.warning(f"Failed to parse trace for collaboration ratio: {e}")
        collaboration_ratio = float(share_count) / float(total_actions) if total_actions else 0.0

        team_record = {
            "ts": datetime.utcnow().isoformat() + "Z",
            "episode_index": episode_index,
            "team_graph": team_graph,
            "team_evidence": team_evidence,
            "team_cost": cost,
            "team_fitness": team_fitness,
            "replication_all_pass": replication_all_pass,
            "replication_penalty": replication_penalty if not replication_all_pass else 1.0,
            "collaboration_ratio": collaboration_ratio,
        }

        try:
            with open(leaderboard_json_path, "w", encoding="utf-8") as f:
                f.write(json.dumps({"ts": summary["ts"], "leaderboard": flat_rows}, ensure_ascii=False, indent=2))
                f.write("\n")
            with open(leaderboard_csv_path, "w", encoding="utf-8") as f:
                f.write("rank,agent_id,paper_id,f1,graph_score,evidence_score,replication_ok,fitness,exp_count,hypothesis\n")
                for row in flat_rows:
                    hypothesis = row.get("hypothesis") or []
                    hypothesis_str = "|".join(str(v) for v in hypothesis)
                    f.write(
                        f'{row["rank"]},{row["agent_id"]},{row.get("paper_id")},{row.get("f1")},{row.get("graph_score")},{row.get("evidence_score")},{row.get("replication_ok")},{row.get("fitness")},{row.get("exp_count")},{hypothesis_str}\n'
                    )
            with open(team_metrics_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(team_record, ensure_ascii=False) + "\n")

            logger.info(f"Episode leaderboard written with {len(records_sorted)} agents.")
            plot_result = generate_trend_dashboard(base_dir=base, out_dir=out_dir)
            logger.info(f"Trend dashboard generated: {plot_result}")
            return {"ok": True, "team": team_record}
        except Exception as e:
            logger.error(f"Failed to write leaderboard: {e}")
            return {"ok": False, "error": str(e)}
