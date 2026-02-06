"""Example custom controller"""

from __future__ import annotations

import json
import os
from datetime import datetime

from agentkernel_standalone.mas.controller.controller import ControllerImpl
from agentkernel_standalone.toolkit.logger import get_logger

logger = get_logger(__name__)


class CustomController(ControllerImpl):
    """Controller extension"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    async def update_agents_status(self) -> None:
        """Trigger each pod to refresh agent status within the environment.

        Returns:
            None
        """
        logger.info("Updating agent status...")

    async def finalize_episode(self) -> None:
        """Aggregate agent fitness and write a leaderboard for the episode."""
        base = os.getenv("MAS_PROJECT_ABS_PATH", ".")
        out_dir = os.path.join(base, "logs", "app", "simulation")
        os.makedirs(out_dir, exist_ok=True)
        # leaderboard_path = os.path.join(out_dir, "leaderboard.jsonl")
        leaderboard_json_path = os.path.join(out_dir, "leaderboard.json")
        leaderboard_csv_path = os.path.join(out_dir, "leaderboard.csv")

        agents = self.get_agent_ids()
        records = []
        for aid in agents:
            fitness = await self.run_agent_method(aid, "state", "get_state", "last_fitness")
            policy = await self.run_agent_method(aid, "state", "get_state", "policy")
            exp_count = await self.run_agent_method(aid, "state", "get_state", "exp_count")
            hypothesis = await self.run_agent_method(aid, "state", "get_state", "hypothesis")
            record = {
                "agent_id": aid,
                "fitness": fitness,
                "exp_count": exp_count,
                "hypothesis": hypothesis,
                "policy": policy,
            }
            records.append(record)

        def fitness_score(rec):
            fit = rec.get("fitness") or {}
            return fit.get("fitness", float("-inf"))

        records_sorted = sorted(records, key=fitness_score, reverse=True)

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
                    "f1": fit.get("f1"),
                    "fitness": fit.get("fitness"),
                    "exp_count": rec.get("exp_count"),
                    "hypothesis": rec.get("hypothesis"),
                }
            )

        try:
            with open(leaderboard_json_path, "w", encoding="utf-8") as f:
                f.write(json.dumps({"ts": summary["ts"], "leaderboard": flat_rows}, ensure_ascii=False, indent=2))
                f.write("\n")
            with open(leaderboard_csv_path, "w", encoding="utf-8") as f:
                f.write("rank,agent_id,f1,fitness,exp_count,hypothesis\n")
                for row in flat_rows:
                    hypothesis = row.get("hypothesis") or []
                    hypothesis_str = "|".join(str(v) for v in hypothesis)
                    f.write(
                        f'{row["rank"]},{row["agent_id"]},{row.get("f1")},{row.get("fitness")},{row.get("exp_count")},{hypothesis_str}\n'
                    )

            logger.info(f"Episode leaderboard written with {len(records_sorted)} agents.")
        except Exception as e:
            logger.error(f"Failed to write leaderboard: {e}")

