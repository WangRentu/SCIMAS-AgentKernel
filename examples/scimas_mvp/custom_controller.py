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
        self._last_team_record: Dict[str, Any] | None = None
        self._log_mode = (os.getenv("SCIMAS_LOG_MODE", "compact") or "compact").strip().lower()
        
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

    def _minmax(self, rows: List[Dict[str, Any]], key: str) -> Dict[str, float]:
        values = [float(r.get(key, 0.0) or 0.0) for r in rows]
        if not values:
            return {}
        lo = min(values)
        hi = max(values)
        if abs(hi - lo) < 1e-9:
            return {str(r.get("agent_id")): 0.0 for r in rows}
        return {str(r.get("agent_id")): (float(r.get(key, 0.0) or 0.0) - lo) / (hi - lo) for r in rows}

    def _selection_score(self, rec: Dict[str, Any], team_ctx: Dict[str, Any], norms: Dict[str, Dict[str, float]]) -> float:
        aid = str(rec.get("agent_id"))
        individual = norms["individual"].get(aid, 0.0)
        contrib = norms["contrib"].get(aid, 0.0)
        collab = norms["collab"].get(aid, 0.0)
        target_collab = float(os.getenv("SCIMAS_TARGET_COLLAB_RATIO", "0.15"))
        team_collab = float(team_ctx.get("collaboration_ratio", 0.0) or 0.0)
        team_fitness = float(team_ctx.get("team_fitness", 0.0) or 0.0)

        w_ind = float(os.getenv("SCIMAS_EVOLVE_W_INDIV", "0.70"))
        w_contrib = float(os.getenv("SCIMAS_EVOLVE_W_CONTRIB", "0.20"))
        w_collab = float(os.getenv("SCIMAS_EVOLVE_W_COLLAB", "0.10"))
        if team_collab < target_collab:
            # Under-collaboration: shift selection toward collaboration and contribution signals.
            gap = min(1.0, max(0.0, (target_collab - team_collab) / max(target_collab, 1e-6)))
            shift = 0.20 * gap
            w_ind = max(0.40, w_ind - shift)
            w_collab = min(0.30, w_collab + 0.5 * shift)
            w_contrib = min(0.35, w_contrib + 0.5 * shift)
        rep_pass_rate = team_ctx.get("replication_pass_rate")
        if rep_pass_rate is None:
            rep_pass_rate = 1.0 if bool(team_ctx.get("replication_all_pass", False)) else 0.0
        rep_pass_rate = float(rep_pass_rate or 0.0)
        target_rep_pass_rate = float(os.getenv("SCIMAS_TARGET_REPLICATION_PASS_RATE", "0.30"))
        if rep_pass_rate < target_rep_pass_rate:
            w_contrib += 0.05
            w_ind = max(0.35, w_ind - 0.05)

        # Team fitness modulates strength of social signals instead of adding a useless constant.
        team_gain = 1.0 + max(-0.5, min(0.5, team_fitness))
        score = (w_ind * individual) + (w_contrib * contrib * team_gain) + (w_collab * collab * team_gain)
        return float(score)

    async def evolve_population(self, top_ratio: float = 0.2, noise_scale: float = 0.03) -> Dict[str, Any]:
        agents = self.get_agent_ids()
        if len(agents) < 2:
            return {"ok": False, "reason": "not_enough_agents"}

        records: List[Dict[str, Any]] = []
        team_ctx = dict(self._last_team_record or {})
        for aid in agents:
            fitness = await self.run_agent_method(aid, "state", "get_state", "last_fitness")
            policy = await self.run_agent_method(aid, "state", "get_state", "policy")
            contribution_credit = await self.run_agent_method(aid, "state", "get_state", "contribution_credit_total")
            shares_e = await self.run_agent_method(aid, "state", "get_state", "share_sent_evidence_count")
            shares_o = await self.run_agent_method(aid, "state", "get_state", "share_sent_observation_count")
            collab_count = float(shares_e or 0) + float(shares_o or 0)
            individual_fit = self._fit_value({"fitness": fitness})
            if individual_fit == float("-inf"):
                individual_fit = 0.0
            rec = {
                "agent_id": aid,
                "fitness": fitness,
                "policy": policy or {},
                "individual_fitness": float(individual_fit),
                "contribution_credit_total": float(contribution_credit or 0.0),
                "collab_count": float(collab_count),
            }
            records.append(rec)

        norms = {
            "individual": self._minmax(records, "individual_fitness"),
            "contrib": self._minmax(records, "contribution_credit_total"),
            "collab": self._minmax(records, "collab_count"),
        }
        for rec in records:
            score = self._selection_score(rec, team_ctx=team_ctx, norms=norms)
            rec["selection_score"] = score
            await self.run_agent_method(rec["agent_id"], "state", "set_state", "last_selection_score", score)
        ranked = sorted(records, key=lambda r: float(r.get("selection_score", 0.0)), reverse=True)
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
                    "receiver_selection_score": rec.get("selection_score"),
                    "donor_selection_score": donor.get("selection_score"),
                }
            )
        result = {
            "ts": datetime.utcnow().isoformat() + "Z",
            "ok": True,
            "top_k": k,
            "noise_scale": noise_scale,
            "selection_mode": "mixed_individual_contribution_collab",
            "team_context": {
                "team_fitness": team_ctx.get("team_fitness"),
                "collaboration_ratio": team_ctx.get("collaboration_ratio"),
                "replication_all_pass": team_ctx.get("replication_all_pass"),
                "replication_pass_rate": team_ctx.get("replication_pass_rate"),
                "replication_verified_rate": team_ctx.get("replication_verified_rate"),
                "publishable_rate": team_ctx.get("publishable_rate"),
            },
            "top_donors": [
                {
                    "agent_id": d.get("agent_id"),
                    "selection_score": d.get("selection_score"),
                    "individual_fitness": d.get("individual_fitness"),
                    "contribution_credit_total": d.get("contribution_credit_total"),
                    "collab_count": d.get("collab_count"),
                }
                for d in donors[: min(5, len(donors))]
            ],
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
            await self.run_agent_method(aid, "state", "set_state", "last_effective_action", None)
            await self.run_agent_method(aid, "state", "set_state", "last_reward", 0.0)
            await self.run_agent_method(aid, "state", "set_state", "last_learning_reward", 0.0)
            await self.run_agent_method(aid, "state", "set_state", "last_reward_components", {})
            await self.run_agent_method(aid, "state", "set_state", "last_fitness", None)
            await self.run_agent_method(aid, "state", "set_state", "last_paper_id", None)
            await self.run_agent_method(aid, "state", "set_state", "current_task_id", None)
            await self.run_agent_method(aid, "state", "set_state", "episode_reward_ledger", {})
            await self.run_agent_method(aid, "state", "set_state", "episode_action_counts", {})
            await self.run_agent_method(aid, "state", "set_state", "credit_buffer", 0.0)
            await self.run_agent_method(aid, "state", "set_state", "contribution_credit_total", 0.0)
            await self.run_agent_method(aid, "state", "set_state", "share_sent_evidence_count", 0)
            await self.run_agent_method(aid, "state", "set_state", "share_sent_observation_count", 0)
            await self.run_agent_method(aid, "state", "set_state", "paper_write_count", 0)
            await self.run_agent_method(aid, "state", "set_state", "review_count", 0)
            await self.run_agent_method(aid, "state", "set_state", "replication_count", 0)
            await self.run_agent_method(aid, "state", "set_state", "last_selection_score", None)

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
            contribution_credit_total = await self.run_agent_method(aid, "state", "get_state", "contribution_credit_total")
            share_sent_evidence_count = await self.run_agent_method(aid, "state", "get_state", "share_sent_evidence_count")
            share_sent_observation_count = await self.run_agent_method(aid, "state", "get_state", "share_sent_observation_count")
            episode_action_counts = await self.run_agent_method(aid, "state", "get_state", "episode_action_counts")
            record = {
                "agent_id": aid,
                "fitness": fitness,
                "exp_count": exp_count,
                "hypothesis": hypothesis,
                "policy": policy,
                "paper_id": paper_id,
                "contribution_credit_total": float(contribution_credit_total or 0.0),
                "share_sent_evidence_count": int(share_sent_evidence_count or 0),
                "share_sent_observation_count": int(share_sent_observation_count or 0),
                "episode_action_counts": episode_action_counts or {},
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
        evidence_cov_vals = [float((r.get("fitness") or {}).get("evidence_coverage_score", 0.0) or 0.0) for r in records]
        replication_flags = [bool((r.get("fitness") or {}).get("replication_ok", False)) for r in records if r.get("fitness")]
        replication_verified_flags = [
            bool((r.get("fitness") or {}).get("replication_verified", False)) for r in records if r.get("fitness")
        ]
        publishable_flags = [bool((r.get("fitness") or {}).get("publishable", False)) for r in records if r.get("fitness")]
        preprint_flags = [bool((r.get("fitness") or {}).get("gate_preprint_pass", False)) for r in records if r.get("fitness")]
        readiness_vals = [float((r.get("fitness") or {}).get("readiness_score", 0.0) or 0.0) for r in records]
        replication_support_vals = [
            float((r.get("fitness") or {}).get("replication_support_score", 0.0) or 0.0) for r in records if r.get("fitness")
        ]
        budget_vals = [int((r.get("fitness") or {}).get("budget", 10) or 10) for r in records if r.get("fitness")]
        mean_budget = sum(budget_vals) / len(budget_vals) if budget_vals else 10
        cost = sum(float(rec.get("exp_count") or 0) / max(1.0, float(mean_budget)) for rec in records) / max(1, len(records))
        team_graph = sum(graph_vals) / max(1, len(graph_vals))
        team_evidence = sum(evidence_vals) / max(1, len(evidence_vals))
        team_evidence_coverage = sum(evidence_cov_vals) / max(1, len(evidence_cov_vals))
        replication_all_pass = all(replication_flags) if replication_flags else False
        replication_pass_rate = (
            sum(1.0 for x in replication_flags if x) / len(replication_flags) if replication_flags else 0.0
        )
        replication_verified_rate = (
            sum(1.0 for x in replication_verified_flags if x) / len(replication_verified_flags)
            if replication_verified_flags
            else 0.0
        )
        publishable_rate = (
            sum(1.0 for x in publishable_flags if x) / len(publishable_flags) if publishable_flags else 0.0
        )
        preprint_ready_rate = (
            sum(1.0 for x in preprint_flags if x) / len(preprint_flags) if preprint_flags else 0.0
        )
        team_readiness = sum(readiness_vals) / max(1, len(readiness_vals))
        team_replication_support = sum(replication_support_vals) / max(1, len(replication_support_vals)) if replication_support_vals else 0.0

        # Team protocol v4: smoother rates preserve evolutionary signal under strict pipelines.
        team_quality = (
            0.60 * team_graph
            + 0.15 * team_evidence
            + 0.05 * team_evidence_coverage
            + 0.10 * replication_pass_rate
            + 0.10 * publishable_rate
        )
        team_fitness = team_quality + (0.10 * team_readiness) + (0.05 * replication_verified_rate) - cost
        # Smooth replication penalty replaces hard all-pass collapse.
        replication_penalty = 0.80 + (0.20 * replication_pass_rate)
        team_fitness *= replication_penalty
        if replication_verified_rate <= 0.0:
            team_fitness -= 0.05

        share_count = 0
        total_actions = 0
        for rec in records:
            counts = rec.get("episode_action_counts") or {}
            if not isinstance(counts, dict):
                continue
            total_actions += sum(int(v or 0) for v in counts.values())
            share_count += int(counts.get("share_evidence", 0) or 0) + int(counts.get("share_observation", 0) or 0)
        collaboration_ratio = float(share_count) / float(total_actions) if total_actions else 0.0
        team_contrib_credit = sum(float(rec.get("contribution_credit_total") or 0.0) for rec in records)
        team_share_sent_evidence = sum(int(rec.get("share_sent_evidence_count") or 0) for rec in records)
        team_share_sent_observation = sum(int(rec.get("share_sent_observation_count") or 0) for rec in records)

        team_record = {
            "ts": datetime.utcnow().isoformat() + "Z",
            "episode_index": episode_index,
            "team_graph": team_graph,
            "team_evidence": team_evidence,
            "team_evidence_coverage": team_evidence_coverage,
            "team_cost": cost,
            "team_fitness": team_fitness,
            "replication_all_pass": replication_all_pass,
            "replication_penalty": replication_penalty,
            "replication_pass_rate": replication_pass_rate,
            "replication_verified_rate": replication_verified_rate,
            "team_replication_support": team_replication_support,
            "publishable_rate": publishable_rate,
            "preprint_ready_rate": preprint_ready_rate,
            "team_readiness": team_readiness,
            "collaboration_ratio": collaboration_ratio,
            "team_contribution_credit": team_contrib_credit,
            "team_share_sent_evidence": team_share_sent_evidence,
            "team_share_sent_observation": team_share_sent_observation,
        }
        self._last_team_record = dict(team_record)

        try:
            if self._log_mode != "minimal":
                with open(leaderboard_json_path, "w", encoding="utf-8") as f:
                    f.write(json.dumps({"ts": summary["ts"], "leaderboard": flat_rows}, ensure_ascii=False, indent=2))
                    f.write("\n")
            if self._log_mode == "verbose":
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
