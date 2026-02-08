import json
import os
import random
from datetime import datetime
from typing import Any, Dict, List, Optional

from agentkernel_standalone.mas.action.base.plugin_base import OtherActionsPlugin
from agentkernel_standalone.toolkit.logger import get_logger
from agentkernel_standalone.toolkit.utils.annotation import AgentCall, ServiceCall
from agentkernel_standalone.types.schemas.action import ActionResult

logger = get_logger(__name__)


class ResearchActionsPlugin(OtherActionsPlugin):
    """
    Minimal research action plugin for causal discovery.
    """

    def __init__(self):
        super().__init__()
        self._rng = random.Random()
        base = os.getenv("MAS_PROJECT_ABS_PATH", ".")
        self._trace_path = os.path.join(base, "logs", "app", "action", "trace.jsonl")
        self._research_log_dir = os.path.join(base, "logs", "app", "research")
        self._cards_log_path = os.path.join(self._research_log_dir, "evidence_cards.jsonl")
        self._papers_log_path = os.path.join(self._research_log_dir, "papers.jsonl")

    async def init(self, model_router=None, controller=None):
        self.model = model_router
        self.controller = controller
        os.makedirs(os.path.dirname(self._trace_path), exist_ok=True)
        os.makedirs(self._research_log_dir, exist_ok=True)

    async def _log_action(self, *args, **kwargs):
        return None

    @ServiceCall
    async def save_to_db(self):
        return ActionResult.success(method_name="save_to_db", message="No state to save.")

    @ServiceCall
    async def load_from_db(self):
        return ActionResult.success(method_name="load_from_db", message="No state to load.")

    async def _get_state(self, agent_id: str, key: str) -> Any:
        return await self.controller.run_agent_method(agent_id, "state", "get_state", key)

    async def _set_state(self, agent_id: str, key: str, value: Any) -> None:
        await self.controller.run_agent_method(agent_id, "state", "set_state", key, value)

    async def _get_claimed_task(self, agent_id: str, task_id: str) -> Optional[Dict[str, Any]]:
        listed = await self.controller.run_environment("science", "task_list", status="claimed", agent_id=agent_id)
        tasks = (listed or {}).get("tasks", []) if isinstance(listed, dict) else []
        for task in tasks:
            if task.get("task_id") == task_id:
                return task
        return None

    async def _execute_task_action(
        self,
        agent_id: str,
        task_action: str,
        task_payload: Optional[Dict[str, Any]] = None,
    ) -> ActionResult:
        payload = task_payload or {}
        if task_action == "read":
            return await self.read(agent_id=agent_id, topic=payload.get("topic"))
        if task_action == "hypothesize":
            return await self.hypothesize(agent_id=agent_id)
        if task_action == "experiment":
            return await self.experiment(
                agent_id=agent_id,
                intervention=payload.get("intervention"),
                n_samples=int(payload.get("n_samples", 50)),
            )
        if task_action == "replicate":
            return await self.replicate(
                agent_id=agent_id,
                intervention=payload.get("intervention"),
                n_samples=int(payload.get("n_samples", 50)),
            )
        if task_action == "write":
            return await self.write(agent_id=agent_id)
        if task_action == "review":
            return await self.review(agent_id=agent_id, paper_id=payload.get("paper_id"))
        if task_action == "share_evidence":
            return await self.share_evidence(
                agent_id=agent_id,
                to_agent_id=payload.get("to_agent_id"),
                max_hints=int(payload.get("max_hints", 3)),
            )
        if task_action == "share_observation":
            return await self.share_observation(agent_id=agent_id, to_agent_id=payload.get("to_agent_id"))
        return ActionResult.success(method_name=task_action, message="No-op task action.", data={"reward": 0.0})

    def _pick_recipient(self, agent_id: str, to_agent_id: Optional[str] = None) -> Optional[str]:
        if to_agent_id and to_agent_id != agent_id:
            return to_agent_id
        others = [aid for aid in self.controller.get_agent_ids() if aid != agent_id]
        if not others:
            return None
        return self._rng.choice(others)

    async def _append_trace(self, agent_id: str, action: str, reward: float, detail: Dict[str, Any]):
        tick = await self.controller.run_system("timer", "get_tick")
        world_spec = await self.controller.run_environment("science", "get_world_spec")
        fitness = await self._get_state(agent_id, "last_fitness")
        exp_count = await self._get_state(agent_id, "exp_count") or 0
        hypothesis = await self._get_state(agent_id, "hypothesis") or []
        record = {
            "ts": datetime.utcnow().isoformat() + "Z",
            "tick": tick,
            "episode_id": world_spec.get("episode_id"),
            "agent_id": agent_id,
            "action": action,
            "reward": reward,
            "exp_count": exp_count,
            "hypothesis": hypothesis,
            "last_fitness": fitness,
            "detail": detail,
        }
        try:
            with open(self._trace_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.error(f"Failed to write trace: {e}")

    async def _append_jsonl(self, path: str, record: Dict[str, Any]) -> None:
        try:
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.error(f"Failed to write jsonl {path}: {e}")

    async def _grant_credits(
        self,
        credit_by_agent: Dict[str, float],
        source: str,
        reference_id: Optional[str] = None,
    ) -> None:
        for recipient, credit in credit_by_agent.items():
            if not recipient or credit <= 0:
                continue
            try:
                prev = await self._get_state(recipient, "last_reward") or 0.0
                updated = float(prev) + float(credit)
                await self._set_state(recipient, "last_reward", updated)
                await self._set_state(
                    recipient,
                    "last_credit",
                    {
                        "source": source,
                        "value": float(credit),
                        "reference_id": reference_id,
                    },
                )
            except Exception as e:
                logger.warning(f"Credit assignment failed for {recipient}: {e}")

    def _build_citation_owner_map(
        self,
        *,
        agent_id: str,
        local_notes: List[Dict[str, Any]],
        shared_notes: List[Dict[str, Any]],
    ) -> Dict[str, str]:
        owner_by_citation: Dict[str, str] = {}
        for note in local_notes or []:
            for card in (note or {}).get("cards", []) or []:
                cid = card.get("citation_id")
                if cid and cid not in owner_by_citation:
                    owner_by_citation[cid] = agent_id
        for note in shared_notes or []:
            owner = (note or {}).get("source_agent")
            if not owner:
                continue
            for card in (note or {}).get("cards", []) or []:
                cid = card.get("citation_id")
                if cid and cid not in owner_by_citation:
                    owner_by_citation[cid] = owner
        return owner_by_citation

    def _compute_contribution_credit(
        self,
        *,
        agent_id: str,
        paper: Dict[str, Any],
        metrics: Dict[str, Any],
        shared_observations: List[Dict[str, Any]],
    ) -> Dict[str, float]:
        credit_by_agent: Dict[str, float] = {}
        citation_owner_map = paper.get("citation_owner_map") or {}
        cited = paper.get("citations") or []
        for cid in cited:
            owner = citation_owner_map.get(cid)
            if owner and owner != agent_id:
                credit_by_agent[owner] = credit_by_agent.get(owner, 0.0) + 0.02

        obs_refs = set(paper.get("observation_refs") or [])
        for obs in shared_observations or []:
            owner = (obs or {}).get("source_agent")
            tick = (obs or {}).get("tick")
            if owner and tick is not None:
                ref_key = f"OBS@{owner}@{tick}"
                if ref_key in obs_refs and owner != agent_id:
                    credit_by_agent[owner] = credit_by_agent.get(owner, 0.0) + 0.01

        if bool(metrics.get("replication_ok", False)):
            for cid in cited:
                owner = citation_owner_map.get(cid)
                if owner and owner != agent_id:
                    credit_by_agent[owner] = credit_by_agent.get(owner, 0.0) + 0.03
        return credit_by_agent

    async def _log_evidence_cards(
        self,
        agent_id: str,
        literature: Dict[str, Any],
        source: str = "read",
    ) -> None:
        tick = await self.controller.run_system("timer", "get_tick")
        world_spec = await self.controller.run_environment("science", "get_world_spec")
        cards = literature.get("cards") or []
        record = {
            "ts": datetime.utcnow().isoformat() + "Z",
            "tick": tick,
            "episode_id": world_spec.get("episode_id"),
            "agent_id": agent_id,
            "source": source,
            "topic": literature.get("topic"),
            "agent_view_size": literature.get("agent_view_size", len(cards)),
            "cards": cards,
        }
        await self._append_jsonl(self._cards_log_path, record)

    async def _log_paper_result(
        self,
        agent_id: str,
        paper_id: Optional[str],
        paper: Dict[str, Any],
        metrics: Dict[str, Any],
        source: str,
    ) -> None:
        tick = await self.controller.run_system("timer", "get_tick")
        world_spec = await self.controller.run_environment("science", "get_world_spec")
        record = {
            "ts": datetime.utcnow().isoformat() + "Z",
            "tick": tick,
            "episode_id": world_spec.get("episode_id"),
            "agent_id": agent_id,
            "source": source,
            "paper_id": paper_id,
            "paper": paper,
            "metrics": metrics,
        }
        await self._append_jsonl(self._papers_log_path, record)

    def _build_paper_payload(
        self,
        *,
        target: str,
        hypothesis: List[str],
        notes: List[Dict[str, Any]],
        observations: List[Dict[str, Any]],
        exp_count: int,
    ) -> Dict[str, Any]:
        claimed_edges = [f"{var}->{target}" for var in hypothesis]
        citations: List[str] = []
        evidence_map: Dict[str, List[str]] = {}

        card_pool = []
        for note in notes:
            for card in (note or {}).get("cards", []) or []:
                card_pool.append(card)

        for edge in claimed_edges:
            src = edge.split("->", 1)[0]
            cited = []
            for card in card_pool:
                cid = card.get("citation_id")
                if card.get("var") == src and cid:
                    cited.append(cid)
            unique_cited = list(dict.fromkeys(cited))
            if unique_cited:
                evidence_map[edge] = unique_cited
                citations.extend(unique_cited)

        citations = list(dict.fromkeys(citations))
        observation_refs = []
        for idx, obs in enumerate(observations or []):
            tick = obs.get("tick", idx)
            source_agent = obs.get("source_agent")
            if source_agent:
                observation_refs.append(f"OBS@{source_agent}@{tick}")
            else:
                observation_refs.append(f"OBS@{tick}")

        return {
            "claimed_edges": claimed_edges,
            "citations": citations,
            "evidence_map": evidence_map,
            "observation_refs": observation_refs,
            "method_section": "rule_based_hypothesis + intervention_heuristic",
            "exp_count": exp_count,
        }

    def _infer_hypothesis_rule_based(
        self,
        *,
        target: str,
        candidates: List[str],
        notes: List[Dict[str, Any]],
        observations: List[Dict[str, Any]],
        max_parents: int,
    ) -> Dict[str, Any]:
        """
        Explainable heuristic:
        - Literature prior: count +/- evidence per Xi.
        - Experiment evidence: average |effect| / |delta| for experiments intervening on Xi.
        """
        lit_score: Dict[str, float] = {c: 0.0 for c in candidates}
        for note in notes or []:
            for hint in (note or {}).get("hints", []) or []:
                for c in candidates:
                    if hint.startswith(f"{c} likely causes {target}."):
                        lit_score[c] += 1.0
                    elif hint.startswith(f"No evidence that {c} causes {target}."):
                        lit_score[c] -= 1.0

        exp_strength: Dict[str, List[float]] = {c: [] for c in candidates}
        for obs in observations or []:
            effects = (obs or {}).get("effects") or {}
            intervention = (obs or {}).get("intervention") or {}
            for c in candidates:
                delta = intervention.get(c)
                if delta is None or delta == 0:
                    continue
                effect = effects.get(c)
                if effect is None:
                    continue
                exp_strength[c].append(abs(float(effect)) / max(1e-6, abs(float(delta))))

        exp_score: Dict[str, float] = {
            c: (sum(vals) / len(vals) if vals else 0.0) for c, vals in exp_strength.items()
        }

        # Normalise literature to [-1, 1] roughly (cap extremes).
        lit_norm: Dict[str, float] = {c: max(-1.0, min(1.0, lit_score[c] / 3.0)) for c in candidates}

        # Combine: if we have any experimental evidence, trust it more.
        has_exp = any(exp_score[c] > 0 for c in candidates)
        w_exp = 0.8 if has_exp else 0.2
        w_lit = 1.0 - w_exp
        combined: Dict[str, float] = {c: w_lit * lit_norm[c] + w_exp * exp_score[c] for c in candidates}

        ranked = sorted(candidates, key=lambda c: combined[c], reverse=True)
        chosen = ranked[: max(1, max_parents)]

        explanation = [
            {
                "var": c,
                "literature_score": lit_score[c],
                "experiment_score": exp_score[c],
                "combined_score": combined[c],
            }
            for c in ranked
        ]

        return {"hypothesis": chosen, "explanation": explanation, "weights": {"w_lit": w_lit, "w_exp": w_exp}}

    @AgentCall
    async def read(self, agent_id: str, topic: Optional[str] = None) -> ActionResult:
        literature = await self.controller.run_environment("science", "read_literature", agent_id=agent_id, topic=topic)
        notes = await self._get_state(agent_id, "notes") or []
        notes.append(literature)
        await self._set_state(agent_id, "notes", notes)
        await self._log_evidence_cards(agent_id, literature, source="read")
        logger.info(f"Agent {agent_id} read literature: {literature}")
        ar = ActionResult.success(
            method_name="read",
            message="Literature retrieved.",
            data={"note": literature, "reward": 0.0},
        )
        await self._append_trace(agent_id, "read", 0.0, ar.data or {})
        return ar

    @AgentCall
    async def hypothesize(self, agent_id: str, hypothesis: Optional[List[str]] = None) -> ActionResult:
        inferred: Dict[str, Any] = {}
        if hypothesis is None:
            spec = await self.controller.run_environment("science", "get_world_spec")
            target = spec.get("target")
            candidates = [v for v in spec.get("variables", []) if v != target]
            notes = (await self._get_state(agent_id, "notes") or []) + (await self._get_state(agent_id, "shared_notes") or [])
            observations = (await self._get_state(agent_id, "observations") or []) + (
                await self._get_state(agent_id, "shared_observations") or []
            )
            max_parents = max(1, len(candidates) // 2)

            if not candidates:
                inferred = {"hypothesis": [], "explanation": [], "weights": {"w_lit": 1.0, "w_exp": 0.0}}
            else:
                inferred = self._infer_hypothesis_rule_based(
                    target=target,
                    candidates=candidates,
                    notes=notes,
                    observations=observations,
                    max_parents=max_parents,
                )
            hypothesis = inferred["hypothesis"]
        await self._set_state(agent_id, "hypothesis", hypothesis)
        logger.info(f"Agent {agent_id} proposed hypothesis: {hypothesis}")
        ar = ActionResult.success(
            method_name="hypothesize",
            message="Hypothesis proposed.",
            data={
                "hypothesis": hypothesis,
                "reward": 0.0,
                "explanation": inferred.get("explanation", []),
                "weights": inferred.get("weights", {}),
            },
        )
        await self._append_trace(agent_id, "hypothesize", 0.0, ar.data or {})
        return ar

    @AgentCall
    async def experiment(
        self,
        agent_id: str,
        intervention: Optional[Dict[str, float]] = None,
        n_samples: int = 50,
    ) -> ActionResult:
        exp_count = (await self._get_state(agent_id, "exp_count")) or 0
        exp_count += 1
        intervention = intervention or {}
        result = await self.controller.run_environment(
            "science",
            "run_experiment",
            intervention=intervention,
            n_samples=n_samples,
        )
        await self._set_state(agent_id, "exp_count", exp_count)
        observations = await self._get_state(agent_id, "observations") or []
        tick = await self.controller.run_system("timer", "get_tick")
        observations.append(
            {
                "tick": tick,
                "intervention": intervention,
                **(result or {}),
            }
        )
        await self._set_state(agent_id, "observations", observations)
        logger.info(f"Agent {agent_id} ran experiment #{exp_count}: {result}")
        ar = ActionResult.success(
            method_name="experiment",
            message="Experiment executed.",
            data={"observation": {"intervention": intervention, **(result or {})}, "exp_count": exp_count, "reward": 0.0},
        )
        await self._append_trace(agent_id, "experiment", 0.0, ar.data or {})
        return ar

    @AgentCall
    async def replicate(
        self,
        agent_id: str,
        intervention: Optional[Dict[str, float]] = None,
        n_samples: int = 50,
    ) -> ActionResult:
        intervention = intervention or {}
        if not intervention:
            hypothesis = await self._get_state(agent_id, "hypothesis") or []
            if hypothesis:
                picked = hypothesis[0]
                intervention = {picked: -1.0 if self._rng.random() < 0.5 else 1.0}

        result = await self.controller.run_environment(
            "science",
            "run_experiment",
            intervention=intervention,
            n_samples=n_samples,
        )
        replications = await self._get_state(agent_id, "replications") or []
        tick = await self.controller.run_system("timer", "get_tick")
        replication = {
            "tick": tick,
            "intervention": intervention,
            **(result or {}),
        }
        replications.append(replication)
        await self._set_state(agent_id, "replications", replications)
        ar = ActionResult.success(
            method_name="replicate",
            message="Replication experiment executed.",
            data={"replication": replication, "reward": 0.0},
        )
        await self._append_trace(agent_id, "replicate", 0.0, ar.data or {})
        return ar

    @AgentCall
    async def write(self, agent_id: str) -> ActionResult:
        spec = await self.controller.run_environment("science", "get_world_spec")
        target = spec.get("target") or "Y"
        hypothesis = await self._get_state(agent_id, "hypothesis") or []
        exp_count = await self._get_state(agent_id, "exp_count") or 0
        local_notes = await self._get_state(agent_id, "notes") or []
        shared_notes = await self._get_state(agent_id, "shared_notes") or []
        notes = local_notes + shared_notes
        local_observations = await self._get_state(agent_id, "observations") or []
        shared_observations = await self._get_state(agent_id, "shared_observations") or []
        observations = local_observations + shared_observations
        paper = self._build_paper_payload(
            target=target,
            hypothesis=hypothesis,
            notes=notes,
            observations=observations,
            exp_count=exp_count,
        )
        paper["author_id"] = agent_id
        paper["citation_owner_map"] = self._build_citation_owner_map(
            agent_id=agent_id,
            local_notes=local_notes,
            shared_notes=shared_notes,
        )
        submit_info = await self.controller.run_environment("science", "submit_paper", paper=paper)
        paper_id = (submit_info or {}).get("paper_id")
        metrics = await self.controller.run_environment("science", "evaluate_paper", paper=paper)
        reward = metrics.get("fitness", 0.0)
        contribution_credit = self._compute_contribution_credit(
            agent_id=agent_id,
            paper=paper,
            metrics=metrics,
            shared_observations=shared_observations,
        )
        await self._grant_credits(contribution_credit, source="paper_write", reference_id=paper_id)
        await self._set_state(agent_id, "last_fitness", metrics)
        await self._set_state(agent_id, "last_paper_id", paper_id)
        await self._log_paper_result(agent_id, paper_id, paper, metrics, source="write")
        logger.info(f"Agent {agent_id} write submission: {metrics}")
        ar = ActionResult.success(
            method_name="write",
            message="Submission evaluated.",
            data={
                "metrics": metrics,
                "paper_id": paper_id,
                "paper": paper,
                "credit_by_agent": contribution_credit,
                "reward": reward,
            },
        )
        await self._append_trace(agent_id, "write", reward, ar.data or {})
        return ar

    @AgentCall
    async def review(
        self,
        agent_id: str,
        paper_id: Optional[str] = None,
        submission: Optional[Dict[str, Any]] = None,
    ) -> ActionResult:
        if paper_id:
            paper = await self.controller.run_environment("science", "get_paper", paper_id=paper_id)
            if paper:
                metrics = await self.controller.run_environment("science", "evaluate_paper", paper=paper)
                review_score = 0.5 * metrics.get("graph_score", 0.0) + 0.5 * metrics.get("evidence_score", 0.0)
                reward = 0.1 * review_score
                if bool(metrics.get("replication_ok", False)):
                    author_id = (paper or {}).get("author_id")
                    if author_id:
                        await self._grant_credits({author_id: 0.05}, source="replication_pass", reference_id=paper_id)
                await self._log_paper_result(agent_id, paper_id, paper, metrics, source="review")
                ar = ActionResult.success(
                    method_name="review",
                    message="Paper reviewed.",
                    data={"paper_id": paper_id, "score": review_score, "metrics": metrics, "reward": reward},
                )
                await self._append_trace(agent_id, "review", reward, ar.data or {})
                return ar

        if submission is None:
            submission = {
                "author_id": agent_id,
                "hypothesis": await self._get_state(agent_id, "hypothesis") or [],
            }
        score = await self.controller.run_environment(
            "science",
            "score_hypothesis",
            hypothesis=submission.get("hypothesis", []),
        )
        review_score = score.get("f1", 0.0) if isinstance(score, dict) else 0.0
        logger.info(f"Agent {agent_id} review score: {review_score}")
        ar = ActionResult.success(
            method_name="review",
            message="Review completed.",
            data={"score": review_score, "reward": 0.1 * review_score},
        )
        await self._append_trace(agent_id, "review", 0.1 * review_score, ar.data or {})
        return ar

    @AgentCall
    async def claim_task(
        self,
        agent_id: str,
        task_id: Optional[str] = None,
        task_type: Optional[str] = None,
    ) -> ActionResult:
        selected_task_id = task_id
        if not selected_task_id:
            listed = await self.controller.run_environment("science", "task_list", status="open")
            tasks = (listed or {}).get("tasks", []) if isinstance(listed, dict) else []
            if task_type:
                tasks = [t for t in tasks if t.get("task_type") == task_type]
            if tasks:
                selected_task_id = tasks[0].get("task_id")

        if not selected_task_id:
            ar = ActionResult.success(
                method_name="claim_task",
                message="No open task available.",
                data={"reward": 0.0},
            )
            await self._append_trace(agent_id, "claim_task", 0.0, ar.data or {})
            return ar

        claim_res = await self.controller.run_environment(
            "science",
            "task_claim",
            task_id=selected_task_id,
            agent_id=agent_id,
        )
        ok = bool((claim_res or {}).get("ok"))
        task = (claim_res or {}).get("task")
        reward = 0.01 if ok else 0.0
        if ok:
            await self._set_state(agent_id, "current_task_id", selected_task_id)
        ar = ActionResult.success(
            method_name="claim_task",
            message="Task claimed." if ok else f"Task claim failed: {(claim_res or {}).get('reason')}",
            data={"task_id": selected_task_id, "task": task, "ok": ok, "reward": reward},
        )
        await self._append_trace(agent_id, "claim_task", reward, ar.data or {})
        return ar

    @AgentCall
    async def complete_task(
        self,
        agent_id: str,
        task_id: str,
        task_action: Optional[str] = None,
        task_payload: Optional[Dict[str, Any]] = None,
    ) -> ActionResult:
        task = await self._get_claimed_task(agent_id, task_id)
        action_name = task_action or (task or {}).get("task_type")
        payload = dict((task or {}).get("payload") or {})
        payload.update(task_payload or {})

        action_result = ActionResult.success(method_name="noop", message="No action executed.", data={"reward": 0.0})
        if action_name:
            action_result = await self._execute_task_action(
                agent_id=agent_id,
                task_action=action_name,
                task_payload=payload,
            )

        completion_result = {
            "task_action": action_name,
            "action_status": action_result.status if isinstance(action_result, ActionResult) else None,
            "action_data": action_result.data if isinstance(action_result, ActionResult) else {},
        }
        complete_res = await self.controller.run_environment(
            "science",
            "task_complete",
            task_id=task_id,
            agent_id=agent_id,
            result=completion_result,
        )
        ok = bool((complete_res or {}).get("ok"))
        inner_reward = 0.0
        if isinstance(action_result, ActionResult) and isinstance(action_result.data, dict):
            inner_reward = float(action_result.data.get("reward", 0.0) or 0.0)
        reward = inner_reward + (0.01 if ok else 0.0)
        if ok:
            await self._set_state(agent_id, "current_task_id", None)
        ar = ActionResult.success(
            method_name="complete_task",
            message="Task completed." if ok else f"Task completion failed: {(complete_res or {}).get('reason')}",
            data={
                "task_id": task_id,
                "task_action": action_name,
                "task": (complete_res or {}).get("task"),
                "ok": ok,
                "inner_action_status": action_result.status if isinstance(action_result, ActionResult) else None,
                "inner_action_message": action_result.message if isinstance(action_result, ActionResult) else None,
                "reward": reward,
            },
        )
        await self._append_trace(agent_id, "complete_task", reward, ar.data or {})
        return ar

    @AgentCall
    async def share_evidence(
        self,
        agent_id: str,
        to_agent_id: Optional[str] = None,
        max_hints: int = 3,
    ) -> ActionResult:
        notes = await self._get_state(agent_id, "notes") or []
        if not notes:
            ar = ActionResult.success(
                method_name="share_evidence",
                message="No local evidence to share.",
                data={"reward": 0.0},
            )
            await self._append_trace(agent_id, "share_evidence", 0.0, ar.data or {})
            return ar

        recipient = self._pick_recipient(agent_id, to_agent_id)
        if not recipient:
            ar = ActionResult.success(
                method_name="share_evidence",
                message="No recipient available.",
                data={"reward": 0.0},
            )
            await self._append_trace(agent_id, "share_evidence", 0.0, ar.data or {})
            return ar

        latest_note = notes[-1]
        payload = {
            "type": "evidence_share",
            "from_agent": agent_id,
            "evidence": {
                "topic": latest_note.get("topic"),
                "hints": (latest_note.get("hints") or [])[: max(1, max_hints)],
                "cards": (latest_note.get("cards") or [])[: max(1, max_hints)],
            },
        }
        await self._log_evidence_cards(agent_id, payload["evidence"], source="share_evidence")
        content = json.dumps(payload, ensure_ascii=False)
        send_result = await self.controller.run_action(
            "communication",
            "send_message",
            from_id=agent_id,
            to_id=recipient,
            content=content,
        )
        ok = isinstance(send_result, ActionResult) and send_result.is_successful()
        reward = 0.02 if ok else 0.0
        ar = ActionResult.success(
            method_name="share_evidence",
            message="Evidence shared." if ok else "Evidence share attempted.",
            data={"to_agent_id": recipient, "shared_type": "note", "reward": reward},
        )
        await self._append_trace(agent_id, "share_evidence", reward, ar.data or {})
        return ar

    @AgentCall
    async def share_observation(
        self,
        agent_id: str,
        to_agent_id: Optional[str] = None,
    ) -> ActionResult:
        observations = await self._get_state(agent_id, "observations") or []
        if not observations:
            ar = ActionResult.success(
                method_name="share_observation",
                message="No local observations to share.",
                data={"reward": 0.0},
            )
            await self._append_trace(agent_id, "share_observation", 0.0, ar.data or {})
            return ar

        recipient = self._pick_recipient(agent_id, to_agent_id)
        if not recipient:
            ar = ActionResult.success(
                method_name="share_observation",
                message="No recipient available.",
                data={"reward": 0.0},
            )
            await self._append_trace(agent_id, "share_observation", 0.0, ar.data or {})
            return ar

        latest_obs = observations[-1]
        payload = {
            "type": "observation_share",
            "from_agent": agent_id,
            "observation": latest_obs,
        }
        content = json.dumps(payload, ensure_ascii=False)
        send_result = await self.controller.run_action(
            "communication",
            "send_message",
            from_id=agent_id,
            to_id=recipient,
            content=content,
        )
        ok = isinstance(send_result, ActionResult) and send_result.is_successful()
        reward = 0.02 if ok else 0.0
        ar = ActionResult.success(
            method_name="share_observation",
            message="Observation shared." if ok else "Observation share attempted.",
            data={"to_agent_id": recipient, "shared_type": "observation", "reward": reward},
        )
        await self._append_trace(agent_id, "share_observation", reward, ar.data or {})
        return ar
