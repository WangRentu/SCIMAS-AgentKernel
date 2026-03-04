import json
from typing import Any, Dict, List, Optional

from agentkernel_standalone.toolkit.logger import get_logger

logger = get_logger(__name__)


class ContributionService:
    """Paper payload and credit-allocation helpers."""

    def __init__(self, plugin: Any):
        self.plugin = plugin

    def build_citation_owner_map(
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

    async def grant_credits(
        self,
        credit_by_agent: Dict[str, float],
        source: str,
        reference_id: Optional[str] = None,
    ) -> None:
        for recipient, credit in credit_by_agent.items():
            if not recipient or credit <= 0:
                continue
            try:
                await self.plugin._inc_state_number(recipient, "credit_buffer", float(credit))
                await self.plugin._inc_state_number(recipient, "contribution_credit_total", float(credit))
                await self.plugin._set_state(
                    recipient,
                    "last_credit",
                    {"source": source, "value": float(credit), "reference_id": reference_id},
                )
            except Exception as e:  # pragma: no cover
                logger.warning(f"Credit assignment failed for {recipient}: {e}")

    def compute_contribution_credit(
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
            run_id = (obs or {}).get("run_id")
            if owner and run_id:
                ref_key = f"RUN@{run_id}"
                if ref_key in obs_refs and owner != agent_id:
                    credit_by_agent[owner] = credit_by_agent.get(owner, 0.0) + 0.01

        if bool(metrics.get("replication_ok", False)):
            for cid in cited:
                owner = citation_owner_map.get(cid)
                if owner and owner != agent_id:
                    credit_by_agent[owner] = credit_by_agent.get(owner, 0.0) + 0.03
        return credit_by_agent

    def build_paper_payload(
        self,
        *,
        task_name: str,
        metric_name: str,
        best_run: Dict[str, Any],
        notes: List[Dict[str, Any]],
        observations: List[Dict[str, Any]],
        hypothesis: List[str],
        plan_spec: Dict[str, Any],
        exp_count: int,
        llm_write_spec: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        citations: List[str] = []
        for note in notes or []:
            for card in (note or {}).get("cards", []) or []:
                cid = card.get("citation_id")
                if cid and cid not in citations:
                    citations.append(cid)

        observation_refs: List[str] = []
        for obs in observations or []:
            run_id = obs.get("run_id")
            if run_id:
                observation_refs.append(f"RUN@{run_id}")

        evidence_map = {
            "claimed_result": citations[: min(8, len(citations))],
            "best_run": [f"RUN@{best_run.get('run_id')}"] if best_run.get("run_id") else [],
        }

        method_summary = {
            "hypothesis_tags": hypothesis,
            "strategy": (plan_spec or {}).get("strategy", "default_baseline"),
            "rationale": (plan_spec or {}).get("rationale", []),
            "risk": (plan_spec or {}).get("risk", []),
        }
        llm_write_spec = llm_write_spec or {}
        llm_evidence_map = llm_write_spec.get("evidence_map")
        if isinstance(llm_evidence_map, dict) and llm_evidence_map:
            for key, refs in llm_evidence_map.items():
                if not isinstance(refs, list):
                    continue
                safe_refs = [str(ref) for ref in refs[:10] if isinstance(ref, (str, int, float))]
                if safe_refs:
                    evidence_map[str(key)] = safe_refs

        return {
            "task_name": task_name,
            "title": (
                self.plugin._truncate(llm_write_spec.get("title"), 220)
                if llm_write_spec.get("title")
                else f"{task_name} iterative study"
            ),
            "abstract": self.plugin._truncate(llm_write_spec.get("abstract"), 1200),
            "claimed_results": {
                "metric_name": metric_name,
                "raw_score": float(best_run.get("raw_score", 0.0) or 0.0),
                "score_norm": float(best_run.get("score_norm", 0.0) or 0.0),
                "dev_score": best_run.get("dev_score"),
                "dev_score_norm": best_run.get("dev_score_norm"),
                "run_id": best_run.get("run_id"),
            },
            "artifacts": {
                "submission_path": best_run.get("submission_path"),
                "run_id": best_run.get("run_id"),
            },
            "citations": citations,
            "evidence_map": evidence_map,
            "observation_refs": observation_refs,
            "observation_evidence_map": {"claimed_result": observation_refs[:3]},
            "key_claims": self.plugin._safe_text_list(llm_write_spec.get("key_claims"), limit=6, item_limit=320),
            "limitations": self.plugin._safe_text_list(llm_write_spec.get("limitations"), limit=6, item_limit=260),
            "replication_checklist": self.plugin._safe_text_list(
                llm_write_spec.get("replication_checklist"),
                limit=8,
                item_limit=220,
            ),
            "next_experiments": self.plugin._safe_text_list(llm_write_spec.get("next_experiments"), limit=6, item_limit=220),
            "method_section": self.plugin._truncate(
                llm_write_spec.get("method_section") or json.dumps(method_summary, ensure_ascii=False),
                4000,
            ),
            "exp_count": int(exp_count or 0),
        }
