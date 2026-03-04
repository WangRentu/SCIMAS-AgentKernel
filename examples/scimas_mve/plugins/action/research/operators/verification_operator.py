from typing import Any, Dict, Optional

from agentkernel_standalone.types.schemas.action import ActionResult


class VerificationOperator:
    """Encapsulates verify_strength / verify_issue execution paths."""

    def __init__(self, plugin: Any):
        self.plugin = plugin

    async def execute_strength(
        self,
        agent_id: str,
        paper_id: Optional[str] = None,
        reviewer_id: Optional[str] = None,
        strength: Optional[Dict[str, Any]] = None,
        test: Optional[Dict[str, Any]] = None,
    ) -> ActionResult:
        if not paper_id:
            ar = self.plugin._action_error(
                "verify_strength",
                "paper_id is required for strength verification.",
                effective_action="verify_strength",
                detail={"precondition_failed": True, "reason": "paper_id_required"},
            )
            await self.plugin._append_trace(agent_id, "verify_strength", 0.0, ar.data or {})
            return ar

        paper = await self.plugin.controller.run_environment("science", "get_paper", paper_id=paper_id)
        if not isinstance(paper, dict):
            ar = self.plugin._action_error(
                "verify_strength",
                f"Paper {paper_id} not found.",
                effective_action="verify_strength",
                detail={"precondition_failed": True, "reason": "paper_not_found", "paper_id": paper_id},
            )
            await self.plugin._append_trace(agent_id, "verify_strength", 0.0, ar.data or {})
            return ar

        metrics = await self.plugin.controller.run_environment("science", "evaluate_paper", paper=paper, paper_id=paper_id)
        score_norm = float(metrics.get("score_norm", 0.0) or 0.0)
        replication_ok = bool(metrics.get("replication_ok", False))
        test_kind = str((test or {}).get("kind") or "replicate").strip().lower()
        passed = False
        evidence = [f"score_norm={score_norm:.4f}", f"replication_ok={replication_ok}"]
        replication_submit = None

        if test_kind == "replicate":
            replication_submit = await self.plugin.controller.run_environment(
                "science",
                "submit_replication",
                paper_id=paper_id,
                agent_id=agent_id,
                replication={"mode": "score_consistency", "source": "verify_strength"},
                source="verify_strength",
            )
            support = (replication_submit or {}).get("support") or {}
            support_ratio = float(support.get("support_ratio", 0.0) or 0.0)
            evidence.append(f"support_ratio={support_ratio:.4f}")
            passed = bool((replication_submit or {}).get("ok")) and support_ratio >= 0.5
        elif test_kind == "static_check":
            passed = bool(replication_ok and score_norm >= 0.3)
        else:
            passed = bool(score_norm >= 0.5)

        reward = 0.03 if passed else -0.015
        if reviewer_id and str(reviewer_id) != str(agent_id):
            reviewer_delta = 0.02 if passed else -0.02
            await self.plugin._inc_state_number(str(reviewer_id), "credit_buffer", reviewer_delta)
            await self.plugin._inc_state_number(str(reviewer_id), "contribution_credit_total", reviewer_delta)
            await self.plugin._set_state(
                str(reviewer_id),
                "last_credit",
                {"source": "verify_strength", "value": float(reviewer_delta), "reference_id": paper_id},
            )

        verification_result = {
            "paper_id": paper_id,
            "reviewer_id": reviewer_id,
            "strength_id": (strength or {}).get("id"),
            "test_kind": test_kind,
            "passed": passed,
            "evidence": evidence,
            "replication_submit": replication_submit,
        }
        ar = ActionResult.success(
            method_name="verify_strength",
            message="Strength verification executed.",
            data={
                "verification_result": verification_result,
                "reward": reward,
                "effective_action": "verify_strength",
                "reward_components": {
                    "verify_strength_reward": float(reward),
                    "learning_reward": float(reward),
                    "verify_strength_pass": float(1.0 if passed else 0.0),
                },
            },
        )
        await self.plugin._append_trace(agent_id, "verify_strength", reward, ar.data or {})
        return ar

    async def execute_issue(
        self,
        agent_id: str,
        paper_id: Optional[str] = None,
        reviewer_id: Optional[str] = None,
        issue: Optional[Dict[str, Any]] = None,
        test: Optional[Dict[str, Any]] = None,
    ) -> ActionResult:
        if not paper_id:
            ar = self.plugin._action_error(
                "verify_issue",
                "paper_id is required for issue verification.",
                effective_action="verify_issue",
                detail={"precondition_failed": True, "reason": "paper_id_required"},
            )
            await self.plugin._append_trace(agent_id, "verify_issue", 0.0, ar.data or {})
            return ar

        paper = await self.plugin.controller.run_environment("science", "get_paper", paper_id=paper_id)
        if not isinstance(paper, dict):
            ar = self.plugin._action_error(
                "verify_issue",
                f"Paper {paper_id} not found.",
                effective_action="verify_issue",
                detail={"precondition_failed": True, "reason": "paper_not_found", "paper_id": paper_id},
            )
            await self.plugin._append_trace(agent_id, "verify_issue", 0.0, ar.data or {})
            return ar

        metrics = await self.plugin.controller.run_environment("science", "evaluate_paper", paper=paper, paper_id=paper_id)
        test_kind = str((test or {}).get("kind") or "replicate").strip().lower()
        issue_type = str((issue or {}).get("type") or "")
        severity = self.plugin._clamp01((issue or {}).get("severity", 0.6))
        issue_validated = False
        evidence = []
        replication_submit = None

        if test_kind == "replicate":
            replication_submit = await self.plugin.controller.run_environment(
                "science",
                "submit_replication",
                paper_id=paper_id,
                agent_id=agent_id,
                replication={"mode": "score_consistency", "source": "verify_issue"},
                source="verify_issue",
            )
            support = (replication_submit or {}).get("support") or {}
            support_ratio = float(support.get("support_ratio", 0.0) or 0.0)
            issue_validated = bool((replication_submit or {}).get("ok")) and support_ratio < 0.5
            evidence.append(f"support_ratio={support_ratio:.4f}")
        elif test_kind == "static_check":
            evidence_score = float(metrics.get("evidence_score", 0.0) or 0.0)
            issue_validated = evidence_score < 0.45
            evidence.append(f"evidence_score={evidence_score:.4f}")
        elif test_kind == "ablation":
            score_norm = float(metrics.get("score_norm", 0.0) or 0.0)
            issue_validated = score_norm < 0.5
            evidence.append(f"score_norm={score_norm:.4f}")
        else:
            issue_validated = not bool(metrics.get("replication_ok", False))
            evidence.append(f"replication_ok={bool(metrics.get('replication_ok', False))}")

        reward = (0.04 if issue_validated else -0.02) * (0.7 + 0.3 * severity)
        if reviewer_id and str(reviewer_id) != str(agent_id):
            reviewer_delta = 0.025 if issue_validated else -0.025
            await self.plugin._inc_state_number(str(reviewer_id), "credit_buffer", reviewer_delta)
            await self.plugin._inc_state_number(str(reviewer_id), "contribution_credit_total", reviewer_delta)
            await self.plugin._set_state(
                str(reviewer_id),
                "last_credit",
                {"source": "verify_issue", "value": float(reviewer_delta), "reference_id": paper_id},
            )

        verification_result = {
            "paper_id": paper_id,
            "reviewer_id": reviewer_id,
            "issue_id": (issue or {}).get("id"),
            "issue_type": issue_type,
            "test_kind": test_kind,
            "validated": issue_validated,
            "evidence": evidence,
            "replication_submit": replication_submit,
        }
        ar = ActionResult.success(
            method_name="verify_issue",
            message="Issue verification executed.",
            data={
                "verification_result": verification_result,
                "reward": reward,
                "effective_action": "verify_issue",
                "reward_components": {
                    "verify_issue_reward": float(reward),
                    "learning_reward": float(reward),
                    "verify_issue_validated": float(1.0 if issue_validated else 0.0),
                },
            },
        )
        await self.plugin._append_trace(agent_id, "verify_issue", reward, ar.data or {})
        return ar
