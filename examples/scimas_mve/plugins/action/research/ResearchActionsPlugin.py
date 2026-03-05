import asyncio
import hashlib
import inspect
import json
import os
import re
from datetime import datetime
from typing import Any, Dict, List, Optional

from agentkernel_standalone.mas.action.base.plugin_base import OtherActionsPlugin
from agentkernel_standalone.toolkit.logger import get_logger
from agentkernel_standalone.toolkit.utils.annotation import AgentCall, ServiceCall
from agentkernel_standalone.types.schemas.action import ActionResult

try:
    from .models import OperatorInput, OperatorOutput, FailureDiagnosis, RecoveryDecision
    from .operators import (
        CollaborationOperator,
        DataOpsOperator,
        ExperimentOperator,
        HypothesizeOperator,
        ReplicationOperator,
        RetrieveOperator,
        ReviewOperator,
        TaskboardOperator,
        VerificationOperator,
        WriteOperator,
    )
    from .services import (
        AuditService,
        ConfigService,
        ContextService,
        ContributionService,
        DiagnosisService,
        LlmService,
        PlanningService,
        PromptService,
        RagService,
        RecoveryService,
        ReviewQualityService,
        RewardService,
        UtilityService,
        VDHService,
    )
except Exception:  # pragma: no cover
    OperatorInput = None  # type: ignore
    OperatorOutput = None  # type: ignore
    FailureDiagnosis = None  # type: ignore
    RecoveryDecision = None  # type: ignore
    ExperimentOperator = None  # type: ignore
    CollaborationOperator = None  # type: ignore
    DataOpsOperator = None  # type: ignore
    HypothesizeOperator = None  # type: ignore
    ReplicationOperator = None  # type: ignore
    RetrieveOperator = None  # type: ignore
    ReviewOperator = None  # type: ignore
    TaskboardOperator = None  # type: ignore
    VerificationOperator = None  # type: ignore
    WriteOperator = None  # type: ignore
    AuditService = None  # type: ignore
    ConfigService = None  # type: ignore
    ContextService = None  # type: ignore
    ContributionService = None  # type: ignore
    DiagnosisService = None  # type: ignore
    LlmService = None  # type: ignore
    PlanningService = None  # type: ignore
    PromptService = None  # type: ignore
    RagService = None  # type: ignore
    RecoveryService = None  # type: ignore
    ReviewQualityService = None  # type: ignore
    RewardService = None  # type: ignore
    UtilityService = None  # type: ignore
    VDHService = None  # type: ignore

logger = get_logger(__name__)


class ResearchActionsPlugin(OtherActionsPlugin):
    """AIRS-benchmark research action plugin.

    Keeps the same action surface as previous MVE implementation while mapping
    action semantics to AIRS workflow:
      read -> task cards
      hypothesize -> method plan
      experiment -> baseline submission + evaluation run
      write -> structured paper artifact
      review/replicate -> robustness and quality check
    """

    def __init__(self):
        super().__init__()
        if ConfigService is None:
            raise RuntimeError("ConfigService unavailable")
        ConfigService(self).apply()

    async def init(self, model_router=None, controller=None):
        self.model = model_router
        self.controller = controller
        log_dirs = [
            os.path.dirname(self._trace_path),
            os.path.dirname(self._code_loop_log_path),
            os.path.dirname(self._code_diagnosis_log_path),
            os.path.dirname(self._precondition_gate_log_path),
            os.path.dirname(self._retrieve_pipeline_log_path),
            os.path.dirname(self._retrieve_guardrail_log_path),
            os.path.dirname(self._retrieve_evidence_log_path),
            os.path.dirname(self._vdh_gate_log_path),
            os.path.dirname(self._review_gate_log_path),
            os.path.dirname(self._rag_index_log_path),
            os.path.dirname(self._rag_query_log_path),
            os.path.dirname(self._rag_usage_log_path),
            os.path.dirname(self._rag_health_log_path),
            os.path.dirname(self._rag_alert_log_path),
            self._research_log_dir,
            self._llm_log_dir,
            self._audit_log_dir,
        ]
        for path in log_dirs:
            os.makedirs(path, exist_ok=True)
        self._init_operator_modules()
        self._init_rag_clients()
        try:
            world_spec = await self.controller.run_environment("science", "get_world_spec")
        except Exception:
            world_spec = {}
        await self._rag_startup_health_check(world_spec=world_spec)
        if self._audit_io_enable:
            try:
                tick = await self.controller.run_system("timer", "get_tick")
            except Exception:
                tick = None
            ts_info = self._audit_timestamp_fields()
            session_record = {
                "meta": {
                    "ts": ts_info["ts"],
                    "tick": tick,
                    "episode_id": (world_spec or {}).get("episode_id"),
                    "task_name": (world_spec or {}).get("task_name"),
                    "kind": "session_start",
                },
                "input": {
                    "llm_enabled": bool(self._llm_enabled),
                    "rag_enabled": bool(self._rag_enable),
                    "llm_actions": dict(self._llm_actions),
                },
                "output": {
                    "status": "ready",
                    "audit_markdown": bool(self._audit_markdown_enable),
                    "llm_audit_jsonl": self._llm_audit_jsonl_path,
                    "rag_audit_jsonl": self._rag_audit_jsonl_path,
                },
            }
            await self._append_jsonl(self._llm_audit_jsonl_path, session_record)
            await self._append_jsonl(self._rag_audit_jsonl_path, session_record)

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

    async def _inc_state_number(self, agent_id: str, key: str, delta: float = 1.0) -> float:
        current = await self._get_state(agent_id, key)
        current_val = float(current or 0.0)
        updated = current_val + float(delta)
        if isinstance(current, int) or float(delta).is_integer():
            if abs(updated - round(updated)) < 1e-9:
                await self._set_state(agent_id, key, int(round(updated)))
                return float(int(round(updated)))
        await self._set_state(agent_id, key, updated)
        return updated

    async def _append_jsonl(self, path: str, record: Dict[str, Any]) -> None:
        try:
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.error(f"Failed to write jsonl {path}: {e}")

    def _clip_text_for_audit(self, text: Any, limit: int) -> Dict[str, Any]:
        value = str(text or "")
        truncated = len(value) > limit
        return {
            "text": value[:limit] if truncated else value,
            "chars": len(value),
            "truncated": bool(truncated),
        }

    def _safe_jsonable(self, value: Any) -> Any:
        try:
            return json.loads(json.dumps(value, ensure_ascii=False, default=str))
        except Exception:
            return str(value)

    def _audit_timestamp_fields(self) -> Dict[str, str]:
        # Keep audit timestamps aligned with terminal log style.
        return {"ts": datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S")}

    async def _append_markdown_audit(
        self,
        *,
        path: str,
        title: str,
        meta: Dict[str, Any],
        request_payload: Dict[str, Any],
        response_payload: Dict[str, Any],
    ) -> None:
        if not self._audit_markdown_enable:
            return
        try:
            lines: List[str] = []
            lines.append(f"## {title}")
            lines.append("")
            lines.append("### Meta")
            lines.append("```json")
            lines.append(json.dumps(self._safe_jsonable(meta), ensure_ascii=False, indent=2))
            lines.append("```")
            lines.append("")
            lines.append("### Input")
            lines.append("```json")
            lines.append(json.dumps(self._safe_jsonable(request_payload), ensure_ascii=False, indent=2))
            lines.append("```")
            lines.append("")
            lines.append("### Output")
            lines.append("```json")
            lines.append(json.dumps(self._safe_jsonable(response_payload), ensure_ascii=False, indent=2))
            lines.append("```")
            lines.append("")
            with open(path, "a", encoding="utf-8") as f:
                f.write("\n".join(lines))
        except Exception as e:
            logger.error(f"Failed to write markdown audit {path}: {e}")

    async def _append_trace(self, agent_id: str, action: str, reward: float, detail: Dict[str, Any]):
        audit_service = self._get_service("_audit_service", AuditService)
        if audit_service is not None:
            return await audit_service.append_trace(agent_id=agent_id, action=action, reward=reward, data=detail or {})

    async def _log_precondition_gate(
        self,
        *,
        agent_id: str,
        action: str,
        phase: str,
        failures: List[str],
        summary: Optional[Dict[str, Any]] = None,
    ) -> None:
        audit_service = self._get_service("_audit_service", AuditService)
        if audit_service is not None:
            return await audit_service.log_precondition_gate(
                agent_id=agent_id,
                action=action,
                phase=phase,
                failures=failures,
                summary=summary,
            )

    async def _log_vdh_gate(
        self,
        *,
        agent_id: str,
        vdh_report: Dict[str, Any],
        reward_components: Optional[Dict[str, Any]] = None,
    ) -> None:
        audit_service = self._get_service("_audit_service", AuditService)
        if audit_service is not None:
            return await audit_service.log_vdh_gate(
                agent_id=agent_id,
                vdh_report=vdh_report,
                reward_components=reward_components,
            )

    async def _log_review_gate(
        self,
        *,
        agent_id: str,
        paper_id: Optional[str],
        run_id: Optional[str],
        gate: Dict[str, Any],
    ) -> None:
        audit_service = self._get_service("_audit_service", AuditService)
        if audit_service is not None:
            return await audit_service.log_review_gate(
                agent_id=agent_id,
                paper_id=paper_id,
                run_id=run_id,
                gate=gate,
            )

    async def _log_code_loop(
        self,
        *,
        agent_id: str,
        attempts: List[Dict[str, Any]],
        best_dev_score_norm: Optional[float],
    ) -> None:
        audit_service = self._get_service("_audit_service", AuditService)
        if audit_service is not None:
            return await audit_service.log_code_loop(
                agent_id=agent_id,
                attempts=attempts,
                best_dev_score_norm=best_dev_score_norm,
            )

    async def _log_code_diagnosis(
        self,
        *,
        agent_id: str,
        phase: str,
        run_id: Optional[str],
        diagnosis: Dict[str, Any],
        template_fix: Dict[str, Any],
        decision: str,
        score_norm: Optional[float] = None,
        dev_score_norm: Optional[float] = None,
        code_agent_ok: Optional[bool] = None,
        fallback_solver_used: Optional[bool] = None,
        execution_path: Optional[str] = None,
    ) -> None:
        audit_service = self._get_service("_audit_service", AuditService)
        if audit_service is not None:
            return await audit_service.log_code_diagnosis(
                agent_id=agent_id,
                phase=phase,
                run_id=run_id,
                diagnosis=diagnosis,
                template_fix=template_fix,
                decision=decision,
                score_norm=score_norm,
                dev_score_norm=dev_score_norm,
                code_agent_ok=code_agent_ok,
                fallback_solver_used=fallback_solver_used,
                execution_path=execution_path,
            )

    def _action_error(
        self,
        method_name: str,
        message: str,
        *,
        effective_action: Optional[str] = None,
        detail: Optional[Dict[str, Any]] = None,
    ) -> ActionResult:
        data = {
            "reward": 0.0,
            "effective_action": effective_action or method_name,
            "reward_components": {"learning_reward": 0.0},
        }
        if detail:
            data.update(detail)
        return ActionResult.error(method_name=method_name, message=message, data=data)

    async def _operator_action_or_error(
        self,
        *,
        action_key: str,
        operator_attr: str,
        operator_method: str,
        method_name: str,
        error_message: str,
        reason: str,
        trace_action: Optional[str] = None,
        **kwargs,
    ) -> ActionResult:
        if self._operator_enabled(action_key):
            operator = getattr(self, operator_attr, None)
            handler = getattr(operator, operator_method, None) if operator is not None else None
            if handler is not None:
                return await handler(**kwargs)
        ar = self._action_error(
            method_name,
            error_message,
            effective_action=action_key,
            detail={"precondition_failed": True, "reason": reason},
        )
        agent_id = str(kwargs.get("agent_id") or "")
        await self._append_trace(agent_id, trace_action or action_key, 0.0, ar.data or {})
        return ar

    def _get_service(self, attr_name: str, service_cls: Any) -> Any:
        service = getattr(self, attr_name, None)
        if service is None and service_cls is not None:
            service = service_cls(self)
            setattr(self, attr_name, service)
        return service

    def _service_sync(
        self,
        attr_name: str,
        service_cls: Any,
        method_name: str,
        *,
        default: Any,
        **kwargs: Any,
    ) -> Any:
        service = self._get_service(attr_name, service_cls)
        if service is None or not hasattr(service, method_name):
            return default
        return getattr(service, method_name)(**kwargs)

    async def _service_async(
        self,
        attr_name: str,
        service_cls: Any,
        method_name: str,
        *,
        default: Any,
        **kwargs: Any,
    ) -> Any:
        service = self._get_service(attr_name, service_cls)
        if service is None or not hasattr(service, method_name):
            return default
        result = getattr(service, method_name)(**kwargs)
        if inspect.isawaitable(result):
            return await result
        return result

    def _llm_ready(self, action_name: str) -> bool:
        if not self._llm_enabled:
            return False
        if not self._llm_actions.get(action_name, False):
            return False
        return self.model is not None and hasattr(self.model, "chat")

    def _extract_json_candidate(self, text: str) -> str:
        llm_service = self._get_service("_llm_service", LlmService)
        if llm_service is not None:
            return llm_service.extract_json_candidate(text)
        return str(text or "").strip()

    def _safe_json_loads(self, text: str) -> Any:
        llm_service = self._get_service("_llm_service", LlmService)
        if llm_service is not None:
            return llm_service.safe_json_loads(text)
        candidate = self._extract_json_candidate(text)
        return json.loads(candidate)

    async def _log_llm_call(self, record: Dict[str, Any]) -> None:
        llm_service = self._get_service("_llm_service", LlmService)
        if llm_service is not None:
            await llm_service.log_llm_call(record)
            return
        if self._llm_log_enabled:
            await self._append_jsonl(self._llm_log_path, record)

    async def _log_llm_audit(
        self,
        *,
        ts: str,
        tick: int,
        world_spec: Dict[str, Any],
        agent_id: str,
        action_name: str,
        prompt: str,
        raw_response: Any,
        parsed_data: Any,
        ok: bool,
        reason: str = "",
    ) -> None:
        llm_service = self._get_service("_llm_service", LlmService)
        if llm_service is not None:
            await llm_service.log_llm_audit(
                ts=ts,
                tick=tick,
                world_spec=world_spec,
                agent_id=agent_id,
                action_name=action_name,
                prompt=prompt,
                raw_response=raw_response,
                parsed_data=parsed_data,
                ok=ok,
                reason=reason,
            )
            return

    async def _call_llm_json(self, *, agent_id: str, action_name: str, prompt: str) -> Dict[str, Any]:
        llm_service = self._get_service("_llm_service", LlmService)
        if llm_service is not None:
            return await llm_service.call_llm_json(agent_id=agent_id, action_name=action_name, prompt=prompt)
        return {"ok": False, "reason": "llm_service_unavailable"}

    async def _log_evidence_cards(self, agent_id: str, literature: Dict[str, Any], source: str = "read") -> None:
        audit_service = self._get_service("_audit_service", AuditService)
        if audit_service is not None:
            return await audit_service.log_evidence_cards(agent_id=agent_id, literature=literature, source=source)

    async def _log_paper_result(
        self,
        agent_id: str,
        paper_id: Optional[str],
        paper: Dict[str, Any],
        metrics: Dict[str, Any],
        source: str,
    ) -> None:
        audit_service = self._get_service("_audit_service", AuditService)
        if audit_service is not None:
            return await audit_service.log_paper_result(
                agent_id=agent_id,
                paper_id=paper_id,
                paper=paper,
                metrics=metrics,
                source=source,
            )

    def _init_operator_modules(self) -> None:
        all_attrs = [
            "_diagnosis_service",
            "_llm_service",
            "_reward_service",
            "_audit_service",
            "_context_service",
            "_contribution_service",
            "_planning_service",
            "_prompt_service",
            "_rag_service",
            "_recovery_service",
            "_review_quality_service",
            "_utility_service",
            "_vdh_service",
            "_experiment_operator",
            "_hypothesize_operator",
            "_review_operator",
            "_verification_operator",
            "_replication_operator",
            "_write_operator",
            "_retrieve_operator",
            "_taskboard_operator",
            "_collaboration_operator",
            "_data_ops_operator",
        ]
        for attr in all_attrs:
            setattr(self, attr, None)

        service_specs = [
            ("_diagnosis_service", DiagnosisService, True),
            ("_llm_service", LlmService, True),
            ("_reward_service", RewardService, False),
            ("_audit_service", AuditService, True),
            ("_context_service", ContextService, True),
            ("_contribution_service", ContributionService, True),
            ("_planning_service", PlanningService, True),
            ("_prompt_service", PromptService, True),
            ("_rag_service", RagService, True),
            ("_recovery_service", RecoveryService, True),
            ("_review_quality_service", ReviewQualityService, True),
            ("_utility_service", UtilityService, True),
            ("_vdh_service", VDHService, True),
        ]
        for attr, cls, with_plugin in service_specs:
            if cls is None:
                continue
            setattr(self, attr, cls(self) if with_plugin else cls())

        operator_specs = [
            ("_experiment_operator", ExperimentOperator),
            ("_hypothesize_operator", HypothesizeOperator),
            ("_review_operator", ReviewOperator),
            ("_verification_operator", VerificationOperator),
            ("_replication_operator", ReplicationOperator),
            ("_write_operator", WriteOperator),
            ("_retrieve_operator", RetrieveOperator),
            ("_taskboard_operator", TaskboardOperator),
            ("_collaboration_operator", CollaborationOperator),
            ("_data_ops_operator", DataOpsOperator),
        ]
        for attr, cls in operator_specs:
            if cls is not None:
                setattr(self, attr, cls(self))

    def _operator_enabled(self, action: str) -> bool:
        action_key = str(action or "").strip().lower()
        if action_key == "experiment":
            return bool(self._operator_experiment_v2 and self._experiment_operator is not None)
        if action_key == "hypothesize":
            return bool(self._operator_hypothesize_v2 and self._hypothesize_operator is not None)
        if action_key == "review":
            return bool(self._operator_review_v2 and self._review_operator is not None)
        if action_key in {"verify_strength", "verify_issue"}:
            return bool(self._operator_verify_v2 and self._verification_operator is not None)
        if action_key == "replicate":
            return bool(self._operator_replicate_v2 and self._replication_operator is not None)
        if action_key == "write":
            return bool(self._operator_write_v2 and self._write_operator is not None)
        if action_key == "retrieve_literature":
            return bool(self._operator_retrieve_v2 and self._retrieve_operator is not None)
        if action_key in {"claim_task", "complete_task"}:
            return bool(self._operator_taskboard_v2 and self._taskboard_operator is not None)
        if action_key in {"share_evidence", "share_observation"}:
            return bool(self._operator_collaboration_v2 and self._collaboration_operator is not None)
        if action_key in {"read", "profile_data", "prepare_data"}:
            return bool(self._operator_dataops_v2 and self._data_ops_operator is not None)
        return False

    async def _load_research_context(self, agent_id: str, include_shared: bool = True):
        context = await self._service_async(
            "_context_service",
            ContextService,
            "load_context",
            default=None,
            agent_id=agent_id,
            include_shared=include_shared,
        )
        if context is not None:
            return context
        return {
            "agent_id": agent_id,
            "world_spec": {},
            "notes": [],
            "observations": [],
            "local_notes": [],
            "shared_notes": [],
            "local_observations": [],
            "shared_observations": [],
            "hypothesis": [],
            "data_card": None,
            "method_card": None,
            "plan_spec": {},
        }

    def _init_rag_clients(self) -> None:
        # RagService.init_rag_clients mutates self._rag_store/_rag_retriever in-place.
        # Do not reset to None after invocation, otherwise RAG stays permanently disabled.
        self._service_sync("_rag_service", RagService, "init_rag_clients", default=None)

    async def _rag_startup_health_check(self, *, world_spec: Dict[str, Any]) -> None:
        await self._service_async(
            "_rag_service",
            RagService,
            "rag_startup_health_check",
            default=None,
            world_spec=world_spec,
        )

    async def _rag_track_runtime_health(
        self,
        *,
        world_spec: Dict[str, Any],
        agent_id: str,
        action: str,
        status: str,
        source: str,
    ) -> None:
        await self._service_async(
            "_rag_service",
            RagService,
            "rag_track_runtime_health",
            default=None,
            world_spec=world_spec,
            agent_id=agent_id,
            action=action,
            status=status,
            source=source,
        )

    async def _rag_retrieve_recovery_paused(self) -> bool:
        result = await self._service_async(
            "_rag_service",
            RagService,
            "rag_retrieve_recovery_paused",
            default=None,
        )
        if result is not None:
            return bool(result)
        until = int(self._rag_degraded_pause_until_tick or 0)
        return until > 0

    def _rag_hash(self, text: str) -> str:
        result = self._service_sync("_rag_service", RagService, "rag_hash", default=None, text=text)
        if result is not None:
            return str(result)
        return hashlib.sha1((text or "").encode("utf-8")).hexdigest()

    def _rag_doc_base(
        self,
        *,
        world_spec: Dict[str, Any],
        agent_id: str,
        source_type: str,
        source_id: str,
        action: str,
        tags: Optional[List[str]] = None,
        quality: Optional[float] = None,
    ) -> Dict[str, Any]:
        payload = dict(locals())
        payload.pop("self", None)
        return self._service_sync("_rag_service", RagService, "rag_doc_base", default={}, **payload)

    def _rag_docs_from_note(
        self,
        *,
        world_spec: Dict[str, Any],
        agent_id: str,
        note: Dict[str, Any],
        action: str,
    ) -> List[Dict[str, Any]]:
        payload = dict(locals())
        payload.pop("self", None)
        return self._service_sync("_rag_service", RagService, "rag_docs_from_note", default=[], **payload)

    def _rag_docs_from_method_card(
        self,
        *,
        world_spec: Dict[str, Any],
        agent_id: str,
        method_card: Dict[str, Any],
        action: str,
    ) -> List[Dict[str, Any]]:
        payload = dict(locals())
        payload.pop("self", None)
        return self._service_sync("_rag_service", RagService, "rag_docs_from_method_card", default=[], **payload)

    def _rag_docs_from_data_card(
        self,
        *,
        world_spec: Dict[str, Any],
        agent_id: str,
        data_card: Dict[str, Any],
        action: str,
    ) -> List[Dict[str, Any]]:
        payload = dict(locals())
        payload.pop("self", None)
        return self._service_sync("_rag_service", RagService, "rag_docs_from_data_card", default=[], **payload)

    def _rag_docs_from_world_spec(
        self,
        *,
        world_spec: Dict[str, Any],
        agent_id: str,
        action: str,
    ) -> List[Dict[str, Any]]:
        payload = dict(locals())
        payload.pop("self", None)
        return self._service_sync("_rag_service", RagService, "rag_docs_from_world_spec", default=[], **payload)

    def _rag_docs_from_observation(
        self,
        *,
        world_spec: Dict[str, Any],
        agent_id: str,
        observation: Dict[str, Any],
        action: str,
    ) -> List[Dict[str, Any]]:
        payload = dict(locals())
        payload.pop("self", None)
        return self._service_sync("_rag_service", RagService, "rag_docs_from_observation", default=[], **payload)

    def _rag_docs_from_paper(
        self,
        *,
        world_spec: Dict[str, Any],
        agent_id: str,
        paper: Dict[str, Any],
        paper_id: Optional[str],
        action: str,
    ) -> List[Dict[str, Any]]:
        payload = dict(locals())
        payload.pop("self", None)
        return self._service_sync("_rag_service", RagService, "rag_docs_from_paper", default=[], **payload)

    async def _rag_log_query(
        self,
        *,
        agent_id: str,
        action: str,
        run_id: Optional[str],
        paper_id: Optional[str],
        query_text: str,
        topk: int,
        result: Dict[str, Any],
        latency_ms: float,
    ) -> None:
        payload = dict(locals())
        payload.pop("self", None)
        await self._service_async("_rag_service", RagService, "rag_log_query", default=None, **payload)

    async def _log_rag_audit(
        self,
        *,
        world_spec: Dict[str, Any],
        agent_id: str,
        action: str,
        operation: str,
        run_id: Optional[str],
        paper_id: Optional[str],
        request_payload: Dict[str, Any],
        response_payload: Dict[str, Any],
    ) -> None:
        payload = dict(locals())
        payload.pop("self", None)
        await self._service_async("_rag_service", RagService, "log_rag_audit", default=None, **payload)

    async def _rag_log_usage(
        self,
        *,
        agent_id: str,
        action: str,
        run_id: Optional[str],
        paper_id: Optional[str],
        result: Dict[str, Any],
    ) -> None:
        payload = dict(locals())
        payload.pop("self", None)
        await self._service_async("_rag_service", RagService, "rag_log_usage", default=None, **payload)

    async def _rag_index_documents(
        self,
        *,
        agent_id: str,
        action: str,
        docs: List[Dict[str, Any]],
        run_id: Optional[str] = None,
        paper_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        payload = dict(locals())
        payload.pop("self", None)
        return await self._service_async(
            "_rag_service",
            RagService,
            "rag_index_documents",
            default={"ok": False, "status": "disabled", "indexed_points": 0},
            **payload,
        )

    def _rag_local_docs(
        self,
        *,
        world_spec: Dict[str, Any],
        agent_id: str,
        notes: List[Dict[str, Any]],
        observations: List[Dict[str, Any]],
        data_card: Optional[Dict[str, Any]],
        method_card: Optional[Dict[str, Any]],
        paper: Optional[Dict[str, Any]] = None,
        paper_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        payload = dict(locals())
        payload.pop("self", None)
        return self._service_sync("_rag_service", RagService, "rag_local_docs", default=[], **payload)

    async def _rag_retrieve_context(
        self,
        *,
        agent_id: str,
        action: str,
        run_id: Optional[str],
        paper_id: Optional[str],
        query_text: str,
        quotas: Dict[str, int],
        notes: List[Dict[str, Any]],
        observations: List[Dict[str, Any]],
        data_card: Optional[Dict[str, Any]],
        method_card: Optional[Dict[str, Any]],
        paper: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        payload = dict(locals())
        payload.pop("self", None)
        return await self._service_async(
            "_rag_service",
            RagService,
            "rag_retrieve_context",
            default={
                "status": "disabled",
                "fallback_reason": "rag_service_unavailable",
                "all_results": [],
                "selected": [],
                "context": "",
                "refs": [],
            },
            **payload,
        )

    async def _rag_bootstrap_episode_knowledge(
        self,
        *,
        agent_id: str,
        world_spec: Dict[str, Any],
        data_card: Optional[Dict[str, Any]],
        method_card: Optional[Dict[str, Any]],
    ) -> None:
        payload = dict(locals())
        payload.pop("self", None)
        await self._service_async("_rag_service", RagService, "rag_bootstrap_episode_knowledge", default=None, **payload)

    def _format_rag_prompt_block(self, *, result: Dict[str, Any]) -> Dict[str, Any]:
        value = self._service_sync("_rag_service", RagService, "format_rag_prompt_block", default=None, result=result)
        if value is not None:
            return value
        refs = list((result or {}).get("refs") or [])
        context = str((result or {}).get("context") or "").strip()
        if not context:
            context = "(no high-confidence retrieval results)"
        return {
            "context": context,
            "refs": refs,
            "status": str((result or {}).get("status") or "empty"),
            "usage_constraint": "Prioritize retrieved evidence. Do not fabricate citations or run references.",
        }

    def _truncate(self, text: Any, limit: int = 280) -> str:
        utility_service = self._get_service("_utility_service", UtilityService)
        if utility_service is not None:
            return utility_service.truncate(text, limit)
        value = str(text or "")
        if len(value) <= limit:
            return value
        return value[: limit - 3] + "..."

    def _safe_task_types(self, values: Any, *, fallback: Optional[List[str]] = None) -> List[str]:
        utility_service = self._get_service("_utility_service", UtilityService)
        if utility_service is not None:
            return utility_service.safe_task_types(values, fallback=fallback)
        return list(fallback or [])

    def _safe_text_list(self, values: Any, *, limit: int = 5, item_limit: int = 220) -> List[str]:
        utility_service = self._get_service("_utility_service", UtilityService)
        if utility_service is not None:
            return utility_service.safe_text_list(values, limit=limit, item_limit=item_limit)
        return []

    def _text_tokens(self, text: str) -> List[str]:
        utility_service = self._get_service("_utility_service", UtilityService)
        if utility_service is not None:
            return utility_service.text_tokens(text)
        return []

    def _counter_cosine(self, a: Any, b: Any) -> float:
        utility_service = self._get_service("_utility_service", UtilityService)
        if utility_service is not None:
            return float(utility_service.counter_cosine(a, b))
        return 0.0

    def _note_to_text(self, note: Dict[str, Any]) -> str:
        utility_service = self._get_service("_utility_service", UtilityService)
        if utility_service is not None:
            return utility_service.note_to_text(note)
        return ""

    def _has_method_signal(self, text: str) -> bool:
        utility_service = self._get_service("_utility_service", UtilityService)
        if utility_service is not None:
            return bool(utility_service.has_method_signal(text))
        return False

    async def _tei_embed(self, text: str) -> Optional[List[float]]:
        utility_service = self._get_service("_utility_service", UtilityService)
        if utility_service is not None:
            return await utility_service.tei_embed(text)
        return None

    async def _qdrant_max_similarity(self, vector: List[float]) -> Optional[float]:
        utility_service = self._get_service("_utility_service", UtilityService)
        if utility_service is not None:
            return await utility_service.qdrant_max_similarity(vector)
        return None

    async def _compute_read_reward(
        self,
        *,
        existing_notes: List[Dict[str, Any]],
        new_note: Dict[str, Any],
    ) -> Dict[str, Any]:
        utility_service = self._get_service("_utility_service", UtilityService)
        if utility_service is not None:
            return await utility_service.compute_read_reward(existing_notes=existing_notes, new_note=new_note)
        return {
            "reward": 0.0,
            "novelty": 0.0,
            "local_similarity": 0.0,
            "qdrant_similarity": None,
            "method_bonus": 0.0,
            "duplicate": False,
        }

    def _hypothesis_feasibility(self, world_spec: Dict[str, Any], plan_spec: Dict[str, Any]) -> Dict[str, Any]:
        return self._service_sync(
            "_vdh_service",
            VDHService,
            "hypothesis_feasibility",
            default={
            "feasibility_score": 0.0,
            "schema_safe": False,
            "resource_safe": False,
            "schema_bonus": 0.0,
            "resource_bonus": 0.0,
            "reward": 0.0,
            "code_memory_mb": world_spec.get("code_memory_mb"),
            },
            world_spec=world_spec,
            plan_spec=plan_spec,
        )

    async def _vdh_qdrant_schema_constraints(self, world_spec: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        return await self._service_async(
            "_vdh_service",
            VDHService,
            "vdh_qdrant_schema_constraints",
            default=None,
            world_spec=world_spec,
        )

    def _vdh_constraints_from_manifest_file(self, world_spec: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        return self._service_sync(
            "_vdh_service",
            VDHService,
            "vdh_constraints_from_manifest_file",
            default=None,
            world_spec=world_spec,
        )

    def _vdh_normalize_constraints(
        self,
        *,
        source: str,
        raw: Optional[Dict[str, Any]],
        world_spec: Dict[str, Any],
        notes: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        return self._service_sync(
            "_vdh_service",
            VDHService,
            "vdh_normalize_constraints",
            default={"ok": True, "source": source, "constraints": {}, "warnings": []},
            source=source,
            raw=raw,
            world_spec=world_spec,
            notes=notes,
        )

    async def _vdh_metadata_alignment(
        self,
        *,
        world_spec: Dict[str, Any],
        notes: Optional[List[Dict[str, Any]]],
        plan_spec: Dict[str, Any],
    ) -> Dict[str, Any]:
        return await self._service_async(
            "_vdh_service",
            VDHService,
            "vdh_metadata_alignment",
            default={"ok": True, "source": "disabled", "constraints": {}, "warnings": [], "errors": []},
            world_spec=world_spec,
            notes=notes,
            plan_spec=plan_spec,
        )

    def _vdh_plan_validator(
        self,
        *,
        world_spec: Dict[str, Any],
        plan_spec: Dict[str, Any],
        data_card: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        return self._service_sync(
            "_vdh_service",
            VDHService,
            "vdh_plan_validator",
            default={"ok": True, "errors": [], "resource_estimate": {}},
            world_spec=world_spec,
            plan_spec=plan_spec,
            data_card=data_card,
        )

    async def _vdh_embed_text(self, text: str) -> Optional[List[float]]:
        return await self._service_async(
            "_vdh_service",
            VDHService,
            "vdh_embed_text",
            default=None,
            text=text,
        )

    def _vector_cosine(self, a: Optional[List[float]], b: Optional[List[float]]) -> float:
        return float(
            self._service_sync("_vdh_service", VDHService, "vector_cosine", default=0.0, a=a, b=b)
        )

    async def _vdh_evidence_coverage(
        self,
        *,
        hypothesis: List[str],
        plan_spec: Dict[str, Any],
        notes: List[Dict[str, Any]],
        observations: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        return await self._service_async(
            "_vdh_service",
            VDHService,
            "vdh_evidence_coverage",
            default={
                "ok": True,
                "coverage_score": 1.0,
                "threshold": float(self._vdh_evidence_threshold),
                "source": "disabled",
                "token_overlap": 0.0,
                "keyword_coverage": 0.0,
                "vector_similarity": None,
                "errors": [],
            },
            hypothesis=hypothesis,
            plan_spec=plan_spec,
            notes=notes,
            observations=observations,
        )

    async def _evaluate_vdh_gates(
        self,
        *,
        world_spec: Dict[str, Any],
        hypothesis: List[str],
        plan_spec: Dict[str, Any],
        notes: List[Dict[str, Any]],
        observations: List[Dict[str, Any]],
        data_card: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        return await self._service_async(
            "_vdh_service",
            VDHService,
            "evaluate_vdh_gates",
            default={
                "gate_a": {"ok": True},
                "gate_b": {"ok": True},
                "gate_c": {"ok": True},
                "final_ok": True,
                "failures": [],
                "policy": self._vdh_gate_policy,
            },
            world_spec=world_spec,
            hypothesis=hypothesis,
            plan_spec=plan_spec,
            notes=notes,
            observations=observations,
            data_card=data_card,
        )

    async def _enqueue_vdh_recovery_tasks(self, *, vdh_report: Dict[str, Any]) -> Dict[str, Any]:
        return await self._service_async(
            "_vdh_service",
            VDHService,
            "enqueue_vdh_recovery_tasks",
            default={"created_task_ids": [], "requested": [], "skipped": []},
            vdh_report=vdh_report,
        )

    def _experiment_error_flags(self, result: Dict[str, Any]) -> Dict[str, bool]:
        utility_service = self._get_service("_utility_service", UtilityService)
        if utility_service is not None:
            return utility_service.experiment_error_flags(result or {})
        return {"oom": False, "typeerror": False}

    def _is_first_pass_success(self, *, code_attempts: Any, ok: bool) -> bool:
        utility_service = self._get_service("_utility_service", UtilityService)
        if utility_service is not None:
            return bool(utility_service.is_first_pass_success(code_attempts=code_attempts, ok=ok))
        return False

    def _estimate_vram_efficiency(self, *, result: Dict[str, Any], world_spec: Dict[str, Any]) -> Optional[float]:
        utility_service = self._get_service("_utility_service", UtilityService)
        if utility_service is not None:
            return utility_service.estimate_vram_efficiency(result=result or {}, world_spec=world_spec or {})
        return None

    def _compact_data_card(self, data_card: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        planning_service = self._get_service("_planning_service", PlanningService)
        if planning_service is not None:
            return planning_service.compact_data_card(data_card)
        return {}

    def _compact_method_card(self, method_card: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        planning_service = self._get_service("_planning_service", PlanningService)
        if planning_service is not None:
            return planning_service.compact_method_card(method_card)
        return {}

    def _default_solver_plan(self, world_spec: Dict[str, Any]) -> Dict[str, Any]:
        planning_service = self._get_service("_planning_service", PlanningService)
        if planning_service is not None:
            return planning_service.default_solver_plan(world_spec)
        return {
            "strategy": "iterative_solver_baseline",
            "solver_spec": {"model_family": "tfidf_logreg", "seed": 42, "preprocess": {}, "hyperparams": {}, "input_columns": []},
            "rationale": [],
            "risk": [],
            "experiment_protocol": {"primary_knob": "", "ablation_axis": "", "format_checks": []},
            "replication_plan": [],
        }

    def _merge_solver_plan(self, base_plan: Dict[str, Any], candidate: Dict[str, Any]) -> Dict[str, Any]:
        planning_service = self._get_service("_planning_service", PlanningService)
        if planning_service is not None:
            return planning_service.merge_solver_plan(base_plan, candidate)
        return dict(base_plan or {})

    def _derive_next_solver_plan_from_history(
        self,
        plan_spec: Dict[str, Any],
        run_history: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        planning_service = self._get_service("_planning_service", PlanningService)
        if planning_service is not None:
            return planning_service.derive_next_solver_plan_from_history(plan_spec, run_history)
        return self._merge_solver_plan(plan_spec or {}, {})

    def _clamp01(self, value: Any) -> float:
        planning_service = self._get_service("_planning_service", PlanningService)
        if planning_service is not None:
            return planning_service.clamp01(value)
        try:
            parsed = float(value)
        except Exception:
            return 0.0
        return max(0.0, min(1.0, parsed))

    def _extract_evidence_refs(self, text: Any) -> List[str]:
        planning_service = self._get_service("_planning_service", PlanningService)
        if planning_service is not None:
            return planning_service.extract_evidence_refs(text)
        return []

    def _normalize_review_issues(self, review_note: Dict[str, Any]) -> List[Dict[str, Any]]:
        return self._service_sync(
            "_review_quality_service",
            ReviewQualityService,
            "normalize_review_issues",
            default=[],
            review_note=review_note,
        )

    def _heuristic_review_note(self, *, paper: Dict[str, Any], metrics: Dict[str, Any]) -> Dict[str, Any]:
        return self._service_sync(
            "_review_quality_service",
            ReviewQualityService,
            "heuristic_review_note",
            default={"summary": "", "strengths": [], "issues": [], "revision_actions": [], "paper_id": paper.get("paper_id")},
            paper=paper,
            metrics=metrics,
        )

    def _score_review_quality(
        self,
        *,
        review_note: Dict[str, Any],
        issues: List[Dict[str, Any]],
        self_review: bool,
        replication_ok: bool,
    ) -> Dict[str, Any]:
        return self._service_sync(
            "_review_quality_service",
            ReviewQualityService,
            "score_review_quality",
            default={"critique_score": 0.0, "needs_revision": False, "issue_count": len(issues)},
            review_note=review_note,
            issues=issues,
            self_review=self_review,
            replication_ok=replication_ok,
        )

    async def _qdrant_search_similarity(self, *, vector: List[float], collection: str) -> Optional[float]:
        return await self._service_async(
            "_review_quality_service",
            ReviewQualityService,
            "qdrant_search_similarity",
            default=None,
            vector=vector,
            collection=collection,
        )

    async def _qgr_relevance_score(
        self,
        *,
        review_note: Dict[str, Any],
        context_text: str,
    ) -> Dict[str, Any]:
        return await self._service_async(
            "_review_quality_service",
            ReviewQualityService,
            "qgr_relevance_score",
            default={"score": 0.0, "source": "disabled", "token_score": 0.0, "vector_score": None},
            review_note=review_note,
            context_text=context_text,
        )

    async def _qgr_fact_check(self, issues: List[Dict[str, Any]]) -> Dict[str, Any]:
        return await self._service_async(
            "_review_quality_service",
            ReviewQualityService,
            "qgr_fact_check",
            default={"hallucinated_count": 0, "checked": 0, "source": "disabled"},
            issues=issues,
        )

    async def _qgr_validate_review(
        self,
        *,
        review_note: Dict[str, Any],
        issues: List[Dict[str, Any]],
        context_text: str,
        stage: Optional[str] = None,
    ) -> Dict[str, Any]:
        return await self._service_async(
            "_review_quality_service",
            ReviewQualityService,
            "qgr_validate_review",
            default={"valid": True, "thresholds": {}, "metrics": {}, "checks": {}, "fact_check": {}},
            review_note=review_note,
            issues=issues,
            context_text=context_text,
            stage=stage,
        )

    def _qgr_predictive_bonus(
        self,
        *,
        issues: List[Dict[str, Any]],
        run_history: List[Dict[str, Any]],
        target_run_id: Optional[str],
    ) -> Dict[str, Any]:
        return self._service_sync(
            "_review_quality_service",
            ReviewQualityService,
            "qgr_predictive_bonus",
            default={"matched_issue_ids": [], "bonus": 0.0},
            issues=issues,
            run_history=run_history,
            target_run_id=target_run_id,
        )

    async def _spawn_qgr_followup_tasks(
        self,
        *,
        paper_id: Optional[str],
        run_id: Optional[str],
        score: float,
        issues: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        return await self._service_async(
            "_review_quality_service",
            ReviewQualityService,
            "spawn_qgr_followup_tasks",
            default={"created": [], "reason": "review_quality_service_unavailable"},
            paper_id=paper_id,
            run_id=run_id,
            score=score,
            issues=issues,
        )

    async def _spawn_review_validation_tasks(
        self,
        *,
        paper_id: str,
        reviewer_id: str,
        review_note: Dict[str, Any],
        critique_quality: Dict[str, Any],
    ) -> Dict[str, Any]:
        return await self._service_async(
            "_review_quality_service",
            ReviewQualityService,
            "spawn_review_validation_tasks",
            default={"created": [], "write_task_id": None, "skipped_existing": []},
            paper_id=paper_id,
            reviewer_id=reviewer_id,
            review_note=review_note,
            critique_quality=critique_quality,
        )

    def _build_task_role_prompt(
        self,
        *,
        world_spec: Dict[str, Any],
        open_tasks: List[Dict[str, Any]],
        hypothesis: List[str],
        notes_count: int,
        observations_count: int,
    ) -> str:
        payload = dict(locals())
        payload.pop("self", None)
        return self._service_sync(
            "_prompt_service",
            PromptService,
            "build_task_role_prompt",
            default="{}",
            **payload,
        )

    def _build_plan_prompt(
        self,
        world_spec: Dict[str, Any],
        cards: List[Dict[str, Any]],
        recent_runs: List[Dict[str, Any]],
        data_card: Optional[Dict[str, Any]] = None,
        method_card: Optional[Dict[str, Any]] = None,
        rag_context: str = "",
        rag_refs: Optional[List[str]] = None,
        rag_status: str = "",
    ) -> str:
        payload = dict(locals())
        payload.pop("self", None)
        return self._service_sync("_prompt_service", PromptService, "build_plan_prompt", default="{}", **payload)

    def _build_experiment_prompt(
        self,
        *,
        world_spec: Dict[str, Any],
        hypothesis: List[str],
        plan_spec: Dict[str, Any],
        notes: List[Dict[str, Any]],
        observations: List[Dict[str, Any]],
        data_card: Optional[Dict[str, Any]],
        method_card: Optional[Dict[str, Any]],
        exp_count: int,
        budget: int,
        rag_context: str = "",
        rag_refs: Optional[List[str]] = None,
        rag_status: str = "",
    ) -> str:
        payload = dict(locals())
        payload.pop("self", None)
        return self._service_sync("_prompt_service", PromptService, "build_experiment_prompt", default="{}", **payload)

    def _build_retrieve_method_prompt(
        self,
        *,
        world_spec: Dict[str, Any],
        data_card: Optional[Dict[str, Any]],
        notes: List[Dict[str, Any]],
        observations: List[Dict[str, Any]],
        evidence_pack: List[Dict[str, Any]],
        query_bundle: Dict[str, Any],
        rag_status: str = "",
    ) -> str:
        payload = dict(locals())
        payload.pop("self", None)
        return self._service_sync("_prompt_service", PromptService, "build_retrieve_method_prompt", default="{}", **payload)

    def _build_write_prompt(
        self,
        *,
        world_spec: Dict[str, Any],
        best_run: Dict[str, Any],
        hypothesis: List[str],
        plan_spec: Dict[str, Any],
        citations: List[str],
        observation_refs: List[str],
        notes: List[Dict[str, Any]],
        observations: List[Dict[str, Any]],
        rag_context: str = "",
        rag_refs: Optional[List[str]] = None,
        rag_status: str = "",
    ) -> str:
        payload = dict(locals())
        payload.pop("self", None)
        return self._service_sync("_prompt_service", PromptService, "build_write_prompt", default="{}", **payload)

    def _build_review_prompt(self, *, paper: Dict[str, Any], metrics: Dict[str, Any]) -> str:
        payload = dict(locals())
        payload.pop("self", None)
        return self._service_sync("_prompt_service", PromptService, "build_review_prompt", default="{}", **payload)

    def _build_replication_prompt(
        self,
        *,
        world_spec: Dict[str, Any],
        paper_id: str,
        paper: Dict[str, Any],
        claimed_metrics: Dict[str, Any],
    ) -> str:
        payload = dict(locals())
        payload.pop("self", None)
        return self._service_sync(
            "_prompt_service",
            PromptService,
            "build_replication_prompt",
            default="{}",
            **payload,
        )

    def _normalize_code_plan(self, payload: Any) -> Dict[str, Any]:
        diagnosis_service = self._get_service("_diagnosis_service", DiagnosisService)
        if diagnosis_service is not None:
            return diagnosis_service.normalize_code_plan(payload)
        return {}

    def _classify_experiment_failure(
        self,
        *,
        result: Dict[str, Any],
        code_plan: Optional[Dict[str, Any]],
        world_spec: Dict[str, Any],
    ) -> Dict[str, Any]:
        payload = {
            "result": result,
            "code_plan": code_plan or {},
            "world_spec": world_spec or {},
        }
        fallback = {
            "error_class": "unknown",
            "error_codes": ["unknown_failure"],
            "severity": "medium",
            "retryable": True,
            "root_cause": "diagnosis_service_unavailable",
            "evidence": {},
            "repair_hints": [],
            "code_plan_brief": {
                "has_files": bool((code_plan or {}).get("files")),
                "run_cmd": self._truncate((code_plan or {}).get("run_cmd"), 240),
            },
        }
        return self._service_sync(
            "_diagnosis_service",
            DiagnosisService,
            "classify_experiment_failure",
            default=fallback,
            **payload,
        )

    def _apply_failure_template_fixes(
        self,
        *,
        code_plan: Dict[str, Any],
        diagnosis: Dict[str, Any],
        world_spec: Dict[str, Any],
    ) -> Dict[str, Any]:
        return self._service_sync(
            "_diagnosis_service",
            DiagnosisService,
            "apply_failure_template_fixes",
            default={
                "applied": False,
                "rules_hit": [],
                "mutated_files": [],
                "summary": "diagnosis_service_unavailable",
                "code_plan": code_plan,
            },
            code_plan=code_plan,
            diagnosis=diagnosis,
            world_spec=world_spec or {},
        )

    def _build_repair_context(
        self,
        *,
        diagnosis: Dict[str, Any],
        template_fix: Dict[str, Any],
        prev_plan: Optional[Dict[str, Any]],
    ) -> str:
        return self._service_sync(
            "_diagnosis_service",
            DiagnosisService,
            "build_repair_context",
            default=json.dumps(
                {
                    "diagnosis": diagnosis,
                    "template_fix": template_fix,
                    "previous_plan_brief": {
                        "run_cmd": (prev_plan or {}).get("run_cmd"),
                        "file_count": len((prev_plan or {}).get("files") or []),
                    },
                },
                ensure_ascii=False,
            ),
            diagnosis=diagnosis,
            template_fix=template_fix,
            prev_plan=prev_plan,
        )

    def _should_enter_optimize(self, attempts: List[Dict[str, Any]]) -> bool:
        return bool(
            self._service_sync(
                "_diagnosis_service",
                DiagnosisService,
                "should_enter_optimize",
                default=True,
                attempts=attempts,
            )
        )

    def _build_code_experiment_prompt(
        self,
        *,
        world_spec: Dict[str, Any],
        hypothesis: List[str],
        notes: List[Dict[str, Any]],
        observations: List[Dict[str, Any]],
        data_card: Optional[Dict[str, Any]],
        method_card: Optional[Dict[str, Any]],
        plan_spec: Dict[str, Any],
        exp_count: int,
        budget: int,
        phase: str,
        round_idx: int,
        max_rounds: int,
        previous_plan: Optional[Dict[str, Any]] = None,
        failure_context: Optional[str] = None,
        failure_diagnosis: Optional[Dict[str, Any]] = None,
        template_fix: Optional[Dict[str, Any]] = None,
        best_dev_score_norm: Optional[float] = None,
        rag_context: str = "",
        rag_refs: Optional[List[str]] = None,
        rag_status: str = "",
        data_columns: Optional[Dict[str, List[str]]] = None,
    ) -> str:
        payload = dict(locals())
        payload.pop("self", None)
        return self._service_sync(
            "_prompt_service",
            PromptService,
            "build_code_experiment_prompt",
            default="{}",
            **payload,
        )

    async def _run_code_research_loop(
        self,
        *,
        agent_id: str,
        world_spec: Dict[str, Any],
        hypothesis: List[str],
        plan_spec: Dict[str, Any],
        notes: List[Dict[str, Any]],
        prior_observations: List[Dict[str, Any]],
        data_card: Optional[Dict[str, Any]],
        method_card: Optional[Dict[str, Any]],
        exp_count: int,
        budget: int,
        base_run_config: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        payload = dict(locals())
        payload.pop("self", None)
        experiment_operator = getattr(self, "_experiment_operator", None)
        if experiment_operator is None:
            return None
        fn = getattr(experiment_operator, "_run_code_research_loop", None)
        if fn is None:
            return None
        return await fn(**payload)

    def _experiment_precondition_failures(
        self,
        *,
        hypothesis: List[str],
        notes: List[Dict[str, Any]],
        data_card: Optional[Dict[str, Any]],
        method_card: Optional[Dict[str, Any]],
    ) -> List[str]:
        recovery_service = self._get_service("_recovery_service", RecoveryService)
        if recovery_service is not None:
            return recovery_service.experiment_precondition_failures(
                hypothesis=hypothesis,
                notes=notes,
                data_card=data_card,
                method_card=method_card,
            )
        return []

    async def _hydrate_experiment_prerequisites(
        self,
        *,
        agent_id: str,
        hypothesis: List[str],
        notes: List[Dict[str, Any]],
        data_card: Optional[Dict[str, Any]],
        method_card: Optional[Dict[str, Any]],
        failures: List[str],
    ) -> Dict[str, Any]:
        recovery_service = self._get_service("_recovery_service", RecoveryService)
        if recovery_service is not None:
            return await recovery_service.hydrate_experiment_prerequisites(
                agent_id=agent_id,
                hypothesis=hypothesis,
                notes=notes,
                data_card=data_card,
                method_card=method_card,
                failures=failures,
            )
        return {"used_shared_artifacts": False, "hydrate_steps": [], "remaining_failures": list(failures or [])}

    async def _enqueue_prereq_recovery_tasks(self, *, failures: List[str]) -> Dict[str, Any]:
        recovery_service = self._get_service("_recovery_service", RecoveryService)
        if recovery_service is not None:
            return await recovery_service.enqueue_prereq_recovery_tasks(failures=failures)
        return {"created_task_ids": [], "required_task_types": []}

    def _build_citation_owner_map(
        self,
        *,
        agent_id: str,
        local_notes: List[Dict[str, Any]],
        shared_notes: List[Dict[str, Any]],
    ) -> Dict[str, str]:
        contribution_service = self._get_service("_contribution_service", ContributionService)
        if contribution_service is not None:
            return contribution_service.build_citation_owner_map(
                agent_id=agent_id,
                local_notes=local_notes,
                shared_notes=shared_notes,
            )
        return {}

    async def _grant_credits(self, credit_by_agent: Dict[str, float], source: str, reference_id: Optional[str] = None) -> None:
        contribution_service = self._get_service("_contribution_service", ContributionService)
        if contribution_service is not None:
            return await contribution_service.grant_credits(
                credit_by_agent=credit_by_agent,
                source=source,
                reference_id=reference_id,
            )

    def _compute_contribution_credit(
        self,
        *,
        agent_id: str,
        paper: Dict[str, Any],
        metrics: Dict[str, Any],
        shared_observations: List[Dict[str, Any]],
    ) -> Dict[str, float]:
        contribution_service = self._get_service("_contribution_service", ContributionService)
        if contribution_service is not None:
            return contribution_service.compute_contribution_credit(
                agent_id=agent_id,
                paper=paper,
                metrics=metrics,
                shared_observations=shared_observations,
            )
        return {}

    def _build_paper_payload(
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
        contribution_service = self._get_service("_contribution_service", ContributionService)
        if contribution_service is not None:
            return contribution_service.build_paper_payload(
                task_name=task_name,
                metric_name=metric_name,
                best_run=best_run,
                notes=notes,
                observations=observations,
                hypothesis=hypothesis,
                plan_spec=plan_spec,
                exp_count=exp_count,
                llm_write_spec=llm_write_spec,
            )
        return {}

    @AgentCall
    async def read(self, agent_id: str, topic: Optional[str] = None) -> ActionResult:
        return await self._operator_action_or_error(
            action_key="read",
            operator_attr="_data_ops_operator",
            operator_method="execute_read",
            method_name="read",
            error_message="DataOps operator unavailable.",
            reason="dataops_operator_unavailable",
            agent_id=agent_id,
            topic=topic,
        )

    @AgentCall
    async def profile_data(
        self,
        agent_id: str,
        focus_cols: Optional[List[str]] = None,
        refresh: bool = False,
    ) -> ActionResult:
        return await self._operator_action_or_error(
            action_key="profile_data",
            operator_attr="_data_ops_operator",
            operator_method="execute_profile_data",
            method_name="profile_data",
            error_message="DataOps operator unavailable.",
            reason="dataops_operator_unavailable",
            agent_id=agent_id,
            focus_cols=focus_cols,
            refresh=refresh,
        )

    @AgentCall
    async def prepare_data(
        self,
        agent_id: str,
        refresh: bool = False,
    ) -> ActionResult:
        return await self._operator_action_or_error(
            action_key="prepare_data",
            operator_attr="_data_ops_operator",
            operator_method="execute_prepare_data",
            method_name="prepare_data",
            error_message="DataOps operator unavailable.",
            reason="dataops_operator_unavailable",
            agent_id=agent_id,
            refresh=refresh,
        )

    @AgentCall
    async def retrieve_literature(
        self,
        agent_id: str,
        topic: Optional[str] = None,
        refresh: bool = False,
    ) -> ActionResult:
        return await self._operator_action_or_error(
            action_key="retrieve_literature",
            operator_attr="_retrieve_operator",
            operator_method="execute",
            method_name="retrieve_literature",
            error_message="Retrieve operator unavailable.",
            reason="retrieve_operator_unavailable",
            agent_id=agent_id,
            topic=topic,
            refresh=refresh,
        )

    @AgentCall
    async def hypothesize(self, agent_id: str, hypothesis: Optional[List[str]] = None) -> ActionResult:
        return await self._operator_action_or_error(
            action_key="hypothesize",
            operator_attr="_hypothesize_operator",
            operator_method="execute",
            method_name="hypothesize",
            error_message="Hypothesize operator unavailable.",
            reason="hypothesize_operator_unavailable",
            agent_id=agent_id,
            hypothesis=hypothesis,
        )

    @AgentCall
    async def experiment(
        self,
        agent_id: str,
        config: Optional[Dict[str, Any]] = None,
        intervention: Optional[Dict[str, float]] = None,
        n_samples: int = 50,
    ) -> ActionResult:
        return await self._operator_action_or_error(
            action_key="experiment",
            operator_attr="_experiment_operator",
            operator_method="execute",
            method_name="experiment",
            error_message="Experiment operator unavailable.",
            reason="experiment_operator_unavailable",
            agent_id=agent_id,
            config=config,
            intervention=intervention,
            n_samples=n_samples,
        )

    @AgentCall
    async def write(self, agent_id: str) -> ActionResult:
        return await self._operator_action_or_error(
            action_key="write",
            operator_attr="_write_operator",
            operator_method="execute",
            method_name="write",
            error_message="Write operator unavailable.",
            reason="write_operator_unavailable",
            agent_id=agent_id,
        )

    @AgentCall
    async def review(
        self,
        agent_id: str,
        paper_id: Optional[str] = None,
        run_id: Optional[str] = None,
        submission: Optional[Dict[str, Any]] = None,
    ) -> ActionResult:
        return await self._operator_action_or_error(
            action_key="review",
            operator_attr="_review_operator",
            operator_method="execute",
            method_name="review",
            error_message="Review operator unavailable.",
            reason="review_operator_unavailable",
            agent_id=agent_id,
            paper_id=paper_id,
            run_id=run_id,
            submission=submission,
        )

    @AgentCall
    async def verify_strength(
        self,
        agent_id: str,
        paper_id: Optional[str] = None,
        reviewer_id: Optional[str] = None,
        strength: Optional[Dict[str, Any]] = None,
        test: Optional[Dict[str, Any]] = None,
    ) -> ActionResult:
        return await self._operator_action_or_error(
            action_key="verify_strength",
            operator_attr="_verification_operator",
            operator_method="execute_strength",
            method_name="verify_strength",
            error_message="Verification operator unavailable.",
            reason="verification_operator_unavailable",
            agent_id=agent_id,
            paper_id=paper_id,
            reviewer_id=reviewer_id,
            strength=strength,
            test=test,
        )

    @AgentCall
    async def verify_issue(
        self,
        agent_id: str,
        paper_id: Optional[str] = None,
        reviewer_id: Optional[str] = None,
        issue: Optional[Dict[str, Any]] = None,
        test: Optional[Dict[str, Any]] = None,
    ) -> ActionResult:
        return await self._operator_action_or_error(
            action_key="verify_issue",
            operator_attr="_verification_operator",
            operator_method="execute_issue",
            method_name="verify_issue",
            error_message="Verification operator unavailable.",
            reason="verification_operator_unavailable",
            agent_id=agent_id,
            paper_id=paper_id,
            reviewer_id=reviewer_id,
            issue=issue,
            test=test,
        )

    @AgentCall
    async def replicate(
        self,
        agent_id: str,
        paper_id: Optional[str] = None,
        intervention: Optional[Dict[str, float]] = None,
        n_samples: int = 50,
    ) -> ActionResult:
        return await self._operator_action_or_error(
            action_key="replicate",
            operator_attr="_replication_operator",
            operator_method="execute",
            method_name="replicate",
            error_message="Replication operator unavailable.",
            reason="replication_operator_unavailable",
            agent_id=agent_id,
            paper_id=paper_id,
            intervention=intervention,
            n_samples=n_samples,
        )

    @AgentCall
    async def claim_task(
        self,
        agent_id: str,
        task_id: Optional[str] = None,
        task_type: Optional[str] = None,
    ) -> ActionResult:
        return await self._operator_action_or_error(
            action_key="claim_task",
            operator_attr="_taskboard_operator",
            operator_method="execute_claim_task",
            method_name="claim_task",
            error_message="Taskboard operator unavailable.",
            reason="taskboard_operator_unavailable",
            agent_id=agent_id,
            task_id=task_id,
            task_type=task_type,
        )

    @AgentCall
    async def complete_task(
        self,
        agent_id: str,
        task_id: str,
        task_action: Optional[str] = None,
        task_payload: Optional[Dict[str, Any]] = None,
    ) -> ActionResult:
        return await self._operator_action_or_error(
            action_key="complete_task",
            operator_attr="_taskboard_operator",
            operator_method="execute_complete_task",
            method_name="complete_task",
            error_message="Taskboard operator unavailable.",
            reason="taskboard_operator_unavailable",
            agent_id=agent_id,
            task_id=task_id,
            task_action=task_action,
            task_payload=task_payload,
        )

    @AgentCall
    async def share_evidence(
        self,
        agent_id: str,
        to_agent_id: Optional[str] = None,
        max_hints: int = 3,
    ) -> ActionResult:
        return await self._operator_action_or_error(
            action_key="share_evidence",
            operator_attr="_collaboration_operator",
            operator_method="execute_share_evidence",
            method_name="share_evidence",
            error_message="Collaboration operator unavailable.",
            reason="collaboration_operator_unavailable",
            agent_id=agent_id,
            to_agent_id=to_agent_id,
            max_hints=max_hints,
        )

    @AgentCall
    async def share_observation(
        self,
        agent_id: str,
        to_agent_id: Optional[str] = None,
    ) -> ActionResult:
        return await self._operator_action_or_error(
            action_key="share_observation",
            operator_attr="_collaboration_operator",
            operator_method="execute_share_observation",
            method_name="share_observation",
            error_message="Collaboration operator unavailable.",
            reason="collaboration_operator_unavailable",
            agent_id=agent_id,
            to_agent_id=to_agent_id,
        )
