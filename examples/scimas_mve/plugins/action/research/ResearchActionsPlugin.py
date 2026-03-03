import asyncio
import csv
import hashlib
import math
import json
import os
import random
import re
import time
import textwrap
import urllib.error
import urllib.request
from collections import Counter
from datetime import datetime
from typing import Any, Dict, List, Optional

from agentkernel_standalone.mas.action.base.plugin_base import OtherActionsPlugin
from agentkernel_standalone.toolkit.logger import get_logger
from agentkernel_standalone.toolkit.utils.annotation import AgentCall, ServiceCall
from agentkernel_standalone.types.schemas.action import ActionResult

try:
    from .rag_store import RagStore, RagStoreConfig
    from .rag_retriever import RagRetriever
except Exception:  # pragma: no cover
    RagStore = None  # type: ignore
    RagStoreConfig = None  # type: ignore
    RagRetriever = None  # type: ignore

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
        self._rng = random.Random()
        base = os.getenv("MAS_PROJECT_ABS_PATH", ".")
        self._log_mode = (os.getenv("SCIMAS_LOG_MODE", "compact") or "compact").strip().lower()
        self._verbose_action_logs = os.getenv("SCIMAS_VERBOSE_ACTION_LOGS", "0").lower() in {"1", "true", "yes"}
        self._trace_enabled = os.getenv("SCIMAS_TRACE_ENABLE", "1" if self._log_mode == "verbose" else "0").lower() not in {
            "0",
            "false",
            "no",
        }

        self._trace_path = os.path.join(base, "logs", "app", "action", "trace.jsonl")
        self._code_loop_log_path = os.path.join(base, "logs", "app", "action", "code_loop.jsonl")
        self._code_diagnosis_log_path = os.path.join(base, "logs", "app", "action", "code_diagnosis.jsonl")
        self._precondition_gate_log_path = os.path.join(base, "logs", "app", "action", "precondition_gate.jsonl")
        self._research_log_dir = os.path.join(base, "logs", "app", "research")
        self._cards_log_path = os.path.join(self._research_log_dir, "evidence_cards.jsonl")
        self._papers_log_path = os.path.join(self._research_log_dir, "papers.jsonl")

        self._llm_log_dir = os.path.join(base, "logs", "app", "llm")
        self._llm_log_path = os.path.join(self._llm_log_dir, "llm_calls.jsonl")
        self._audit_log_dir = os.path.join(base, "logs", "app", "audit")
        self._llm_audit_jsonl_path = os.path.join(self._audit_log_dir, "llm_io.jsonl")
        self._llm_audit_md_path = os.path.join(self._audit_log_dir, "llm_io.md")
        self._rag_audit_jsonl_path = os.path.join(self._audit_log_dir, "rag_io.jsonl")
        self._rag_audit_md_path = os.path.join(self._audit_log_dir, "rag_io.md")
        self._llm_timeout_s = float(os.getenv("SCIMAS_LLM_TIMEOUT_S", "15"))
        self._llm_enabled = os.getenv("SCIMAS_LLM_ENABLE", "1").lower() not in {"0", "false", "no"}
        self._llm_actions = {
            "claim_task": os.getenv("SCIMAS_LLM_CLAIM_TASK", "1").lower() not in {"0", "false", "no"},
            "hypothesize": os.getenv("SCIMAS_LLM_HYPOTHESIZE", "1").lower() not in {"0", "false", "no"},
            "experiment": os.getenv("SCIMAS_LLM_EXPERIMENT", "1").lower() not in {"0", "false", "no"},
            "write": os.getenv("SCIMAS_LLM_WRITE", "1").lower() not in {"0", "false", "no"},
            "review": os.getenv("SCIMAS_LLM_REVIEW", "1").lower() not in {"0", "false", "no"},
            "replicate": os.getenv("SCIMAS_LLM_REPLICATE", "1").lower() not in {"0", "false", "no"},
        }
        self._llm_log_enabled = os.getenv("SCIMAS_LLM_LOG_ENABLE", "0" if self._log_mode == "compact" else "1").lower() not in {
            "0",
            "false",
            "no",
        }
        self._audit_io_enable = os.getenv("SCIMAS_AUDIT_IO_ENABLE", "1").lower() not in {"0", "false", "no"}
        self._audit_markdown_enable = os.getenv("SCIMAS_AUDIT_MARKDOWN_ENABLE", "1").lower() not in {
            "0",
            "false",
            "no",
        }
        self._audit_llm_max_chars = int(max(2000, int(os.getenv("SCIMAS_AUDIT_LLM_MAX_CHARS", "300000"))))
        self._audit_rag_max_chars = int(max(2000, int(os.getenv("SCIMAS_AUDIT_RAG_MAX_CHARS", "300000"))))
        self._audit_rag_max_rows = int(max(1, int(os.getenv("SCIMAS_AUDIT_RAG_MAX_ROWS", "20"))))
        self._llm_max_cards = int(os.getenv("SCIMAS_LLM_MAX_CARDS", "10"))
        self._llm_max_runs = int(os.getenv("SCIMAS_LLM_MAX_RUNS", "8"))
        self._code_loop_enabled = os.getenv("SCIMAS_CODE_AGENT_ENABLE", "1").lower() not in {"0", "false", "no"}
        self._code_debug_rounds = int(max(1, int(os.getenv("SCIMAS_CODE_DEBUG_ROUNDS", "4"))))
        self._code_optimize_after_success = os.getenv("SCIMAS_CODE_OPTIMIZE_AFTER_SUCCESS", "1").lower() not in {
            "0",
            "false",
            "no",
        }
        self._code_diag_enable = os.getenv("SCIMAS_CODE_DIAG_ENABLE", "1").lower() not in {"0", "false", "no"}
        self._code_template_fix_enable = os.getenv("SCIMAS_CODE_TEMPLATE_FIX_ENABLE", "1").lower() not in {
            "0",
            "false",
            "no",
        }
        self._code_optimize_guard_enable = os.getenv("SCIMAS_CODE_OPTIMIZE_GUARD_ENABLE", "1").lower() not in {
            "0",
            "false",
            "no",
        }
        self._code_optimize_patience = int(max(1, int(os.getenv("SCIMAS_CODE_OPTIMIZE_PATIENCE", "2"))))
        self._code_max_files = int(max(1, int(os.getenv("SCIMAS_CODE_MAX_FILES", "8"))))
        self._code_max_file_chars = int(max(2000, int(os.getenv("SCIMAS_CODE_MAX_FILE_CHARS", "60000"))))
        self._code_error_tail_chars = int(max(300, int(os.getenv("SCIMAS_CODE_ERROR_TAIL_CHARS", "3000"))))
        self._cards_log_enabled = os.getenv("SCIMAS_EVIDENCE_LOG_ENABLE", "1" if self._log_mode == "verbose" else "0").lower() not in {
            "0",
            "false",
            "no",
        }
        self._papers_log_enabled = os.getenv("SCIMAS_PAPER_LOG_ENABLE", "1" if self._log_mode == "verbose" else "0").lower() not in {
            "0",
            "false",
            "no",
        }

        self._strict_task_dependencies = os.getenv("SCIMAS_STRICT_TASK_DEPENDENCIES", "1").lower() not in {
            "0",
            "false",
            "no",
        }
        self._write_min_notes = int(os.getenv("SCIMAS_WRITE_MIN_NOTES", "1"))
        self._write_min_observations = int(os.getenv("SCIMAS_WRITE_MIN_OBS", "1"))
        self._write_min_hypothesis = int(os.getenv("SCIMAS_WRITE_MIN_HYP", "1"))
        self._experiment_require_data_card = os.getenv("SCIMAS_EXPERIMENT_REQUIRE_DATA_CARD", "1").lower() not in {
            "0",
            "false",
            "no",
        }
        self._experiment_require_method_card = os.getenv("SCIMAS_EXPERIMENT_REQUIRE_METHOD_CARD", "1").lower() not in {
            "0",
            "false",
            "no",
        }
        self._experiment_min_notes = int(max(0, int(os.getenv("SCIMAS_EXPERIMENT_MIN_NOTES", "1"))))
        self._experiment_min_hypothesis = int(max(0, int(os.getenv("SCIMAS_EXPERIMENT_MIN_HYP", "0"))))
        self._claim_backoff_base = int(max(1, int(os.getenv("SCIMAS_CLAIM_BACKOFF_BASE", "1"))))
        self._claim_backoff_max = int(max(self._claim_backoff_base, int(os.getenv("SCIMAS_CLAIM_BACKOFF_MAX", "8"))))
        self._claim_cost = float(max(0.0, float(os.getenv("SCIMAS_CLAIM_COST", "0.002"))))
        self._claim_dispatch_enabled = os.getenv("SCIMAS_CLAIM_DISPATCH_ENABLE", "1").lower() not in {"0", "false", "no"}
        dispatch_raw = str(os.getenv("SCIMAS_CLAIM_DISPATCH_TASK_TYPES", "experiment") or "").strip()
        self._claim_dispatch_task_types = {
            t.strip().lower() for t in dispatch_raw.split(",") if t.strip()
        } or {"experiment"}
        self._task_heartbeat_enabled = os.getenv("SCIMAS_TASK_HEARTBEAT_ENABLE", "1").lower() not in {"0", "false", "no"}
        self._review_min_issue_count = int(max(1, int(os.getenv("SCIMAS_REVIEW_MIN_ISSUES", "2"))))
        self._review_min_revision_actions = int(max(1, int(os.getenv("SCIMAS_REVIEW_MIN_ACTIONS", "2"))))
        self._review_revision_trigger_score = float(
            max(0.0, min(1.0, float(os.getenv("SCIMAS_REVIEW_REVISION_TRIGGER", "0.62"))))
        )
        self._strict_review_mode = os.getenv("SCIMAS_STRICT_REVIEW", "1").lower() not in {"0", "false", "no"}
        self._qgr_min_issue_count = int(max(1, int(os.getenv("SCIMAS_QGR_MIN_ISSUES", "2"))))
        self._qgr_min_citations = int(max(1, int(os.getenv("SCIMAS_QGR_MIN_CITATIONS", "3"))))
        self._qgr_relevance_threshold = float(max(0.0, min(1.0, float(os.getenv("SCIMAS_QGR_RELEVANCE_THRESHOLD", "0.75")))))
        self._qgr_fact_support_threshold = float(max(0.0, min(1.0, float(os.getenv("SCIMAS_QGR_FACT_SUPPORT_THRESHOLD", "0.20")))))
        self._qgr_base_reward = float(os.getenv("SCIMAS_QGR_BASE_REWARD", "0.2"))
        self._qgr_quality_bonus = float(os.getenv("SCIMAS_QGR_QUALITY_BONUS", "0.5"))
        self._qgr_predictive_bonus_reward = float(os.getenv("SCIMAS_QGR_PREDICTIVE_BONUS", "1.5"))
        self._review_flattery_penalty = float(max(0.0, float(os.getenv("SCIMAS_REVIEW_FLATTERY_PENALTY", "0.03"))))
        self._review_shallow_penalty = float(max(0.0, float(os.getenv("SCIMAS_REVIEW_SHALLOW_PENALTY", "0.02"))))
        self._review_self_review_penalty = float(max(0.0, float(os.getenv("SCIMAS_REVIEW_SELF_PENALTY", "0.02"))))
        self._dense_reward_enable = os.getenv("SCIMAS_DENSE_REWARD_ENABLE", "1").lower() not in {"0", "false", "no"}
        self._read_reward_alpha = float(max(0.0, float(os.getenv("SCIMAS_READ_REWARD_ALPHA", "0.35"))))
        self._read_reward_base = float(max(0.0, float(os.getenv("SCIMAS_READ_REWARD_BASE", "0.20"))))
        self._read_reward_max = float(max(self._read_reward_base, float(os.getenv("SCIMAS_READ_REWARD_MAX", "0.50"))))
        self._read_method_bonus = float(max(0.0, float(os.getenv("SCIMAS_READ_METHOD_BONUS", "0.08"))))
        self._read_dup_threshold = float(max(0.0, min(1.0, float(os.getenv("SCIMAS_READ_DUP_THRESHOLD", "0.90")))))
        self._reward_tei_url = str(os.getenv("SCIMAS_REWARD_TEI_URL", "")).strip()
        self._reward_qdrant_url = str(os.getenv("SCIMAS_REWARD_QDRANT_URL", "")).strip().rstrip("/")
        self._reward_qdrant_collection = str(os.getenv("SCIMAS_REWARD_QDRANT_COLLECTION", "notes")).strip()
        self._reward_qdrant_api_key = str(os.getenv("SCIMAS_REWARD_QDRANT_API_KEY", "")).strip()
        self._hypothesis_schema_bonus = float(max(0.0, float(os.getenv("SCIMAS_HYPOTHESIS_SCHEMA_BONUS", "0.5"))))
        self._hypothesis_resource_bonus = float(max(0.0, float(os.getenv("SCIMAS_HYPOTHESIS_RESOURCE_BONUS", "0.3"))))
        self._experiment_success_reward = float(os.getenv("SCIMAS_EXPERIMENT_SUCCESS_REWARD", "1.0"))
        self._experiment_oom_penalty = float(max(0.0, float(os.getenv("SCIMAS_EXPERIMENT_OOM_PENALTY", "0.8"))))
        self._experiment_typeerror_penalty = float(max(0.0, float(os.getenv("SCIMAS_EXPERIMENT_TYPEERROR_PENALTY", "0.2"))))
        self._experiment_first_pass_bonus = float(max(0.0, float(os.getenv("SCIMAS_EXPERIMENT_FIRST_PASS_BONUS", "0.25"))))
        self._experiment_vram_reward_weight = float(max(0.0, float(os.getenv("SCIMAS_EXPERIMENT_VRAM_WEIGHT", "0.2"))))
        self._write_eval_success_bonus = float(max(0.0, float(os.getenv("SCIMAS_WRITE_EVAL_SUCCESS_BONUS", "2.0"))))
        self._write_format_pass_reward = float(max(0.0, float(os.getenv("SCIMAS_WRITE_FORMAT_PASS_REWARD", "0.3"))))
        self._write_cache_repeat_penalty = float(max(0.0, float(os.getenv("SCIMAS_WRITE_CACHE_REPEAT_PENALTY", "0.05"))))
        self._write_defer_on_system_error = os.getenv("SCIMAS_WRITE_DEFER_ON_SYSTEM_ERROR", "1").lower() not in {
            "0",
            "false",
            "no",
        }
        self._replicate_high_support_reward = float(max(0.0, float(os.getenv("SCIMAS_REPLICATE_HIGH_SUPPORT_REWARD", "5.0"))))
        self._replicate_support_threshold = float(max(0.0, min(1.0, float(os.getenv("SCIMAS_REPLICATE_SUPPORT_THRESHOLD", "0.8")))))
        self._vdh_enable = os.getenv("SCIMAS_VDH_ENABLE", "1").lower() not in {"0", "false", "no"}
        self._vdh_gate_policy = str(os.getenv("SCIMAS_VDH_GATE_POLICY", "hard_fail") or "hard_fail").strip().lower()
        self._vdh_evidence_threshold = float(
            max(0.0, min(1.0, float(os.getenv("SCIMAS_VDH_EVIDENCE_THRESHOLD", "0.60"))))
        )
        self._vdh_qdrant_enable = os.getenv("SCIMAS_VDH_QDRANT_ENABLE", "1").lower() not in {"0", "false", "no"}
        self._vdh_qdrant_url = str(os.getenv("SCIMAS_VDH_QDRANT_URL", "")).strip().rstrip("/")
        self._vdh_qdrant_collection = str(os.getenv("SCIMAS_VDH_QDRANT_COLLECTION", "schema_collection")).strip()
        self._vdh_qdrant_api_key = str(os.getenv("SCIMAS_VDH_QDRANT_API_KEY", "")).strip()
        self._vdh_tei_enable = os.getenv("SCIMAS_VDH_TEI_ENABLE", "1").lower() not in {"0", "false", "no"}
        self._vdh_tei_url = str(os.getenv("SCIMAS_VDH_TEI_URL", "")).strip()
        self._vdh_oom_ratio_threshold = float(
            max(0.1, min(2.0, float(os.getenv("SCIMAS_VDH_OOM_RATIO_THRESHOLD", "0.90"))))
        )
        self._vdh_schema_pass_reward = float(max(0.0, float(os.getenv("SCIMAS_VDH_SCHEMA_PASS_REWARD", "0.5"))))
        self._vdh_evidence_high_reward = float(max(0.0, float(os.getenv("SCIMAS_VDH_EVIDENCE_HIGH_REWARD", "0.8"))))
        self._vdh_oom_penalty = float(max(0.0, float(os.getenv("SCIMAS_VDH_OOM_PENALTY", "1.0"))))
        self._vdh_gate_penalty = float(max(0.0, float(os.getenv("SCIMAS_VDH_GATE_PENALTY", "0.2"))))
        self._vdh_gate_log_path = os.path.join(base, "logs", "app", "action", "vdh_gate.jsonl")
        self._review_gate_log_path = os.path.join(base, "logs", "app", "action", "review_gate.jsonl")
        self._rag_index_log_path = os.path.join(base, "logs", "app", "action", "rag_index.jsonl")
        self._rag_query_log_path = os.path.join(base, "logs", "app", "action", "rag_query.jsonl")
        self._rag_usage_log_path = os.path.join(base, "logs", "app", "action", "rag_usage.jsonl")
        self._rag_health_log_path = os.path.join(base, "logs", "app", "action", "rag_health.jsonl")
        self._rag_alert_log_path = os.path.join(base, "logs", "app", "action", "rag_alert.jsonl")

        self._rag_enable = os.getenv("SCIMAS_RAG_ENABLE", "1").lower() not in {"0", "false", "no"}
        self._rag_qdrant_url = str(os.getenv("SCIMAS_RAG_QDRANT_URL", "http://127.0.0.1:6333")).strip().rstrip("/")
        self._rag_qdrant_api_key = str(os.getenv("SCIMAS_RAG_QDRANT_API_KEY", "")).strip()
        self._rag_collection = str(os.getenv("SCIMAS_RAG_COLLECTION", "scimas_local_knowledge_v1")).strip()
        self._rag_embed_url = str(os.getenv("SCIMAS_RAG_EMBED_URL", "http://127.0.0.1:8001/v1/embeddings")).strip()
        self._rag_embed_model = str(os.getenv("SCIMAS_RAG_EMBED_MODEL", "Qwen/Qwen3-Embedding-4B")).strip()
        self._rag_topk = int(max(1, int(os.getenv("SCIMAS_RAG_TOPK", "8"))))
        self._rag_max_context_chars = int(max(500, int(os.getenv("SCIMAS_RAG_MAX_CONTEXT_CHARS", "9000"))))
        self._rag_chunk_chars = int(max(200, int(os.getenv("SCIMAS_RAG_CHUNK_CHARS", "1200"))))
        self._rag_chunk_overlap = int(max(0, int(os.getenv("SCIMAS_RAG_CHUNK_OVERLAP", "180"))))
        self._rag_batch_size = int(max(1, int(os.getenv("SCIMAS_RAG_BATCH_SIZE", "32"))))
        self._rag_timeout_s = float(max(1.0, float(os.getenv("SCIMAS_RAG_TIMEOUT_S", "8"))))
        self._rag_min_score = float(max(0.0, min(1.0, float(os.getenv("SCIMAS_RAG_MIN_SCORE", "0.25")))))
        self._rag_index_on_read = os.getenv("SCIMAS_RAG_INDEX_ON_READ", "1").lower() not in {"0", "false", "no"}
        self._rag_index_on_experiment = os.getenv("SCIMAS_RAG_INDEX_ON_EXPERIMENT", "1").lower() not in {"0", "false", "no"}
        self._rag_index_on_write = os.getenv("SCIMAS_RAG_INDEX_ON_WRITE", "1").lower() not in {"0", "false", "no"}
        self._rag_degraded_alert_threshold = int(max(1, int(os.getenv("SCIMAS_RAG_DEGRADED_STREAK_ALERT", "3"))))
        self._rag_degraded_pause_ticks = int(max(1, int(os.getenv("SCIMAS_RAG_DEGRADED_PAUSE_TICKS", "12"))))
        self._rag_store = None
        self._rag_retriever = None
        self._rag_bootstrap_episode: Optional[int] = None
        self._rag_health_checked = False
        self._rag_degraded_streak = 0
        self._rag_degraded_last_status = ""
        self._rag_degraded_pause_until_tick = 0

    async def init(self, model_router=None, controller=None):
        self.model = model_router
        self.controller = controller
        os.makedirs(os.path.dirname(self._trace_path), exist_ok=True)
        os.makedirs(os.path.dirname(self._code_loop_log_path), exist_ok=True)
        os.makedirs(os.path.dirname(self._code_diagnosis_log_path), exist_ok=True)
        os.makedirs(os.path.dirname(self._precondition_gate_log_path), exist_ok=True)
        os.makedirs(os.path.dirname(self._vdh_gate_log_path), exist_ok=True)
        os.makedirs(os.path.dirname(self._review_gate_log_path), exist_ok=True)
        os.makedirs(os.path.dirname(self._rag_index_log_path), exist_ok=True)
        os.makedirs(os.path.dirname(self._rag_query_log_path), exist_ok=True)
        os.makedirs(os.path.dirname(self._rag_usage_log_path), exist_ok=True)
        os.makedirs(os.path.dirname(self._rag_health_log_path), exist_ok=True)
        os.makedirs(os.path.dirname(self._rag_alert_log_path), exist_ok=True)
        os.makedirs(self._research_log_dir, exist_ok=True)
        os.makedirs(self._llm_log_dir, exist_ok=True)
        os.makedirs(self._audit_log_dir, exist_ok=True)
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
            session_record = {
                "meta": {
                    "ts": datetime.utcnow().isoformat() + "Z",
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
        if not self._trace_enabled:
            return
        if self._log_mode == "compact":
            important = {"write", "review", "replicate", "complete_task"}
            is_failure = bool((detail or {}).get("precondition_failed")) or bool((detail or {}).get("ok") is False)
            if action not in important and not is_failure:
                return

        tick = await self.controller.run_system("timer", "get_tick")
        world_spec = await self.controller.run_environment("science", "get_world_spec")
        fitness = await self._get_state(agent_id, "last_fitness")
        exp_count = await self._get_state(agent_id, "exp_count") or 0
        hypothesis = await self._get_state(agent_id, "hypothesis") or []

        record = {
            "ts": datetime.utcnow().isoformat() + "Z",
            "tick": tick,
            "episode_id": world_spec.get("episode_id"),
            "task_name": world_spec.get("task_name"),
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

    async def _log_precondition_gate(
        self,
        *,
        agent_id: str,
        action: str,
        phase: str,
        failures: List[str],
        summary: Optional[Dict[str, Any]] = None,
    ) -> None:
        tick = await self.controller.run_system("timer", "get_tick")
        world_spec = await self.controller.run_environment("science", "get_world_spec")
        await self._append_jsonl(
            self._precondition_gate_log_path,
            {
                "ts": datetime.utcnow().isoformat() + "Z",
                "tick": tick,
                "episode_id": world_spec.get("episode_id"),
                "task_name": world_spec.get("task_name"),
                "agent_id": agent_id,
                "action": action,
                "phase": phase,
                "failures": [str(x) for x in (failures or [])],
                "summary": summary or {},
            },
        )

    async def _log_vdh_gate(
        self,
        *,
        agent_id: str,
        vdh_report: Dict[str, Any],
        reward_components: Optional[Dict[str, Any]] = None,
    ) -> None:
        tick = await self.controller.run_system("timer", "get_tick")
        world_spec = await self.controller.run_environment("science", "get_world_spec")
        await self._append_jsonl(
            self._vdh_gate_log_path,
            {
                "ts": datetime.utcnow().isoformat() + "Z",
                "tick": tick,
                "episode_id": world_spec.get("episode_id"),
                "task_name": world_spec.get("task_name"),
                "task_path": world_spec.get("task_path"),
                "agent_id": agent_id,
                "action": "hypothesize",
                "vdh": vdh_report or {},
                "reward_components": reward_components or {},
            },
        )

    async def _log_review_gate(
        self,
        *,
        agent_id: str,
        paper_id: Optional[str],
        run_id: Optional[str],
        gate: Dict[str, Any],
    ) -> None:
        tick = await self.controller.run_system("timer", "get_tick")
        world_spec = await self.controller.run_environment("science", "get_world_spec")
        await self._append_jsonl(
            self._review_gate_log_path,
            {
                "ts": datetime.utcnow().isoformat() + "Z",
                "tick": tick,
                "episode_id": world_spec.get("episode_id"),
                "task_name": world_spec.get("task_name"),
                "agent_id": agent_id,
                "paper_id": paper_id,
                "run_id": run_id,
                "gate": gate or {},
            },
        )

    async def _log_code_loop(
        self,
        *,
        agent_id: str,
        attempts: List[Dict[str, Any]],
        best_dev_score_norm: Optional[float],
    ) -> None:
        tick = await self.controller.run_system("timer", "get_tick")
        world_spec = await self.controller.run_environment("science", "get_world_spec")
        compact_attempts: List[Dict[str, Any]] = []
        for a in (attempts or []):
            r = (a or {}).get("result") if isinstance((a or {}).get("result"), dict) else {}
            compact_attempts.append(
                {
                    "round": a.get("round"),
                    "phase": a.get("phase"),
                    "llm_ok": bool(a.get("llm_ok", False)),
                    "llm_reason": self._truncate(a.get("llm_reason"), 180),
                    "run_id": r.get("run_id"),
                    "ok": bool(r.get("ok", False)) if isinstance(r, dict) else False,
                    "error": self._truncate((r or {}).get("error"), 220) if isinstance(r, dict) else "",
                    "dev_score_norm": (r or {}).get("dev_score_norm") if isinstance(r, dict) else None,
                    "score_norm": (r or {}).get("score_norm") if isinstance(r, dict) else None,
                    "solver_mode": (r or {}).get("solver_mode") if isinstance(r, dict) else None,
                    "error_class": ((a.get("diagnosis") or {}).get("error_class") if isinstance(a.get("diagnosis"), dict) else None),
                    "error_codes": ((a.get("diagnosis") or {}).get("error_codes") if isinstance(a.get("diagnosis"), dict) else []),
                    "template_rules": ((a.get("template_fix") or {}).get("rules_hit") if isinstance(a.get("template_fix"), dict) else []),
                }
            )
        await self._append_jsonl(
            self._code_loop_log_path,
            {
                "ts": datetime.utcnow().isoformat() + "Z",
                "tick": tick,
                "episode_id": world_spec.get("episode_id"),
                "task_name": world_spec.get("task_name"),
                "agent_id": agent_id,
                "attempt_count": len(compact_attempts),
                "best_dev_score_norm": best_dev_score_norm,
                "attempts": compact_attempts,
            },
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
    ) -> None:
        tick = await self.controller.run_system("timer", "get_tick")
        world_spec = await self.controller.run_environment("science", "get_world_spec")
        await self._append_jsonl(
            self._code_diagnosis_log_path,
            {
                "ts": datetime.utcnow().isoformat() + "Z",
                "tick": tick,
                "episode_id": world_spec.get("episode_id"),
                "task_name": world_spec.get("task_name"),
                "agent_id": agent_id,
                "phase": phase,
                "run_id": run_id,
                "decision": decision,
                "score_norm": score_norm,
                "dev_score_norm": dev_score_norm,
                "diagnosis": diagnosis or {},
                "template_fix": template_fix or {},
            },
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

    def _llm_ready(self, action_name: str) -> bool:
        if not self._llm_enabled:
            return False
        if not self._llm_actions.get(action_name, False):
            return False
        return self.model is not None and hasattr(self.model, "chat")

    def _extract_json_candidate(self, text: str) -> str:
        if not text:
            return ""
        stripped = text.strip()
        fenced = re.search(r"```(?:json)?\s*(\{.*\}|\[.*\])\s*```", stripped, re.DOTALL)
        if fenced:
            return fenced.group(1).strip()
        first_obj = stripped.find("{")
        last_obj = stripped.rfind("}")
        if first_obj != -1 and last_obj != -1 and last_obj > first_obj:
            return stripped[first_obj : last_obj + 1]
        first_arr = stripped.find("[")
        last_arr = stripped.rfind("]")
        if first_arr != -1 and last_arr != -1 and last_arr > first_arr:
            return stripped[first_arr : last_arr + 1]
        return stripped

    def _safe_json_loads(self, text: str) -> Any:
        candidate = self._extract_json_candidate(text)
        return json.loads(candidate)

    async def _log_llm_call(self, record: Dict[str, Any]) -> None:
        if not self._llm_log_enabled:
            return
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
        if not self._audit_io_enable:
            return
        req = {
            "prompt": self._clip_text_for_audit(prompt, self._audit_llm_max_chars),
        }
        raw_text = raw_response if isinstance(raw_response, str) else str(raw_response or "")
        resp = {
            "ok": bool(ok),
            "reason": str(reason or ""),
            "raw_response": self._clip_text_for_audit(raw_text, self._audit_llm_max_chars),
            "parsed_json": self._safe_jsonable(parsed_data),
        }
        meta = {
            "ts": ts,
            "tick": tick,
            "episode_id": world_spec.get("episode_id"),
            "task_name": world_spec.get("task_name"),
            "agent_id": agent_id,
            "action": action_name,
            "kind": "llm_chat_json",
        }
        record = {
            "meta": meta,
            "input": req,
            "output": resp,
        }
        await self._append_jsonl(self._llm_audit_jsonl_path, record)
        await self._append_markdown_audit(
            path=self._llm_audit_md_path,
            title=f"[{ts}] action={action_name} agent={agent_id} ok={bool(ok)}",
            meta=meta,
            request_payload=req,
            response_payload=resp,
        )

    async def _call_llm_json(self, *, agent_id: str, action_name: str, prompt: str) -> Dict[str, Any]:
        tick = await self.controller.run_system("timer", "get_tick")
        world_spec = await self.controller.run_environment("science", "get_world_spec")
        ts = datetime.utcnow().isoformat() + "Z"

        if not self._llm_ready(action_name):
            await self._log_llm_audit(
                ts=ts,
                tick=tick,
                world_spec=world_spec,
                agent_id=agent_id,
                action_name=action_name,
                prompt=prompt,
                raw_response="",
                parsed_data={},
                ok=False,
                reason="llm_disabled_or_missing",
            )
            return {"ok": False, "reason": "llm_disabled_or_missing"}
        try:
            raw = await asyncio.wait_for(self.model.chat(prompt), timeout=self._llm_timeout_s)
            parsed = self._safe_json_loads(raw if isinstance(raw, str) else str(raw))
            await self._log_llm_call(
                {
                    "ts": ts,
                    "tick": tick,
                    "episode_id": world_spec.get("episode_id"),
                    "task_name": world_spec.get("task_name"),
                    "agent_id": agent_id,
                    "action": action_name,
                    "ok": True,
                    "prompt_chars": len(prompt),
                    "response_chars": len(raw if isinstance(raw, str) else str(raw)),
                    "prompt_preview": prompt[:500],
                    "response_preview": (raw if isinstance(raw, str) else str(raw))[:500],
                }
            )
            await self._log_llm_audit(
                ts=ts,
                tick=tick,
                world_spec=world_spec,
                agent_id=agent_id,
                action_name=action_name,
                prompt=prompt,
                raw_response=raw,
                parsed_data=parsed,
                ok=True,
            )
            return {"ok": True, "raw": raw, "data": parsed}
        except Exception as e:
            await self._log_llm_call(
                {
                    "ts": ts,
                    "tick": tick,
                    "episode_id": world_spec.get("episode_id"),
                    "task_name": world_spec.get("task_name"),
                    "agent_id": agent_id,
                    "action": action_name,
                    "ok": False,
                    "error": str(e),
                    "prompt_chars": len(prompt),
                    "prompt_preview": prompt[:500],
                }
            )
            await self._log_llm_audit(
                ts=ts,
                tick=tick,
                world_spec=world_spec,
                agent_id=agent_id,
                action_name=action_name,
                prompt=prompt,
                raw_response="",
                parsed_data={},
                ok=False,
                reason=str(e),
            )
            return {"ok": False, "reason": str(e)}

    async def _log_evidence_cards(self, agent_id: str, literature: Dict[str, Any], source: str = "read") -> None:
        if not self._cards_log_enabled:
            return
        tick = await self.controller.run_system("timer", "get_tick")
        world_spec = await self.controller.run_environment("science", "get_world_spec")
        cards = literature.get("cards") or []
        record = {
            "ts": datetime.utcnow().isoformat() + "Z",
            "tick": tick,
            "episode_id": world_spec.get("episode_id"),
            "task_name": world_spec.get("task_name"),
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
        if not self._papers_log_enabled:
            return
        tick = await self.controller.run_system("timer", "get_tick")
        world_spec = await self.controller.run_environment("science", "get_world_spec")
        record = {
            "ts": datetime.utcnow().isoformat() + "Z",
            "tick": tick,
            "episode_id": world_spec.get("episode_id"),
            "task_name": world_spec.get("task_name"),
            "agent_id": agent_id,
            "source": source,
            "paper_id": paper_id,
            "paper": paper,
            "metrics": metrics,
        }
        await self._append_jsonl(self._papers_log_path, record)

    def _init_rag_clients(self) -> None:
        self._rag_store = None
        self._rag_retriever = None
        if not self._rag_enable:
            return
        if RagStoreConfig is None or RagStore is None or RagRetriever is None:
            return
        cfg = RagStoreConfig(
            enable=bool(self._rag_enable),
            qdrant_url=self._rag_qdrant_url,
            qdrant_api_key=self._rag_qdrant_api_key,
            collection=self._rag_collection,
            embed_url=self._rag_embed_url,
            embed_model=self._rag_embed_model,
            timeout_s=float(self._rag_timeout_s),
            chunk_chars=int(self._rag_chunk_chars),
            chunk_overlap=int(self._rag_chunk_overlap),
            batch_size=int(self._rag_batch_size),
        )
        self._rag_store = RagStore(cfg)
        self._rag_retriever = RagRetriever(self._rag_store, max_context_chars=self._rag_max_context_chars)

    async def _rag_startup_health_check(self, *, world_spec: Dict[str, Any]) -> None:
        if self._rag_health_checked:
            return
        self._rag_health_checked = True

        ts = datetime.utcnow().isoformat() + "Z"
        request_payload = {
            "rag_enable": bool(self._rag_enable),
            "qdrant_url": self._rag_qdrant_url,
            "embed_url": self._rag_embed_url,
            "collection": self._rag_collection,
            "embed_model": self._rag_embed_model,
            "timeout_s": float(self._rag_timeout_s),
        }

        if not self._rag_enable:
            response_payload = {"ok": False, "status": "disabled", "reason": "SCIMAS_RAG_ENABLE=0"}
        elif self._rag_store is None:
            response_payload = {"ok": False, "status": "degraded:init_failed", "reason": "rag_clients_unavailable"}
        else:
            try:
                response_payload = await self._rag_store.health_check()
            except Exception as e:
                response_payload = {"ok": False, "status": "degraded:health_check_exception", "error": str(e)}

        await self._append_jsonl(
            self._rag_health_log_path,
            {
                "ts": ts,
                "episode_id": (world_spec or {}).get("episode_id"),
                "task_name": (world_spec or {}).get("task_name"),
                "request": request_payload,
                "result": response_payload,
            },
        )

        if self._audit_io_enable:
            await self._log_rag_audit(
                world_spec=world_spec or {},
                agent_id="system",
                action="system",
                operation="health_check",
                run_id=None,
                paper_id=None,
                request_payload=request_payload,
                response_payload=response_payload,
            )

        if bool(response_payload.get("ok")):
            logger.info(
                f"RAG health check OK: qdrant={self._rag_qdrant_url} "
                f"embed={self._rag_embed_url} collection={self._rag_collection}"
            )
        else:
            logger.warning(
                f"RAG health check degraded: status={response_payload.get('status')} "
                f"qdrant={self._rag_qdrant_url} embed={self._rag_embed_url}"
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
        st = str(status or "").strip().lower()
        degraded = st.startswith("degraded:")
        if not degraded:
            if self._rag_degraded_streak > 0 or self._rag_degraded_pause_until_tick > 0:
                self._rag_degraded_streak = 0
                self._rag_degraded_last_status = st
                self._rag_degraded_pause_until_tick = 0
            return

        self._rag_degraded_streak += 1
        self._rag_degraded_last_status = st

        try:
            tick = int(await self.controller.run_system("timer", "get_tick") or 0)
        except Exception:
            tick = 0
        if self._rag_degraded_streak < self._rag_degraded_alert_threshold:
            return
        if tick > 0 and tick < self._rag_degraded_pause_until_tick:
            return

        self._rag_degraded_pause_until_tick = max(
            self._rag_degraded_pause_until_tick,
            tick + self._rag_degraded_pause_ticks,
        )
        alert_record = {
            "ts": datetime.utcnow().isoformat() + "Z",
            "episode_id": (world_spec or {}).get("episode_id"),
            "task_name": (world_spec or {}).get("task_name"),
            "agent_id": agent_id,
            "action": action,
            "source": source,
            "rag_status": st,
            "degraded_streak": int(self._rag_degraded_streak),
            "threshold": int(self._rag_degraded_alert_threshold),
            "pause_until_tick": int(self._rag_degraded_pause_until_tick),
            "reason": "rag_degraded_streak_threshold_reached",
        }
        await self._append_jsonl(self._rag_alert_log_path, alert_record)
        logger.error(
            f"RAG HARD ALERT: status={st} streak={self._rag_degraded_streak} "
            f"threshold={self._rag_degraded_alert_threshold} "
            f"pause_retrieve_until_tick={self._rag_degraded_pause_until_tick}"
        )

    async def _rag_retrieve_recovery_paused(self) -> bool:
        until = int(self._rag_degraded_pause_until_tick or 0)
        if until <= 0:
            return False
        try:
            tick = int(await self.controller.run_system("timer", "get_tick") or 0)
        except Exception:
            tick = 0
        if tick <= 0:
            return True
        return tick < until

    def _rag_hash(self, text: str) -> str:
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
        return {
            "episode_id": world_spec.get("episode_id"),
            "task_name": world_spec.get("task_name"),
            "agent_id": agent_id,
            "source_type": source_type,
            "source_id": source_id,
            "action": action,
            "tags": list(tags or []),
            "quality": quality,
            "ts": datetime.utcnow().isoformat() + "Z",
            "version": "v1",
        }

    def _rag_docs_from_note(
        self,
        *,
        world_spec: Dict[str, Any],
        agent_id: str,
        note: Dict[str, Any],
        action: str,
    ) -> List[Dict[str, Any]]:
        if not isinstance(note, dict):
            return []
        text = self._note_to_text(note).strip()
        if not text:
            return []
        source_id = f"note:{self._rag_hash(text)[:16]}"
        d = self._rag_doc_base(
            world_spec=world_spec,
            agent_id=agent_id,
            source_type="note",
            source_id=source_id,
            action=action,
            tags=self._safe_text_list(note.get("hints"), limit=6, item_limit=80),
            quality=0.5,
        )
        d["text"] = text
        return [d]

    def _rag_docs_from_method_card(
        self,
        *,
        world_spec: Dict[str, Any],
        agent_id: str,
        method_card: Dict[str, Any],
        action: str,
    ) -> List[Dict[str, Any]]:
        if not isinstance(method_card, dict):
            return []
        docs: List[Dict[str, Any]] = []
        topic = str(method_card.get("topic") or method_card.get("task_name") or "method").strip() or "method"
        baselines = method_card.get("recommended_baselines") if isinstance(method_card.get("recommended_baselines"), list) else []
        if baselines:
            for idx, b in enumerate(baselines[:8], start=1):
                if not isinstance(b, dict):
                    continue
                text = " | ".join(
                    [
                        f"name={self._truncate(b.get('name'), 120)}",
                        f"use_when={self._truncate(b.get('use_when'), 180)}",
                        "key_steps=" + "; ".join(self._safe_text_list(b.get("key_steps"), limit=5, item_limit=120)),
                        "pitfalls=" + "; ".join(self._safe_text_list(b.get("pitfalls"), limit=4, item_limit=120)),
                    ]
                ).strip(" |")
                if not text:
                    continue
                d = self._rag_doc_base(
                    world_spec=world_spec,
                    agent_id=agent_id,
                    source_type="method_card",
                    source_id=f"method:{topic}:{idx}",
                    action=action,
                    tags=[topic, str(method_card.get("metric") or "")],
                    quality=0.8,
                )
                d["text"] = text
                docs.append(d)
        else:
            text = self._truncate(json.dumps(method_card, ensure_ascii=False), 4000)
            if text:
                d = self._rag_doc_base(
                    world_spec=world_spec,
                    agent_id=agent_id,
                    source_type="method_card",
                    source_id=f"method:{topic}",
                    action=action,
                    tags=[topic],
                    quality=0.7,
                )
                d["text"] = text
                docs.append(d)
        return docs

    def _rag_docs_from_data_card(
        self,
        *,
        world_spec: Dict[str, Any],
        agent_id: str,
        data_card: Dict[str, Any],
        action: str,
    ) -> List[Dict[str, Any]]:
        if not isinstance(data_card, dict):
            return []
        summary = self._compact_data_card(data_card)
        text = self._truncate(json.dumps(summary, ensure_ascii=False), 8000)
        if not text:
            return []
        d = self._rag_doc_base(
            world_spec=world_spec,
            agent_id=agent_id,
            source_type="data_card",
            source_id=f"data_card:{self._rag_hash(text)[:16]}",
            action=action,
            tags=[str(summary.get("target_column") or "target")],
            quality=0.85,
        )
        d["text"] = text
        return [d]

    def _rag_docs_from_observation(
        self,
        *,
        world_spec: Dict[str, Any],
        agent_id: str,
        observation: Dict[str, Any],
        action: str,
    ) -> List[Dict[str, Any]]:
        if not isinstance(observation, dict):
            return []
        run_id = str(observation.get("run_id") or "").strip()
        source_id = run_id or f"obs:{self._rag_hash(json.dumps(observation, ensure_ascii=False))[:16]}"
        obs_blob = {
            "run_id": observation.get("run_id"),
            "ok": observation.get("ok"),
            "metric_name": observation.get("metric_name"),
            "score_norm": observation.get("score_norm"),
            "dev_score_norm": observation.get("dev_score_norm"),
            "strategy": observation.get("strategy"),
            "error": self._truncate(observation.get("error"), 300),
            "stderr_tail": self._truncate(observation.get("stderr_tail"), 600),
        }
        docs: List[Dict[str, Any]] = []
        d_obs = self._rag_doc_base(
            world_spec=world_spec,
            agent_id=agent_id,
            source_type="observation",
            source_id=source_id,
            action=action,
            tags=[str(observation.get("metric_name") or ""), str(observation.get("solver_mode") or "")],
            quality=0.75 if bool(observation.get("ok")) else 0.55,
        )
        d_obs["text"] = self._truncate(json.dumps(obs_blob, ensure_ascii=False), 2000)
        docs.append(d_obs)

        error_text = str(observation.get("error") or "")
        if error_text or observation.get("stderr_tail"):
            d_diag = self._rag_doc_base(
                world_spec=world_spec,
                agent_id=agent_id,
                source_type="diagnosis",
                source_id=source_id,
                action=action,
                tags=["failure" if not bool(observation.get("ok")) else "success"],
                quality=0.65,
            )
            d_diag["text"] = self._truncate(
                f"error={error_text}\nstderr={observation.get('stderr_tail')}\nstrategy={observation.get('strategy')}",
                2400,
            )
            docs.append(d_diag)
        return docs

    def _rag_docs_from_paper(
        self,
        *,
        world_spec: Dict[str, Any],
        agent_id: str,
        paper: Dict[str, Any],
        paper_id: Optional[str],
        action: str,
    ) -> List[Dict[str, Any]]:
        if not isinstance(paper, dict):
            return []
        source_id = str(paper_id or paper.get("paper_id") or f"paper:{self._rag_hash(json.dumps(paper, ensure_ascii=False))[:16]}")
        docs: List[Dict[str, Any]] = []
        summary = {
            "title": paper.get("title"),
            "abstract": paper.get("abstract"),
            "key_claims": paper.get("key_claims"),
            "limitations": paper.get("limitations"),
            "evidence_map": paper.get("evidence_map"),
            "observation_refs": paper.get("observation_refs"),
        }
        d = self._rag_doc_base(
            world_spec=world_spec,
            agent_id=agent_id,
            source_type="paper",
            source_id=source_id,
            action=action,
            tags=self._safe_text_list(paper.get("citations"), limit=6, item_limit=40),
            quality=0.8,
        )
        d["text"] = self._truncate(json.dumps(summary, ensure_ascii=False), 6000)
        docs.append(d)
        return docs

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
        world_spec = await self.controller.run_environment("science", "get_world_spec")
        await self._append_jsonl(
            self._rag_query_log_path,
            {
                "ts": datetime.utcnow().isoformat() + "Z",
                "episode_id": world_spec.get("episode_id"),
                "task_name": world_spec.get("task_name"),
                "agent_id": agent_id,
                "action": action,
                "run_id": run_id,
                "paper_id": paper_id,
                "query_text_hash": self._rag_hash(query_text),
                "topk": int(topk),
                "latency_ms": float(round(latency_ms, 2)),
                "rag_status": result.get("status"),
                "fallback_reason": result.get("fallback_reason"),
                "result_count": len(result.get("all_results") or []),
                "selected_count": len(result.get("selected") or []),
            },
        )

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
        if not self._audit_io_enable:
            return
        ts = datetime.utcnow().isoformat() + "Z"
        meta = {
            "ts": ts,
            "episode_id": world_spec.get("episode_id"),
            "task_name": world_spec.get("task_name"),
            "agent_id": agent_id,
            "action": action,
            "operation": operation,
            "run_id": run_id,
            "paper_id": paper_id,
            "kind": "rag_io",
        }
        record = {
            "meta": meta,
            "input": self._safe_jsonable(request_payload),
            "output": self._safe_jsonable(response_payload),
        }
        await self._append_jsonl(self._rag_audit_jsonl_path, record)
        await self._append_markdown_audit(
            path=self._rag_audit_md_path,
            title=f"[{ts}] action={action} op={operation} agent={agent_id}",
            meta=meta,
            request_payload=request_payload,
            response_payload=response_payload,
        )

    async def _rag_log_usage(
        self,
        *,
        agent_id: str,
        action: str,
        run_id: Optional[str],
        paper_id: Optional[str],
        result: Dict[str, Any],
    ) -> None:
        world_spec = await self.controller.run_environment("science", "get_world_spec")
        await self._append_jsonl(
            self._rag_usage_log_path,
            {
                "ts": datetime.utcnow().isoformat() + "Z",
                "episode_id": world_spec.get("episode_id"),
                "task_name": world_spec.get("task_name"),
                "agent_id": agent_id,
                "action": action,
                "run_id": run_id,
                "paper_id": paper_id,
                "rag_status": result.get("status"),
                "used_refs": result.get("refs") or [],
                "used_count": len(result.get("refs") or []),
            },
        )

    async def _rag_index_documents(
        self,
        *,
        agent_id: str,
        action: str,
        docs: List[Dict[str, Any]],
        run_id: Optional[str] = None,
        paper_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        world_spec = await self.controller.run_environment("science", "get_world_spec")
        if not self._rag_enable or self._rag_store is None:
            payload = {"ok": False, "status": "disabled", "indexed_points": 0}
        else:
            payload = await self._rag_store.upsert_documents(docs)
        await self._append_jsonl(
            self._rag_index_log_path,
            {
                "ts": datetime.utcnow().isoformat() + "Z",
                "episode_id": world_spec.get("episode_id"),
                "task_name": world_spec.get("task_name"),
                "agent_id": agent_id,
                "action": action,
                "run_id": run_id,
                "paper_id": paper_id,
                "doc_count": len(docs or []),
                "indexed_points": int((payload or {}).get("indexed_points", 0) or 0),
                "rag_status": (payload or {}).get("status"),
                "ok": bool((payload or {}).get("ok", False)),
            },
        )
        sample_docs = []
        for doc in (docs or [])[: self._audit_rag_max_rows]:
            sample_docs.append(
                {
                    "source_type": doc.get("source_type"),
                    "source_id": doc.get("source_id"),
                    "tags": list(doc.get("tags") or []),
                    "quality": doc.get("quality"),
                    "text": self._clip_text_for_audit(doc.get("text"), self._audit_rag_max_chars),
                }
            )
        await self._log_rag_audit(
            world_spec=world_spec,
            agent_id=agent_id,
            action=action,
            operation="index",
            run_id=run_id,
            paper_id=paper_id,
            request_payload={
                "collection": self._rag_collection,
                "doc_count": len(docs or []),
                "docs_sample": sample_docs,
            },
            response_payload={
                "ok": bool((payload or {}).get("ok", False)),
                "status": (payload or {}).get("status"),
                "indexed_points": int((payload or {}).get("indexed_points", 0) or 0),
                "documents": int((payload or {}).get("documents", 0) or 0),
                "error": (payload or {}).get("error"),
            },
        )
        await self._rag_track_runtime_health(
            world_spec=world_spec,
            agent_id=agent_id,
            action=action,
            status=str((payload or {}).get("status") or ""),
            source="index",
        )
        return payload

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
        docs: List[Dict[str, Any]] = []
        if isinstance(data_card, dict):
            docs.extend(self._rag_docs_from_data_card(world_spec=world_spec, agent_id=agent_id, data_card=data_card, action="local"))
        if isinstance(method_card, dict):
            docs.extend(self._rag_docs_from_method_card(world_spec=world_spec, agent_id=agent_id, method_card=method_card, action="local"))
        for note in (notes or [])[-12:]:
            if isinstance(note, dict):
                docs.extend(self._rag_docs_from_note(world_spec=world_spec, agent_id=agent_id, note=note, action="local"))
        for obs in (observations or [])[-12:]:
            if isinstance(obs, dict):
                docs.extend(self._rag_docs_from_observation(world_spec=world_spec, agent_id=agent_id, observation=obs, action="local"))
        if isinstance(paper, dict):
            docs.extend(self._rag_docs_from_paper(world_spec=world_spec, agent_id=agent_id, paper=paper, paper_id=paper_id, action="local"))
        return docs

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
        world_spec = await self.controller.run_environment("science", "get_world_spec")
        local_docs = self._rag_local_docs(
            world_spec=world_spec,
            agent_id=agent_id,
            notes=notes,
            observations=observations,
            data_card=data_card,
            method_card=method_card,
            paper=paper,
            paper_id=paper_id,
        )
        t0 = time.perf_counter()
        if self._rag_retriever is None:
            result = {
                "status": "disabled",
                "fallback_reason": "retriever_unavailable",
                "all_results": [],
                "selected": [],
                "context": "",
                "refs": [],
            }
        else:
            result = await self._rag_retriever.retrieve(
                action=action,
                query_text=query_text,
                topk=self._rag_topk,
                min_score=self._rag_min_score,
                quotas=quotas,
                local_docs=local_docs,
            )
        latency_ms = (time.perf_counter() - t0) * 1000.0
        await self._rag_log_query(
            agent_id=agent_id,
            action=action,
            run_id=run_id,
            paper_id=paper_id,
            query_text=query_text,
            topk=self._rag_topk,
            result=result,
            latency_ms=latency_ms,
        )
        await self._rag_log_usage(
            agent_id=agent_id,
            action=action,
            run_id=run_id,
            paper_id=paper_id,
            result=result,
        )
        selected_rows = []
        for row in (result.get("selected") or [])[: self._audit_rag_max_rows]:
            selected_rows.append(
                {
                    "source_type": row.get("source_type"),
                    "source_id": row.get("source_id"),
                    "score": row.get("score"),
                    "tags": list(row.get("tags") or []),
                    "text": self._clip_text_for_audit(row.get("text"), self._audit_rag_max_chars),
                }
            )
        await self._log_rag_audit(
            world_spec=world_spec,
            agent_id=agent_id,
            action=action,
            operation="query",
            run_id=run_id,
            paper_id=paper_id,
            request_payload={
                "query_text": self._clip_text_for_audit(query_text, self._audit_rag_max_chars),
                "topk": int(self._rag_topk),
                "min_score": float(self._rag_min_score),
                "quotas": dict(quotas or {}),
                "local_docs_count": len(local_docs or []),
            },
            response_payload={
                "status": result.get("status"),
                "fallback_reason": result.get("fallback_reason"),
                "result_count": len(result.get("all_results") or []),
                "selected_count": len(result.get("selected") or []),
                "refs": list(result.get("refs") or []),
                "context": self._clip_text_for_audit(result.get("context"), self._audit_rag_max_chars),
                "selected_rows": selected_rows,
            },
        )
        await self._rag_track_runtime_health(
            world_spec=world_spec,
            agent_id=agent_id,
            action=action,
            status=str(result.get("status") or ""),
            source="query",
        )
        return result

    async def _rag_bootstrap_episode_knowledge(
        self,
        *,
        agent_id: str,
        world_spec: Dict[str, Any],
        data_card: Optional[Dict[str, Any]],
        method_card: Optional[Dict[str, Any]],
    ) -> None:
        ep = int(world_spec.get("episode_id") or 0)
        if ep <= 0 or self._rag_bootstrap_episode == ep:
            return
        docs: List[Dict[str, Any]] = []
        if isinstance(data_card, dict):
            docs.extend(self._rag_docs_from_data_card(world_spec=world_spec, agent_id=agent_id, data_card=data_card, action="bootstrap"))
        if isinstance(method_card, dict):
            docs.extend(
                self._rag_docs_from_method_card(world_spec=world_spec, agent_id=agent_id, method_card=method_card, action="bootstrap")
            )
        if docs:
            await self._rag_index_documents(agent_id=agent_id, action="bootstrap", docs=docs)
        self._rag_bootstrap_episode = ep

    def _format_rag_prompt_block(self, *, result: Dict[str, Any]) -> Dict[str, Any]:
        refs = list(result.get("refs") or [])
        context = str(result.get("context") or "").strip()
        if not context:
            context = "(no high-confidence retrieval results)"
        return {
            "context": context,
            "refs": refs,
            "status": str(result.get("status") or "empty"),
            "usage_constraint": "Prioritize retrieved evidence. Do not fabricate citations or run references.",
        }

    def _truncate(self, text: Any, limit: int = 280) -> str:
        value = str(text or "")
        if len(value) <= limit:
            return value
        return value[: limit - 3] + "..."

    def _safe_task_types(self, values: Any, *, fallback: Optional[List[str]] = None) -> List[str]:
        allowed = {
            "read",
            "prepare_data",
            "profile_data",
            "retrieve_literature",
            "hypothesize",
            "experiment",
            "write",
            "review",
            "replicate",
            "verify_strength",
            "verify_issue",
        }
        result: List[str] = []
        if isinstance(values, list):
            for item in values:
                name = str(item or "").strip().lower()
                if name in allowed and name not in result:
                    result.append(name)
        if result:
            return result
        return list(fallback or [])

    def _safe_text_list(self, values: Any, *, limit: int = 5, item_limit: int = 220) -> List[str]:
        if not isinstance(values, list):
            return []
        out: List[str] = []
        for item in values[: max(0, limit)]:
            text = str(item or "").strip()
            if text:
                out.append(self._truncate(text, item_limit))
        return out

    def _text_tokens(self, text: str) -> List[str]:
        return [tok for tok in re.split(r"[^a-z0-9_]+", (text or "").lower()) if len(tok) >= 2]

    def _counter_cosine(self, a: Counter, b: Counter) -> float:
        if not a or not b:
            return 0.0
        dot = 0.0
        for k, v in a.items():
            dot += float(v) * float(b.get(k, 0.0))
        if dot <= 0:
            return 0.0
        na = math.sqrt(sum(float(v) * float(v) for v in a.values()))
        nb = math.sqrt(sum(float(v) * float(v) for v in b.values()))
        if na <= 0.0 or nb <= 0.0:
            return 0.0
        return max(0.0, min(1.0, dot / (na * nb)))

    def _note_to_text(self, note: Dict[str, Any]) -> str:
        parts: List[str] = [str(note.get("topic") or "")]
        for hint in (note.get("hints") or [])[:6]:
            parts.append(str(hint))
        for card in (note.get("cards") or [])[:8]:
            if not isinstance(card, dict):
                continue
            parts.append(str(card.get("title") or ""))
            parts.append(str(card.get("text") or ""))
        return "\n".join(parts)

    def _has_method_signal(self, text: str) -> bool:
        s = (text or "").lower()
        keywords = (
            "baseline",
            "method",
            "ablation",
            "protocol",
            "pitfall",
            "mase",
            "timeseriessplit",
            "submission",
            "evaluation",
            "scoring_column",
        )
        return any(k in s for k in keywords)

    async def _tei_embed(self, text: str) -> Optional[List[float]]:
        if not self._reward_tei_url:
            return None
        payload = json.dumps({"inputs": text}).encode("utf-8")
        req = urllib.request.Request(
            self._reward_tei_url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        def _send() -> Optional[List[float]]:
            with urllib.request.urlopen(req, timeout=6) as resp:
                body = resp.read().decode("utf-8", errors="replace")
            data = json.loads(body)
            if isinstance(data, list) and data and isinstance(data[0], (int, float)):
                return [float(x) for x in data]
            if isinstance(data, list) and data and isinstance(data[0], list):
                return [float(x) for x in data[0]]
            return None
        try:
            return await asyncio.to_thread(_send)
        except Exception:
            return None

    async def _qdrant_max_similarity(self, vector: List[float]) -> Optional[float]:
        if not self._reward_qdrant_url or not self._reward_qdrant_collection or not vector:
            return None
        url = f"{self._reward_qdrant_url}/collections/{self._reward_qdrant_collection}/points/search"
        headers = {"Content-Type": "application/json"}
        if self._reward_qdrant_api_key:
            headers["api-key"] = self._reward_qdrant_api_key
        payload = {
            "vector": vector,
            "limit": 1,
            "with_payload": False,
            "with_vector": False,
        }
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        def _send() -> Optional[float]:
            with urllib.request.urlopen(req, timeout=6) as resp:
                body = resp.read().decode("utf-8", errors="replace")
            data = json.loads(body)
            result = data.get("result")
            if isinstance(result, list) and result:
                score = (result[0] or {}).get("score")
                if isinstance(score, (int, float)):
                    return float(score)
            return None
        try:
            return await asyncio.to_thread(_send)
        except Exception:
            return None

    async def _compute_read_reward(
        self,
        *,
        existing_notes: List[Dict[str, Any]],
        new_note: Dict[str, Any],
    ) -> Dict[str, Any]:
        new_text = self._note_to_text(new_note)
        new_counter = Counter(self._text_tokens(new_text))
        local_max_sim = 0.0
        for note in (existing_notes or [])[-24:]:
            if not isinstance(note, dict):
                continue
            sim = self._counter_cosine(new_counter, Counter(self._text_tokens(self._note_to_text(note))))
            if sim > local_max_sim:
                local_max_sim = sim

        remote_max_sim = None
        if self._dense_reward_enable and self._reward_tei_url and self._reward_qdrant_url:
            vec = await self._tei_embed(new_text)
            if isinstance(vec, list) and vec:
                remote_max_sim = await self._qdrant_max_similarity(vec)

        max_sim = max(local_max_sim, float(remote_max_sim or 0.0))
        novelty = max(0.0, 1.0 - max_sim)
        method_bonus = self._read_method_bonus if self._has_method_signal(new_text) else 0.0
        if max_sim >= self._read_dup_threshold:
            reward = 0.0
        else:
            reward = self._read_reward_base + (self._read_reward_alpha * novelty) + method_bonus
            reward = max(0.0, min(self._read_reward_max, reward))
        return {
            "reward": float(reward),
            "novelty": float(novelty),
            "local_similarity": float(local_max_sim),
            "qdrant_similarity": float(remote_max_sim) if isinstance(remote_max_sim, (int, float)) else None,
            "method_bonus": float(method_bonus),
            "duplicate": bool(max_sim >= self._read_dup_threshold),
        }

    def _hypothesis_feasibility(self, world_spec: Dict[str, Any], plan_spec: Dict[str, Any]) -> Dict[str, Any]:
        plan_text = json.dumps(plan_spec or {}, ensure_ascii=False).lower()
        schema_markers = (
            "scoring_column",
            "task_manifest",
            "list",
            "target_column_hint",
            "submission csv",
        )
        schema_safe = sum(1 for m in schema_markers if m in plan_text) >= 2
        resource_markers = ("sample", "batch", "chunk", "window", "stream")
        resource_by_text = any(m in plan_text for m in resource_markers)
        solver_spec = (plan_spec or {}).get("solver_spec") if isinstance((plan_spec or {}).get("solver_spec"), dict) else {}
        preprocess = solver_spec.get("preprocess") if isinstance(solver_spec.get("preprocess"), dict) else {}
        max_features = preprocess.get("max_features")
        resource_by_config = isinstance(max_features, (int, float)) and float(max_features) <= 50000
        schema_bonus = self._hypothesis_schema_bonus if schema_safe else 0.0
        resource_bonus = self._hypothesis_resource_bonus if (resource_by_text or resource_by_config) else 0.0
        feasibility = schema_bonus + resource_bonus
        reward = max(-0.02, min(0.08, 0.10 * feasibility - 0.01))
        return {
            "feasibility_score": float(feasibility),
            "schema_safe": bool(schema_safe),
            "resource_safe": bool(resource_by_text or resource_by_config),
            "schema_bonus": float(schema_bonus),
            "resource_bonus": float(resource_bonus),
            "reward": float(reward),
            "code_memory_mb": world_spec.get("code_memory_mb"),
        }

    async def _vdh_qdrant_schema_constraints(self, world_spec: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if not self._vdh_qdrant_enable or not self._vdh_qdrant_url or not self._vdh_qdrant_collection:
            return None
        task_name = str(world_spec.get("task_name") or "").strip()
        if not task_name:
            return None
        url = f"{self._vdh_qdrant_url}/collections/{self._vdh_qdrant_collection}/points/scroll"
        headers = {"Content-Type": "application/json"}
        if self._vdh_qdrant_api_key:
            headers["api-key"] = self._vdh_qdrant_api_key
        payload = {
            "with_payload": True,
            "with_vector": False,
            "limit": 1,
            "filter": {
                "must": [
                    {
                        "key": "task_name",
                        "match": {"value": task_name},
                    }
                ]
            },
        }
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )

        def _send() -> Optional[Dict[str, Any]]:
            with urllib.request.urlopen(req, timeout=6) as resp:
                body = resp.read().decode("utf-8", errors="replace")
            data = json.loads(body)
            result = data.get("result")
            points = result.get("points") if isinstance(result, dict) else result
            if isinstance(points, list) and points:
                payload_obj = (points[0] or {}).get("payload")
                if isinstance(payload_obj, dict):
                    return payload_obj
            return None

        try:
            return await asyncio.to_thread(_send)
        except Exception:
            return None

    def _vdh_constraints_from_manifest_file(self, world_spec: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        task_path = str(world_spec.get("task_path") or "").strip()
        if not task_path:
            return None
        candidates = [
            os.path.join(task_path, ".task_manifest.json"),
            os.path.join(task_path, "task_manifest.json"),
            os.path.join(task_path, "metadata.yaml"),
            os.path.join(task_path, "metadata.yml"),
        ]
        for path in candidates:
            if not os.path.exists(path):
                continue
            try:
                if path.endswith(".json"):
                    with open(path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                else:
                    import yaml  # local import to keep optional dependency bounded

                    with open(path, "r", encoding="utf-8") as f:
                        data = yaml.safe_load(f) or {}
                if not isinstance(data, dict):
                    continue
                return data
            except Exception:
                continue
        return None

    def _vdh_normalize_constraints(
        self,
        *,
        source: str,
        raw: Optional[Dict[str, Any]],
        world_spec: Dict[str, Any],
        notes: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        warnings: List[str] = []
        raw = raw if isinstance(raw, dict) else {}
        scoring = raw.get("scoring_column")
        if not isinstance(scoring, list):
            info = raw.get("logging_info") if isinstance(raw.get("logging_info"), dict) else {}
            scoring = info.get("scoring_column")
        if not isinstance(scoring, list):
            scoring = []
        scoring_is_list = bool(isinstance(scoring, list) and len(scoring) > 0)

        target_hint = ""
        if isinstance(raw.get("target_column"), str):
            target_hint = str(raw.get("target_column"))
        elif isinstance(raw.get("label_column"), str):
            target_hint = str(raw.get("label_column"))
        elif isinstance(raw.get("submission"), dict) and isinstance((raw.get("submission") or {}).get("target"), str):
            target_hint = str((raw.get("submission") or {}).get("target"))
        elif scoring_is_list:
            target_hint = str(scoring[0])
        metric = str(raw.get("metric") or world_spec.get("metric") or "").strip()
        submission_requirements = []
        if isinstance(raw.get("submission_requirements"), list):
            submission_requirements = [str(x) for x in raw.get("submission_requirements")[:8]]
        if not submission_requirements and isinstance(raw.get("submission"), dict):
            submission_requirements = [self._truncate(json.dumps(raw.get("submission"), ensure_ascii=False), 280)]
        if not scoring_is_list and notes:
            notes_text = " ".join(self._note_to_text(n) for n in notes[-6:] if isinstance(n, dict)).lower()
            if "scoring_column" in notes_text and "list" in notes_text:
                scoring_is_list = True
                warnings.append("inferred_scoring_column_list_from_notes")
        if not target_hint:
            target_hint = "target"
            warnings.append("target_column_hint_fallback")
        return {
            "ok": True,
            "source": source,
            "constraints": {
                "scoring_column_is_list": bool(scoring_is_list),
                "target_column_hint": target_hint,
                "metric": metric,
                "submission_requirements": submission_requirements,
            },
            "warnings": warnings,
        }

    async def _vdh_metadata_alignment(
        self,
        *,
        world_spec: Dict[str, Any],
        notes: Optional[List[Dict[str, Any]]],
        plan_spec: Dict[str, Any],
    ) -> Dict[str, Any]:
        source = "fallback"
        raw = None
        if self._vdh_qdrant_enable:
            raw = await self._vdh_qdrant_schema_constraints(world_spec)
            if isinstance(raw, dict):
                source = "qdrant"
        if not isinstance(raw, dict):
            raw = self._vdh_constraints_from_manifest_file(world_spec)
            if isinstance(raw, dict):
                source = "manifest"
        normalized = self._vdh_normalize_constraints(source=source, raw=raw, world_spec=world_spec, notes=notes)

        constraints = normalized.get("constraints") if isinstance(normalized.get("constraints"), dict) else {}
        scoring_is_list = bool(constraints.get("scoring_column_is_list", False))
        plan_text = json.dumps(plan_spec or {}, ensure_ascii=False).lower()
        target_cols = (plan_spec or {}).get("target_cols")
        solver_spec = (plan_spec or {}).get("solver_spec") if isinstance((plan_spec or {}).get("solver_spec"), dict) else {}
        solver_target_cols = solver_spec.get("target_cols")
        handles_list = False
        if isinstance(target_cols, list) and target_cols:
            handles_list = True
        if isinstance(solver_target_cols, list) and solver_target_cols:
            handles_list = True
        markers = (
            "scoring_column[0]",
            "scoring_column'][0]",
            "manifest['scoring_column'][0]",
            "manifest[\"scoring_column\"][0]",
            "target_cols",
        )
        if any(m in plan_text for m in markers):
            handles_list = True
        errors: List[str] = []
        if scoring_is_list and not handles_list:
            errors.append("scoring_column_is_list_but_plan_not_handled")
        normalized["ok"] = len(errors) == 0
        normalized["errors"] = errors
        normalized["plan_handles_scoring_list"] = bool(handles_list)
        return normalized

    def _vdh_plan_validator(
        self,
        *,
        world_spec: Dict[str, Any],
        plan_spec: Dict[str, Any],
        data_card: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        errors: List[str] = []
        solver_spec = (plan_spec or {}).get("solver_spec")
        if not isinstance(solver_spec, dict):
            errors.append("solver_spec must be object")
            solver_spec = {}
        for key in ("input_columns", "target_cols"):
            if key in plan_spec and not isinstance(plan_spec.get(key), list):
                errors.append(f"{key} must be list")
            if key in solver_spec and not isinstance(solver_spec.get(key), list):
                errors.append(f"solver_spec.{key} must be list")

        batch_size = plan_spec.get("batch_size", solver_spec.get("batch_size", 32))
        if not isinstance(batch_size, (int, float)) or float(batch_size) <= 0:
            errors.append("batch_size must be positive number")
            batch_size = 32
        sample_ratio = plan_spec.get("sample_ratio", solver_spec.get("sample_ratio"))
        if sample_ratio is not None:
            if not isinstance(sample_ratio, (int, float)) or not (0 < float(sample_ratio) <= 1.0):
                errors.append("sample_ratio must be in (0,1]")
        max_features = None
        preprocess = solver_spec.get("preprocess") if isinstance(solver_spec.get("preprocess"), dict) else {}
        if "max_features" in preprocess:
            max_features = preprocess.get("max_features")
            if not isinstance(max_features, (int, float)) or float(max_features) <= 0:
                errors.append("preprocess.max_features must be positive number")

        model_family = str(solver_spec.get("model_family") or plan_spec.get("model_family") or "").strip().lower()
        model_param_defaults = {
            "tfidf_logreg": 6_000_000,
            "linear_svc": 5_000_000,
            "tfidf_ridge": 4_000_000,
            "naive_series": 500_000,
        }
        model_params = plan_spec.get("model_params", solver_spec.get("model_params"))
        if not isinstance(model_params, (int, float)):
            model_params = model_param_defaults.get(model_family, 3_000_000)
        seq_len = plan_spec.get("sequence_length", solver_spec.get("sequence_length", 128))
        if not isinstance(seq_len, (int, float)) or float(seq_len) <= 0:
            seq_len = 128

        split_stats = (data_card or {}).get("split_stats") if isinstance((data_card or {}).get("split_stats"), dict) else {}
        approx_rows = 0
        for info in split_stats.values():
            if isinstance(info, dict) and isinstance(info.get("rows"), (int, float)):
                approx_rows += int(info.get("rows"))
        if approx_rows <= 0:
            approx_rows = 145_000
        effective_rows = int(approx_rows * float(sample_ratio if isinstance(sample_ratio, (int, float)) else 1.0))
        estimated_mb = (
            512.0
            + float(batch_size) * (8.0 + 0.02 * float(seq_len))
            + (float(model_params) / 1_000_000.0) * 18.0
            + (float(effective_rows) / 1000.0) * 0.04
        )
        if not isinstance(sample_ratio, (int, float)) and approx_rows >= 100_000:
            estimated_mb += 1400.0

        limit_mb = world_spec.get("code_memory_mb")
        if not isinstance(limit_mb, (int, float)) or float(limit_mb) <= 0:
            limit_mb = 8192.0
        ratio = float(estimated_mb) / float(limit_mb)
        risk = "low"
        if ratio >= self._vdh_oom_ratio_threshold:
            risk = "high"
            errors.append("potential_oom")
        elif ratio >= 0.7:
            risk = "medium"

        return {
            "ok": len(errors) == 0,
            "errors": errors,
            "resource_estimate": {
                "estimated_mb": round(float(estimated_mb), 2),
                "limit_mb": round(float(limit_mb), 2),
                "ratio": round(float(ratio), 4),
                "risk": risk,
                "rows": int(approx_rows),
                "effective_rows": int(effective_rows),
            },
        }

    async def _vdh_embed_text(self, text: str) -> Optional[List[float]]:
        endpoint = self._vdh_tei_url or self._reward_tei_url
        if not self._vdh_tei_enable or not endpoint:
            return None
        payload = json.dumps({"inputs": text}).encode("utf-8")
        req = urllib.request.Request(
            endpoint,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        def _send() -> Optional[List[float]]:
            with urllib.request.urlopen(req, timeout=6) as resp:
                body = resp.read().decode("utf-8", errors="replace")
            data = json.loads(body)
            if isinstance(data, list) and data and isinstance(data[0], (int, float)):
                return [float(x) for x in data]
            if isinstance(data, list) and data and isinstance(data[0], list):
                return [float(x) for x in data[0]]
            return None

        try:
            return await asyncio.to_thread(_send)
        except Exception:
            return None

    def _vector_cosine(self, a: Optional[List[float]], b: Optional[List[float]]) -> float:
        if not isinstance(a, list) or not isinstance(b, list) or not a or not b:
            return 0.0
        n = min(len(a), len(b))
        if n <= 0:
            return 0.0
        dot = 0.0
        na = 0.0
        nb = 0.0
        for i in range(n):
            x = float(a[i])
            y = float(b[i])
            dot += x * y
            na += x * x
            nb += y * y
        if na <= 0 or nb <= 0:
            return 0.0
        return max(0.0, min(1.0, dot / ((na ** 0.5) * (nb ** 0.5))))

    async def _vdh_evidence_coverage(
        self,
        *,
        hypothesis: List[str],
        plan_spec: Dict[str, Any],
        notes: List[Dict[str, Any]],
        observations: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        hypo_parts: List[str] = []
        for tag in (hypothesis or [])[:10]:
            hypo_parts.append(str(tag))
        for key in ("strategy", "rationale", "risk", "schema_assumptions", "memory_safety", "evidence_refs"):
            val = (plan_spec or {}).get(key)
            if isinstance(val, str):
                hypo_parts.append(val)
            elif isinstance(val, list):
                hypo_parts.extend([str(x) for x in val[:8]])
            elif isinstance(val, dict):
                hypo_parts.append(json.dumps(val, ensure_ascii=False))
        hypothesis_text = "\n".join([x for x in hypo_parts if str(x).strip()]).strip()

        evidence_chunks: List[str] = []
        for note in (notes or [])[-12:]:
            if isinstance(note, dict):
                evidence_chunks.append(self._note_to_text(note))
        for obs in (observations or [])[-10:]:
            if not isinstance(obs, dict):
                continue
            blob = {
                "strategy": obs.get("strategy"),
                "ok": obs.get("ok"),
                "error": obs.get("error"),
                "dev_score_norm": obs.get("dev_score_norm"),
                "score_norm": obs.get("score_norm"),
            }
            evidence_chunks.append(json.dumps(blob, ensure_ascii=False))
        evidence_text = "\n".join([x for x in evidence_chunks if x.strip()])

        h_counter = Counter(self._text_tokens(hypothesis_text))
        e_counter = Counter(self._text_tokens(evidence_text))
        token_overlap = self._counter_cosine(h_counter, e_counter)
        keyword_cov = 0.0
        h_tokens = set(h_counter.keys())
        if h_tokens:
            keyword_cov = len(h_tokens & set(e_counter.keys())) / max(1, len(h_tokens))
        fallback_score = max(0.0, min(1.0, 0.5 * token_overlap + 0.5 * keyword_cov))

        vector_score = None
        h_vec = await self._vdh_embed_text(hypothesis_text) if hypothesis_text else None
        if isinstance(h_vec, list) and h_vec:
            sims: List[float] = []
            for chunk in evidence_chunks[-8:]:
                e_vec = await self._vdh_embed_text(chunk)
                if isinstance(e_vec, list) and e_vec:
                    sims.append(self._vector_cosine(h_vec, e_vec))
            if sims:
                vector_score = max(sims)

        if isinstance(vector_score, (int, float)):
            coverage_score = max(0.0, min(1.0, 0.6 * float(vector_score) + 0.4 * fallback_score))
            source = "tei+token"
        else:
            coverage_score = fallback_score
            source = "token_fallback"

        return {
            "ok": bool(coverage_score >= self._vdh_evidence_threshold),
            "coverage_score": float(round(coverage_score, 4)),
            "threshold": float(self._vdh_evidence_threshold),
            "source": source,
            "token_overlap": float(round(token_overlap, 4)),
            "keyword_coverage": float(round(keyword_cov, 4)),
            "vector_similarity": float(round(float(vector_score), 4)) if isinstance(vector_score, (int, float)) else None,
            "errors": [] if coverage_score >= self._vdh_evidence_threshold else ["evidence_coverage_below_threshold"],
        }

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
        gate_a = await self._vdh_metadata_alignment(world_spec=world_spec, notes=notes, plan_spec=plan_spec)
        gate_b = self._vdh_plan_validator(world_spec=world_spec, plan_spec=plan_spec, data_card=data_card)
        gate_c = await self._vdh_evidence_coverage(
            hypothesis=hypothesis,
            plan_spec=plan_spec,
            notes=notes,
            observations=observations,
        )
        errors: List[str] = []
        for gate_name, gate_obj in (("gate_a", gate_a), ("gate_b", gate_b), ("gate_c", gate_c)):
            if not bool((gate_obj or {}).get("ok", False)):
                for e in (gate_obj or {}).get("errors", []) or [f"{gate_name}_failed"]:
                    errors.append(f"{gate_name}:{e}")
        return {
            "gate_a": gate_a,
            "gate_b": gate_b,
            "gate_c": gate_c,
            "final_ok": len(errors) == 0,
            "failures": errors,
            "policy": self._vdh_gate_policy,
        }

    async def _enqueue_vdh_recovery_tasks(self, *, vdh_report: Dict[str, Any]) -> Dict[str, Any]:
        failures = [str(x) for x in (vdh_report or {}).get("failures", [])]
        required: List[Dict[str, Any]] = []
        if any("gate_a" in f for f in failures):
            required.append({"task_type": "read", "payload": {"topic": "task_requirements"}})
        if any("gate_c" in f for f in failures):
            required.append({"task_type": "retrieve_literature", "payload": {"topic": "task_baselines"}})
            required.append({"task_type": "read", "payload": {"topic": "task_requirements"}})
        if any("potential_oom" in f or "gate_b" in f for f in failures):
            required.append({"task_type": "read", "payload": {"topic": "memory_safety"}})

        open_list = await self.controller.run_environment("science", "task_list", status="open")
        claimed_list = await self.controller.run_environment("science", "task_list", status="claimed")
        known_types = set()
        for listing in (open_list, claimed_list):
            tasks = (listing or {}).get("tasks", []) if isinstance(listing, dict) else []
            for task in tasks:
                known_types.add(str((task or {}).get("task_type") or ""))

        created: List[str] = []
        skipped: List[Dict[str, Any]] = []
        dedup_types: set[str] = set()
        for item in required:
            task_type = str(item.get("task_type") or "")
            if not task_type or task_type in dedup_types:
                continue
            if task_type == "retrieve_literature":
                paused = await self._rag_retrieve_recovery_paused()
                if paused:
                    skipped.append(
                        {
                            "task_type": task_type,
                            "reason": "rag_degraded_pause_active",
                            "pause_until_tick": int(self._rag_degraded_pause_until_tick or 0),
                        }
                    )
                    continue
            dedup_types.add(task_type)
            if task_type in known_types:
                continue
            created_task = await self.controller.run_environment(
                "science",
                "task_create",
                task_type=task_type,
                payload=dict(item.get("payload") or {}),
                priority=9,
            )
            if isinstance(created_task, dict) and created_task.get("ok"):
                tid = str((created_task.get("task") or {}).get("task_id") or "")
                if tid:
                    created.append(tid)
                known_types.add(task_type)
        return {"created_task_ids": created, "requested": required, "skipped": skipped}

    def _experiment_error_flags(self, result: Dict[str, Any]) -> Dict[str, bool]:
        err = str((result or {}).get("error") or "")
        stderr = str((result or {}).get("stderr_tail") or "")
        merged = (err + "\n" + stderr).lower()
        return {
            "oom": ("killed" in merged) or ("out of memory" in merged) or ("oom" in merged),
            "typeerror": "typeerror" in merged,
        }

    def _is_first_pass_success(self, *, code_attempts: Any, ok: bool) -> bool:
        if not ok:
            return False
        if not isinstance(code_attempts, list) or not code_attempts:
            return False
        first = code_attempts[0] if isinstance(code_attempts[0], dict) else {}
        first_result = first.get("result") if isinstance(first.get("result"), dict) else {}
        first_ok = bool(first_result.get("ok", False))
        failed_before_success = any(
            isinstance(a, dict)
            and isinstance(a.get("result"), dict)
            and not bool((a.get("result") or {}).get("ok", False))
            for a in code_attempts[:1]
        )
        return bool(first_ok and not failed_before_success)

    def _estimate_vram_efficiency(self, *, result: Dict[str, Any], world_spec: Dict[str, Any]) -> Optional[float]:
        limit_mb = world_spec.get("code_memory_mb")
        peak_mb = (result or {}).get("peak_memory_mb")
        if not isinstance(limit_mb, (int, float)) or float(limit_mb) <= 0:
            return None
        if not isinstance(peak_mb, (int, float)):
            return None
        eff = (float(limit_mb) - float(peak_mb)) / float(limit_mb)
        return max(0.0, min(1.0, eff))

    def _compact_data_card(self, data_card: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if not isinstance(data_card, dict):
            return {}
        split_stats = data_card.get("split_stats") if isinstance(data_card.get("split_stats"), dict) else {}
        sampled_rows = data_card.get("sampled_rows") if isinstance(data_card.get("sampled_rows"), dict) else {}
        schema = data_card.get("schema") if isinstance(data_card.get("schema"), list) else []
        schema_short = []
        for col in schema[:8]:
            if not isinstance(col, dict):
                continue
            schema_short.append(
                {
                    "name": col.get("name"),
                    "dtype": col.get("dtype"),
                    "missing_ratio": col.get("missing_ratio"),
                    "unique": col.get("unique"),
                }
            )
        label_profile = data_card.get("label_profile") if isinstance(data_card.get("label_profile"), dict) else {}
        info_dyn = data_card.get("information_dynamics") if isinstance(data_card.get("information_dynamics"), dict) else {}
        dist_stab = data_card.get("distribution_stability") if isinstance(data_card.get("distribution_stability"), dict) else {}
        quality_diag = data_card.get("quality_diagnostics") if isinstance(data_card.get("quality_diagnostics"), dict) else {}
        task_priors = data_card.get("task_priors") if isinstance(data_card.get("task_priors"), dict) else {}
        naive_baseline = data_card.get("naive_baseline") if isinstance(data_card.get("naive_baseline"), dict) else {}

        top_assoc = []
        for item in (info_dyn.get("feature_target_association") or [])[:5]:
            if not isinstance(item, dict):
                continue
            top_assoc.append(
                {
                    "feature": item.get("feature"),
                    "abs_corr": item.get("abs_corr"),
                    "mutual_info": item.get("mutual_info"),
                    "feature_source": item.get("feature_source"),
                }
            )
        top_shift = []
        for item in (dist_stab.get("train_test_shift") or [])[:5]:
            if not isinstance(item, dict):
                continue
            top_shift.append(
                {
                    "feature": item.get("feature"),
                    "psi": item.get("psi"),
                    "ks_stat": item.get("ks_stat"),
                }
            )
        quality_hot = []
        for item in (quality_diag.get("numeric_distribution") or [])[:5]:
            if not isinstance(item, dict):
                continue
            quality_hot.append(
                {
                    "feature": item.get("feature"),
                    "skewness": item.get("skewness"),
                    "iqr_outlier_ratio": item.get("iqr_outlier_ratio"),
                }
            )

        prior_items = []
        for item in (task_priors.get("priors") or [])[:3]:
            if not isinstance(item, dict):
                continue
            prior_items.append(
                {
                    "domain": item.get("domain"),
                    "recommended_features": self._safe_text_list(item.get("recommended_features"), limit=4, item_limit=120),
                    "unit_checks": self._safe_text_list(item.get("unit_checks"), limit=3, item_limit=120),
                    "recommended_protocol": self._safe_text_list(item.get("recommended_protocol"), limit=4, item_limit=120),
                }
            )
        return {
            "target_column": data_card.get("target_column"),
            "split_stats": split_stats,
            "sampled_rows": sampled_rows,
            "label_profile": label_profile,
            "schema": schema_short,
            "information_dynamics": {
                "summary": info_dyn.get("summary"),
                "top_feature_association": top_assoc,
                "leakage_suspects": info_dyn.get("leakage_suspects"),
                "multicollinearity_pairs": (info_dyn.get("multicollinearity_pairs") or [])[:5],
            },
            "distribution_stability": {
                "summary": dist_stab.get("summary"),
                "top_shift_features": top_shift,
                "severe_shift_features": (dist_stab.get("severe_shift_features") or [])[:5],
            },
            "quality_diagnostics": {
                "summary": quality_diag.get("summary"),
                "top_numeric_flags": quality_hot,
            },
            "task_priors": {
                "dataset": task_priors.get("dataset"),
                "category": task_priors.get("category"),
                "priors": prior_items,
            },
            "naive_baseline": {
                "available": naive_baseline.get("available"),
                "best": naive_baseline.get("best"),
                "candidates": (naive_baseline.get("candidates") or [])[:3],
            },
            "risk_flags": self._safe_text_list(data_card.get("risk_flags"), limit=6, item_limit=120),
        }

    def _compact_method_card(self, method_card: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if not isinstance(method_card, dict):
            return {}
        baselines = []
        for item in (method_card.get("recommended_baselines") or [])[:4]:
            if not isinstance(item, dict):
                continue
            baselines.append(
                {
                    "name": item.get("name"),
                    "use_when": self._truncate(item.get("use_when"), 120),
                    "key_steps": self._safe_text_list(item.get("key_steps"), limit=4, item_limit=140),
                    "pitfalls": self._safe_text_list(item.get("pitfalls"), limit=4, item_limit=140),
                }
            )
        return {
            "topic": method_card.get("topic"),
            "metric": method_card.get("metric"),
            "category": method_card.get("category"),
            "recommended_baselines": baselines,
            "evaluation_protocol": self._safe_text_list(method_card.get("evaluation_protocol"), limit=5, item_limit=140),
            "common_pitfalls": self._safe_text_list(method_card.get("common_pitfalls"), limit=6, item_limit=140),
        }

    def _default_solver_plan(self, world_spec: Dict[str, Any]) -> Dict[str, Any]:
        metric = str(world_spec.get("metric") or "").lower()
        category = str(world_spec.get("category") or "").lower()
        model_family = "tfidf_logreg"
        if any(tok in metric for tok in ("mae", "mase", "meanabsoluteerror", "spearman")):
            model_family = "tfidf_ridge"
        if "time series" in category:
            model_family = "naive_series"
        return {
            "strategy": "iterative_solver_baseline",
            "solver_spec": {
                "model_family": model_family,
                "seed": 42,
                "preprocess": {"max_features": 50000, "ngram_range": [1, 2], "min_df": 1},
                "hyperparams": {"C": 1.0, "max_iter": 2000, "class_weight": "balanced", "alpha": 1.0},
                "input_columns": [],
            },
            "rationale": [
                "fit task metric with reproducible baseline",
                "optimize by evidence-driven iteration on dev score",
            ],
            "risk": ["submission format mismatch", "overfitting to random seed"],
            "experiment_protocol": {
                "primary_knob": "solver_hyperparams",
                "ablation_axis": "model_family",
                "format_checks": ["submission csv schema", "metric-specific output constraints"],
            },
            "replication_plan": ["rerun best config on alternate seeds and compare normalized score delta"],
        }

    def _merge_solver_plan(self, base_plan: Dict[str, Any], candidate: Dict[str, Any]) -> Dict[str, Any]:
        merged = dict(base_plan or {})
        if not isinstance(candidate, dict):
            return merged
        strategy = candidate.get("strategy")
        if isinstance(strategy, str) and strategy.strip():
            merged["strategy"] = strategy.strip()[:80]
        for key in ("rationale", "risk", "replication_plan"):
            if isinstance(candidate.get(key), list):
                merged[key] = self._safe_text_list(candidate.get(key), limit=6, item_limit=220)
        if isinstance(candidate.get("experiment_protocol"), dict):
            ep = candidate.get("experiment_protocol") or {}
            merged["experiment_protocol"] = {
                "primary_knob": self._truncate(ep.get("primary_knob"), 120),
                "ablation_axis": self._truncate(ep.get("ablation_axis"), 120),
                "format_checks": self._safe_text_list(ep.get("format_checks"), limit=6, item_limit=180),
            }
        solver_spec = merged.get("solver_spec") if isinstance(merged.get("solver_spec"), dict) else {}
        cand_solver = candidate.get("solver_spec") if isinstance(candidate.get("solver_spec"), dict) else None
        if isinstance(cand_solver, dict):
            solver_spec = dict(solver_spec)
            model_family = cand_solver.get("model_family")
            if isinstance(model_family, str) and model_family.strip():
                solver_spec["model_family"] = model_family.strip()[:80]
            seed = cand_solver.get("seed")
            if isinstance(seed, int):
                solver_spec["seed"] = int(seed)
            if isinstance(cand_solver.get("input_columns"), list):
                solver_spec["input_columns"] = [str(c) for c in cand_solver.get("input_columns") if str(c).strip()][:16]
            if isinstance(cand_solver.get("preprocess"), dict):
                pp_base = solver_spec.get("preprocess") if isinstance(solver_spec.get("preprocess"), dict) else {}
                pp = dict(pp_base)
                for k in ("max_features", "min_df"):
                    val = cand_solver["preprocess"].get(k)
                    if isinstance(val, int) and val > 0:
                        pp[k] = int(val)
                ng = cand_solver["preprocess"].get("ngram_range")
                if isinstance(ng, (list, tuple)) and len(ng) == 2:
                    try:
                        pp["ngram_range"] = [max(1, int(ng[0])), max(1, int(ng[1]))]
                    except Exception:
                        pass
                solver_spec["preprocess"] = pp
            if isinstance(cand_solver.get("hyperparams"), dict):
                hp_base = solver_spec.get("hyperparams") if isinstance(solver_spec.get("hyperparams"), dict) else {}
                hp = dict(hp_base)
                for k in ("C", "alpha"):
                    val = cand_solver["hyperparams"].get(k)
                    if isinstance(val, (int, float)) and float(val) > 0:
                        hp[k] = float(val)
                val = cand_solver["hyperparams"].get("max_iter")
                if isinstance(val, int) and val > 0:
                    hp["max_iter"] = int(val)
                cw = cand_solver["hyperparams"].get("class_weight")
                if isinstance(cw, str):
                    hp["class_weight"] = cw[:40]
                solver_spec["hyperparams"] = hp
        merged["solver_spec"] = solver_spec
        return merged

    def _derive_next_solver_plan_from_history(
        self,
        plan_spec: Dict[str, Any],
        run_history: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        current = self._merge_solver_plan(plan_spec or {}, {})
        solver = current.get("solver_spec") if isinstance(current.get("solver_spec"), dict) else {}
        preprocess = solver.get("preprocess") if isinstance(solver.get("preprocess"), dict) else {}
        hp = solver.get("hyperparams") if isinstance(solver.get("hyperparams"), dict) else {}
        model_family = str(solver.get("model_family") or "tfidf_logreg")

        valid_runs = [r for r in run_history if bool((r or {}).get("ok"))]
        latest = run_history[-1] if run_history else {}
        latest_dev = float((latest or {}).get("dev_score_norm", 0.0) or 0.0)
        best_dev = max([float(r.get("dev_score_norm", 0.0) or 0.0) for r in valid_runs] or [0.0])
        failed = bool(run_history and not bool((latest or {}).get("ok")))

        if failed:
            preprocess["max_features"] = max(5000, int(preprocess.get("max_features", 50000) or 50000) // 2)
            hp["max_iter"] = min(4000, int(hp.get("max_iter", 2000) or 2000) + 500)
            current["strategy"] = "recover_from_failure"
        elif latest_dev + 1e-6 < best_dev:
            # Regression from best -> reduce variance and regularize.
            if model_family == "tfidf_logreg":
                hp["C"] = max(0.1, float(hp.get("C", 1.0) or 1.0) * 0.7)
            elif model_family == "tfidf_ridge":
                hp["alpha"] = min(20.0, float(hp.get("alpha", 1.0) or 1.0) * 1.5)
            preprocess["max_features"] = max(8000, int(preprocess.get("max_features", 50000) or 50000) // 2)
            current["strategy"] = "stability_regularization"
        else:
            # Modest improvement path: explore feature capacity.
            preprocess["max_features"] = min(120000, int(preprocess.get("max_features", 50000) or 50000) + 5000)
            if model_family == "tfidf_logreg":
                hp["C"] = min(5.0, float(hp.get("C", 1.0) or 1.0) * 1.2)
            current["strategy"] = "incremental_capacity_tuning"

        solver["preprocess"] = preprocess
        solver["hyperparams"] = hp
        current["solver_spec"] = solver
        return current

    def _clamp01(self, value: Any) -> float:
        try:
            parsed = float(value)
        except Exception:
            return 0.0
        return max(0.0, min(1.0, parsed))

    def _extract_evidence_refs(self, text: Any) -> List[str]:
        raw = str(text or "")
        if not raw:
            return []
        refs = set()
        for cid in re.findall(r"\bC\d{4}\b", raw):
            refs.add(cid)
        for rid in re.findall(r"\bRUN@[A-Za-z0-9\-_]+\b", raw):
            refs.add(rid)
        return sorted(refs)

    def _normalize_review_issues(self, review_note: Dict[str, Any]) -> List[Dict[str, Any]]:
        issues: List[Dict[str, Any]] = []
        raw = review_note.get("issues")
        if isinstance(raw, list):
            for idx, item in enumerate(raw, start=1):
                if isinstance(item, dict):
                    finding = self._truncate(item.get("claim") or item.get("finding") or item.get("issue"), 320)
                    if not finding:
                        continue
                    issue_type = self._truncate(item.get("type") or item.get("category") or "methodological_risk", 80)
                    severity = self._clamp01(item.get("severity", 0.6))
                    proposed_test = item.get("proposed_test") if isinstance(item.get("proposed_test"), dict) else None
                    if not proposed_test:
                        proposed_test = {
                            "kind": "replicate" if (severity >= 0.75 or "replication" in issue_type) else "static_check",
                            "params": {"focus": issue_type or "issue_validation"},
                        }
                    refs = self._safe_text_list(item.get("evidence"), limit=8, item_limit=120) or self._safe_text_list(
                        item.get("evidence_refs"), limit=8, item_limit=120
                    )
                    if not refs:
                        refs = self._extract_evidence_refs(finding)
                    issues.append(
                        {
                            "id": str(item.get("id") or f"I-{idx:03d}"),
                            "type": issue_type,
                            "severity": severity,
                            "claim": finding,
                            "evidence_refs": refs,
                            "proposed_test": proposed_test,
                            "suggested_fix": self._truncate(item.get("suggested_fix") or item.get("fix"), 260),
                        }
                    )
                else:
                    text = self._truncate(item, 320)
                    if not text:
                        continue
                    issues.append(
                        {
                            "id": f"I-{idx:03d}",
                            "type": "methodological_risk",
                            "severity": 0.6,
                            "claim": text,
                            "evidence_refs": self._extract_evidence_refs(text),
                            "proposed_test": None,
                            "suggested_fix": "",
                        }
                    )

        if not issues:
            major = self._safe_text_list(review_note.get("major_risks"), limit=6, item_limit=320)
            gaps = self._safe_text_list(review_note.get("gaps"), limit=6, item_limit=320)
            for idx, text in enumerate(major, start=1):
                issues.append(
                    {
                        "id": f"I-{idx:03d}",
                        "type": "major_risk",
                        "severity": 0.85,
                        "claim": text,
                        "evidence_refs": self._extract_evidence_refs(text),
                        "proposed_test": {"kind": "replicate", "params": {"mode": "score_consistency"}},
                        "suggested_fix": "",
                    }
                )
            base = len(issues)
            for idx, text in enumerate(gaps, start=1):
                issues.append(
                    {
                        "id": f"I-{base + idx:03d}",
                        "type": "gap",
                        "severity": 0.65,
                        "claim": text,
                        "evidence_refs": self._extract_evidence_refs(text),
                        "proposed_test": {"kind": "static_check", "params": {"focus": "evidence_coverage"}},
                        "suggested_fix": "",
                    }
                )
        return issues

    def _heuristic_review_note(self, *, paper: Dict[str, Any], metrics: Dict[str, Any]) -> Dict[str, Any]:
        issues: List[Dict[str, Any]] = []
        rid = 1
        replication_ok = bool(metrics.get("replication_ok", False))
        replication_verified = bool(metrics.get("replication_verified", False))
        evidence_score = self._clamp01(metrics.get("evidence_score"))
        graph_score = self._clamp01(metrics.get("graph_score"))
        score_norm = self._clamp01(metrics.get("score_norm"))

        if (not replication_verified) or (not replication_ok):
            issues.append(
                {
                    "id": f"I-{rid:03d}",
                    "type": "replication_risk",
                    "severity": 0.9,
                    "claim": "Replication support is weak or unverified; claims may not generalize.",
                    "evidence_refs": [],
                    "proposed_test": {"kind": "replicate", "params": {"mode": "score_consistency", "extra_seeds": [2027, 2028]}},
                    "suggested_fix": "Run additional replication and compare normalized score delta with tolerance.",
                }
            )
            rid += 1
        if evidence_score < 0.45:
            issues.append(
                {
                    "id": f"I-{rid:03d}",
                    "type": "insufficient_evidence",
                    "severity": 0.78,
                    "claim": "Evidence coverage is low relative to claimed results.",
                    "evidence_refs": [],
                    "proposed_test": {"kind": "static_check", "params": {"focus": "citation_and_observation_coverage"}},
                    "suggested_fix": "Attach stronger citation/observation mapping for each claim.",
                }
            )
            rid += 1
        if graph_score < 0.4:
            issues.append(
                {
                    "id": f"I-{rid:03d}",
                    "type": "quality_risk",
                    "severity": 0.72,
                    "claim": "Overall quality score is still weak for publication readiness.",
                    "evidence_refs": [],
                    "proposed_test": {"kind": "ablation", "params": {"focus": "error_reduction"}},
                    "suggested_fix": "Prioritize correction experiments before additional write attempts.",
                }
            )
            rid += 1

        strengths = []
        if score_norm >= 0.5:
            strengths.append(
                {
                    "id": "S-001",
                    "claim": "Model demonstrates non-trivial benchmark signal above weak baseline.",
                    "evidence": [f"score_norm={score_norm:.4f}"],
                    "confidence": 0.7,
                    "verification": {"kind": "replicate", "params": {"mode": "score_consistency"}},
                }
            )
        if evidence_score >= 0.5:
            strengths.append(
                {
                    "id": "S-002",
                    "claim": "Evidence chain is partially traceable.",
                    "evidence": [f"evidence_score={evidence_score:.4f}"],
                    "confidence": 0.6,
                    "verification": {"kind": "static_check", "params": {"focus": "evidence_map"}},
                }
            )

        if not strengths:
            strengths = [
                {
                    "id": "S-001",
                    "claim": "Submission pipeline executed and produced evaluable artifact.",
                    "evidence": [f"raw_score={float(metrics.get('raw_score', 0.0) or 0.0):.6f}"],
                    "confidence": 0.55,
                    "verification": {"kind": "static_check", "params": {"focus": "submission_validity"}},
                }
            ]

        revision_actions = [i.get("suggested_fix") for i in issues if i.get("suggested_fix")]
        return {
            "summary": "Evidence-based review with both validated strengths and falsifiable concerns.",
            "stance": "weak_accept" if not issues else "weak_reject",
            "strengths": strengths[:4],
            "issues": issues[:8],
            "revision_actions": [x for x in revision_actions[:6] if x],
            "replication_focus": "score_consistency_under_new_seeds",
            "anti_flattery": {"non_evidence_praise": False},
            "paper_id": paper.get("paper_id"),
        }

    def _score_review_quality(
        self,
        *,
        review_note: Dict[str, Any],
        issues: List[Dict[str, Any]],
        self_review: bool,
        replication_ok: bool,
    ) -> Dict[str, Any]:
        strengths = review_note.get("strengths")
        if not isinstance(strengths, list):
            strengths = []

        revision_actions = self._safe_text_list(review_note.get("revision_actions"), limit=10, item_limit=220)
        issue_count = len(issues)
        major_count = sum(1 for i in issues if self._clamp01(i.get("severity")) >= 0.8)
        evidence_linked_count = sum(1 for i in issues if bool(i.get("evidence_refs")))
        actionable_count = len(revision_actions)
        severity_avg = (
            sum(self._clamp01(i.get("severity")) for i in issues) / max(1, issue_count)
            if issue_count
            else 0.0
        )
        evidence_ratio = evidence_linked_count / max(1, issue_count)
        issue_density = min(1.0, issue_count / max(1, self._review_min_issue_count))
        actionability = min(1.0, actionable_count / max(1, self._review_min_revision_actions))

        praise_words = (
            "excellent",
            "outstanding",
            "impressive",
            "great",
            "fantastic",
            "solid",
            "well done",
            "novel",
            "strong",
        )
        praise_blob = " ".join(
            [
                str(review_note.get("summary") or ""),
                " ".join(str(x) for x in strengths if not isinstance(x, dict)),
                " ".join(str((x or {}).get("claim") or "") for x in strengths if isinstance(x, dict)),
            ]
        ).lower()
        praise_hits = sum(praise_blob.count(word) for word in praise_words)

        flattery_penalty = 0.0
        if praise_hits > 0 and issue_count == 0:
            flattery_penalty += self._review_flattery_penalty
        elif praise_hits > (issue_count + 1):
            flattery_penalty += self._review_flattery_penalty * 0.5

        shallow_penalty = 0.0
        if issue_count < self._review_min_issue_count:
            shallow_penalty += self._review_shallow_penalty
        if actionable_count < self._review_min_revision_actions:
            shallow_penalty += self._review_shallow_penalty * 0.5

        self_penalty = self._review_self_review_penalty if self_review else 0.0
        no_issue_but_failed_replication = (issue_count == 0) and (not replication_ok)
        if no_issue_but_failed_replication:
            shallow_penalty += self._review_shallow_penalty

        critique_score = (
            0.40 * issue_density
            + 0.20 * severity_avg
            + 0.20 * evidence_ratio
            + 0.20 * actionability
            - flattery_penalty
            - shallow_penalty
            - self_penalty
        )
        critique_score = max(0.0, min(1.0, critique_score))

        needs_revision = bool(
            (major_count > 0)
            or (critique_score < self._review_revision_trigger_score)
            or (not replication_ok)
        )
        return {
            "issue_count": issue_count,
            "major_issue_count": major_count,
            "evidence_linked_issue_count": evidence_linked_count,
            "actionable_count": actionable_count,
            "severity_avg": severity_avg,
            "praise_hits": praise_hits,
            "flattery_penalty": flattery_penalty,
            "shallow_penalty": shallow_penalty,
            "self_review_penalty": self_penalty,
            "no_issue_but_failed_replication": no_issue_but_failed_replication,
            "critique_score": critique_score,
            "needs_revision": needs_revision,
        }

    async def _qdrant_search_similarity(self, *, vector: List[float], collection: str) -> Optional[float]:
        if not self._vdh_qdrant_url or not collection or not vector:
            return None
        url = f"{self._vdh_qdrant_url}/collections/{collection}/points/search"
        headers = {"Content-Type": "application/json"}
        if self._vdh_qdrant_api_key:
            headers["api-key"] = self._vdh_qdrant_api_key
        payload = {
            "vector": vector,
            "limit": 1,
            "with_payload": False,
            "with_vector": False,
        }
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )

        def _send() -> Optional[float]:
            with urllib.request.urlopen(req, timeout=6) as resp:
                body = resp.read().decode("utf-8", errors="replace")
            data = json.loads(body)
            result = data.get("result")
            if isinstance(result, list) and result:
                score = (result[0] or {}).get("score")
                if isinstance(score, (int, float)):
                    return float(score)
            return None

        try:
            return await asyncio.to_thread(_send)
        except Exception:
            return None

    async def _qgr_relevance_score(
        self,
        *,
        review_note: Dict[str, Any],
        context_text: str,
    ) -> Dict[str, Any]:
        review_text = " ".join(
            [
                str(review_note.get("summary") or ""),
                " ".join(str((i or {}).get("claim") or "") for i in (review_note.get("issues") or []) if isinstance(i, dict)),
                " ".join(self._safe_text_list(review_note.get("revision_actions"), limit=10, item_limit=220)),
            ]
        ).strip()
        token_score = self._counter_cosine(
            Counter(self._text_tokens(review_text)),
            Counter(self._text_tokens(context_text)),
        )
        vector_score = None
        if self._vdh_tei_enable and (self._vdh_tei_url or self._reward_tei_url):
            rv = await self._vdh_embed_text(review_text)
            cv = await self._vdh_embed_text(context_text)
            vector_score = self._vector_cosine(rv, cv)
        if isinstance(vector_score, (int, float)):
            score = 0.7 * float(vector_score) + 0.3 * float(token_score)
            source = "tei+token"
        else:
            score = float(token_score)
            source = "token_fallback"
        return {
            "score": max(0.0, min(1.0, score)),
            "source": source,
            "token_score": float(token_score),
            "vector_score": float(vector_score) if isinstance(vector_score, (int, float)) else None,
        }

    async def _qgr_fact_check(self, issues: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not self._vdh_qdrant_enable or not self._vdh_qdrant_url:
            return {"hallucinated_count": 0, "checked": 0, "source": "disabled"}
        checked = 0
        hallucinated = 0
        details: List[Dict[str, Any]] = []
        for issue in (issues or [])[:8]:
            if not isinstance(issue, dict):
                continue
            issue_type = str(issue.get("type") or "").lower()
            claim = str(issue.get("claim") or "")
            if not claim.strip():
                continue
            if not any(k in issue_type for k in ("schema", "format", "type", "data", "replication")):
                continue
            checked += 1
            vec = await self._vdh_embed_text(claim)
            if not isinstance(vec, list) or not vec:
                details.append(
                    {
                        "issue_id": issue.get("id"),
                        "issue_type": issue.get("type"),
                        "support_score": None,
                        "supported": True,
                        "skipped": "embedding_unavailable",
                    }
                )
                continue
            score = await self._qdrant_search_similarity(vector=vec or [], collection=self._vdh_qdrant_collection)
            score_val = float(score) if isinstance(score, (int, float)) else 0.0
            supported = score_val >= self._qgr_fact_support_threshold
            if not supported:
                hallucinated += 1
            details.append(
                {
                    "issue_id": issue.get("id"),
                    "issue_type": issue.get("type"),
                    "support_score": score_val,
                    "supported": supported,
                }
            )
        return {
            "hallucinated_count": hallucinated,
            "checked": checked,
            "details": details,
            "source": "qdrant_schema_collection",
        }

    async def _qgr_validate_review(
        self,
        *,
        review_note: Dict[str, Any],
        issues: List[Dict[str, Any]],
        context_text: str,
    ) -> Dict[str, Any]:
        citation_set = set()
        for issue in issues:
            if not isinstance(issue, dict):
                continue
            for ref in (issue.get("evidence_refs") or []):
                ref_s = str(ref)
                if ref_s:
                    citation_set.add(ref_s)
        min_issue_ok = len(issues) >= self._qgr_min_issue_count
        min_citation_ok = len(citation_set) >= self._qgr_min_citations
        relevance = await self._qgr_relevance_score(review_note=review_note, context_text=context_text)
        relevance_ok = float(relevance.get("score", 0.0) or 0.0) >= self._qgr_relevance_threshold
        fact_check = await self._qgr_fact_check(issues)
        fact_ok = int(fact_check.get("hallucinated_count", 0) or 0) == 0
        valid = bool(min_issue_ok and min_citation_ok and relevance_ok and fact_ok)
        return {
            "valid": valid,
            "thresholds": {
                "min_issues": self._qgr_min_issue_count,
                "min_citations": self._qgr_min_citations,
                "min_relevance": self._qgr_relevance_threshold,
            },
            "metrics": {
                "issue_count": len(issues),
                "citation_count": len(citation_set),
                "relevance_score": float(relevance.get("score", 0.0) or 0.0),
                "relevance_source": relevance.get("source"),
                "hallucinated_count": int(fact_check.get("hallucinated_count", 0) or 0),
                "fact_checked_count": int(fact_check.get("checked", 0) or 0),
            },
            "checks": {
                "issue_count_ok": min_issue_ok,
                "citation_count_ok": min_citation_ok,
                "relevance_ok": relevance_ok,
                "fact_ok": fact_ok,
            },
            "fact_check": fact_check,
        }

    def _qgr_predictive_bonus(
        self,
        *,
        issues: List[Dict[str, Any]],
        run_history: List[Dict[str, Any]],
        target_run_id: Optional[str],
    ) -> Dict[str, Any]:
        target = None
        if target_run_id:
            for run in reversed(run_history or []):
                if str(run.get("run_id") or "") == str(target_run_id):
                    target = run
                    break
        if target is None and run_history:
            target = run_history[-1]
        merged = ""
        if isinstance(target, dict):
            merged = "\n".join(
                [
                    str(target.get("error") or ""),
                    str(target.get("stderr_tail") or ""),
                    str(target.get("fallback_reason") or ""),
                ]
            ).lower()
        matched = []
        for issue in issues:
            if not isinstance(issue, dict):
                continue
            claim = str(issue.get("claim") or "").lower()
            issue_type = str(issue.get("type") or "").lower()
            if ("oom" in claim or "memory" in claim or "oom" in issue_type) and ("killed" in merged or "out of memory" in merged):
                matched.append(str(issue.get("id") or ""))
            if ("schema" in claim or "format" in claim or "type" in claim) and (
                "typeerror" in merged or "unhashable" in merged or "submission" in merged
            ):
                matched.append(str(issue.get("id") or ""))
        matched = [m for m in matched if m]
        return {
            "matched_issue_ids": sorted(set(matched)),
            "bonus": float(self._qgr_predictive_bonus_reward if matched else 0.0),
        }

    async def _spawn_qgr_followup_tasks(
        self,
        *,
        paper_id: Optional[str],
        run_id: Optional[str],
        score: float,
        issues: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        created: List[str] = []
        if score >= 0.6 or not issues:
            return {"created": created, "reason": "score_ok_or_no_issues"}
        if paper_id:
            try:
                write_res = await self.controller.run_environment(
                    "science",
                    "task_create",
                    task_type="write",
                    payload={"paper_id": paper_id, "revision_reason": "qgr_low_score"},
                    priority=9,
                )
                if isinstance(write_res, dict) and write_res.get("ok"):
                    tid = ((write_res.get("task") or {}).get("task_id"))
                    if tid:
                        created.append(str(tid))
            except Exception:
                pass
        if run_id:
            try:
                exp_res = await self.controller.run_environment(
                    "science",
                    "task_create",
                    task_type="experiment",
                    payload={"from_run_id": run_id, "revision_reason": "qgr_low_score"},
                    priority=8,
                )
                if isinstance(exp_res, dict) and exp_res.get("ok"):
                    tid = ((exp_res.get("task") or {}).get("task_id"))
                    if tid:
                        created.append(str(tid))
            except Exception:
                pass
        return {"created": created, "reason": "low_score_spawned"}

    async def _spawn_review_validation_tasks(
        self,
        *,
        paper_id: str,
        reviewer_id: str,
        review_note: Dict[str, Any],
        critique_quality: Dict[str, Any],
    ) -> Dict[str, Any]:
        listed = await self.controller.run_environment("science", "task_list")
        tasks = (listed or {}).get("tasks", []) if isinstance(listed, dict) else []
        existing = []
        for task in tasks:
            payload = task.get("payload") or {}
            if str(payload.get("paper_id") or "") != str(paper_id):
                continue
            if str(payload.get("revision_reason") or "") != "review_validation":
                continue
            if str(task.get("task_type") or "") in {"verify_strength", "verify_issue", "write"}:
                existing.append(task.get("task_id"))
        if existing:
            return {"created": [], "skipped_existing": existing}

        created: List[str] = []
        strengths = review_note.get("strengths")
        if not isinstance(strengths, list):
            strengths = []
        issues = review_note.get("issues")
        if not isinstance(issues, list):
            issues = []

        for strength in strengths[:6]:
            if not isinstance(strength, dict):
                continue
            test = strength.get("verification")
            if not isinstance(test, dict):
                continue
            created_task = await self.controller.run_environment(
                "science",
                "task_create",
                task_type="verify_strength",
                payload={
                    "paper_id": paper_id,
                    "reviewer_id": reviewer_id,
                    "strength": strength,
                    "test": test,
                    "revision_reason": "review_validation",
                },
                priority=8,
            )
            if isinstance(created_task, dict):
                tid = ((created_task.get("task") or {}).get("task_id")) if created_task.get("ok") else None
                if tid:
                    created.append(tid)

        for issue in issues[:8]:
            if not isinstance(issue, dict):
                continue
            test = issue.get("proposed_test")
            if not isinstance(test, dict):
                continue
            sev = self._clamp01(issue.get("severity", 0.5))
            created_task = await self.controller.run_environment(
                "science",
                "task_create",
                task_type="verify_issue",
                payload={
                    "paper_id": paper_id,
                    "reviewer_id": reviewer_id,
                    "issue": issue,
                    "test": test,
                    "revision_reason": "review_validation",
                },
                priority=10 if sev >= 0.8 else 9,
            )
            if isinstance(created_task, dict):
                tid = ((created_task.get("task") or {}).get("task_id")) if created_task.get("ok") else None
                if tid:
                    created.append(tid)

        write_task_id = None
        if bool(critique_quality.get("needs_revision")):
            deps = created[: min(3, len(created))]
            write_task = await self.controller.run_environment(
                "science",
                "task_create",
                task_type="write",
                payload={
                    "paper_id": paper_id,
                    "revision_reason": "review_validation",
                    "reviewer_id": reviewer_id,
                    "critique_score": float(critique_quality.get("critique_score", 0.0) or 0.0),
                },
                priority=10,
                depends_on=deps,
            )
            if isinstance(write_task, dict) and write_task.get("ok"):
                write_task_id = (write_task.get("task") or {}).get("task_id")
                if write_task_id:
                    created.append(write_task_id)

        return {"created": created, "write_task_id": write_task_id, "skipped_existing": []}

    def _build_task_role_prompt(
        self,
        *,
        world_spec: Dict[str, Any],
        open_tasks: List[Dict[str, Any]],
        hypothesis: List[str],
        notes_count: int,
        observations_count: int,
    ) -> str:
        task_view = [
            {
                "task_id": t.get("task_id"),
                "task_type": t.get("task_type"),
                "priority": t.get("priority"),
                "ready": t.get("ready", True),
                "blocked_by": (t.get("blocked_by") or [])[:2],
            }
            for t in open_tasks[:14]
        ]
        return textwrap.dedent(
            f"""
            You are a principal investigator assigning ONE agent role in a multi-agent AIRS-Bench research lab.
            The agent must pick a sustainable specialization that improves team-level publication probability.

            Global task context:
            - AIRS task: {world_spec.get('task_name')}
            - Metric: {world_spec.get('metric')}
            - Category: {world_spec.get('category')}
            - Budget: {world_spec.get('budget')}
            - Taskboard summary: {json.dumps(world_spec.get('taskboard') or {}, ensure_ascii=False)}

            Agent local context:
            - hypothesis_tags: {json.dumps(hypothesis[:6], ensure_ascii=False)}
            - notes_count: {notes_count}
            - observations_count: {observations_count}

            Open tasks snapshot:
            {json.dumps(task_view, ensure_ascii=False)}

            Requirements:
            1) Respect strict dependency constraints; do not choose blocked tasks as primary.
            2) Maximize expected contribution to reproducible performance, not short-term reward.
            3) Provide explicit risk controls (format validity, metric alignment, replication readiness).
            4) Produce a stable role profile used across subsequent task claims.

            Return ONLY JSON:
            {{
              "role_name": "methodologist|experimenter|writer|reviewer|replicator|reader",
              "preferred_task_types": ["prepare_data", "profile_data", "retrieve_literature", "experiment", "hypothesize", "write", "verify_issue"],
              "primary_task_id": "Txxx",
              "selection_rationale": ["...", "..."],
              "risk_controls": ["...", "..."],
              "fallback_if_blocked": ["verify_issue", "verify_strength", "prepare_data", "profile_data", "retrieve_literature", "read", "hypothesize"]
            }}
            """
        ).strip()

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
        cards_short = [
            {
                "id": c.get("citation_id"),
                "kind": c.get("kind"),
                "title": c.get("title"),
                "text": self._truncate(c.get("text"), 180),
            }
            for c in cards[-self._llm_max_cards :]
        ]
        runs_short = [
            {
                "run_id": r.get("run_id"),
                "metric_name": r.get("metric_name"),
                "raw_score": r.get("raw_score"),
                "score_norm": r.get("score_norm"),
                "ok": r.get("ok"),
                "strategy": r.get("strategy"),
                "error": self._truncate(r.get("error"), 120),
            }
            for r in recent_runs[-self._llm_max_runs :]
        ]
        data_card_short = self._compact_data_card(data_card)
        method_card_short = self._compact_method_card(method_card)
        return textwrap.dedent(
            f"""
            You are an AIRS-Bench senior scientist drafting a rigorous research hypothesis and protocol.

            Task:
            - Name: {world_spec.get('task_name')}
            - Metric: {world_spec.get('metric')}
            - Category: {world_spec.get('category')}
            - Research problem: {world_spec.get('research_problem')}
            - Dataset: {world_spec.get('dataset')}

            Evidence cards:
            {json.dumps(cards_short, ensure_ascii=False)}

            Data card (structured dataset evidence):
            {json.dumps(data_card_short, ensure_ascii=False)}

            Method card (task-type baselines and pitfalls):
            {json.dumps(method_card_short, ensure_ascii=False)}

            RAG_CONTEXT (status={rag_status or 'n/a'}):
            {rag_context or "(none)"}

            RAG_EVIDENCE_REFERENCES:
            {json.dumps(list(rag_refs or [])[:32], ensure_ascii=False)}

            RAG_USAGE_CONSTRAINT:
            Prioritize retrieved evidence. Do not fabricate citations or run references.

            Recent experimental traces:
            {json.dumps(runs_short, ensure_ascii=False)}

            Constraints:
            1) Explicitly align method with official metric and submission format.
            2) Include reproducibility safeguards and expected failure modes.
            3) Favor incremental, testable, falsifiable hypotheses.
            4) Keep strategy executable within limited budget.
            5) CRITICAL: if task manifest uses list scoring_column, plan must handle manifest['scoring_column'][0].
            6) CRITICAL: include memory-safe strategy for large datasets (sampling/batching required).

            Return ONLY JSON with concise but technical content:
            {{
              "hypothesis_tags": ["..."],
              "strategy": "...",
              "schema_assumptions": ["..."],
              "memory_safety": ["..."],
              "evidence_refs": ["..."],
              "solver_spec": {{
                "model_family": "tfidf_logreg|linear_svc|tfidf_ridge",
                "seed": 42,
                "preprocess": {{"max_features": 50000, "ngram_range": [1,2], "min_df": 1}},
                "hyperparams": {{"C": 1.0, "max_iter": 2000, "class_weight": "balanced", "alpha": 1.0}}
              }},
              "rationale": ["...", "..."],
              "risk": ["...", "..."],
              "experiment_protocol": {{
                "primary_knob": "...",
                "ablation_axis": "...",
                "format_checks": ["...", "..."]
              }},
              "replication_plan": ["...", "..."]
            }}
            """
        ).strip()

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
        notes_short = []
        for n in notes[-4:]:
            cards = (n or {}).get("cards", []) or []
            notes_short.append(
                {
                    "topic": n.get("topic"),
                    "hints": (n.get("hints") or [])[:3],
                    "cards": [
                        {
                            "citation_id": c.get("citation_id"),
                            "title": c.get("title"),
                            "text": self._truncate(c.get("text"), 120),
                        }
                        for c in cards[:3]
                    ],
                }
            )
        obs_short = [
            {
                "run_id": o.get("run_id"),
                "score_norm": o.get("score_norm"),
                "ok": o.get("ok"),
                "strategy": o.get("strategy"),
                "error": self._truncate(o.get("error"), 120),
            }
            for o in observations[-self._llm_max_runs :]
        ]
        data_card_short = self._compact_data_card(data_card)
        method_card_short = self._compact_method_card(method_card)
        return textwrap.dedent(
            f"""
            You are an experimental ML researcher designing the NEXT AIRS run.

            Task context:
            - task_name: {world_spec.get('task_name')}
            - metric: {world_spec.get('metric')}
            - category: {world_spec.get('category')}
            - budget_used: {exp_count}/{budget}
            - current_strategy: {plan_spec.get('strategy')}

            Current hypothesis tags:
            {json.dumps(hypothesis[:8], ensure_ascii=False)}

            Evidence snippets:
            {json.dumps(notes_short, ensure_ascii=False)}

            Data card:
            {json.dumps(data_card_short, ensure_ascii=False)}

            Method card:
            {json.dumps(method_card_short, ensure_ascii=False)}

            RAG_CONTEXT (status={rag_status or 'n/a'}):
            {rag_context or "(none)"}

            RAG_EVIDENCE_REFERENCES:
            {json.dumps(list(rag_refs or [])[:32], ensure_ascii=False)}

            RAG_USAGE_CONSTRAINT:
            Prioritize retrieved evidence. Do not fabricate citations or run references.

            Previous run outcomes:
            {json.dumps(obs_short, ensure_ascii=False)}

            Requirements:
            1) Propose one high-value, reproducible run.
            2) Include explicit submission format checks.
            3) If previous runs failed, prioritize validity recovery before novelty.
            4) Keep config compact and executable.

            Return ONLY JSON:
            {{
              "strategy": "...",
              "config": {{
                "model_family": "tfidf_logreg|linear_svc|tfidf_ridge",
                "seed": 42,
                "preprocess": {{"max_features": 50000, "ngram_range": [1,2], "min_df": 1}},
                "hyperparams": {{"C": 1.0, "max_iter": 2000, "class_weight": "balanced", "alpha": 1.0}}
              }},
              "expected_signal": "...",
              "validity_checks": ["...", "..."],
              "failure_modes": ["...", "..."]
            }}
            """
        ).strip()

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
        evidence_short = []
        for n in notes[-4:]:
            cards = (n or {}).get("cards", []) or []
            for c in cards[:2]:
                evidence_short.append(
                    {
                        "citation_id": c.get("citation_id"),
                        "title": c.get("title"),
                        "text": self._truncate(c.get("text"), 120),
                    }
                )
        obs_short = [
            {
                "run_id": o.get("run_id"),
                "score_norm": o.get("score_norm"),
                "ok": o.get("ok"),
                "strategy": o.get("strategy"),
            }
            for o in observations[-self._llm_max_runs :]
        ]
        return textwrap.dedent(
            f"""
            You are writing a concise but rigorous AIRS research report for internal peer review.

            Task context:
            - task_name: {world_spec.get('task_name')}
            - metric: {world_spec.get('metric')}
            - category: {world_spec.get('category')}
            - best_run: {json.dumps({{
                "run_id": best_run.get("run_id"),
                "raw_score": best_run.get("raw_score"),
                "score_norm": best_run.get("score_norm"),
                "strategy": best_run.get("strategy"),
            }}, ensure_ascii=False)}
            - current_strategy: {plan_spec.get('strategy')}

            Hypothesis tags:
            {json.dumps(hypothesis[:8], ensure_ascii=False)}

            Candidate citations:
            {json.dumps(citations[:16], ensure_ascii=False)}

            Observation refs:
            {json.dumps(observation_refs[:12], ensure_ascii=False)}

            Evidence snippets:
            {json.dumps(evidence_short[:10], ensure_ascii=False)}

            RAG_CONTEXT (status={rag_status or 'n/a'}):
            {rag_context or "(none)"}

            RAG_EVIDENCE_REFERENCES:
            {json.dumps(list(rag_refs or [])[:32], ensure_ascii=False)}

            RAG_USAGE_CONSTRAINT:
            Prioritize retrieved evidence. Do not fabricate citations or run references.

            Experiment history:
            {json.dumps(obs_short, ensure_ascii=False)}

            Requirements:
            1) Claims must be directly linked to evidence.
            2) Distinguish observed results vs speculative interpretation.
            3) Include threats to validity and replication checklist.
            4) Output must be structured JSON only.

            Return ONLY JSON:
            {{
              "title": "...",
              "abstract": "...",
              "key_claims": ["...", "..."],
              "method_section": "...",
              "evidence_map": {{
                "claimed_result": ["C0001", "RUN@RUN001-0001"]
              }},
              "limitations": ["...", "..."],
              "replication_checklist": ["...", "..."],
              "next_experiments": ["...", "..."]
            }}
            """
        ).strip()

    def _build_review_prompt(self, *, paper: Dict[str, Any], metrics: Dict[str, Any]) -> str:
        return textwrap.dedent(
            f"""
            You are a rigorous AIRS-Bench reviewer in a scientific community.
            Your job is balanced: provide evidence-backed support for what is correct,
            and provide falsifiable critiques for weak points.

            Paper object:
            {json.dumps(paper, ensure_ascii=False)}

            Quantitative metrics:
            {json.dumps(metrics, ensure_ascii=False)}

            Review rubric:
            1) Method-metric alignment and dataset handling validity.
            2) Evidence sufficiency for each claim.
            3) Reproducibility and replication risk.
            4) Actionable revisions ranked by expected impact.
            5) No emotional praise. Every support statement must be verifiable.

            Mandatory constraints:
            - Include at least one strength with evidence and verification plan.
            - Include at least one issue with evidence and proposed test,
              unless replication and evidence are both clearly strong.
            - If issues are zero, you must provide stronger verification for strengths.
            - Keep claims falsifiable and machine-checkable.

            Return ONLY JSON:
            {{
              "summary": "evidence-based summary only",
              "stance": "accept|weak_accept|borderline|weak_reject|reject",
              "strengths": [
                {{
                  "id": "S-001",
                  "claim": "...",
                  "evidence": ["RUN@...", "C0001"],
                  "confidence": 0.75,
                  "verification": {{"kind": "replicate|static_check", "params": {{}}}}
                }}
              ],
              "issues": [
                {{
                  "id": "I-001",
                  "type": "replication_risk|metric_mismatch|insufficient_evidence|runtime_risk",
                  "severity": 0.85,
                  "claim": "...",
                  "evidence_refs": ["RUN@...", "C0002"],
                  "proposed_test": {{"kind": "replicate|ablation|static_check", "params": {{}}}},
                  "suggested_fix": "..."
                }}
              ],
              "revision_actions": ["...", "..."],
              "replication_focus": "...",
              "anti_flattery": {{"non_evidence_praise": false}}
            }}
            """
        ).strip()

    def _build_replication_prompt(
        self,
        *,
        world_spec: Dict[str, Any],
        paper_id: str,
        paper: Dict[str, Any],
        claimed_metrics: Dict[str, Any],
    ) -> str:
        return textwrap.dedent(
            f"""
            You are a reproducibility lead designing a replication protocol for AIRS-Bench.

            Task context:
            - task_name: {world_spec.get('task_name')}
            - metric: {world_spec.get('metric')}
            - category: {world_spec.get('category')}
            - paper_id: {paper_id}

            Paper summary:
            {json.dumps(paper, ensure_ascii=False)}

            Claimed metrics:
            {json.dumps(claimed_metrics, ensure_ascii=False)}

            Requirements:
            1) Define replication checks that can falsify over-claimed results.
            2) Prioritize metric consistency and submission validity.
            3) Output machine-readable protocol with clear pass criteria.

            Return ONLY JSON:
            {{
              "mode": "score_consistency",
              "protocol_name": "...",
              "stress_tests": ["...", "..."],
              "pass_criteria": {{
                "max_delta_norm": 0.08,
                "require_format_valid": true
              }},
              "failure_signals": ["...", "..."],
              "notes": ["...", "..."]
            }}
            """
        ).strip()

    def _normalize_code_plan(self, payload: Any) -> Dict[str, Any]:
        if not isinstance(payload, dict):
            return {}
        files_raw = payload.get("files")
        files: List[Dict[str, str]] = []
        if isinstance(files_raw, list):
            for item in files_raw[: self._code_max_files]:
                if not isinstance(item, dict):
                    continue
                rel_path = str(item.get("path") or "").replace("\\", "/").strip()
                content = item.get("content")
                if not rel_path or not isinstance(content, str):
                    continue
                if rel_path.startswith("/") or rel_path.startswith("../") or "/../" in rel_path:
                    continue
                files.append({"path": rel_path[:220], "content": content[: self._code_max_file_chars]})
        run_cmd = str(payload.get("run_cmd") or "").strip()
        if not run_cmd:
            run_cmd = "python src/main.py --data-dir ./data --output-dir ./outputs --task-manifest ./.task_manifest.json"
        plan = {
            "run_cmd": run_cmd[:600],
            "files": files,
        }
        if isinstance(payload.get("notes"), str):
            plan["notes"] = self._truncate(payload.get("notes"), 500)
        return plan

    def _classify_experiment_failure(
        self,
        *,
        result: Dict[str, Any],
        code_plan: Optional[Dict[str, Any]],
        world_spec: Dict[str, Any],
    ) -> Dict[str, Any]:
        error = str((result or {}).get("error") or "")
        stderr_tail = str((result or {}).get("stderr_tail") or "")
        stdout_tail = str((result or {}).get("stdout_tail") or "")
        merged = "\n".join([error, stderr_tail, stdout_tail]).lower()
        fallback_reason = str((result or {}).get("fallback_reason") or "").lower()

        error_class = "unknown"
        error_codes: List[str] = []
        severity = "medium"
        retryable = True
        root_cause = "unknown_failure"
        repair_hints: List[str] = []

        if "hard_stop:" in merged:
            error_class = "runtime"
            error_codes.append("hard_stop")
            severity = "fatal"
            retryable = False
            root_cause = "episode_budget_or_step_hard_stop"
            repair_hints.append("wait_for_new_episode_or_reduce_compute_cost")
        elif "timed out" in merged or "timeout" in merged:
            error_class = "timeout"
            error_codes.append("execution_timeout")
            severity = "high"
            root_cause = "run_timeout"
            repair_hints.extend(["reduce_data_scale", "reduce_model_complexity", "add_fast_path"])
        elif "killed" in merged or "out of memory" in merged or "oom" in merged:
            error_class = "oom"
            error_codes.append("process_killed_or_oom")
            severity = "high"
            root_cause = "memory_pressure"
            repair_hints.extend(["enable_sampling", "reduce_batch_size", "reduce_feature_count"])
        elif "no such file or directory" in merged or "filenotfounderror" in merged:
            error_class = "io"
            severity = "medium"
            root_cause = "input_or_output_path_missing"
            if "./data/train.csv" in merged:
                error_codes.append("missing_train_csv")
                repair_hints.append("prefer_load_from_disk_train_with_csv_fallback")
            if "submission.csv" in merged:
                error_codes.append("missing_submission_csv")
                repair_hints.append("ensure_outputs_submission_csv_written")
            if not error_codes:
                error_codes.append("missing_path")
                repair_hints.append("check_manifest_and_runtime_paths")
        elif "modulenotfounderror" in merged or "no module named" in merged:
            error_class = "dependency"
            severity = "medium"
            root_cause = "missing_runtime_dependency"
            m = re.search(r"no module named ['\"]([^'\"]+)['\"]", merged)
            if m:
                error_codes.append(f"module_missing_{m.group(1)}")
            else:
                error_codes.append("module_missing_unknown")
            repair_hints.append("avoid_optional_dependency_or_use_stdlib_fallback")
        elif "column object has no attribute tolist" in merged:
            error_class = "schema"
            error_codes.append("column_tolist_misuse")
            root_cause = "dataset_column_type_mismatch"
            repair_hints.append("convert_column_with_list_or_numpy_array")
        elif "unhashable type: 'list'" in merged or "unhashable type: \"list\"" in merged:
            error_class = "schema"
            error_codes.append("unhashable_list_schema")
            root_cause = "list_used_in_hash_context"
            repair_hints.append("avoid_value_counts_on_list_column")
        elif "scoring_column" in merged and ("list" in merged or "[0]" in merged):
            error_class = "schema"
            error_codes.append("scoring_column_list_misuse")
            root_cause = "scoring_column_list_not_indexed"
            repair_hints.append("read_manifest_and_use_scoring_column_index0")
        elif "evaluate.py failed" in merged or "submission" in merged and "format" in merged:
            error_class = "format"
            error_codes.append("submission_format_invalid")
            root_cause = "submission_schema_mismatch"
            repair_hints.append("align_submission_columns_with_manifest")
        elif "syntaxerror" in merged or "typeerror" in merged or "valueerror" in merged:
            error_class = "runtime"
            error_codes.append("python_runtime_exception")
            root_cause = "python_exception"
            repair_hints.append("fix_exception_using_traceback_line")

        if "code_agent:" in fallback_reason and not error_codes:
            error_codes.append("code_agent_fallback")

        return {
            "error_class": error_class,
            "error_codes": error_codes or ["unknown_failure"],
            "severity": severity,
            "retryable": bool(retryable),
            "root_cause": root_cause,
            "evidence": {
                "error": self._truncate(error, 800),
                "stderr_tail": self._truncate(stderr_tail, self._code_error_tail_chars),
                "stdout_tail": self._truncate(stdout_tail, self._code_error_tail_chars),
                "run_meta": {
                    "task_name": world_spec.get("task_name"),
                    "metric": world_spec.get("metric"),
                    "code_memory_mb": world_spec.get("code_memory_mb"),
                    "solver_mode": (result or {}).get("solver_mode"),
                    "fallback_reason": (result or {}).get("fallback_reason"),
                },
            },
            "repair_hints": repair_hints[:8],
            "code_plan_brief": {
                "has_files": bool((code_plan or {}).get("files")),
                "run_cmd": self._truncate((code_plan or {}).get("run_cmd"), 240),
            },
        }

    def _apply_failure_template_fixes(
        self,
        *,
        code_plan: Dict[str, Any],
        diagnosis: Dict[str, Any],
        world_spec: Dict[str, Any],
    ) -> Dict[str, Any]:
        if not self._code_template_fix_enable:
            return {"applied": False, "rules_hit": [], "mutated_files": [], "summary": "template_fix_disabled", "code_plan": code_plan}
        plan = json.loads(json.dumps(code_plan or {}))
        files = plan.get("files")
        if not isinstance(files, list):
            return {"applied": False, "rules_hit": [], "mutated_files": [], "summary": "no_files", "code_plan": plan}

        rules_hit: List[str] = []
        mutated_files: List[str] = []
        error_codes = {str(x) for x in (diagnosis or {}).get("error_codes", [])}
        error_class = str((diagnosis or {}).get("error_class") or "")

        for file_obj in files:
            if not isinstance(file_obj, dict):
                continue
            path = str(file_obj.get("path") or "")
            content = file_obj.get("content")
            if not isinstance(content, str):
                continue
            updated = content

            if "missing_train_csv" in error_codes or error_class == "io":
                if "./data/train.csv" in updated and "load_from_disk('./data/train')" not in updated:
                    updated = updated.replace(
                        "pd.read_csv('./data/train.csv')",
                        "load_from_disk('./data/train').to_pandas() if os.path.exists('./data/train') else pd.read_csv('./data/train.csv')",
                    )
                    rules_hit.append("io_train_csv_to_load_from_disk")

            if "scoring_column_list_misuse" in error_codes:
                if "scoring_column" in updated and "[0]" not in updated:
                    updated = updated.replace("manifest.get('scoring_column')", "(manifest.get('scoring_column') or ['prediction'])[0]")
                    rules_hit.append("schema_scoring_column_index0")

            if "column_tolist_misuse" in error_codes and ".tolist()" in updated:
                updated = updated.replace(".tolist()", " if hasattr(train_data['target'], '__iter__') else []")
                rules_hit.append("schema_column_tolist_guard")

            if error_class == "oom":
                if "max_features" in updated and "50000" in updated:
                    updated = updated.replace("50000", "15000")
                    rules_hit.append("oom_reduce_max_features")
                if "sample_ratio" not in updated:
                    updated += "\n\n# template fix: OOM guard\nsample_ratio = 0.1\n"
                    rules_hit.append("oom_add_sample_ratio")

            if error_class == "format":
                if "outputs/submission.csv" not in updated:
                    updated += "\n\n# template fix: enforce submission path\nsubmission_path = os.path.join('./outputs', 'submission.csv')\n"
                    rules_hit.append("format_force_submission_path")

            if updated != content:
                file_obj["content"] = updated[: self._code_max_file_chars]
                mutated_files.append(path or "unknown_path")

        applied = bool(mutated_files)
        return {
            "applied": applied,
            "rules_hit": sorted(set(rules_hit)),
            "mutated_files": mutated_files,
            "summary": "template_fix_applied" if applied else "no_rule_matched",
            "code_plan": plan,
            "task_name": world_spec.get("task_name"),
        }

    def _build_repair_context(
        self,
        *,
        diagnosis: Dict[str, Any],
        template_fix: Dict[str, Any],
        prev_plan: Optional[Dict[str, Any]],
    ) -> str:
        payload = {
            "diagnosis": {
                "error_class": diagnosis.get("error_class"),
                "error_codes": diagnosis.get("error_codes"),
                "severity": diagnosis.get("severity"),
                "retryable": diagnosis.get("retryable"),
                "root_cause": diagnosis.get("root_cause"),
                "repair_hints": diagnosis.get("repair_hints"),
                "evidence": diagnosis.get("evidence"),
            },
            "template_fix": {
                "applied": template_fix.get("applied"),
                "rules_hit": template_fix.get("rules_hit"),
                "mutated_files": template_fix.get("mutated_files"),
                "summary": template_fix.get("summary"),
            },
            "previous_plan_brief": {
                "run_cmd": (prev_plan or {}).get("run_cmd"),
                "file_count": len((prev_plan or {}).get("files") or []),
            },
        }
        return json.dumps(payload, ensure_ascii=False)

    def _should_enter_optimize(self, attempts: List[Dict[str, Any]]) -> bool:
        if not self._code_optimize_guard_enable:
            return True
        if not isinstance(attempts, list) or len(attempts) < 2:
            return False
        successes = [
            a for a in attempts
            if isinstance(a, dict)
            and isinstance(a.get("result"), dict)
            and bool((a.get("result") or {}).get("ok", False))
        ]
        if len(successes) < 1:
            return False
        last_two = attempts[-2:]
        for item in last_two:
            if not isinstance(item, dict) or not isinstance(item.get("result"), dict):
                return False
            r = item.get("result") or {}
            if not bool(r.get("ok", False)):
                return False
            if not isinstance(r.get("dev_score_norm"), (int, float)) and not isinstance(r.get("score_norm"), (int, float)):
                return False
        return True

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
    ) -> str:
        notes_short = []
        for n in notes[-4:]:
            cards = (n or {}).get("cards", []) or []
            notes_short.append(
                {
                    "topic": n.get("topic"),
                    "hints": (n.get("hints") or [])[:3],
                    "cards": [
                        {
                            "citation_id": c.get("citation_id"),
                            "title": c.get("title"),
                            "text": self._truncate(c.get("text"), 120),
                        }
                        for c in cards[:3]
                    ],
                }
            )
        obs_short = [
            {
                "run_id": o.get("run_id"),
                "ok": o.get("ok"),
                "dev_score_norm": o.get("dev_score_norm"),
                "score_norm": o.get("score_norm"),
                "solver_mode": o.get("solver_mode"),
                "error": self._truncate(o.get("error"), 180),
            }
            for o in observations[-self._llm_max_runs :]
        ]
        data_card_short = self._compact_data_card(data_card)
        method_card_short = self._compact_method_card(method_card)
        phase_guidance = {
            "generate": "Write first executable baseline code for this task.",
            "repair": "Fix execution/runtime errors and keep scientific validity.",
            "optimize": "Improve dev score while preserving reproducibility and format validity.",
        }.get(phase, "Write executable research code.")

        return textwrap.dedent(
            f"""
            You are an autonomous ML researcher working in a controlled AIRS code sandbox.
            Objective: produce runnable code, iterate from errors, and improve dev metrics.

            Phase: {phase}
            Round: {round_idx}/{max_rounds}
            Guidance: {phase_guidance}

            Task context:
            - task_name: {world_spec.get('task_name')}
            - metric: {world_spec.get('metric')}
            - category: {world_spec.get('category')}
            - research_problem: {world_spec.get('research_problem')}
            - budget_used: {exp_count}/{budget}
            - best_dev_score_norm_so_far: {best_dev_score_norm if best_dev_score_norm is not None else "N/A"}
            - current_strategy: {plan_spec.get('strategy')}

            Scientific hypotheses:
            {json.dumps(hypothesis[:8], ensure_ascii=False)}

            Evidence cards / notes:
            {json.dumps(notes_short, ensure_ascii=False)}

            Data card:
            {json.dumps(data_card_short, ensure_ascii=False)}

            Method card:
            {json.dumps(method_card_short, ensure_ascii=False)}

            Previous run summaries:
            {json.dumps(obs_short, ensure_ascii=False)}

            Previous code plan:
            {json.dumps(previous_plan or {}, ensure_ascii=False)}

            Failure context (if any):
            {failure_context or "N/A"}

            Diagnosis JSON (for repair/optimize):
            {json.dumps(failure_diagnosis or {}, ensure_ascii=False)}

            TemplateFix summary:
            {json.dumps(template_fix or {}, ensure_ascii=False)}

            RAG_CONTEXT (status={rag_status or 'n/a'}):
            {rag_context or "(none)"}

            RAG_EVIDENCE_REFERENCES:
            {json.dumps(list(rag_refs or [])[:32], ensure_ascii=False)}

            RAG_USAGE_CONSTRAINT:
            Prioritize retrieved evidence. Do not fabricate citations or run references.

            Sandbox contract:
            1) You must write file-level code updates.
            2) Code must run via one command.
            3) Must output `outputs/submission.csv` for test-format predictions.
            4) Should output `outputs/dev_predictions.csv` for dev evaluation.
            5) No network calls, no package installs, no external downloads.
            6) Keep code deterministic (set seeds if applicable).
            7) IMPORTANT: Do NOT assume `./data/train.csv` or `./data/test.csv` always exist.
               AIRS data is often HuggingFace `datasets.save_to_disk` format under:
               - `./data/train/` and `./data/test/` (arrow + state.json)
               Prefer robust loading:
               - first try `datasets.load_from_disk('./data/train')`
               - fallback to `pd.read_csv('./data/train.csv')` only if CSV exists.
            8) Read `.task_manifest.json` to get metric/category/scoring_column and format submission accordingly.
            9) Do NOT repeat previously failed error_codes if provided in Diagnosis JSON.
            10) Keep valid existing logic unless a rule in TemplateFix explicitly changes it.

            Return ONLY JSON:
            {{
              "run_cmd": "python src/main.py --data-dir ./data --output-dir ./outputs --task-manifest ./.task_manifest.json",
              "files": [
                {{"path": "src/main.py", "content": "FULL PYTHON CODE"}},
                {{"path": "src/feature_engineering.py", "content": "OPTIONAL"}}
              ],
              "notes": "brief explanation of this iteration",
              "fixed_error_codes": ["..."],
              "risk_left": ["..."]
            }}
            """
        ).strip()

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
        if not self._code_loop_enabled or not self._llm_ready("experiment"):
            return None
        if not bool(world_spec.get("code_agent_enable", False)):
            return None

        attempts: List[Dict[str, Any]] = []
        best_result: Optional[Dict[str, Any]] = None
        best_score = -1.0
        best_plan: Dict[str, Any] = {}
        previous_plan: Dict[str, Any] = {}
        failure_context = ""
        last_diagnosis: Dict[str, Any] = {}
        last_template_fix: Dict[str, Any] = {}
        no_improve_rounds = 0

        max_rounds = min(self._code_debug_rounds, max(1, int(budget or self._code_debug_rounds)))

        for idx in range(max_rounds):
            has_success = best_result is not None
            if not has_success and idx == 0:
                phase = "generate"
            elif not has_success:
                phase = "repair"
            else:
                phase = "optimize"
                if not self._code_optimize_after_success:
                    break
                if not self._should_enter_optimize(attempts):
                    phase = "repair"

            recent_errors = [
                self._truncate((a.get("result") or {}).get("error"), 180)
                for a in attempts[-5:]
                if isinstance(a, dict)
                and isinstance(a.get("result"), dict)
                and str(((a.get("result") or {}).get("error") or "")).strip()
            ]
            rag_query = " | ".join(
                [
                    f"task={world_spec.get('task_name')}",
                    f"phase={phase}",
                    f"strategy={plan_spec.get('strategy')}",
                    f"round={idx + 1}/{max_rounds}",
                    f"failure_codes={json.dumps((last_diagnosis or {}).get('error_codes', []), ensure_ascii=False)}",
                    f"recent_errors={json.dumps(recent_errors, ensure_ascii=False)}",
                ]
            )
            rag_result = await self._rag_retrieve_context(
                agent_id=agent_id,
                action="experiment",
                run_id=str((best_result or {}).get("run_id") or ""),
                paper_id=None,
                query_text=rag_query,
                quotas={"observation": 5, "diagnosis": 5, "method_card": 2, "data_card": 1, "note": 1},
                notes=notes,
                observations=prior_observations + [a.get("result") or {} for a in attempts],
                data_card=data_card,
                method_card=method_card,
            )
            rag_block = self._format_rag_prompt_block(result=rag_result)

            prompt = self._build_code_experiment_prompt(
                world_spec=world_spec,
                hypothesis=hypothesis,
                notes=notes,
                observations=prior_observations + [a.get("result") or {} for a in attempts],
                data_card=data_card,
                method_card=method_card,
                plan_spec=plan_spec,
                exp_count=exp_count + idx,
                budget=budget,
                phase=phase,
                round_idx=idx + 1,
                max_rounds=max_rounds,
                previous_plan=previous_plan,
                failure_context=failure_context,
                failure_diagnosis=last_diagnosis,
                template_fix=last_template_fix,
                best_dev_score_norm=best_score if best_score >= 0.0 else None,
                rag_context=rag_block.get("context", ""),
                rag_refs=rag_block.get("refs", []),
                rag_status=rag_block.get("status", ""),
            )
            llm_result = await self._call_llm_json(agent_id=agent_id, action_name="experiment", prompt=prompt)
            if not llm_result.get("ok") or not isinstance(llm_result.get("data"), dict):
                attempts.append(
                    {
                        "round": idx + 1,
                        "phase": phase,
                        "llm_ok": False,
                        "llm_reason": llm_result.get("reason"),
                    }
                )
                break

            code_plan = self._normalize_code_plan(llm_result.get("data") or {})
            llm_fixed_error_codes = self._safe_text_list((llm_result.get("data") or {}).get("fixed_error_codes"), limit=10, item_limit=120)
            llm_risk_left = self._safe_text_list((llm_result.get("data") or {}).get("risk_left"), limit=10, item_limit=120)
            previous_plan = code_plan
            if not bool(code_plan.get("files")):
                attempts.append(
                    {
                        "round": idx + 1,
                        "phase": phase,
                        "llm_ok": True,
                        "llm_reason": "code_plan_files_empty",
                        "diagnosis": {"error_class": "runtime", "error_codes": ["code_plan_files_empty"]},
                    }
                )
                failure_context = "LLM returned empty files list; provide full runnable files."
                last_diagnosis = {
                    "error_class": "runtime",
                    "error_codes": ["code_plan_files_empty"],
                    "severity": "medium",
                    "retryable": True,
                    "root_cause": "llm_output_invalid",
                    "repair_hints": ["return_non_empty_files"],
                    "evidence": {"error": "code_plan_files_empty"},
                }
                last_template_fix = {"applied": False, "rules_hit": [], "mutated_files": [], "summary": "no_file_to_fix"}
                if self._code_diag_enable:
                    await self._log_code_diagnosis(
                        agent_id=agent_id,
                        phase=phase,
                        run_id=None,
                        diagnosis=last_diagnosis,
                        template_fix=last_template_fix,
                        decision="repair",
                    )
                continue

            current_tick = int(await self.controller.run_system("timer", "get_tick"))
            run_config = dict(base_run_config or {})
            run_config["strategy"] = "code_agent_iterative"
            run_config["code_phase"] = phase
            run_config["code_round"] = idx + 1
            run_config["code_plan"] = code_plan

            result = await self.controller.run_environment(
                "science",
                "run_experiment",
                config=run_config,
                agent_id=agent_id,
                current_tick=current_tick,
            )
            attempts.append(
                {
                    "round": idx + 1,
                    "phase": phase,
                    "llm_ok": True,
                    "code_plan": code_plan,
                    "llm_fixed_error_codes": llm_fixed_error_codes,
                    "llm_risk_left": llm_risk_left,
                    "result": result,
                }
            )

            ok = bool((result or {}).get("ok", False))
            if not ok:
                diagnosis = self._classify_experiment_failure(result=result or {}, code_plan=code_plan, world_spec=world_spec)
                template_fix = self._apply_failure_template_fixes(
                    code_plan=code_plan,
                    diagnosis=diagnosis,
                    world_spec=world_spec,
                )
                if isinstance(template_fix.get("code_plan"), dict):
                    previous_plan = template_fix.get("code_plan") or previous_plan
                failure_context = self._build_repair_context(
                    diagnosis=diagnosis,
                    template_fix=template_fix,
                    prev_plan=previous_plan,
                )
                last_diagnosis = diagnosis
                last_template_fix = template_fix
                attempts[-1]["diagnosis"] = diagnosis
                attempts[-1]["template_fix"] = {
                    "applied": template_fix.get("applied"),
                    "rules_hit": template_fix.get("rules_hit"),
                    "mutated_files": template_fix.get("mutated_files"),
                    "summary": template_fix.get("summary"),
                }
                if self._code_diag_enable:
                    await self._log_code_diagnosis(
                        agent_id=agent_id,
                        phase=phase,
                        run_id=str((result or {}).get("run_id") or ""),
                        diagnosis=diagnosis,
                        template_fix=template_fix,
                        decision="repair",
                        score_norm=(result or {}).get("score_norm"),
                        dev_score_norm=(result or {}).get("dev_score_norm"),
                    )
                if not bool(diagnosis.get("retryable", True)):
                    break
                continue

            score = (result or {}).get("dev_score_norm")
            if not isinstance(score, (int, float)):
                score = (result or {}).get("score_norm")
            score_f = self._clamp01(score or 0.0)
            if best_result is None or score_f > best_score + 1e-6:
                best_result = result
                best_score = score_f
                best_plan = code_plan
                no_improve_rounds = 0
            else:
                no_improve_rounds += 1

            if self._code_diag_enable:
                await self._log_code_diagnosis(
                    agent_id=agent_id,
                    phase=phase,
                    run_id=str((result or {}).get("run_id") or ""),
                    diagnosis={
                        "error_class": "none",
                        "error_codes": [],
                        "severity": "low",
                        "retryable": True,
                        "root_cause": "success",
                        "repair_hints": [],
                        "evidence": {},
                    },
                    template_fix={"applied": False, "rules_hit": [], "mutated_files": [], "summary": "success"},
                    decision="optimize" if self._should_enter_optimize(attempts) else "repair",
                    score_norm=(result or {}).get("score_norm"),
                    dev_score_norm=(result or {}).get("dev_score_norm"),
                )

            if has_success and no_improve_rounds >= self._code_optimize_patience:
                break

        if not attempts:
            return None

        await self._log_code_loop(
            agent_id=agent_id,
            attempts=attempts,
            best_dev_score_norm=(best_score if best_score >= 0.0 else None),
        )

        final_result = best_result
        final_plan = best_plan
        if final_result is None:
            last = attempts[-1]
            if isinstance(last.get("result"), dict):
                final_result = last.get("result")
            final_plan = last.get("code_plan") if isinstance(last.get("code_plan"), dict) else {}
        if not isinstance(final_result, dict):
            return None

        return {
            "result": final_result,
            "run_config": {
                "strategy": "code_agent_iterative",
                "code_plan": final_plan,
            },
            "llm_experiment_plan": {
                "mode": "code_agent_loop",
                "attempts": attempts,
                "best_dev_score_norm": best_score if best_score >= 0.0 else None,
            },
            "code_attempts": attempts,
        }

    def _experiment_precondition_failures(
        self,
        *,
        hypothesis: List[str],
        notes: List[Dict[str, Any]],
        data_card: Optional[Dict[str, Any]],
        method_card: Optional[Dict[str, Any]],
    ) -> List[str]:
        failures: List[str] = []
        if self._experiment_require_data_card and not isinstance(data_card, dict):
            failures.append("need_data_card")
        if self._experiment_require_method_card and not isinstance(method_card, dict):
            failures.append("need_method_card")
        if self._experiment_min_notes > 0 and len(notes or []) < self._experiment_min_notes:
            failures.append(f"need_notes>={self._experiment_min_notes}")
        if self._experiment_min_hypothesis > 0 and len(hypothesis or []) < self._experiment_min_hypothesis:
            failures.append(f"need_hypothesis>={self._experiment_min_hypothesis}")
        return failures

    def _has_notes_failure(self, failures: List[str]) -> bool:
        return any(str(item).startswith("need_notes>=") for item in (failures or []))

    def _build_method_note_from_card(self, method_card: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if not isinstance(method_card, dict):
            return None
        baselines = method_card.get("recommended_baselines") if isinstance(method_card.get("recommended_baselines"), list) else []
        cards: List[Dict[str, Any]] = []
        hints: List[str] = []
        for idx, baseline in enumerate(baselines[: self._llm_max_cards], start=1):
            if not isinstance(baseline, dict):
                continue
            citation_id = f"M{idx:04d}"
            title = str(baseline.get("name") or f"baseline_{idx}")
            steps = self._safe_text_list(baseline.get("key_steps"), limit=3, item_limit=120)
            pitfalls = self._safe_text_list(baseline.get("pitfalls"), limit=2, item_limit=120)
            text_parts = [f"use_when={self._truncate(baseline.get('use_when'), 120)}"]
            if steps:
                text_parts.append("steps=" + "; ".join(steps))
            if pitfalls:
                text_parts.append("pitfalls=" + "; ".join(pitfalls))
            cards.append(
                {
                    "citation_id": citation_id,
                    "kind": "method_card",
                    "title": title,
                    "text": " | ".join(text_parts),
                }
            )
            hints.append(f"[{citation_id}] {title}")
        if not cards and isinstance(method_card.get("task_evidence_refs"), list):
            for ref in method_card.get("task_evidence_refs")[: self._llm_max_cards]:
                if not isinstance(ref, dict):
                    continue
                cid = str(ref.get("citation_id") or "")
                title = str(ref.get("title") or "method_ref")
                cards.append(
                    {
                        "citation_id": cid or f"MR{len(cards) + 1:04d}",
                        "kind": "method_ref",
                        "title": title,
                        "text": self._truncate(ref.get("snippet"), 220),
                    }
                )
                hints.append(f"[{cards[-1]['citation_id']}] {title}")
        if not cards:
            return None
        return {
            "topic": "task_baselines",
            "hints": hints,
            "cards": cards,
            "task_name": method_card.get("task_name"),
            "source": "method_card",
        }

    def _build_data_note_from_card(self, data_card: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if not isinstance(data_card, dict):
            return None
        split_stats = data_card.get("split_stats") if isinstance(data_card.get("split_stats"), dict) else {}
        schema = data_card.get("schema") if isinstance(data_card.get("schema"), list) else []
        cards: List[Dict[str, Any]] = []
        hints: List[str] = []
        cards.append(
            {
                "citation_id": "D0001",
                "kind": "data_profile",
                "title": "split_stats",
                "text": self._truncate(json.dumps(split_stats, ensure_ascii=False), 320),
            }
        )
        hints.append("[D0001] split_stats")
        for idx, col in enumerate(schema[:6], start=2):
            if not isinstance(col, dict):
                continue
            name = str(col.get("name") or f"col_{idx}")
            desc = {
                "dtype": col.get("dtype"),
                "missing_ratio": col.get("missing_ratio"),
                "unique": col.get("unique"),
            }
            cards.append(
                {
                    "citation_id": f"D{idx:04d}",
                    "kind": "data_schema",
                    "title": name,
                    "text": self._truncate(json.dumps(desc, ensure_ascii=False), 220),
                }
            )
            hints.append(f"[D{idx:04d}] {name}")
        return {
            "topic": "data_profile",
            "hints": hints,
            "cards": cards,
            "task_name": data_card.get("task_name"),
            "source": "data_card",
        }

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
        """Synchronize episode-level shared artifacts into agent-local state before experiment."""
        summary: Dict[str, Any] = {
            "used_shared_artifacts": False,
            "hydrate_steps": [],
            "remaining_failures": list(failures or []),
        }
        if not failures:
            return summary

        local_notes = await self._get_state(agent_id, "notes") or []
        changed_notes = False
        shared_artifacts: Dict[str, Any] = {}

        need_data = "need_data_card" in failures and not isinstance(data_card, dict)
        need_method = "need_method_card" in failures and not isinstance(method_card, dict)
        need_notes = self._has_notes_failure(failures)

        if need_data or need_method or need_notes:
            try:
                shared_artifacts = await self.controller.run_environment(
                    "science",
                    "get_shared_artifacts",
                    include_cards=True,
                    max_refs=max(4, min(12, self._llm_max_cards)),
                )
                if isinstance(shared_artifacts, dict) and shared_artifacts.get("ok"):
                    summary["used_shared_artifacts"] = True
                    summary["hydrate_steps"].append("fetched_shared_artifacts")
            except Exception as e:
                summary["hydrate_steps"].append(f"shared_artifacts_failed:{self._truncate(str(e), 160)}")

        if need_data and not isinstance(data_card, dict):
            shared_data = shared_artifacts.get("data_card") if isinstance(shared_artifacts.get("data_card"), dict) else None
            if isinstance(shared_data, dict):
                data_card = dict(shared_data)
                await self._set_state(agent_id, "data_card", data_card)
                summary["hydrate_steps"].append("data_card_from_shared_cache")
            else:
                prof = await self.controller.run_environment("science", "profile_data", agent_id=agent_id, refresh=False)
                if isinstance(prof, dict) and bool(prof.get("ok", False)):
                    data_card = prof
                    await self._set_state(agent_id, "data_card", data_card)
                    summary["hydrate_steps"].append("data_card_from_profile_data")
                else:
                    summary["hydrate_steps"].append(
                        "data_card_unavailable:" + self._truncate((prof or {}).get("reason"), 140)
                    )

        if need_method and not isinstance(method_card, dict):
            shared_method = shared_artifacts.get("method_card") if isinstance(shared_artifacts.get("method_card"), dict) else None
            if isinstance(shared_method, dict):
                method_card = dict(shared_method)
                await self._set_state(agent_id, "method_card", method_card)
                summary["hydrate_steps"].append("method_card_from_shared_cache")
            else:
                method = await self.controller.run_environment(
                    "science",
                    "retrieve_method_card",
                    agent_id=agent_id,
                    topic="task_baselines",
                    refresh=False,
                )
                if isinstance(method, dict) and bool(method.get("ok", False)):
                    method_card = method
                    await self._set_state(agent_id, "method_card", method_card)
                    summary["hydrate_steps"].append("method_card_from_retrieve")
                else:
                    summary["hydrate_steps"].append(
                        "method_card_unavailable:" + self._truncate((method or {}).get("reason"), 140)
                    )

        if need_notes:
            note_count = len((local_notes or []) + (await self._get_state(agent_id, "shared_notes") or []))
            if note_count < self._experiment_min_notes:
                shared_template = (
                    shared_artifacts.get("notes_template")
                    if isinstance(shared_artifacts.get("notes_template"), dict)
                    else None
                )
                if isinstance(shared_template, dict):
                    local_notes.append(shared_template)
                    changed_notes = True
                    summary["hydrate_steps"].append("notes_from_shared_template")
            note_count = len((local_notes or []) + (await self._get_state(agent_id, "shared_notes") or []))
            if note_count < self._experiment_min_notes:
                method_note = self._build_method_note_from_card(method_card)
                if isinstance(method_note, dict):
                    local_notes.append(method_note)
                    changed_notes = True
                    summary["hydrate_steps"].append("notes_from_method_card")
            note_count = len((local_notes or []) + (await self._get_state(agent_id, "shared_notes") or []))
            if note_count < self._experiment_min_notes:
                data_note = self._build_data_note_from_card(data_card)
                if isinstance(data_note, dict):
                    local_notes.append(data_note)
                    changed_notes = True
                    summary["hydrate_steps"].append("notes_from_data_card")
            note_count = len((local_notes or []) + (await self._get_state(agent_id, "shared_notes") or []))
            if note_count < self._experiment_min_notes:
                lit = await self.controller.run_environment(
                    "science",
                    "read_literature",
                    agent_id=agent_id,
                    topic="task_requirements",
                )
                if isinstance(lit, dict):
                    local_notes.append(lit)
                    changed_notes = True
                    summary["hydrate_steps"].append("notes_from_read_literature")

        if changed_notes:
            await self._set_state(agent_id, "notes", local_notes)

        refreshed_hypothesis = await self._get_state(agent_id, "hypothesis") or hypothesis
        refreshed_notes = (await self._get_state(agent_id, "notes") or []) + (await self._get_state(agent_id, "shared_notes") or [])
        refreshed_data = await self._get_state(agent_id, "data_card")
        refreshed_method = await self._get_state(agent_id, "method_card")
        summary["remaining_failures"] = self._experiment_precondition_failures(
            hypothesis=refreshed_hypothesis,
            notes=refreshed_notes,
            data_card=refreshed_data if isinstance(refreshed_data, dict) else None,
            method_card=refreshed_method if isinstance(refreshed_method, dict) else None,
        )
        summary["counts"] = {
            "hypothesis": len(refreshed_hypothesis or []),
            "notes": len(refreshed_notes or []),
            "has_data_card": isinstance(refreshed_data, dict),
            "has_method_card": isinstance(refreshed_method, dict),
        }
        return summary

    async def _enqueue_prereq_recovery_tasks(self, *, failures: List[str]) -> Dict[str, Any]:
        """Create recovery tasks only when the required task type is absent on taskboard."""
        open_list = await self.controller.run_environment("science", "task_list", status="open")
        claimed_list = await self.controller.run_environment("science", "task_list", status="claimed")
        known_types = set()
        for listing in (open_list, claimed_list):
            tasks = (listing or {}).get("tasks", []) if isinstance(listing, dict) else []
            for task in tasks:
                known_types.add(str((task or {}).get("task_type") or ""))

        created: List[str] = []
        required_task_types: List[str] = []
        if "need_data_card" in failures:
            required_task_types.append("profile_data")
        if "need_method_card" in failures:
            required_task_types.append("retrieve_literature")
        if self._has_notes_failure(failures):
            required_task_types.append("read")
        if any(str(f).startswith("need_hypothesis>=") for f in failures):
            required_task_types.append("hypothesize")
        if "need_plan_spec" in failures:
            required_task_types.append("hypothesize")

        for task_type in required_task_types:
            if task_type in known_types:
                continue
            created_task = await self.controller.run_environment(
                "science",
                "task_create",
                task_type=task_type,
                payload={},
                priority=8,
            )
            if isinstance(created_task, dict) and created_task.get("ok"):
                tid = str((created_task.get("task") or {}).get("task_id") or "")
                if tid:
                    created.append(tid)
                known_types.add(task_type)

        return {"created_task_ids": created, "required_task_types": required_task_types}

    def _write_precondition_failures(
        self,
        *,
        hypothesis: List[str],
        notes: List[Dict[str, Any]],
        observations: List[Dict[str, Any]],
    ) -> List[str]:
        failures: List[str] = []
        if len(hypothesis or []) < self._write_min_hypothesis:
            failures.append(f"need_hypothesis>={self._write_min_hypothesis}")
        if len(notes or []) < self._write_min_notes:
            failures.append(f"need_notes>={self._write_min_notes}")
        if len(observations or []) < self._write_min_observations:
            failures.append(f"need_observations>={self._write_min_observations}")
        return failures

    def _local_submission_format_check(self, submission_path: str) -> Dict[str, Any]:
        if not submission_path or not os.path.exists(submission_path):
            return {
                "ok": False,
                "error_code": "submission_not_found",
                "message": f"submission missing: {submission_path}",
            }
        rows = 0
        try:
            with open(submission_path, "r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                fieldnames = list(reader.fieldnames or [])
                if not fieldnames:
                    return {
                        "ok": False,
                        "error_code": "missing_header",
                        "message": "submission.csv missing header row",
                    }
                for _ in reader:
                    rows += 1
            if rows <= 0:
                return {
                    "ok": False,
                    "error_code": "empty_submission",
                    "message": "submission.csv has no data rows",
                    "columns": fieldnames,
                }
            try:
                digest = hashlib.sha256()
                with open(submission_path, "rb") as f:
                    while True:
                        chunk = f.read(1024 * 1024)
                        if not chunk:
                            break
                        digest.update(chunk)
                file_hash = digest.hexdigest()
            except Exception:
                file_hash = ""
            return {
                "ok": True,
                "columns": fieldnames,
                "row_count": rows,
                "submission_hash": file_hash,
            }
        except Exception as e:
            return {
                "ok": False,
                "error_code": "csv_parse_error",
                "message": f"failed to parse submission.csv: {e}",
            }

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

    async def _grant_credits(self, credit_by_agent: Dict[str, float], source: str, reference_id: Optional[str] = None) -> None:
        for recipient, credit in credit_by_agent.items():
            if not recipient or credit <= 0:
                continue
            try:
                await self._inc_state_number(recipient, "credit_buffer", float(credit))
                await self._inc_state_number(recipient, "contribution_credit_total", float(credit))
                await self._set_state(
                    recipient,
                    "last_credit",
                    {"source": source, "value": float(credit), "reference_id": reference_id},
                )
            except Exception as e:
                logger.warning(f"Credit assignment failed for {recipient}: {e}")

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
                safe_refs = [str(r) for r in refs[:10] if isinstance(r, (str, int, float))]
                if safe_refs:
                    evidence_map[str(key)] = safe_refs

        return {
            "task_name": task_name,
            "title": self._truncate(llm_write_spec.get("title"), 220) if llm_write_spec.get("title") else f"{task_name} iterative study",
            "abstract": self._truncate(llm_write_spec.get("abstract"), 1200),
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
            "key_claims": self._safe_text_list(llm_write_spec.get("key_claims"), limit=6, item_limit=320),
            "limitations": self._safe_text_list(llm_write_spec.get("limitations"), limit=6, item_limit=260),
            "replication_checklist": self._safe_text_list(
                llm_write_spec.get("replication_checklist"),
                limit=8,
                item_limit=220,
            ),
            "next_experiments": self._safe_text_list(llm_write_spec.get("next_experiments"), limit=6, item_limit=220),
            "method_section": self._truncate(
                llm_write_spec.get("method_section") or json.dumps(method_summary, ensure_ascii=False),
                4000,
            ),
            "exp_count": int(exp_count or 0),
        }

    async def _get_claimed_task(
        self,
        agent_id: str,
        task_id: str,
        current_tick: Optional[int] = None,
    ) -> Optional[Dict[str, Any]]:
        try:
            direct = await self.controller.run_environment(
                "science",
                "task_get",
                task_id=task_id,
                current_tick=current_tick,
            )
            task = (direct or {}).get("task") if isinstance(direct, dict) else None
            if isinstance(task, dict):
                status = str(task.get("status") or "")
                if status in {"claimed", "running"} and str(task.get("claimed_by") or "") == str(agent_id):
                    return task
        except Exception:
            pass

        listed = await self.controller.run_environment(
            "science",
            "task_list",
            status="claimed",
            agent_id=agent_id,
            current_tick=current_tick,
        )
        tasks = (listed or {}).get("tasks", []) if isinstance(listed, dict) else []
        for task in tasks:
            if task.get("task_id") == task_id:
                return task
        listed_running = await self.controller.run_environment(
            "science",
            "task_list",
            status="running",
            agent_id=agent_id,
            current_tick=current_tick,
        )
        tasks_running = (listed_running or {}).get("tasks", []) if isinstance(listed_running, dict) else []
        for task in tasks_running:
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
        if task_action == "prepare_data":
            return await self.prepare_data(agent_id=agent_id, refresh=bool(payload.get("refresh")))
        if task_action == "profile_data":
            return await self.profile_data(agent_id=agent_id, focus_cols=payload.get("focus_cols"), refresh=bool(payload.get("refresh")))
        if task_action == "retrieve_literature":
            return await self.retrieve_literature(
                agent_id=agent_id,
                topic=payload.get("topic"),
                refresh=bool(payload.get("refresh")),
            )
        if task_action == "hypothesize":
            result = await self.hypothesize(agent_id=agent_id)
            if isinstance(result, ActionResult) and result.is_successful():
                data = result.data if isinstance(result.data, dict) else {}
                plan_spec = data.get("plan_spec")
                if not isinstance(plan_spec, dict):
                    return ActionResult.error(
                        method_name="hypothesize",
                        message="hypothesize returned invalid plan_spec format.",
                        data={
                            "ok": False,
                            "precondition_failed": True,
                            "reason": "invalid_plan_spec_format",
                            "reward": 0.0,
                            "effective_action": "hypothesize",
                            "reward_components": {"learning_reward": 0.0, "hypothesize_reward": 0.0},
                        },
                    )
            return result
        if task_action == "experiment":
            if self._vdh_enable and self._vdh_gate_policy == "hard_fail":
                last_vdh = await self._get_state(agent_id, "last_vdh_report")
                if isinstance(last_vdh, dict) and last_vdh and not bool(last_vdh.get("final_ok", True)):
                    return ActionResult.error(
                        method_name="experiment",
                        message="Experiment blocked by VDH gate failure.",
                        data={
                            "ok": False,
                            "precondition_failed": True,
                            "reason": "vdh_gate_failed",
                            "vdh": last_vdh,
                            "reward": 0.0,
                            "effective_action": "experiment",
                            "reward_components": {
                                "learning_reward": 0.0,
                                "experiment_reward": 0.0,
                            },
                        },
                    )
            return await self.experiment(agent_id=agent_id, config=payload.get("config"))
        if task_action == "replicate":
            return await self.replicate(agent_id=agent_id, paper_id=payload.get("paper_id"))
        if task_action == "write":
            return await self.write(agent_id=agent_id)
        if task_action == "review":
            return await self.review(agent_id=agent_id, paper_id=payload.get("paper_id"), run_id=payload.get("run_id"))
        if task_action == "verify_strength":
            return await self.verify_strength(
                agent_id=agent_id,
                paper_id=payload.get("paper_id"),
                reviewer_id=payload.get("reviewer_id"),
                strength=payload.get("strength"),
                test=payload.get("test"),
            )
        if task_action == "verify_issue":
            return await self.verify_issue(
                agent_id=agent_id,
                paper_id=payload.get("paper_id"),
                reviewer_id=payload.get("reviewer_id"),
                issue=payload.get("issue"),
                test=payload.get("test"),
            )
        if task_action == "share_evidence":
            return await self.share_evidence(agent_id=agent_id, to_agent_id=payload.get("to_agent_id"))
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

    @AgentCall
    async def read(self, agent_id: str, topic: Optional[str] = None) -> ActionResult:
        existing_notes = await self._get_state(agent_id, "notes") or []
        literature = await self.controller.run_environment("science", "read_literature", agent_id=agent_id, topic=topic)
        notes = existing_notes
        notes.append(literature)
        await self._set_state(agent_id, "notes", notes)
        await self._log_evidence_cards(agent_id, literature, source="read")
        if self._rag_index_on_read:
            world_spec = await self.controller.run_environment("science", "get_world_spec")
            rag_docs = self._rag_docs_from_note(
                world_spec=world_spec,
                agent_id=agent_id,
                note=literature if isinstance(literature, dict) else {},
                action="read",
            )
            if rag_docs:
                await self._rag_index_documents(agent_id=agent_id, action="read", docs=rag_docs)
        read_reward_info = await self._compute_read_reward(existing_notes=existing_notes, new_note=literature)
        read_reward = float(read_reward_info.get("reward", 0.0) or 0.0)

        ar = ActionResult.success(
            method_name="read",
            message="Task cards retrieved.",
            data={
                "note": literature,
                "reward": read_reward,
                "effective_action": "read",
                "reward_components": {
                    "learning_reward": float(read_reward),
                    "read_reward": float(read_reward),
                    "read_novelty": float(read_reward_info.get("novelty", 0.0) or 0.0),
                    "read_method_bonus": float(read_reward_info.get("method_bonus", 0.0) or 0.0),
                },
                "read_reward_info": read_reward_info,
            },
        )
        await self._append_trace(agent_id, "read", read_reward, ar.data or {})
        return ar

    @AgentCall
    async def profile_data(
        self,
        agent_id: str,
        focus_cols: Optional[List[str]] = None,
        refresh: bool = False,
    ) -> ActionResult:
        data_card = await self.controller.run_environment(
            "science",
            "profile_data",
            agent_id=agent_id,
            focus_cols=focus_cols,
            refresh=bool(refresh),
        )
        ok = isinstance(data_card, dict) and bool(data_card.get("ok", False))
        reward = 0.01 if ok else -0.01
        if ok:
            await self._set_state(agent_id, "data_card", data_card)

        ar = ActionResult.success(
            method_name="profile_data",
            message="Data card generated." if ok else "Data profiling failed.",
            data={
                "ok": ok,
                "data_card": data_card,
                "reward": reward,
                "effective_action": "profile_data",
                "reward_components": {
                    "learning_reward": float(reward),
                    "profile_data_reward": float(reward),
                },
            },
        )
        await self._append_trace(agent_id, "profile_data", reward, ar.data or {})
        return ar

    @AgentCall
    async def prepare_data(
        self,
        agent_id: str,
        refresh: bool = False,
    ) -> ActionResult:
        prep = await self.controller.run_environment(
            "science",
            "prepare_data",
            agent_id=agent_id,
            refresh=bool(refresh),
        )
        ok = isinstance(prep, dict) and bool(prep.get("ok", False))
        reward = 0.008 if ok else -0.01
        if ok:
            await self._set_state(agent_id, "prepare_data_ready", True)

        ar = ActionResult.success(
            method_name="prepare_data",
            message="Prepared AIRS data cache is ready." if ok else "prepare_data failed.",
            data={
                "ok": ok,
                "prepare_data": prep,
                "reward": reward,
                "effective_action": "prepare_data",
                "reward_components": {
                    "learning_reward": float(reward),
                    "prepare_data_reward": float(reward),
                },
            },
        )
        await self._append_trace(agent_id, "prepare_data", reward, ar.data or {})
        return ar

    @AgentCall
    async def retrieve_literature(
        self,
        agent_id: str,
        topic: Optional[str] = None,
        refresh: bool = False,
    ) -> ActionResult:
        method_card = await self.controller.run_environment(
            "science",
            "retrieve_method_card",
            agent_id=agent_id,
            topic=topic,
            refresh=bool(refresh),
        )
        ok = isinstance(method_card, dict) and bool(method_card.get("ok", False))
        reward = 0.01 if ok else -0.01
        if ok:
            await self._set_state(agent_id, "method_card", method_card)
            notes = await self._get_state(agent_id, "notes") or []
            baselines = (method_card.get("recommended_baselines") or []) if isinstance(method_card, dict) else []
            cards = []
            hints = []
            for idx, baseline in enumerate(baselines[: self._llm_max_cards], start=1):
                if not isinstance(baseline, dict):
                    continue
                citation_id = f"M{idx:04d}"
                title = str(baseline.get("name") or f"baseline_{idx}")
                steps = self._safe_text_list(baseline.get("key_steps"), limit=3, item_limit=120)
                pitfalls = self._safe_text_list(baseline.get("pitfalls"), limit=2, item_limit=120)
                text_parts = [f"use_when={self._truncate(baseline.get('use_when'), 120)}"]
                if steps:
                    text_parts.append("steps=" + "; ".join(steps))
                if pitfalls:
                    text_parts.append("pitfalls=" + "; ".join(pitfalls))
                cards.append(
                    {
                        "citation_id": citation_id,
                        "kind": "method_card",
                        "title": title,
                        "text": " | ".join(text_parts),
                    }
                )
                hints.append(f"[{citation_id}] {title}")
            method_note = {
                "topic": topic or "task_baselines",
                "hints": hints,
                "cards": cards,
                "task_name": method_card.get("task_name"),
                "source": "method_card",
            }
            notes.append(method_note)
            await self._set_state(agent_id, "notes", notes)
            await self._log_evidence_cards(agent_id, method_note, source="retrieve_literature")
            if self._rag_index_on_read:
                world_spec = await self.controller.run_environment("science", "get_world_spec")
                rag_docs = self._rag_docs_from_method_card(
                    world_spec=world_spec,
                    agent_id=agent_id,
                    method_card=method_card if isinstance(method_card, dict) else {},
                    action="retrieve_literature",
                )
                rag_docs.extend(
                    self._rag_docs_from_note(
                        world_spec=world_spec,
                        agent_id=agent_id,
                        note=method_note,
                        action="retrieve_literature",
                    )
                )
                if rag_docs:
                    await self._rag_index_documents(agent_id=agent_id, action="retrieve_literature", docs=rag_docs)

        ar = ActionResult.success(
            method_name="retrieve_literature",
            message="Method card retrieved." if ok else "Method retrieval failed.",
            data={
                "ok": ok,
                "method_card": method_card,
                "reward": reward,
                "effective_action": "retrieve_literature",
                "reward_components": {
                    "learning_reward": float(reward),
                    "retrieve_literature_reward": float(reward),
                },
            },
        )
        await self._append_trace(agent_id, "retrieve_literature", reward, ar.data or {})
        return ar

    @AgentCall
    async def hypothesize(self, agent_id: str, hypothesis: Optional[List[str]] = None) -> ActionResult:
        notes = (await self._get_state(agent_id, "notes") or []) + (await self._get_state(agent_id, "shared_notes") or [])
        observations = (await self._get_state(agent_id, "observations") or []) + (
            await self._get_state(agent_id, "shared_observations") or []
        )
        data_card = await self._get_state(agent_id, "data_card")
        method_card = await self._get_state(agent_id, "method_card")
        world_spec = await self.controller.run_environment("science", "get_world_spec")
        await self._rag_bootstrap_episode_knowledge(
            agent_id=agent_id,
            world_spec=world_spec,
            data_card=data_card if isinstance(data_card, dict) else None,
            method_card=method_card if isinstance(method_card, dict) else None,
        )
        existing_plan = await self._get_state(agent_id, "plan_spec") or {}
        plan_spec = self._merge_solver_plan(self._default_solver_plan(world_spec), existing_plan if isinstance(existing_plan, dict) else {})
        inferred = ["metric_alignment", "format_safety"]
        rag_block = {"context": "", "refs": [], "status": "disabled"}

        if hypothesis is None and self._llm_ready("hypothesize"):
            recent_failure_modes = [
                self._truncate((o or {}).get("error"), 180)
                for o in observations[-5:]
                if isinstance(o, dict) and not bool((o or {}).get("ok", False)) and str((o or {}).get("error") or "").strip()
            ]
            rag_query = " | ".join(
                [
                    f"task={world_spec.get('task_name')}",
                    f"metric={world_spec.get('metric')}",
                    f"current_strategy={plan_spec.get('strategy')}",
                    f"existing_hypothesis={json.dumps(hypothesis or [], ensure_ascii=False)}",
                    f"failure_modes={json.dumps(recent_failure_modes[:4], ensure_ascii=False)}",
                ]
            )
            rag_result = await self._rag_retrieve_context(
                agent_id=agent_id,
                action="hypothesize",
                run_id=None,
                paper_id=None,
                query_text=rag_query,
                quotas={"data_card": 2, "method_card": 3, "observation": 2, "diagnosis": 2, "note": 1},
                notes=notes,
                observations=observations,
                data_card=data_card if isinstance(data_card, dict) else None,
                method_card=method_card if isinstance(method_card, dict) else None,
            )
            rag_block = self._format_rag_prompt_block(result=rag_result)
            cards = []
            for n in notes[-4:]:
                cards.extend((n or {}).get("cards", [])[:3])
            prompt = self._build_plan_prompt(
                world_spec=world_spec,
                cards=cards,
                recent_runs=observations,
                data_card=data_card,
                method_card=method_card,
                rag_context=rag_block.get("context", ""),
                rag_refs=rag_block.get("refs", []),
                rag_status=rag_block.get("status", ""),
            )
            llm_result = await self._call_llm_json(agent_id=agent_id, action_name="hypothesize", prompt=prompt)
            if llm_result.get("ok") and isinstance(llm_result.get("data"), dict):
                data = llm_result.get("data") or {}
                tags = data.get("hypothesis_tags")
                if isinstance(tags, list) and tags:
                    inferred = [str(t) for t in tags[:5]]
                if isinstance(data.get("config"), dict):
                    data["solver_spec"] = data.get("config")
                plan_spec = self._merge_solver_plan(plan_spec, data)

        if hypothesis is None:
            hypothesis = inferred

        await self._set_state(agent_id, "hypothesis", hypothesis)
        await self._set_state(agent_id, "plan_spec", plan_spec)
        feasibility = self._hypothesis_feasibility(world_spec=world_spec, plan_spec=plan_spec)
        vdh_report: Dict[str, Any] = {
            "final_ok": True,
            "policy": self._vdh_gate_policy,
            "gate_a": {"ok": True, "source": "disabled", "constraints": {}, "warnings": []},
            "gate_b": {"ok": True, "errors": [], "resource_estimate": {}},
            "gate_c": {
                "ok": True,
                "coverage_score": 1.0,
                "threshold": self._vdh_evidence_threshold,
                "source": "disabled",
                "errors": [],
            },
            "failures": [],
        }
        if self._vdh_enable:
            vdh_report = await self._evaluate_vdh_gates(
                world_spec=world_spec,
                hypothesis=hypothesis,
                plan_spec=plan_spec,
                notes=notes,
                observations=observations,
                data_card=data_card if isinstance(data_card, dict) else None,
            )
        await self._set_state(agent_id, "last_vdh_report", vdh_report)

        gate_a_ok = bool(((vdh_report.get("gate_a") or {}).get("ok", False)))
        coverage_score = float(((vdh_report.get("gate_c") or {}).get("coverage_score", 0.0) or 0.0))
        gate_b_errors = list(((vdh_report.get("gate_b") or {}).get("errors") or []))
        potential_oom = any(str(e) == "potential_oom" for e in gate_b_errors)
        final_ok = bool(vdh_report.get("final_ok", True))

        vdh_schema_pass_reward = self._vdh_schema_pass_reward if gate_a_ok else 0.0
        if coverage_score > 0.8:
            vdh_evidence_coverage_reward = self._vdh_evidence_high_reward * min(1.0, coverage_score)
        else:
            vdh_evidence_coverage_reward = 0.0
        vdh_oom_penalty = -self._vdh_oom_penalty if potential_oom else 0.0
        vdh_gate_penalty = -self._vdh_gate_penalty if (self._vdh_enable and not final_ok) else 0.0

        if self._vdh_enable:
            hypo_reward = vdh_schema_pass_reward + vdh_evidence_coverage_reward + vdh_oom_penalty + vdh_gate_penalty
        else:
            hypo_reward = float(feasibility.get("reward", 0.0) or 0.0)
        hypo_reward = float(max(-1.2, min(1.2, hypo_reward)))

        reward_components = {
            "learning_reward": float(hypo_reward),
            "hypothesize_reward": float(hypo_reward),
            "hypothesis_feasibility_score": float(feasibility.get("feasibility_score", 0.0) or 0.0),
            "hypothesis_schema_bonus": float(feasibility.get("schema_bonus", 0.0) or 0.0),
            "hypothesis_resource_bonus": float(feasibility.get("resource_bonus", 0.0) or 0.0),
            "vdh_schema_pass_reward": float(vdh_schema_pass_reward),
            "vdh_evidence_coverage_reward": float(vdh_evidence_coverage_reward),
            "vdh_oom_penalty": float(vdh_oom_penalty),
            "vdh_gate_penalty": float(vdh_gate_penalty),
        }
        await self._log_vdh_gate(agent_id=agent_id, vdh_report=vdh_report, reward_components=reward_components)

        if self._vdh_enable and self._vdh_gate_policy == "hard_fail" and not final_ok:
            recovery = await self._enqueue_vdh_recovery_tasks(vdh_report=vdh_report)
            ar = ActionResult.error(
                method_name="hypothesize",
                message="VDH gate failed; hypothesis rejected and recovery tasks queued.",
                data={
                    "ok": False,
                    "precondition_failed": True,
                    "reason": "vdh_gate_failed",
                    "hypothesis": hypothesis,
                    "plan_spec": plan_spec,
                    "vdh": vdh_report,
                    "recovery_tasks": recovery,
                    "reward": hypo_reward,
                    "effective_action": "hypothesize",
                    "reward_components": reward_components,
                    "feasibility": feasibility,
                },
            )
            await self._append_trace(agent_id, "hypothesize", hypo_reward, ar.data or {})
            return ar

        ar = ActionResult.success(
            method_name="hypothesize",
            message="AIRS strategy plan updated.",
            data={
                "hypothesis": hypothesis,
                "plan_spec": plan_spec,
                "vdh": vdh_report,
                "reward": hypo_reward,
                "effective_action": "hypothesize",
                "reward_components": reward_components,
                "feasibility": feasibility,
                "rag_status": rag_block.get("status"),
                "rag_refs": rag_block.get("refs", []),
            },
        )
        await self._append_trace(agent_id, "hypothesize", hypo_reward, ar.data or {})
        return ar

    @AgentCall
    async def experiment(
        self,
        agent_id: str,
        config: Optional[Dict[str, Any]] = None,
        intervention: Optional[Dict[str, float]] = None,
        n_samples: int = 50,
    ) -> ActionResult:
        del intervention, n_samples
        exp_count = int((await self._get_state(agent_id, "exp_count")) or 0) + 1
        plan_spec = await self._get_state(agent_id, "plan_spec") or {}
        world_spec = await self.controller.run_environment("science", "get_world_spec")
        budget = int((await self._get_state(agent_id, "budget")) or world_spec.get("budget") or 10)
        notes = (await self._get_state(agent_id, "notes") or []) + (await self._get_state(agent_id, "shared_notes") or [])
        prior_observations = await self._get_state(agent_id, "observations") or []
        hypothesis = await self._get_state(agent_id, "hypothesis") or []
        data_card = await self._get_state(agent_id, "data_card")
        method_card = await self._get_state(agent_id, "method_card")
        precondition_failures = self._experiment_precondition_failures(
            hypothesis=hypothesis,
            notes=notes,
            data_card=data_card if isinstance(data_card, dict) else None,
            method_card=method_card if isinstance(method_card, dict) else None,
        )
        plan_spec_missing = (
            not isinstance(plan_spec, dict)
            or not str((plan_spec or {}).get("strategy") or "").strip()
            or not isinstance((plan_spec or {}).get("solver_spec"), dict)
        )
        if plan_spec_missing:
            precondition_failures.append("need_plan_spec")
        if precondition_failures:
            await self._log_precondition_gate(
                agent_id=agent_id,
                action="experiment",
                phase="initial",
                failures=precondition_failures,
                summary={
                    "hypothesis_count": len(hypothesis or []),
                    "notes_count": len(notes or []),
                    "observation_count": len(prior_observations or []),
                },
            )
            hydrate_summary = await self._hydrate_experiment_prerequisites(
                agent_id=agent_id,
                hypothesis=hypothesis,
                notes=notes,
                data_card=data_card if isinstance(data_card, dict) else None,
                method_card=method_card if isinstance(method_card, dict) else None,
                failures=precondition_failures,
            )
            precondition_failures = list(hydrate_summary.get("remaining_failures") or [])
            await self._log_precondition_gate(
                agent_id=agent_id,
                action="experiment",
                phase="post_hydrate",
                failures=precondition_failures,
                summary=hydrate_summary if isinstance(hydrate_summary, dict) else {},
            )
            if precondition_failures:
                if "need_plan_spec" in precondition_failures:
                    try:
                        await self.controller.run_environment(
                            "science",
                            "task_create",
                            task_type="hypothesize",
                            payload={"reason": "missing_plan_spec"},
                            priority=9,
                        )
                    except Exception as e:
                        logger.warning(f"Failed to create hypothesize recovery task: {e}")
                recovery = await self._enqueue_prereq_recovery_tasks(failures=precondition_failures)
                ar = ActionResult.success(
                    method_name="experiment",
                    message="Experiment deferred: prerequisites pending.",
                    data={
                        "ok": False,
                        "pending_prereq": True,
                        "precondition_failed": True,
                        "precondition_failures": precondition_failures,
                        "hydrate_summary": hydrate_summary,
                        "recovery_tasks": recovery,
                        "counts": {
                            "hypothesis": len(hypothesis),
                            "notes": len(notes),
                            "observations": len(prior_observations),
                        },
                        "reward": 0.0,
                        "effective_action": "experiment",
                        "reward_components": {"learning_reward": 0.0, "experiment_reward": 0.0},
                    },
                )
                await self._append_trace(agent_id, "experiment", 0.0, ar.data or {})
                return ar
            hypothesis = await self._get_state(agent_id, "hypothesis") or hypothesis
            notes = (await self._get_state(agent_id, "notes") or []) + (await self._get_state(agent_id, "shared_notes") or [])
            data_card = await self._get_state(agent_id, "data_card")
            method_card = await self._get_state(agent_id, "method_card")
        run_config = dict(config or {})
        solver_spec = plan_spec.get("solver_spec") if isinstance(plan_spec.get("solver_spec"), dict) else {}
        run_config.setdefault("strategy", plan_spec.get("strategy", "iterative_solver_baseline"))
        if solver_spec:
            run_config.setdefault("solver_spec", json.loads(json.dumps(solver_spec)))
        llm_experiment_plan: Optional[Dict[str, Any]] = None
        code_attempts = None
        code_loop_payload = await self._run_code_research_loop(
            agent_id=agent_id,
            world_spec=world_spec,
            hypothesis=hypothesis,
            plan_spec=plan_spec,
            notes=notes,
            prior_observations=prior_observations,
            data_card=data_card if isinstance(data_card, dict) else None,
            method_card=method_card if isinstance(method_card, dict) else None,
            exp_count=exp_count,
            budget=budget,
            base_run_config=run_config,
        )

        if isinstance(code_loop_payload, dict) and isinstance(code_loop_payload.get("result"), dict):
            result = code_loop_payload.get("result") or {}
            run_config = code_loop_payload.get("run_config") or run_config
            llm_experiment_plan = code_loop_payload.get("llm_experiment_plan")
            code_attempts = code_loop_payload.get("code_attempts")
        else:
            if self._llm_ready("experiment"):
                rag_query = " | ".join(
                    [
                        f"task={world_spec.get('task_name')}",
                        f"metric={world_spec.get('metric')}",
                        f"strategy={plan_spec.get('strategy')}",
                        f"solver_spec={json.dumps(solver_spec or {}, ensure_ascii=False)}",
                        f"recent_errors={json.dumps([self._truncate((o or {}).get('error'), 160) for o in prior_observations[-6:]], ensure_ascii=False)}",
                    ]
                )
                rag_result = await self._rag_retrieve_context(
                    agent_id=agent_id,
                    action="experiment",
                    run_id=None,
                    paper_id=None,
                    query_text=rag_query,
                    quotas={"observation": 5, "diagnosis": 5, "method_card": 2, "data_card": 1, "note": 1},
                    notes=notes,
                    observations=prior_observations,
                    data_card=data_card if isinstance(data_card, dict) else None,
                    method_card=method_card if isinstance(method_card, dict) else None,
                )
                rag_block = self._format_rag_prompt_block(result=rag_result)
                prompt = self._build_experiment_prompt(
                    world_spec=world_spec,
                    hypothesis=hypothesis,
                    plan_spec=plan_spec,
                    notes=notes,
                    observations=prior_observations,
                    data_card=data_card if isinstance(data_card, dict) else None,
                    method_card=method_card if isinstance(method_card, dict) else None,
                    exp_count=exp_count,
                    budget=budget,
                    rag_context=rag_block.get("context", ""),
                    rag_refs=rag_block.get("refs", []),
                    rag_status=rag_block.get("status", ""),
                )
                llm_result = await self._call_llm_json(agent_id=agent_id, action_name="experiment", prompt=prompt)
                if llm_result.get("ok") and isinstance(llm_result.get("data"), dict):
                    llm_experiment_plan = llm_result.get("data") or {}
                    strategy = llm_experiment_plan.get("strategy")
                    if isinstance(strategy, str) and strategy.strip():
                        run_config["strategy"] = strategy.strip()[:120]
                    cfg = llm_experiment_plan.get("config")
                    if isinstance(cfg, dict):
                        # Treat LLM config as solver override for this run.
                        run_config["solver_spec"] = self._merge_solver_plan(
                            {"solver_spec": run_config.get("solver_spec", {})},
                            {"solver_spec": cfg},
                        ).get("solver_spec")
                    validity_checks = self._safe_text_list(llm_experiment_plan.get("validity_checks"), limit=6, item_limit=180)
                    failure_modes = self._safe_text_list(llm_experiment_plan.get("failure_modes"), limit=6, item_limit=180)
                    if validity_checks:
                        run_config["validity_checks"] = validity_checks
                    if failure_modes:
                        run_config["failure_modes"] = failure_modes

            current_tick = int(await self.controller.run_system("timer", "get_tick"))
            result = await self.controller.run_environment(
                "science",
                "run_experiment",
                config=run_config,
                agent_id=agent_id,
                current_tick=current_tick,
            )

        observations = await self._get_state(agent_id, "observations") or []
        observation = {
            "run_id": (result or {}).get("run_id"),
            "task_name": (result or {}).get("task_name"),
            "metric_name": (result or {}).get("metric_name"),
            "raw_score": (result or {}).get("raw_score"),
            "score_norm": (result or {}).get("score_norm"),
            "dev_score": (result or {}).get("dev_score"),
            "dev_score_norm": (result or {}).get("dev_score_norm"),
            "submission_path": (result or {}).get("submission_path"),
            "model_path": (result or {}).get("model_path"),
            "solver_log_path": (result or {}).get("solver_log_path"),
            "solver_mode": (result or {}).get("solver_mode"),
            "fallback_reason": (result or {}).get("fallback_reason"),
            "eval_split": (result or {}).get("eval_split", "dev"),
            "stderr_tail": (result or {}).get("stderr_tail"),
            "code_workspace": (result or {}).get("code_workspace"),
            "code_log_path": (result or {}).get("code_log_path"),
            "code_artifacts": (result or {}).get("code_artifacts"),
            "dev_eval": (result or {}).get("dev_eval"),
            "executor_used": (result or {}).get("executor_used"),
            "executor_fallback_used": (result or {}).get("executor_fallback_used"),
            "ok": bool((result or {}).get("ok", False)),
            "error": (result or {}).get("error"),
            "elapsed_s": (result or {}).get("elapsed_s"),
            "strategy": run_config.get("strategy"),
            "config": run_config.get("solver_spec") or run_config,
            "llm_experiment_plan": llm_experiment_plan,
            "code_attempts": code_attempts,
        }
        observations.append(observation)
        await self._set_state(agent_id, "observations", observations)
        await self._set_state(agent_id, "run_history", observations)
        await self._set_state(agent_id, "exp_count", exp_count)
        if self._rag_index_on_experiment:
            rag_obs_docs = self._rag_docs_from_observation(
                world_spec=world_spec,
                agent_id=agent_id,
                observation=observation,
                action="experiment",
            )
            if rag_obs_docs:
                await self._rag_index_documents(
                    agent_id=agent_id,
                    action="experiment",
                    docs=rag_obs_docs,
                    run_id=str(observation.get("run_id") or ""),
                )

        score_norm = float((result or {}).get("score_norm", 0.0) or 0.0)
        dev_score_norm = (result or {}).get("dev_score_norm")
        dev_score_norm = float(dev_score_norm) if isinstance(dev_score_norm, (int, float)) else score_norm
        cost = float((result or {}).get("cost", 0.0) or 0.0)
        prev_best_dev = max(
            [float(o.get("dev_score_norm", 0.0) or 0.0) for o in prior_observations if bool(o.get("ok"))] or [0.0]
        )
        improvement = dev_score_norm - prev_best_dev
        reward = max(-0.06, min(0.12, 0.05 * dev_score_norm + 0.08 * improvement - 0.03 * cost))
        flags = self._experiment_error_flags(result or {})
        first_pass = self._is_first_pass_success(code_attempts=code_attempts, ok=bool((result or {}).get("ok", False))
)
        vram_eff = self._estimate_vram_efficiency(result=result or {}, world_spec=world_spec)
        if bool((result or {}).get("ok", False)):
            reward += self._experiment_success_reward
            if first_pass:
                reward += self._experiment_first_pass_bonus
            if isinstance(vram_eff, (int, float)):
                reward += self._experiment_vram_reward_weight * float(vram_eff)
        if flags.get("oom", False):
            reward -= self._experiment_oom_penalty
        if flags.get("typeerror", False):
            reward -= self._experiment_typeerror_penalty
        if not bool((result or {}).get("ok", False)):
            reward = min(reward, -0.02)
        reward = max(-1.0, min(2.0, float(reward)))

        # Keep an experiment -> review loop on taskboard for iterative refinement.
        if bool((result or {}).get("ok", False)):
            try:
                await self.controller.run_environment(
                    "science",
                    "task_create",
                    task_type="review",
                    payload={
                        "run_id": observation.get("run_id"),
                        "revision_reason": "post_experiment_diagnosis",
                    },
                    priority=6,
                )
            except Exception as e:
                logger.warning(f"Failed to enqueue post-experiment review task: {e}")

        ar = ActionResult.success(
            method_name="experiment",
            message="AIRS experiment executed.",
            data={
                "observation": observation,
                "exp_count": exp_count,
                "reward": reward,
                "effective_action": "experiment",
                "reward_components": {
                    "learning_reward": float(reward),
                    "experiment_reward": float(reward),
                    "experiment_score_norm": float(score_norm),
                    "experiment_dev_score_norm": float(dev_score_norm),
                    "experiment_improvement": float(improvement),
                    "experiment_first_pass_bonus": float(self._experiment_first_pass_bonus if first_pass else 0.0),
                    "experiment_vram_efficiency": float(vram_eff) if isinstance(vram_eff, (int, float)) else 0.0,
                    "experiment_oom_penalty": float(-self._experiment_oom_penalty if flags.get("oom", False) else 0.0),
                    "experiment_typeerror_penalty": float(
                        -self._experiment_typeerror_penalty if flags.get("typeerror", False) else 0.0
                    ),
                },
                "run_config": run_config,
                "llm_experiment_plan": llm_experiment_plan,
                "engineering_diagnostics": {
                    "first_pass_success": bool(first_pass),
                    "oom_flag": bool(flags.get("oom", False)),
                    "typeerror_flag": bool(flags.get("typeerror", False)),
                    "vram_efficiency": float(vram_eff) if isinstance(vram_eff, (int, float)) else None,
                },
            },
        )
        await self._append_trace(agent_id, "experiment", reward, ar.data or {})
        return ar

    @AgentCall
    async def write(self, agent_id: str) -> ActionResult:
        world_spec = await self.controller.run_environment("science", "get_world_spec")
        task_name = str(world_spec.get("task_name") or "unknown_task")

        hypothesis = await self._get_state(agent_id, "hypothesis") or []
        exp_count = await self._get_state(agent_id, "exp_count") or 0
        local_notes = await self._get_state(agent_id, "notes") or []
        shared_notes = await self._get_state(agent_id, "shared_notes") or []
        notes = local_notes + shared_notes
        local_observations = await self._get_state(agent_id, "observations") or []
        shared_observations = await self._get_state(agent_id, "shared_observations") or []
        observations = local_observations + shared_observations
        plan_spec = await self._get_state(agent_id, "plan_spec") or {}

        precondition_failures = self._write_precondition_failures(
            hypothesis=hypothesis,
            notes=notes,
            observations=observations,
        )
        if precondition_failures:
            ar = self._action_error(
                "write",
                "Write preconditions not satisfied.",
                effective_action="write",
                detail={
                    "precondition_failed": True,
                    "precondition_failures": precondition_failures,
                    "counts": {
                        "hypothesis": len(hypothesis),
                        "notes": len(notes),
                        "observations": len(observations),
                    },
                },
            )
            await self._append_trace(agent_id, "write", 0.0, ar.data or {})
            return ar

        valid_runs = [obs for obs in observations if obs.get("ok") and obs.get("run_id")]
        if not valid_runs:
            ar = self._action_error(
                "write",
                "No successful experiment run available for submission.",
                effective_action="write",
                detail={"precondition_failed": True, "reason": "no_successful_run"},
            )
            await self._append_trace(agent_id, "write", 0.0, ar.data or {})
            return ar

        best_run = sorted(
            valid_runs,
            key=lambda r: float(
                r.get("dev_score_norm", r.get("score_norm", 0.0)) or 0.0
            ),
            reverse=True,
        )[0]
        best_run_for_write = dict(best_run)
        local_preflight = self._local_submission_format_check(str(best_run_for_write.get("submission_path") or ""))
        if not bool(local_preflight.get("ok")):
            ar = self._action_error(
                "write",
                "Write preflight failed before environment evaluation.",
                effective_action="write",
                detail={
                    "precondition_failed": True,
                    "reason": "local_format_check_failed",
                    "local_preflight": local_preflight,
                    "best_run_id": best_run_for_write.get("run_id"),
                },
            )
            await self._append_trace(agent_id, "write", 0.0, ar.data or {})
            return ar

        official_eval = await self.controller.run_environment(
            "science",
            "evaluate_submission",
            submission_path=best_run_for_write.get("submission_path"),
        )
        if not isinstance(official_eval, dict) or not bool(official_eval.get("ok")):
            eval_reason = str((official_eval or {}).get("reason") or "")
            eval_error_type = str((official_eval or {}).get("error_type") or "")
            cache_hit = bool((official_eval or {}).get("cache_hit"))
            cache_repeat_penalty = self._write_cache_repeat_penalty if cache_hit else 0.0
            write_reward = -float(cache_repeat_penalty)
            reward_components = {
                "terminal_quality_reward": float(write_reward),
                "learning_reward": float(write_reward),
                "paper_write_reward": float(write_reward),
                "write_eval_success_bonus": 0.0,
                "write_format_pass_reward": 0.0,
                "write_cache_repeat_penalty": float(-cache_repeat_penalty),
            }
            if self._write_defer_on_system_error and (
                eval_error_type == "system_error" or eval_reason.startswith("system_error:")
            ):
                ar = ActionResult.success(
                    method_name="write",
                    message="Write deferred: evaluation environment precheck failed.",
                    data={
                        "ok": False,
                        "deferred": True,
                        "reason": "evaluation_system_error",
                        "best_run_id": best_run_for_write.get("run_id"),
                        "evaluate_submission": official_eval,
                        "local_preflight": local_preflight,
                        "reward": float(write_reward),
                        "effective_action": "write",
                        "reward_components": reward_components,
                    },
                )
                await self._append_trace(agent_id, "write", write_reward, ar.data or {})
                return ar
            ar = self._action_error(
                "write",
                "Final write requires successful test evaluation but evaluate_submission failed.",
                effective_action="write",
                detail={
                    "precondition_failed": True,
                    "reason": "evaluate_submission_failed",
                    "evaluate_submission": official_eval,
                    "local_preflight": local_preflight,
                    "best_run_id": best_run_for_write.get("run_id"),
                    "reward": float(write_reward),
                    "reward_components": reward_components,
                },
            )
            await self._append_trace(agent_id, "write", write_reward, ar.data or {})
            return ar

        best_run_for_write["metric_name"] = official_eval.get("metric_name") or best_run_for_write.get("metric_name")
        best_run_for_write["raw_score"] = official_eval.get("raw_score")
        best_run_for_write["score_norm"] = official_eval.get("score_norm")
        best_run_for_write["official_eval"] = official_eval
        metric_name = str(best_run_for_write.get("metric_name") or world_spec.get("metric") or "score")
        citations_for_prompt: List[str] = []
        for note in notes:
            for card in (note or {}).get("cards", []) or []:
                cid = card.get("citation_id")
                if cid and cid not in citations_for_prompt:
                    citations_for_prompt.append(str(cid))
        obs_refs_for_prompt = []
        for obs in observations:
            run_id = obs.get("run_id")
            if run_id:
                obs_refs_for_prompt.append(f"RUN@{run_id}")

        llm_write_spec: Optional[Dict[str, Any]] = None
        if self._llm_ready("write"):
            write_data_card = await self._get_state(agent_id, "data_card")
            write_method_card = await self._get_state(agent_id, "method_card")
            rag_query = " | ".join(
                [
                    f"task={world_spec.get('task_name')}",
                    f"metric={metric_name}",
                    f"best_run={json.dumps({'run_id': best_run_for_write.get('run_id'), 'score_norm': best_run_for_write.get('score_norm'), 'strategy': best_run_for_write.get('strategy')}, ensure_ascii=False)}",
                    f"paper_claims_seed={json.dumps(hypothesis[:6], ensure_ascii=False)}",
                ]
            )
            rag_result = await self._rag_retrieve_context(
                agent_id=agent_id,
                action="write",
                run_id=str(best_run_for_write.get("run_id") or ""),
                paper_id=None,
                query_text=rag_query,
                quotas={"observation": 3, "data_card": 2, "method_card": 2, "paper": 1, "review": 1, "note": 1},
                notes=notes,
                observations=observations,
                data_card=write_data_card if isinstance(write_data_card, dict) else None,
                method_card=write_method_card if isinstance(write_method_card, dict) else None,
                paper=None,
            )
            rag_block = self._format_rag_prompt_block(result=rag_result)
            prompt = self._build_write_prompt(
                world_spec=world_spec,
                best_run=best_run_for_write,
                hypothesis=hypothesis,
                plan_spec=plan_spec,
                citations=citations_for_prompt,
                observation_refs=obs_refs_for_prompt,
                notes=notes,
                observations=observations,
                rag_context=rag_block.get("context", ""),
                rag_refs=rag_block.get("refs", []),
                rag_status=rag_block.get("status", ""),
            )
            llm_result = await self._call_llm_json(agent_id=agent_id, action_name="write", prompt=prompt)
            if llm_result.get("ok") and isinstance(llm_result.get("data"), dict):
                llm_write_spec = llm_result.get("data") or {}

        paper = self._build_paper_payload(
            task_name=task_name,
            metric_name=metric_name,
            best_run=best_run_for_write,
            notes=notes,
            observations=observations,
            hypothesis=hypothesis,
            plan_spec=plan_spec,
            exp_count=int(exp_count or 0),
            llm_write_spec=llm_write_spec,
        )
        paper["author_id"] = agent_id
        paper["citation_owner_map"] = self._build_citation_owner_map(
            agent_id=agent_id,
            local_notes=local_notes,
            shared_notes=shared_notes,
        )

        submit_info = await self.controller.run_environment("science", "submit_paper", paper=paper)
        paper_id = (submit_info or {}).get("paper_id")
        metrics = await self.controller.run_environment("science", "evaluate_paper", paper=paper, paper_id=paper_id)
        reward = float(metrics.get("fitness", 0.0) or 0.0)
        eval_success_bonus = self._write_eval_success_bonus if bool(official_eval.get("ok", False)) else 0.0
        format_pass_reward = self._write_format_pass_reward if bool((official_eval.get("preflight") or {}).get("ok", True)) else 0.0
        cache_repeat_penalty = self._write_cache_repeat_penalty if (bool(official_eval.get("cache_hit")) and not bool(official_eval.get("ok"))) else 0.0
        reward += float(eval_success_bonus) + float(format_pass_reward) - float(cache_repeat_penalty)

        contribution_credit = self._compute_contribution_credit(
            agent_id=agent_id,
            paper=paper,
            metrics=metrics,
            shared_observations=shared_observations,
        )
        await self._grant_credits(contribution_credit, source="paper_write", reference_id=paper_id)

        await self._set_state(agent_id, "last_fitness", metrics)
        await self._set_state(agent_id, "last_paper_id", paper_id)
        await self._inc_state_number(agent_id, "paper_write_count", 1)

        try:
            await self.controller.run_environment(
                "science",
                "task_create",
                task_type="review",
                payload={"paper_id": paper_id},
                priority=7,
            )
            await self.controller.run_environment(
                "science",
                "task_create",
                task_type="replicate",
                payload={"paper_id": paper_id},
                priority=8,
            )
        except Exception as e:
            logger.warning(f"Failed to create follow-up tasks for {paper_id}: {e}")

        await self._log_paper_result(agent_id, paper_id, paper, metrics, source="write")
        if self._rag_index_on_write:
            rag_paper_docs = self._rag_docs_from_paper(
                world_spec=world_spec,
                agent_id=agent_id,
                paper=paper,
                paper_id=paper_id,
                action="write",
            )
            if rag_paper_docs:
                await self._rag_index_documents(
                    agent_id=agent_id,
                    action="write",
                    docs=rag_paper_docs,
                    run_id=str(best_run_for_write.get("run_id") or ""),
                    paper_id=str(paper_id or ""),
                )
        ar = ActionResult.success(
            method_name="write",
            message="AIRS submission written and evaluated.",
            data={
                "metrics": metrics,
                "paper_id": paper_id,
                "paper": paper,
                "llm_write_spec": llm_write_spec,
                "official_eval": official_eval,
                "credit_by_agent": contribution_credit,
                "reward": reward,
                "effective_action": "write",
                "reward_components": {
                    "terminal_quality_reward": float(reward),
                    "learning_reward": float(reward),
                    "paper_write_reward": float(reward),
                    "write_eval_success_bonus": float(eval_success_bonus),
                    "write_format_pass_reward": float(format_pass_reward),
                    "write_cache_repeat_penalty": float(-cache_repeat_penalty),
                },
            },
        )
        await self._append_trace(agent_id, "write", reward, ar.data or {})
        return ar

    @AgentCall
    async def review(
        self,
        agent_id: str,
        paper_id: Optional[str] = None,
        run_id: Optional[str] = None,
        submission: Optional[Dict[str, Any]] = None,
    ) -> ActionResult:
        if self._strict_review_mode and not paper_id and not run_id:
            ar = ActionResult.success(
                method_name="review",
                message="Review deferred: no paper_id or run_id available in strict mode.",
                data={
                    "ok": False,
                    "review_deferred": True,
                    "reason": "no_artifact_to_review",
                    "reward": 0.0,
                    "effective_action": "review",
                    "reward_components": {
                        "review_reward": 0.0,
                        "learning_reward": 0.0,
                    },
                },
            )
            await self._append_trace(agent_id, "review", 0.0, ar.data or {})
            return ar

        if paper_id:
            paper = await self.controller.run_environment("science", "get_paper", paper_id=paper_id)
            if paper:
                metrics = await self.controller.run_environment("science", "evaluate_paper", paper=paper, paper_id=paper_id)
                author_id = (paper or {}).get("author_id")
                self_review = bool(author_id and str(author_id) == str(agent_id))

                llm_review_note: Optional[Dict[str, Any]] = None
                if self._llm_ready("review"):
                    prompt = self._build_review_prompt(paper=paper, metrics=metrics)
                    llm_result = await self._call_llm_json(agent_id=agent_id, action_name="review", prompt=prompt)
                    if llm_result.get("ok") and isinstance(llm_result.get("data"), dict):
                        llm_review_note = llm_result.get("data") or {}

                review_note = llm_review_note if isinstance(llm_review_note, dict) else {}
                heuristic_note = self._heuristic_review_note(paper=paper, metrics=metrics)
                if not review_note:
                    review_note = heuristic_note

                strengths = review_note.get("strengths")
                if not isinstance(strengths, list) or not strengths:
                    strengths = heuristic_note.get("strengths") or []
                issues = self._normalize_review_issues(review_note)
                if not issues:
                    issues = self._normalize_review_issues(heuristic_note)
                review_note["strengths"] = strengths[:6]
                review_note["issues"] = issues[:10]
                if not review_note.get("revision_actions"):
                    review_note["revision_actions"] = heuristic_note.get("revision_actions") or []
                if not review_note.get("summary"):
                    review_note["summary"] = heuristic_note.get("summary")
                review_note["paper_id"] = paper_id
                review_note["reviewer_id"] = agent_id

                critique_quality = self._score_review_quality(
                    review_note=review_note,
                    issues=issues,
                    self_review=self_review,
                    replication_ok=bool(metrics.get("replication_ok", False)),
                )
                review_score = float(critique_quality.get("critique_score", 0.0) or 0.0)
                paper_context = json.dumps(
                    {
                        "paper": {
                            "title": (paper or {}).get("title"),
                            "abstract": (paper or {}).get("abstract"),
                            "citations": (paper or {}).get("citations"),
                            "observation_refs": (paper or {}).get("observation_refs"),
                        },
                        "metrics": metrics,
                    },
                    ensure_ascii=False,
                )
                qgr_gate = await self._qgr_validate_review(
                    review_note=review_note,
                    issues=issues,
                    context_text=paper_context,
                )
                await self._log_review_gate(agent_id=agent_id, paper_id=paper_id, run_id=None, gate=qgr_gate)
                if not bool(qgr_gate.get("valid", False)):
                    ar = self._action_error(
                        "review",
                        "QGR gate rejected review: quality below threshold.",
                        effective_action="review",
                        detail={
                            "precondition_failed": True,
                            "reason": "review_quality_below_threshold",
                            "paper_id": paper_id,
                            "qgr_gate": qgr_gate,
                            "review_note": review_note,
                        },
                    )
                    await self._append_trace(agent_id, "review", 0.0, ar.data or {})
                    return ar

                reward = float(self._qgr_base_reward)
                quality_bonus = 0.0
                if int((qgr_gate.get("metrics") or {}).get("citation_count", 0) or 0) >= self._qgr_min_citations:
                    quality_bonus = float(self._qgr_quality_bonus)
                    reward += quality_bonus
                local_runs = await self._get_state(agent_id, "observations") or []
                pred = self._qgr_predictive_bonus(
                    issues=issues,
                    run_history=local_runs,
                    target_run_id=((paper or {}).get("claimed_results") or {}).get("run_id") if isinstance((paper or {}).get("claimed_results"), dict) else None,
                )
                predictive_bonus = float(pred.get("bonus", 0.0) or 0.0)
                reward += predictive_bonus
                reward += max(-0.08, min(0.10, 0.10 * review_score - 0.02))
                reward = max(-0.3, min(2.5, reward))

                validation_tasks = await self._spawn_review_validation_tasks(
                    paper_id=str(paper_id),
                    reviewer_id=agent_id,
                    review_note=review_note,
                    critique_quality=critique_quality,
                )
                followup_tasks = await self._spawn_qgr_followup_tasks(
                    paper_id=str(paper_id),
                    run_id=((paper or {}).get("claimed_results") or {}).get("run_id") if isinstance((paper or {}).get("claimed_results"), dict) else None,
                    score=review_score,
                    issues=issues,
                )

                if bool(metrics.get("replication_ok", False)) and int(critique_quality.get("issue_count", 0) or 0) <= 1:
                    if author_id:
                        await self._grant_credits({str(author_id): 0.03}, source="review_verified_strength", reference_id=paper_id)
                if author_id and isinstance(metrics, dict):
                    await self._set_state(str(author_id), "last_fitness", metrics)

                await self._set_state(
                    agent_id,
                    "last_review_quality",
                    {
                        "paper_id": paper_id,
                        "critique_quality": critique_quality,
                        "review_score": review_score,
                        "self_review": self_review,
                    },
                )
                await self._log_paper_result(agent_id, paper_id, paper, metrics, source="review")
                await self._inc_state_number(agent_id, "review_count", 1)
                ar = ActionResult.success(
                    method_name="review",
                    message="Paper reviewed with evidence-backed strengths and falsifiable critiques.",
                    data={
                        "paper_id": paper_id,
                        "score": review_score,
                        "metrics": metrics,
                        "reward": reward,
                        "effective_action": "review",
                        "reward_components": {
                            "review_reward": float(reward),
                            "learning_reward": float(reward),
                            "review_critique_score": float(review_score),
                            "review_flattery_penalty": float(critique_quality.get("flattery_penalty", 0.0) or 0.0),
                            "qgr_base_reward": float(self._qgr_base_reward),
                            "qgr_quality_bonus": float(quality_bonus),
                            "qgr_predictive_bonus": float(predictive_bonus),
                        },
                        "review_note": review_note,
                        "critique_quality": critique_quality,
                        "qgr_gate": qgr_gate,
                        "predictive_match": pred,
                        "validation_tasks": validation_tasks,
                        "followup_tasks": followup_tasks,
                        "llm_used": llm_review_note is not None,
                        "self_review": self_review,
                    },
                )
                await self._append_trace(agent_id, "review", reward, ar.data or {})
                return ar

            if self._strict_task_dependencies:
                ar = self._action_error(
                    "review",
                    f"Paper {paper_id} not found.",
                    effective_action="review",
                    detail={"precondition_failed": True, "paper_id": paper_id, "reason": "paper_not_found"},
                )
                await self._append_trace(agent_id, "review", 0.0, ar.data or {})
                return ar
            if self._strict_review_mode:
                ar = ActionResult.success(
                    method_name="review",
                    message=f"Review deferred: paper {paper_id} not found.",
                    data={
                        "ok": False,
                        "review_deferred": True,
                        "reason": "paper_not_found",
                        "paper_id": paper_id,
                        "reward": 0.0,
                        "effective_action": "review",
                        "reward_components": {"review_reward": 0.0, "learning_reward": 0.0},
                    },
                )
                await self._append_trace(agent_id, "review", 0.0, ar.data or {})
                return ar

        if run_id and submission is None:
            run_history = await self._get_state(agent_id, "observations") or []
            plan_spec = await self._get_state(agent_id, "plan_spec") or {}
            world_spec = await self.controller.run_environment("science", "get_world_spec")
            latest = None
            for run in reversed(run_history):
                if str(run.get("run_id") or "") == str(run_id):
                    latest = run
                    break
            if latest is None and run_history:
                latest = run_history[-1]
            if latest is None:
                latest = {}

            next_plan = self._derive_next_solver_plan_from_history(plan_spec=plan_spec, run_history=run_history)
            await self._set_state(agent_id, "plan_spec", next_plan)

            review_note = {
                "summary": "Iterative run-level diagnosis generated from run history.",
                "strengths": [
                    {
                        "id": "S-001",
                        "claim": "Latest run produced an evaluable submission artifact.",
                        "evidence": [f"RUN@{latest.get('run_id')}", f"score_norm={float(latest.get('score_norm', 0.0) or 0.0):.4f}"],
                        "confidence": 0.6,
                        "verification": {"kind": "replicate", "params": {"mode": "score_consistency"}},
                    }
                ],
                "issues": [
                    {
                        "id": "I-001",
                        "type": "iterative_refinement",
                        "severity": 0.65,
                        "claim": "Current configuration may still be suboptimal; schedule next experiment with updated solver plan.",
                        "evidence_refs": [f"RUN@{latest.get('run_id')}"] if latest.get("run_id") else [],
                        "proposed_test": {"kind": "ablation", "params": {"focus": "solver_hyperparams"}},
                        "suggested_fix": "Execute next experiment using updated plan_spec to validate improvement.",
                    }
                ],
                "revision_actions": ["Launch next experiment with updated solver_spec and compare dev_score_norm trend."],
                "run_id": latest.get("run_id"),
            }

            enqueued_task_id = None
            try:
                create_res = await self.controller.run_environment(
                    "science",
                    "task_create",
                    task_type="experiment",
                    payload={
                        "config": {
                            "strategy": next_plan.get("strategy"),
                            "solver_spec": next_plan.get("solver_spec"),
                            "source": "review_iteration",
                        },
                        "from_run_id": latest.get("run_id"),
                        "revision_reason": "iterative_review",
                    },
                    priority=7,
                )
                if isinstance(create_res, dict) and create_res.get("ok"):
                    enqueued_task_id = (create_res.get("task") or {}).get("task_id")
            except Exception as e:
                logger.warning(f"Failed to enqueue iterative experiment task: {e}")

            reward = 0.02
            run_context = json.dumps({"latest": latest, "next_plan": next_plan}, ensure_ascii=False)
            qgr_gate = await self._qgr_validate_review(
                review_note=review_note,
                issues=self._normalize_review_issues(review_note),
                context_text=run_context,
            )
            await self._log_review_gate(agent_id=agent_id, paper_id=None, run_id=str(latest.get("run_id") or run_id or ""), gate=qgr_gate)
            if not bool(qgr_gate.get("valid", False)):
                ar = self._action_error(
                    "review",
                    "QGR gate rejected run-level review.",
                    effective_action="review",
                    detail={
                        "precondition_failed": True,
                        "reason": "review_quality_below_threshold",
                        "run_id": latest.get("run_id") or run_id,
                        "qgr_gate": qgr_gate,
                        "review_note": review_note,
                    },
                )
                await self._append_trace(agent_id, "review", 0.0, ar.data or {})
                return ar

            ar = ActionResult.success(
                method_name="review",
                message="Run-level review completed and next experiment queued.",
                data={
                    "run_id": latest.get("run_id"),
                    "task_name": world_spec.get("task_name"),
                    "review_note": review_note,
                    "next_plan_spec": next_plan,
                    "enqueued_experiment_task_id": enqueued_task_id,
                    "reward": reward,
                    "effective_action": "review",
                    "reward_components": {
                        "review_reward": float(reward),
                        "learning_reward": float(reward),
                        "review_iterative_planning_reward": float(reward),
                    },
                    "qgr_gate": qgr_gate,
                },
            )
            await self._append_trace(agent_id, "review", reward, ar.data or {})
            return ar

        if self._strict_review_mode:
            ar = ActionResult.success(
                method_name="review",
                message="Review skipped in strict mode: no valid artifact context.",
                data={
                    "ok": False,
                    "review_deferred": True,
                    "reason": "strict_mode_no_valid_artifact",
                    "reward": 0.0,
                    "effective_action": "review",
                    "reward_components": {"review_reward": 0.0, "learning_reward": 0.0},
                },
            )
            await self._append_trace(agent_id, "review", 0.0, ar.data or {})
            return ar

        if submission is None:
            submission = {
                "author_id": agent_id,
                "hypothesis": await self._get_state(agent_id, "hypothesis") or [],
            }
        review_score = min(1.0, len(submission.get("hypothesis") or []) / 4.0)
        review_note = {
            "summary": "Fallback review without paper object.",
            "strengths": [
                {
                    "id": "S-001",
                    "claim": "Hypothesis is explicitly stated.",
                    "evidence": [f"hypothesis_count={len(submission.get('hypothesis') or [])}"],
                    "confidence": 0.5,
                    "verification": {"kind": "static_check", "params": {"focus": "hypothesis_presence"}},
                }
            ],
            "issues": [
                {
                    "id": "I-001",
                    "type": "evidence_missing",
                    "severity": 0.6,
                    "claim": "No full paper artifact available for scientific review.",
                    "evidence_refs": [],
                    "proposed_test": {"kind": "static_check", "params": {"focus": "paper_availability"}},
                    "suggested_fix": "Generate paper artifact before formal peer review.",
                }
            ],
            "revision_actions": ["Produce a paper object and rerun review with evidence map."],
        }
        reward = max(-0.05, min(0.08, 0.08 * review_score - 0.01))
        ar = ActionResult.success(
            method_name="review",
            message="Review completed.",
            data={
                "score": review_score,
                "reward": reward,
                "effective_action": "review",
                "reward_components": {
                    "review_reward": float(reward),
                    "learning_reward": float(reward),
                },
                "review_note": review_note,
            },
        )
        await self._append_trace(agent_id, "review", reward, ar.data or {})
        return ar

    @AgentCall
    async def verify_strength(
        self,
        agent_id: str,
        paper_id: Optional[str] = None,
        reviewer_id: Optional[str] = None,
        strength: Optional[Dict[str, Any]] = None,
        test: Optional[Dict[str, Any]] = None,
    ) -> ActionResult:
        if not paper_id:
            ar = self._action_error(
                "verify_strength",
                "paper_id is required for strength verification.",
                effective_action="verify_strength",
                detail={"precondition_failed": True, "reason": "paper_id_required"},
            )
            await self._append_trace(agent_id, "verify_strength", 0.0, ar.data or {})
            return ar

        paper = await self.controller.run_environment("science", "get_paper", paper_id=paper_id)
        if not isinstance(paper, dict):
            ar = self._action_error(
                "verify_strength",
                f"Paper {paper_id} not found.",
                effective_action="verify_strength",
                detail={"precondition_failed": True, "reason": "paper_not_found", "paper_id": paper_id},
            )
            await self._append_trace(agent_id, "verify_strength", 0.0, ar.data or {})
            return ar

        metrics = await self.controller.run_environment("science", "evaluate_paper", paper=paper, paper_id=paper_id)
        score_norm = float(metrics.get("score_norm", 0.0) or 0.0)
        replication_ok = bool(metrics.get("replication_ok", False))
        test_kind = str((test or {}).get("kind") or "replicate").strip().lower()
        passed = False
        evidence = [f"score_norm={score_norm:.4f}", f"replication_ok={replication_ok}"]
        replication_submit = None

        if test_kind == "replicate":
            replication_submit = await self.controller.run_environment(
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
            await self._inc_state_number(str(reviewer_id), "credit_buffer", reviewer_delta)
            await self._inc_state_number(str(reviewer_id), "contribution_credit_total", reviewer_delta)
            await self._set_state(
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
        await self._append_trace(agent_id, "verify_strength", reward, ar.data or {})
        return ar

    @AgentCall
    async def verify_issue(
        self,
        agent_id: str,
        paper_id: Optional[str] = None,
        reviewer_id: Optional[str] = None,
        issue: Optional[Dict[str, Any]] = None,
        test: Optional[Dict[str, Any]] = None,
    ) -> ActionResult:
        if not paper_id:
            ar = self._action_error(
                "verify_issue",
                "paper_id is required for issue verification.",
                effective_action="verify_issue",
                detail={"precondition_failed": True, "reason": "paper_id_required"},
            )
            await self._append_trace(agent_id, "verify_issue", 0.0, ar.data or {})
            return ar

        paper = await self.controller.run_environment("science", "get_paper", paper_id=paper_id)
        if not isinstance(paper, dict):
            ar = self._action_error(
                "verify_issue",
                f"Paper {paper_id} not found.",
                effective_action="verify_issue",
                detail={"precondition_failed": True, "reason": "paper_not_found", "paper_id": paper_id},
            )
            await self._append_trace(agent_id, "verify_issue", 0.0, ar.data or {})
            return ar

        metrics = await self.controller.run_environment("science", "evaluate_paper", paper=paper, paper_id=paper_id)
        test_kind = str((test or {}).get("kind") or "replicate").strip().lower()
        issue_type = str((issue or {}).get("type") or "")
        severity = self._clamp01((issue or {}).get("severity", 0.6))
        issue_validated = False
        evidence = []
        replication_submit = None

        if test_kind == "replicate":
            replication_submit = await self.controller.run_environment(
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

        # Reward both scientific skepticism and correct criticism.
        reward = (0.04 if issue_validated else -0.02) * (0.7 + 0.3 * severity)
        if reviewer_id and str(reviewer_id) != str(agent_id):
            reviewer_delta = 0.025 if issue_validated else -0.025
            await self._inc_state_number(str(reviewer_id), "credit_buffer", reviewer_delta)
            await self._inc_state_number(str(reviewer_id), "contribution_credit_total", reviewer_delta)
            await self._set_state(
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
        await self._append_trace(agent_id, "verify_issue", reward, ar.data or {})
        return ar

    @AgentCall
    async def replicate(
        self,
        agent_id: str,
        paper_id: Optional[str] = None,
        intervention: Optional[Dict[str, float]] = None,
        n_samples: int = 50,
    ) -> ActionResult:
        del intervention, n_samples

        target_paper_id = paper_id or await self._get_state(agent_id, "last_paper_id")
        if self._strict_task_dependencies and not target_paper_id:
            ar = self._action_error(
                "replicate",
                "Replication requires a target paper_id under strict task dependencies.",
                effective_action="replicate",
                detail={"precondition_failed": True, "reason": "paper_id_required"},
            )
            await self._append_trace(agent_id, "replicate", 0.0, ar.data or {})
            return ar

        world_spec = await self.controller.run_environment("science", "get_world_spec")
        target_paper = await self.controller.run_environment("science", "get_paper", paper_id=target_paper_id)
        claimed_metrics = (target_paper or {}).get("claimed_results") if isinstance(target_paper, dict) else {}
        llm_replication_plan: Optional[Dict[str, Any]] = None
        replication_payload: Dict[str, Any] = {"mode": "score_consistency"}
        if self._llm_ready("replicate") and isinstance(target_paper, dict):
            prompt = self._build_replication_prompt(
                world_spec=world_spec,
                paper_id=str(target_paper_id),
                paper=target_paper,
                claimed_metrics=claimed_metrics or {},
            )
            llm_result = await self._call_llm_json(agent_id=agent_id, action_name="replicate", prompt=prompt)
            if llm_result.get("ok") and isinstance(llm_result.get("data"), dict):
                llm_replication_plan = llm_result.get("data") or {}
                mode = str(llm_replication_plan.get("mode") or "score_consistency").strip()
                if mode:
                    replication_payload["mode"] = mode[:80]
                protocol_name = llm_replication_plan.get("protocol_name")
                if isinstance(protocol_name, str) and protocol_name.strip():
                    replication_payload["protocol_name"] = protocol_name.strip()[:120]
                pass_criteria = llm_replication_plan.get("pass_criteria")
                if isinstance(pass_criteria, dict):
                    safe_criteria = {}
                    for key, value in pass_criteria.items():
                        if isinstance(value, (str, int, float, bool)):
                            safe_criteria[str(key)] = value
                    if safe_criteria:
                        replication_payload["pass_criteria"] = safe_criteria
                stress_tests = self._safe_text_list(llm_replication_plan.get("stress_tests"), limit=6, item_limit=180)
                failure_signals = self._safe_text_list(llm_replication_plan.get("failure_signals"), limit=6, item_limit=180)
                notes = self._safe_text_list(llm_replication_plan.get("notes"), limit=6, item_limit=180)
                if stress_tests:
                    replication_payload["stress_tests"] = stress_tests
                if failure_signals:
                    replication_payload["failure_signals"] = failure_signals
                if notes:
                    replication_payload["notes"] = notes

        replication_submit = await self.controller.run_environment(
            "science",
            "submit_replication",
            paper_id=target_paper_id,
            agent_id=agent_id,
            replication=replication_payload,
            source="agent_replicate",
        )

        reward = -0.01
        paper_metrics_after = None
        replication_signal = 0.0
        contradiction_bonus = 0.0
        confirmation_bonus = 0.0
        support_ratio = 0.0
        if bool((replication_submit or {}).get("ok")):
            support = (replication_submit or {}).get("support") or {}
            support_ratio = self._clamp01(support.get("support_ratio", 0.0))
            replication_signal = abs(support_ratio - 0.5) * 2.0
            contradiction_bonus = max(0.0, 0.5 - support_ratio) / 0.5
            confirmation_bonus = max(0.0, support_ratio - 0.5) / 0.5
            # Reward informative replication in both directions; detecting failure gets slightly higher credit.
            reward = 0.01 + 0.02 * replication_signal + 0.02 * contradiction_bonus + 0.01 * confirmation_bonus
            if support_ratio >= self._replicate_support_threshold:
                reward = max(float(reward), float(self._replicate_high_support_reward))
            paper_obj = await self.controller.run_environment("science", "get_paper", paper_id=target_paper_id)
            if paper_obj:
                paper_metrics_after = await self.controller.run_environment(
                    "science",
                    "evaluate_paper",
                    paper=paper_obj,
                    paper_id=target_paper_id,
                )
                author_id = (paper_obj or {}).get("author_id")
                if author_id and isinstance(paper_metrics_after, dict):
                    await self._set_state(author_id, "last_fitness", paper_metrics_after)

        replications = await self._get_state(agent_id, "replications") or []
        replications.append(
            {
                "paper_id": target_paper_id,
                "support": (replication_submit or {}).get("support") if isinstance(replication_submit, dict) else None,
                "ok": bool((replication_submit or {}).get("ok", False)),
            }
        )
        await self._set_state(agent_id, "replications", replications)
        await self._inc_state_number(agent_id, "replication_count", 1)

        ar = ActionResult.success(
            method_name="replicate",
            message="Replication executed.",
            data={
                "paper_id": target_paper_id,
                "replication_submit": replication_submit,
                "replication_payload": replication_payload,
                "llm_replication_plan": llm_replication_plan,
                "paper_metrics_after_replication": paper_metrics_after,
                "reward": reward,
                "effective_action": "replicate",
                "reward_components": {
                    "learning_reward": float(reward),
                    "replicate_reward": float(reward),
                    "replication_support_reward": float(reward),
                    "replication_signal": float(replication_signal),
                    "replication_contradiction_bonus": float(contradiction_bonus),
                    "replication_confirmation_bonus": float(confirmation_bonus),
                    "replication_support_ratio": float(support_ratio),
                    "replication_high_support_reward": float(
                        self._replicate_high_support_reward if support_ratio >= self._replicate_support_threshold else 0.0
                    ),
                },
            },
        )
        await self._append_trace(agent_id, "replicate", reward, ar.data or {})
        return ar

    @AgentCall
    async def claim_task(
        self,
        agent_id: str,
        task_id: Optional[str] = None,
        task_type: Optional[str] = None,
    ) -> ActionResult:
        current_tick = int(await self.controller.run_system("timer", "get_tick"))
        backoff_until = int((await self._get_state(agent_id, "claim_backoff_until_tick")) or 0)
        fail_streak = int((await self._get_state(agent_id, "claim_fail_streak")) or 0)
        if current_tick < backoff_until:
            wait_ticks = max(0, backoff_until - current_tick)
            ar = ActionResult.success(
                method_name="claim_task",
                message=f"Claim backoff active for {wait_ticks} tick(s).",
                data={
                    "ok": False,
                    "reason": "claim_backoff_active",
                    "wait_ticks": wait_ticks,
                    "reward": -self._claim_cost,
                    "effective_action": "claim_backoff",
                    "reward_components": {
                        "task_claim_reward": float(-self._claim_cost),
                        "task_claim_cost": float(self._claim_cost),
                        "learning_reward": float(-self._claim_cost),
                    },
                },
            )
            await self._append_trace(agent_id, "claim_task", -self._claim_cost, ar.data or {})
            return ar

        selected_task_id = task_id
        llm_selection: Optional[Dict[str, Any]] = None
        if not selected_task_id:
            listed = await self.controller.run_environment("science", "task_list", status="open", current_tick=current_tick)
            tasks = (listed or {}).get("tasks", []) if isinstance(listed, dict) else []
            if task_type:
                tasks = [t for t in tasks if t.get("task_type") == task_type]
            if tasks:
                world_spec = await self.controller.run_environment("science", "get_world_spec", current_tick=current_tick)
                role_plan = await self._get_state(agent_id, "llm_role_plan")
                active_task_name = str(world_spec.get("task_name") or "")
                role_plan_stale = not isinstance(role_plan, dict) or role_plan.get("task_name") != active_task_name
                if self._llm_ready("claim_task") and role_plan_stale:
                    hypothesis = await self._get_state(agent_id, "hypothesis") or []
                    notes = await self._get_state(agent_id, "notes") or []
                    observations = await self._get_state(agent_id, "observations") or []
                    prompt = self._build_task_role_prompt(
                        world_spec=world_spec,
                        open_tasks=tasks,
                        hypothesis=hypothesis,
                        notes_count=len(notes),
                        observations_count=len(observations),
                    )
                    llm_result = await self._call_llm_json(agent_id=agent_id, action_name="claim_task", prompt=prompt)
                    if llm_result.get("ok") and isinstance(llm_result.get("data"), dict):
                        llm_selection = llm_result.get("data") or {}
                        role_plan = {
                            "task_name": active_task_name,
                            "role_name": self._truncate(llm_selection.get("role_name"), 80),
                            "preferred_task_types": self._safe_task_types(
                                llm_selection.get("preferred_task_types"),
                                fallback=[
                                    "prepare_data",
                                    "profile_data",
                                    "retrieve_literature",
                                    "experiment",
                                    "hypothesize",
                                    "write",
                                    "review",
                                    "replicate",
                                    "verify_issue",
                                    "verify_strength",
                                    "read",
                                ],
                            ),
                            "fallback_if_blocked": self._safe_task_types(
                                llm_selection.get("fallback_if_blocked"),
                                fallback=["verify_issue", "verify_strength", "prepare_data", "profile_data", "read", "hypothesize"],
                            ),
                            "primary_task_id": str(llm_selection.get("primary_task_id") or ""),
                            "selection_rationale": self._safe_text_list(
                                llm_selection.get("selection_rationale"),
                                limit=5,
                                item_limit=220,
                            ),
                            "risk_controls": self._safe_text_list(
                                llm_selection.get("risk_controls"),
                                limit=5,
                                item_limit=220,
                            ),
                        }
                        await self._set_state(agent_id, "llm_role_plan", role_plan)

                preferred_types = []
                primary_task_id = ""
                if isinstance(role_plan, dict):
                    preferred_types = self._safe_task_types(role_plan.get("preferred_task_types"))
                    preferred_types.extend(
                        x for x in self._safe_task_types(role_plan.get("fallback_if_blocked")) if x not in preferred_types
                    )
                    primary_task_id = str(role_plan.get("primary_task_id") or "")

                task_map = {str(t.get("task_id")): t for t in tasks}
                if primary_task_id and primary_task_id in task_map:
                    selected_task_id = primary_task_id
                if not selected_task_id and preferred_types:
                    for preferred in preferred_types:
                        preferred_tasks = [t for t in tasks if str(t.get("task_type")) == preferred and bool(t.get("ready", True))]
                        if preferred_tasks:
                            selected_task_id = preferred_tasks[0].get("task_id")
                            break
                if not selected_task_id:
                    ready_tasks = [t for t in tasks if bool(t.get("ready", True))]
                    selected_task_id = (ready_tasks[0] if ready_tasks else tasks[0]).get("task_id")

        if not selected_task_id:
            fail_streak += 1
            delay = min(self._claim_backoff_max, self._claim_backoff_base * (2 ** max(0, fail_streak - 1)))
            await self._set_state(agent_id, "claim_fail_streak", fail_streak)
            await self._set_state(agent_id, "claim_backoff_until_tick", current_tick + delay)
            ar = ActionResult.success(
                method_name="claim_task",
                message="No open task available.",
                data={
                    "ok": False,
                    "reason": "no_open_task",
                    "backoff_ticks": delay,
                    "reward": -self._claim_cost,
                    "effective_action": "claim_task",
                    "reward_components": {
                        "task_claim_reward": float(-self._claim_cost),
                        "task_claim_cost": float(self._claim_cost),
                        "learning_reward": float(-self._claim_cost),
                    },
                },
            )
            await self._append_trace(agent_id, "claim_task", -self._claim_cost, ar.data or {})
            return ar

        claim_res = await self.controller.run_environment(
            "science",
            "task_claim",
            task_id=selected_task_id,
            agent_id=agent_id,
            current_tick=current_tick,
        )
        ok = bool((claim_res or {}).get("ok"))
        task = (claim_res or {}).get("task")
        if ok:
            fail_streak = 0
            await self._set_state(agent_id, "claim_fail_streak", 0)
            await self._set_state(agent_id, "claim_backoff_until_tick", current_tick)
        else:
            fail_streak += 1
            reason = str((claim_res or {}).get("reason") or "")
            if reason == "not_active_worker":
                delay = self._claim_backoff_max
            else:
                delay = min(self._claim_backoff_max, self._claim_backoff_base * (2 ** max(0, fail_streak - 1)))
            await self._set_state(agent_id, "claim_fail_streak", fail_streak)
            await self._set_state(agent_id, "claim_backoff_until_tick", current_tick + delay)
        reward = (0.01 if ok else 0.0) - self._claim_cost
        if ok:
            await self._set_state(agent_id, "current_task_id", selected_task_id)
            await self._set_state(agent_id, "current_task_type", str((task or {}).get("task_type") or ""))

        auto_dispatch: Optional[ActionResult] = None
        task_type_lower = str((task or {}).get("task_type") or "").strip().lower()
        if ok and self._claim_dispatch_enabled and task_type_lower in self._claim_dispatch_task_types:
            try:
                auto_dispatch = await self.complete_task(
                    agent_id=agent_id,
                    task_id=str(selected_task_id),
                    task_action=str((task or {}).get("task_type") or ""),
                    task_payload=dict((task or {}).get("payload") or {}),
                )
            except Exception as e:
                logger.warning(f"Auto-dispatch failed for task {selected_task_id}: {e}", exc_info=True)

        effective_action = "claim_task"
        if isinstance(auto_dispatch, ActionResult) and isinstance(auto_dispatch.data, dict):
            dispatched_effective = str(auto_dispatch.data.get("effective_action") or "").strip()
            if dispatched_effective:
                effective_action = dispatched_effective
            dispatch_reward = auto_dispatch.data.get("reward")
            if isinstance(dispatch_reward, (int, float)):
                reward = float(reward) + float(dispatch_reward)

        ar = ActionResult.success(
            method_name="claim_task",
            message="Task claimed." if ok else f"Task claim failed: {(claim_res or {}).get('reason')}",
            data={
                "task_id": selected_task_id,
                "task": task,
                "ok": ok,
                "reward": reward,
                "llm_selection": llm_selection,
                "effective_action": effective_action,
                "claim_result": claim_res,
                "auto_dispatch": (
                    {
                        "status": auto_dispatch.status,
                        "message": auto_dispatch.message,
                        "data": auto_dispatch.data if isinstance(auto_dispatch.data, dict) else {},
                    }
                    if isinstance(auto_dispatch, ActionResult)
                    else None
                ),
                "reward_components": {
                    "task_claim_reward": float(reward),
                    "task_claim_cost": float(self._claim_cost),
                    "learning_reward": float(reward),
                },
            },
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
        current_tick = int(await self.controller.run_system("timer", "get_tick"))
        task = await self._get_claimed_task(agent_id, task_id, current_tick=current_tick)
        if task is None:
            remembered_task_id = await self._get_state(agent_id, "current_task_id")
            if remembered_task_id and str(remembered_task_id) == str(task_id):
                try:
                    direct = await self.controller.run_environment(
                        "science",
                        "task_get",
                        task_id=task_id,
                        current_tick=current_tick,
                    )
                    candidate = (direct or {}).get("task") if isinstance(direct, dict) else None
                    if isinstance(candidate, dict):
                        st = str(candidate.get("status") or "")
                        if st in {"claimed", "running"} and str(candidate.get("claimed_by") or "") == str(agent_id):
                            task = candidate
                except Exception:
                    task = None
        action_name = task_action or (task or {}).get("task_type")
        payload = dict((task or {}).get("payload") or {})
        payload.update(task_payload or {})

        if task is None:
            ar = ActionResult.error(
                method_name="complete_task",
                message="Task not currently owned by agent (missing or expired lease).",
                data={
                    "task_id": task_id,
                    "task_action": action_name,
                    "ok": False,
                    "reward": 0.0,
                    "effective_action": action_name or "complete_task",
                    "reason": "task_missing_or_expired",
                    "reward_components": {
                        "learning_reward": 0.0,
                        "task_complete_bonus": 0.0,
                        "task_complete_total": 0.0,
                    },
                },
            )
            await self._append_trace(agent_id, "complete_task", 0.0, ar.data or {})
            return ar

        if self._task_heartbeat_enabled:
            try:
                await self.controller.run_environment(
                    "science",
                    "task_start",
                    task_id=task_id,
                    agent_id=agent_id,
                    current_tick=current_tick,
                    phase=f"start:{action_name or 'unknown'}",
                )
            except Exception as e:
                logger.warning(f"task_start failed for {task_id}: {e}")

        action_result = ActionResult.success(method_name="noop", message="No action executed.", data={"reward": 0.0})
        if action_name:
            try:
                action_result = await self._execute_task_action(agent_id=agent_id, task_action=action_name, task_payload=payload)
            except Exception as e:
                action_result = ActionResult.error(
                    method_name=str(action_name),
                    message=f"Inner action raised exception: {self._truncate(str(e), 220)}",
                    data={
                        "ok": False,
                        "exception": self._truncate(str(e), 400),
                        "reward": 0.0,
                        "effective_action": action_name or "complete_task",
                        "reward_components": {"learning_reward": 0.0},
                    },
                )

        if self._task_heartbeat_enabled:
            try:
                await self.controller.run_environment(
                    "science",
                    "task_heartbeat",
                    task_id=task_id,
                    agent_id=agent_id,
                    current_tick=current_tick,
                    phase=f"done:{action_name or 'unknown'}",
                )
            except Exception as e:
                logger.warning(f"task_heartbeat failed for {task_id}: {e}")

        if isinstance(action_result, ActionResult) and not action_result.is_successful():
            release_res = await self.controller.run_environment(
                "science",
                "task_release",
                task_id=task_id,
                agent_id=agent_id,
                reason=f"inner_action_failed:{action_name}",
                current_tick=current_tick,
            )
            ar = ActionResult.error(
                method_name="complete_task",
                message=f"Inner task action failed: {action_result.message}",
                data={
                    "task_id": task_id,
                    "task_action": action_name,
                    "ok": False,
                    "released": bool((release_res or {}).get("ok", False)),
                    "release_result": release_res,
                    "inner_action_status": action_result.status if isinstance(action_result, ActionResult) else None,
                    "inner_action_message": action_result.message if isinstance(action_result, ActionResult) else None,
                    "reward": 0.0,
                    "effective_action": action_name or "complete_task",
                    "reward_components": {
                        "learning_reward": 0.0,
                        "task_complete_bonus": 0.0,
                        "task_complete_total": 0.0,
                    },
                },
            )
            await self._append_trace(agent_id, "complete_task", 0.0, ar.data or {})
            return ar

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
            current_tick=current_tick,
        )
        ok = bool((complete_res or {}).get("ok"))
        inner_reward = 0.0
        if isinstance(action_result, ActionResult) and isinstance(action_result.data, dict):
            inner_reward = float(action_result.data.get("reward", 0.0) or 0.0)
        reward = inner_reward + (0.01 if ok else 0.0)
        if ok:
            await self._set_state(agent_id, "current_task_id", None)
            await self._set_state(agent_id, "current_task_type", None)

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
                "effective_action": action_name or "complete_task",
                "reward_components": {
                    "inner_reward": float(inner_reward),
                    "task_complete_bonus": float(0.01 if ok else 0.0),
                    "task_complete_total": float(reward),
                    "learning_reward": float(inner_reward),
                },
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
            ar = ActionResult.success(method_name="share_evidence", message="No local evidence to share.", data={"reward": 0.0})
            await self._append_trace(agent_id, "share_evidence", 0.0, ar.data or {})
            return ar

        recipient = self._pick_recipient(agent_id, to_agent_id)
        if not recipient:
            ar = ActionResult.success(method_name="share_evidence", message="No recipient available.", data={"reward": 0.0})
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
        if ok:
            await self._inc_state_number(agent_id, "share_sent_evidence_count", 1)

        ar = ActionResult.success(
            method_name="share_evidence",
            message="Evidence shared." if ok else "Evidence share attempted.",
            data={
                "to_agent_id": recipient,
                "shared_type": "note",
                "reward": reward,
                "effective_action": "share_evidence",
                "reward_components": {
                    "collaboration_reward": float(reward),
                    "learning_reward": float(reward),
                    "share_evidence_reward": float(reward),
                },
            },
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
            ar = ActionResult.success(method_name="share_observation", message="No recipient available.", data={"reward": 0.0})
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
        if ok:
            await self._inc_state_number(agent_id, "share_sent_observation_count", 1)

        ar = ActionResult.success(
            method_name="share_observation",
            message="Observation shared." if ok else "Observation share attempted.",
            data={
                "to_agent_id": recipient,
                "shared_type": "observation",
                "reward": reward,
                "effective_action": "share_observation",
                "reward_components": {
                    "collaboration_reward": float(reward),
                    "learning_reward": float(reward),
                    "share_observation_reward": float(reward),
                },
            },
        )
        await self._append_trace(agent_id, "share_observation", reward, ar.data or {})
        return ar
