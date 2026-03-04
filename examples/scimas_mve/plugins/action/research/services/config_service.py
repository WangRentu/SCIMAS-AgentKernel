import os
import random
from typing import Any, Optional
from urllib.parse import urlparse


class ConfigService:
    """Load runtime/env config onto ResearchActionsPlugin instance."""

    def __init__(self, plugin: Any):
        self.plugin = plugin

    @staticmethod
    def _ensure_no_proxy_for_urls(urls: list[str]) -> None:
        hosts: list[str] = []
        for raw in urls:
            value = str(raw or "").strip()
            if not value:
                continue
            try:
                parsed = urlparse(value)
            except Exception:
                continue
            if not parsed.hostname:
                continue
            host = str(parsed.hostname).strip()
            if host:
                hosts.append(host)
            if parsed.port:
                hosts.append(f"{host}:{int(parsed.port)}")
        if not hosts:
            return
        for key in ("no_proxy", "NO_PROXY"):
            current = str(os.environ.get(key, "") or "").strip()
            items = [x.strip() for x in current.split(",") if x.strip()]
            existed = {x.lower() for x in items}
            for host in hosts:
                if host.lower() not in existed:
                    items.append(host)
                    existed.add(host.lower())
            os.environ[key] = ",".join(items)

    def apply(self) -> None:
        p = self.plugin
        p._rng = random.Random()
        base = os.getenv("MAS_PROJECT_ABS_PATH", ".")
        p._log_mode = (os.getenv("SCIMAS_LOG_MODE", "compact") or "compact").strip().lower()
        p._verbose_action_logs = os.getenv("SCIMAS_VERBOSE_ACTION_LOGS", "0").lower() in {"1", "true", "yes"}
        p._trace_enabled = os.getenv("SCIMAS_TRACE_ENABLE", "1" if p._log_mode == "verbose" else "0").lower() not in {
            "0",
            "false",
            "no",
        }

        p._trace_path = os.path.join(base, "logs", "app", "action", "trace.jsonl")
        p._code_loop_log_path = os.path.join(base, "logs", "app", "action", "code_loop.jsonl")
        p._code_diagnosis_log_path = os.path.join(base, "logs", "app", "action", "code_diagnosis.jsonl")
        p._precondition_gate_log_path = os.path.join(base, "logs", "app", "action", "precondition_gate.jsonl")
        p._retrieve_pipeline_log_path = os.path.join(base, "logs", "app", "action", "retrieve_pipeline.jsonl")
        p._retrieve_guardrail_log_path = os.path.join(base, "logs", "app", "action", "retrieve_guardrail.jsonl")
        p._retrieve_evidence_log_path = os.path.join(base, "logs", "app", "action", "retrieve_evidence.jsonl")
        p._research_log_dir = os.path.join(base, "logs", "app", "research")
        p._cards_log_path = os.path.join(p._research_log_dir, "evidence_cards.jsonl")
        p._papers_log_path = os.path.join(p._research_log_dir, "papers.jsonl")

        p._llm_log_dir = os.path.join(base, "logs", "app", "llm")
        p._llm_log_path = os.path.join(p._llm_log_dir, "llm_calls.jsonl")
        p._audit_log_dir = os.path.join(base, "logs", "app", "audit")
        p._llm_audit_jsonl_path = os.path.join(p._audit_log_dir, "llm_io.jsonl")
        p._llm_audit_md_path = os.path.join(p._audit_log_dir, "llm_io.md")
        p._rag_audit_jsonl_path = os.path.join(p._audit_log_dir, "rag_io.jsonl")
        p._rag_audit_md_path = os.path.join(p._audit_log_dir, "rag_io.md")
        p._llm_timeout_s = float(os.getenv("SCIMAS_LLM_TIMEOUT_S", "15"))
        p._llm_enabled = os.getenv("SCIMAS_LLM_ENABLE", "1").lower() not in {"0", "false", "no"}
        p._llm_actions = {
            "claim_task": os.getenv("SCIMAS_LLM_CLAIM_TASK", "1").lower() not in {"0", "false", "no"},
            "retrieve_literature": os.getenv("SCIMAS_LLM_RETRIEVE", "1").lower() not in {"0", "false", "no"},
            "hypothesize": os.getenv("SCIMAS_LLM_HYPOTHESIZE", "1").lower() not in {"0", "false", "no"},
            "experiment": os.getenv("SCIMAS_LLM_EXPERIMENT", "1").lower() not in {"0", "false", "no"},
            "write": os.getenv("SCIMAS_LLM_WRITE", "1").lower() not in {"0", "false", "no"},
            "review": os.getenv("SCIMAS_LLM_REVIEW", "1").lower() not in {"0", "false", "no"},
            "replicate": os.getenv("SCIMAS_LLM_REPLICATE", "1").lower() not in {"0", "false", "no"},
        }
        p._llm_log_enabled = os.getenv("SCIMAS_LLM_LOG_ENABLE", "0" if p._log_mode == "compact" else "1").lower() not in {
            "0",
            "false",
            "no",
        }
        p._audit_io_enable = os.getenv("SCIMAS_AUDIT_IO_ENABLE", "1").lower() not in {"0", "false", "no"}
        p._audit_markdown_enable = os.getenv("SCIMAS_AUDIT_MARKDOWN_ENABLE", "1").lower() not in {
            "0",
            "false",
            "no",
        }
        p._audit_llm_max_chars = int(max(2000, int(os.getenv("SCIMAS_AUDIT_LLM_MAX_CHARS", "300000"))))
        p._audit_rag_max_chars = int(max(2000, int(os.getenv("SCIMAS_AUDIT_RAG_MAX_CHARS", "300000"))))
        p._audit_rag_max_rows = int(max(1, int(os.getenv("SCIMAS_AUDIT_RAG_MAX_ROWS", "20"))))
        p._llm_max_cards = int(os.getenv("SCIMAS_LLM_MAX_CARDS", "10"))
        p._llm_max_runs = int(os.getenv("SCIMAS_LLM_MAX_RUNS", "8"))
        p._code_loop_enabled = os.getenv("SCIMAS_CODE_AGENT_ENABLE", "1").lower() not in {"0", "false", "no"}
        p._code_debug_rounds = int(max(1, int(os.getenv("SCIMAS_CODE_DEBUG_ROUNDS", "4"))))
        p._code_optimize_after_success = os.getenv("SCIMAS_CODE_OPTIMIZE_AFTER_SUCCESS", "1").lower() not in {
            "0",
            "false",
            "no",
        }
        p._code_diag_enable = os.getenv("SCIMAS_CODE_DIAG_ENABLE", "1").lower() not in {"0", "false", "no"}
        p._code_template_fix_enable = os.getenv("SCIMAS_CODE_TEMPLATE_FIX_ENABLE", "1").lower() not in {
            "0",
            "false",
            "no",
        }
        p._code_optimize_guard_enable = os.getenv("SCIMAS_CODE_OPTIMIZE_GUARD_ENABLE", "1").lower() not in {
            "0",
            "false",
            "no",
        }
        p._code_optimize_patience = int(max(1, int(os.getenv("SCIMAS_CODE_OPTIMIZE_PATIENCE", "2"))))
        p._code_max_files = int(max(1, int(os.getenv("SCIMAS_CODE_MAX_FILES", "8"))))
        p._code_max_file_chars = int(max(2000, int(os.getenv("SCIMAS_CODE_MAX_FILE_CHARS", "60000"))))
        p._code_error_tail_chars = int(max(300, int(os.getenv("SCIMAS_CODE_ERROR_TAIL_CHARS", "3000"))))
        p._cards_log_enabled = os.getenv("SCIMAS_EVIDENCE_LOG_ENABLE", "1" if p._log_mode == "verbose" else "0").lower() not in {
            "0",
            "false",
            "no",
        }
        p._papers_log_enabled = os.getenv("SCIMAS_PAPER_LOG_ENABLE", "1" if p._log_mode == "verbose" else "0").lower() not in {
            "0",
            "false",
            "no",
        }

        p._strict_task_dependencies = os.getenv("SCIMAS_STRICT_TASK_DEPENDENCIES", "1").lower() not in {
            "0",
            "false",
            "no",
        }
        p._write_min_notes = int(os.getenv("SCIMAS_WRITE_MIN_NOTES", "1"))
        p._write_min_observations = int(os.getenv("SCIMAS_WRITE_MIN_OBS", "1"))
        p._write_min_hypothesis = int(os.getenv("SCIMAS_WRITE_MIN_HYP", "1"))
        p._experiment_require_data_card = os.getenv("SCIMAS_EXPERIMENT_REQUIRE_DATA_CARD", "1").lower() not in {
            "0",
            "false",
            "no",
        }
        p._experiment_require_method_card = os.getenv("SCIMAS_EXPERIMENT_REQUIRE_METHOD_CARD", "1").lower() not in {
            "0",
            "false",
            "no",
        }
        p._experiment_min_notes = int(max(0, int(os.getenv("SCIMAS_EXPERIMENT_MIN_NOTES", "1"))))
        p._experiment_min_hypothesis = int(max(0, int(os.getenv("SCIMAS_EXPERIMENT_MIN_HYP", "0"))))
        p._experiment_require_code_agent_success = os.getenv(
            "SCIMAS_EXPERIMENT_REQUIRE_CODE_AGENT_SUCCESS",
            "1",
        ).lower() not in {"0", "false", "no"}
        p._experiment_treat_fallback_as_repair = os.getenv(
            "SCIMAS_EXPERIMENT_TREAT_FALLBACK_AS_REPAIR",
            "1",
        ).lower() not in {"0", "false", "no"}
        p._experiment_prompt_enforce_columns = os.getenv(
            "SCIMAS_EXPERIMENT_PROMPT_ENFORCE_COLUMNS",
            "1",
        ).lower() not in {"0", "false", "no"}
        p._claim_backoff_base = int(max(1, int(os.getenv("SCIMAS_CLAIM_BACKOFF_BASE", "1"))))
        p._claim_backoff_max = int(max(p._claim_backoff_base, int(os.getenv("SCIMAS_CLAIM_BACKOFF_MAX", "8"))))
        p._claim_cost = float(max(0.0, float(os.getenv("SCIMAS_CLAIM_COST", "0.002"))))
        p._claim_dispatch_enabled = os.getenv("SCIMAS_CLAIM_DISPATCH_ENABLE", "1").lower() not in {"0", "false", "no"}
        dispatch_raw = str(os.getenv("SCIMAS_CLAIM_DISPATCH_TASK_TYPES", "experiment") or "").strip()
        p._claim_dispatch_task_types = {
            t.strip().lower() for t in dispatch_raw.split(",") if t.strip()
        } or {"experiment"}
        p._task_heartbeat_enabled = os.getenv("SCIMAS_TASK_HEARTBEAT_ENABLE", "1").lower() not in {"0", "false", "no"}
        p._review_min_issue_count = int(max(1, int(os.getenv("SCIMAS_REVIEW_MIN_ISSUES", "2"))))
        p._review_min_revision_actions = int(max(1, int(os.getenv("SCIMAS_REVIEW_MIN_ACTIONS", "2"))))
        p._review_revision_trigger_score = float(
            max(0.0, min(1.0, float(os.getenv("SCIMAS_REVIEW_REVISION_TRIGGER", "0.62"))))
        )
        p._strict_review_mode = os.getenv("SCIMAS_STRICT_REVIEW", "1").lower() not in {"0", "false", "no"}
        p._qgr_min_issue_count = int(max(1, int(os.getenv("SCIMAS_QGR_MIN_ISSUES", "2"))))
        p._qgr_min_citations = int(max(1, int(os.getenv("SCIMAS_QGR_MIN_CITATIONS", "3"))))
        p._qgr_relevance_threshold = float(max(0.0, min(1.0, float(os.getenv("SCIMAS_QGR_RELEVANCE_THRESHOLD", "0.75")))))
        p._qgr_fact_support_threshold = float(max(0.0, min(1.0, float(os.getenv("SCIMAS_QGR_FACT_SUPPORT_THRESHOLD", "0.20")))))
        p._qgr_base_reward = float(os.getenv("SCIMAS_QGR_BASE_REWARD", "0.2"))
        p._qgr_quality_bonus = float(os.getenv("SCIMAS_QGR_QUALITY_BONUS", "0.5"))
        p._qgr_predictive_bonus_reward = float(os.getenv("SCIMAS_QGR_PREDICTIVE_BONUS", "1.5"))
        p._review_flattery_penalty = float(max(0.0, float(os.getenv("SCIMAS_REVIEW_FLATTERY_PENALTY", "0.03"))))
        p._review_shallow_penalty = float(max(0.0, float(os.getenv("SCIMAS_REVIEW_SHALLOW_PENALTY", "0.02"))))
        p._review_self_review_penalty = float(max(0.0, float(os.getenv("SCIMAS_REVIEW_SELF_PENALTY", "0.02"))))
        p._dense_reward_enable = os.getenv("SCIMAS_DENSE_REWARD_ENABLE", "1").lower() not in {"0", "false", "no"}
        p._read_reward_alpha = float(max(0.0, float(os.getenv("SCIMAS_READ_REWARD_ALPHA", "0.35"))))
        p._read_reward_base = float(max(0.0, float(os.getenv("SCIMAS_READ_REWARD_BASE", "0.20"))))
        p._read_reward_max = float(max(p._read_reward_base, float(os.getenv("SCIMAS_READ_REWARD_MAX", "0.50"))))
        p._read_method_bonus = float(max(0.0, float(os.getenv("SCIMAS_READ_METHOD_BONUS", "0.08"))))
        p._read_dup_threshold = float(max(0.0, min(1.0, float(os.getenv("SCIMAS_READ_DUP_THRESHOLD", "0.90")))))
        p._reward_tei_url = str(os.getenv("SCIMAS_REWARD_TEI_URL", "")).strip()
        p._reward_qdrant_url = str(os.getenv("SCIMAS_REWARD_QDRANT_URL", "")).strip().rstrip("/")
        p._reward_qdrant_collection = str(os.getenv("SCIMAS_REWARD_QDRANT_COLLECTION", "notes")).strip()
        p._reward_qdrant_api_key = str(os.getenv("SCIMAS_REWARD_QDRANT_API_KEY", "")).strip()
        p._hypothesis_schema_bonus = float(max(0.0, float(os.getenv("SCIMAS_HYPOTHESIS_SCHEMA_BONUS", "0.5"))))
        p._hypothesis_resource_bonus = float(max(0.0, float(os.getenv("SCIMAS_HYPOTHESIS_RESOURCE_BONUS", "0.3"))))
        p._experiment_success_reward = float(os.getenv("SCIMAS_EXPERIMENT_SUCCESS_REWARD", "1.0"))
        p._experiment_oom_penalty = float(max(0.0, float(os.getenv("SCIMAS_EXPERIMENT_OOM_PENALTY", "0.8"))))
        p._experiment_typeerror_penalty = float(max(0.0, float(os.getenv("SCIMAS_EXPERIMENT_TYPEERROR_PENALTY", "0.2"))))
        p._experiment_first_pass_bonus = float(max(0.0, float(os.getenv("SCIMAS_EXPERIMENT_FIRST_PASS_BONUS", "0.25"))))
        p._experiment_vram_reward_weight = float(max(0.0, float(os.getenv("SCIMAS_EXPERIMENT_VRAM_WEIGHT", "0.2"))))
        p._write_eval_success_bonus = float(max(0.0, float(os.getenv("SCIMAS_WRITE_EVAL_SUCCESS_BONUS", "2.0"))))
        p._write_format_pass_reward = float(max(0.0, float(os.getenv("SCIMAS_WRITE_FORMAT_PASS_REWARD", "0.3"))))
        p._write_cache_repeat_penalty = float(max(0.0, float(os.getenv("SCIMAS_WRITE_CACHE_REPEAT_PENALTY", "0.05"))))
        p._write_defer_on_system_error = os.getenv("SCIMAS_WRITE_DEFER_ON_SYSTEM_ERROR", "1").lower() not in {
            "0",
            "false",
            "no",
        }
        p._replicate_high_support_reward = float(max(0.0, float(os.getenv("SCIMAS_REPLICATE_HIGH_SUPPORT_REWARD", "5.0"))))
        p._replicate_support_threshold = float(max(0.0, min(1.0, float(os.getenv("SCIMAS_REPLICATE_SUPPORT_THRESHOLD", "0.8")))))
        p._vdh_enable = os.getenv("SCIMAS_VDH_ENABLE", "1").lower() not in {"0", "false", "no"}
        p._vdh_gate_policy = str(os.getenv("SCIMAS_VDH_GATE_POLICY", "hard_fail") or "hard_fail").strip().lower()
        p._vdh_evidence_threshold = float(
            max(0.0, min(1.0, float(os.getenv("SCIMAS_VDH_EVIDENCE_THRESHOLD", "0.60"))))
        )
        p._vdh_dynamic_gating_enable = os.getenv("SCIMAS_VDH_DYNAMIC_GATING_ENABLE", "1").lower() not in {
            "0",
            "false",
            "no",
        }
        p._vdh_dynamic_min_threshold = float(
            max(0.0, min(1.0, float(os.getenv("SCIMAS_VDH_DYNAMIC_MIN_THRESHOLD", "0.20"))))
        )
        p._vdh_dynamic_low_evidence_chunks = int(
            max(1, int(os.getenv("SCIMAS_VDH_DYNAMIC_LOW_EVIDENCE_CHUNKS", "8")))
        )
        p._vdh_dynamic_low_evidence_relax = float(
            max(0.0, min(1.0, float(os.getenv("SCIMAS_VDH_DYNAMIC_LOW_EVIDENCE_RELAX", "0.20"))))
        )
        p._vdh_dynamic_rag_degraded_relax = float(
            max(0.0, min(1.0, float(os.getenv("SCIMAS_VDH_DYNAMIC_RAG_DEGRADED_RELAX", "0.15"))))
        )
        p._vdh_dynamic_no_vector_relax = float(
            max(0.0, min(1.0, float(os.getenv("SCIMAS_VDH_DYNAMIC_NO_VECTOR_RELAX", "0.10"))))
        )
        p._vdh_qdrant_enable = os.getenv("SCIMAS_VDH_QDRANT_ENABLE", "1").lower() not in {"0", "false", "no"}
        p._vdh_qdrant_url = str(os.getenv("SCIMAS_VDH_QDRANT_URL", "")).strip().rstrip("/")
        p._vdh_qdrant_collection = str(os.getenv("SCIMAS_VDH_QDRANT_COLLECTION", "schema_collection")).strip()
        p._vdh_qdrant_api_key = str(os.getenv("SCIMAS_VDH_QDRANT_API_KEY", "")).strip()
        p._vdh_tei_enable = os.getenv("SCIMAS_VDH_TEI_ENABLE", "1").lower() not in {"0", "false", "no"}
        p._vdh_tei_url = str(os.getenv("SCIMAS_VDH_TEI_URL", "")).strip()
        p._vdh_oom_ratio_threshold = float(
            max(0.1, min(2.0, float(os.getenv("SCIMAS_VDH_OOM_RATIO_THRESHOLD", "0.90"))))
        )
        p._vdh_schema_pass_reward = float(max(0.0, float(os.getenv("SCIMAS_VDH_SCHEMA_PASS_REWARD", "0.5"))))
        p._vdh_evidence_high_reward = float(max(0.0, float(os.getenv("SCIMAS_VDH_EVIDENCE_HIGH_REWARD", "0.8"))))
        p._vdh_oom_penalty = float(max(0.0, float(os.getenv("SCIMAS_VDH_OOM_PENALTY", "1.0"))))
        p._vdh_gate_penalty = float(max(0.0, float(os.getenv("SCIMAS_VDH_GATE_PENALTY", "0.2"))))
        p._vdh_gate_log_path = os.path.join(base, "logs", "app", "action", "vdh_gate.jsonl")
        p._review_gate_log_path = os.path.join(base, "logs", "app", "action", "review_gate.jsonl")
        p._rag_index_log_path = os.path.join(base, "logs", "app", "action", "rag_index.jsonl")
        p._rag_query_log_path = os.path.join(base, "logs", "app", "action", "rag_query.jsonl")
        p._rag_usage_log_path = os.path.join(base, "logs", "app", "action", "rag_usage.jsonl")
        p._rag_health_log_path = os.path.join(base, "logs", "app", "action", "rag_health.jsonl")
        p._rag_alert_log_path = os.path.join(base, "logs", "app", "action", "rag_alert.jsonl")

        p._rag_enable = os.getenv("SCIMAS_RAG_ENABLE", "1").lower() not in {"0", "false", "no"}
        p._rag_qdrant_url = str(os.getenv("SCIMAS_RAG_QDRANT_URL", "http://127.0.0.1:6333")).strip().rstrip("/")
        p._rag_qdrant_api_key = str(os.getenv("SCIMAS_RAG_QDRANT_API_KEY", "")).strip()
        p._rag_collection = str(os.getenv("SCIMAS_RAG_COLLECTION", "scimas_local_knowledge_v1")).strip()
        p._rag_collection_literature = str(
            os.getenv("SCIMAS_RAG_COLLECTION_LITERATURE", "literature_chunks")
        ).strip()
        p._rag_collection_run_memory = str(os.getenv("SCIMAS_RAG_COLLECTION_RUN_MEMORY", "run_memory")).strip()
        p._rag_retrieve_mode = str(os.getenv("SCIMAS_RAG_RETRIEVE_MODE", "hybrid")).strip().lower()
        p._rag_embed_url = str(os.getenv("SCIMAS_RAG_EMBED_URL", "http://127.0.0.1:8001/v1/embeddings")).strip()
        p._rag_embed_model = str(os.getenv("SCIMAS_RAG_EMBED_MODEL", "Qwen/Qwen3-Embedding-4B")).strip()
        p._rag_topk = int(max(1, int(os.getenv("SCIMAS_RAG_TOPK", "8"))))
        p._rag_max_context_chars = int(max(500, int(os.getenv("SCIMAS_RAG_MAX_CONTEXT_CHARS", "9000"))))
        p._rag_chunk_chars = int(max(200, int(os.getenv("SCIMAS_RAG_CHUNK_CHARS", "1200"))))
        p._rag_chunk_overlap = int(max(0, int(os.getenv("SCIMAS_RAG_CHUNK_OVERLAP", "180"))))
        p._rag_batch_size = int(max(1, int(os.getenv("SCIMAS_RAG_BATCH_SIZE", "32"))))
        p._rag_timeout_s = float(max(1.0, float(os.getenv("SCIMAS_RAG_TIMEOUT_S", "8"))))
        p._rag_min_score = float(max(0.0, min(1.0, float(os.getenv("SCIMAS_RAG_MIN_SCORE", "0.25")))))
        p._rag_index_on_read = os.getenv("SCIMAS_RAG_INDEX_ON_READ", "1").lower() not in {"0", "false", "no"}
        p._rag_index_on_experiment = os.getenv("SCIMAS_RAG_INDEX_ON_EXPERIMENT", "1").lower() not in {"0", "false", "no"}
        p._rag_index_on_write = os.getenv("SCIMAS_RAG_INDEX_ON_WRITE", "1").lower() not in {"0", "false", "no"}
        p._rag_degraded_alert_threshold = int(max(1, int(os.getenv("SCIMAS_RAG_DEGRADED_STREAK_ALERT", "3"))))
        p._rag_degraded_pause_ticks = int(max(1, int(os.getenv("SCIMAS_RAG_DEGRADED_PAUSE_TICKS", "12"))))
        p._rag_store = None
        p._rag_retriever = None
        p._rag_bootstrap_episode: Optional[int] = None
        p._rag_health_checked = False
        p._rag_degraded_streak = 0
        p._rag_degraded_last_status = ""
        p._rag_degraded_pause_until_tick = 0
        # Default VDH endpoints to RAG endpoints when VDH URLs are left blank.
        if p._vdh_qdrant_enable and not p._vdh_qdrant_url:
            p._vdh_qdrant_url = p._rag_qdrant_url
        if p._vdh_tei_enable and not p._vdh_tei_url:
            p._vdh_tei_url = p._rag_embed_url
        # Keep Qdrant/Embedding calls away from system proxies when running in mixed network envs.
        self._ensure_no_proxy_for_urls(
            [
                p._rag_qdrant_url,
                p._rag_embed_url,
                p._reward_qdrant_url,
                p._reward_tei_url,
                p._vdh_qdrant_url,
                p._vdh_tei_url,
            ]
        )
        p._retrieve_v2_enable = os.getenv("SCIMAS_RETRIEVE_V2_ENABLE", "1").lower() not in {"0", "false", "no"}
        p._retrieve_citation_threshold = float(
            max(0.0, min(1.0, float(os.getenv("SCIMAS_RETRIEVE_CITATION_THRESHOLD", "0.80"))))
        )
        p._retrieve_min_baselines = int(max(1, int(os.getenv("SCIMAS_RETRIEVE_MIN_BASELINES", "2"))))
        p._retrieve_min_evidence = int(max(1, int(os.getenv("SCIMAS_RETRIEVE_MIN_EVIDENCE", "3"))))
        p._retrieve_exec_min_required = int(max(1, int(os.getenv("SCIMAS_RETRIEVE_EXECUTABLE_MIN_REQUIRED", "1"))))
        p._retrieve_degrade_allow = os.getenv("SCIMAS_RETRIEVE_DEGRADE_ALLOW", "1").lower() not in {
            "0",
            "false",
            "no",
        }
        p._retrieve_template_fallback = os.getenv("SCIMAS_RETRIEVE_TEMPLATE_FALLBACK", "1").lower() not in {
            "0",
            "false",
            "no",
        }
        p._experiment_require_method_card_quality = os.getenv(
            "SCIMAS_EXPERIMENT_REQUIRE_METHOD_CARD_QUALITY", "1"
        ).lower() not in {"0", "false", "no"}
        p._experiment_allow_l1_bootstrap = os.getenv(
            "SCIMAS_EXPERIMENT_ALLOW_L1_BOOTSTRAP", "1"
        ).lower() not in {"0", "false", "no"}

        p._operator_experiment_v2 = os.getenv("SCIMAS_OPERATOR_EXPERIMENT_V2", "0").lower() not in {"0", "false", "no"}
        p._operator_review_v2 = os.getenv("SCIMAS_OPERATOR_REVIEW_V2", "0").lower() not in {"0", "false", "no"}
        p._operator_replicate_v2 = os.getenv("SCIMAS_OPERATOR_REPLICATE_V2", "0").lower() not in {"0", "false", "no"}
        p._operator_write_v2 = os.getenv("SCIMAS_OPERATOR_WRITE_V2", "0").lower() not in {"0", "false", "no"}
        p._operator_retrieve_v2 = os.getenv("SCIMAS_OPERATOR_RETRIEVE_V2", "0").lower() not in {"0", "false", "no"}
        p._operator_hypothesize_v2 = os.getenv("SCIMAS_OPERATOR_HYPOTHESIZE_V2", "0").lower() not in {"0", "false", "no"}
        p._operator_taskboard_v2 = os.getenv("SCIMAS_OPERATOR_TASKBOARD_V2", "0").lower() not in {"0", "false", "no"}
        p._operator_collaboration_v2 = os.getenv("SCIMAS_OPERATOR_COLLAB_V2", "0").lower() not in {"0", "false", "no"}
        p._operator_dataops_v2 = os.getenv("SCIMAS_OPERATOR_DATAOPS_V2", "0").lower() not in {"0", "false", "no"}
        p._operator_verify_v2 = os.getenv("SCIMAS_OPERATOR_VERIFY_V2", "0").lower() not in {"0", "false", "no"}
        p._orchestrator_v2 = os.getenv("SCIMAS_ORCHESTRATOR_V2", "0").lower() not in {"0", "false", "no"}

        p._diagnosis_service = None
        p._llm_service = None
        p._reward_service = None
        p._audit_service = None
        p._context_service = None
        p._contribution_service = None
        p._planning_service = None
        p._prompt_service = None
        p._rag_service = None
        p._recovery_service = None
        p._review_quality_service = None
        p._vdh_service = None
        p._experiment_operator = None
        p._hypothesize_operator = None
        p._review_operator = None
        p._verification_operator = None
        p._replication_operator = None
        p._write_operator = None
        p._retrieve_operator = None
        p._taskboard_operator = None
        p._collaboration_operator = None
        p._data_ops_operator = None
