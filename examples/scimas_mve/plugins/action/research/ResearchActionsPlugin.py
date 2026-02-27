import asyncio
import json
import os
import random
import re
import textwrap
from datetime import datetime
from typing import Any, Dict, List, Optional

from agentkernel_standalone.mas.action.base.plugin_base import OtherActionsPlugin
from agentkernel_standalone.toolkit.logger import get_logger
from agentkernel_standalone.toolkit.utils.annotation import AgentCall, ServiceCall
from agentkernel_standalone.types.schemas.action import ActionResult

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
        self._research_log_dir = os.path.join(base, "logs", "app", "research")
        self._cards_log_path = os.path.join(self._research_log_dir, "evidence_cards.jsonl")
        self._papers_log_path = os.path.join(self._research_log_dir, "papers.jsonl")

        self._llm_log_dir = os.path.join(base, "logs", "app", "llm")
        self._llm_log_path = os.path.join(self._llm_log_dir, "llm_calls.jsonl")
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
        self._llm_max_cards = int(os.getenv("SCIMAS_LLM_MAX_CARDS", "10"))
        self._llm_max_runs = int(os.getenv("SCIMAS_LLM_MAX_RUNS", "8"))
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
        self._claim_backoff_base = int(max(1, int(os.getenv("SCIMAS_CLAIM_BACKOFF_BASE", "1"))))
        self._claim_backoff_max = int(max(self._claim_backoff_base, int(os.getenv("SCIMAS_CLAIM_BACKOFF_MAX", "8"))))
        self._claim_cost = float(max(0.0, float(os.getenv("SCIMAS_CLAIM_COST", "0.002"))))
        self._review_min_issue_count = int(max(1, int(os.getenv("SCIMAS_REVIEW_MIN_ISSUES", "2"))))
        self._review_min_revision_actions = int(max(1, int(os.getenv("SCIMAS_REVIEW_MIN_ACTIONS", "2"))))
        self._review_revision_trigger_score = float(
            max(0.0, min(1.0, float(os.getenv("SCIMAS_REVIEW_REVISION_TRIGGER", "0.62"))))
        )
        self._review_flattery_penalty = float(max(0.0, float(os.getenv("SCIMAS_REVIEW_FLATTERY_PENALTY", "0.03"))))
        self._review_shallow_penalty = float(max(0.0, float(os.getenv("SCIMAS_REVIEW_SHALLOW_PENALTY", "0.02"))))
        self._review_self_review_penalty = float(max(0.0, float(os.getenv("SCIMAS_REVIEW_SELF_PENALTY", "0.02"))))

    async def init(self, model_router=None, controller=None):
        self.model = model_router
        self.controller = controller
        os.makedirs(os.path.dirname(self._trace_path), exist_ok=True)
        os.makedirs(self._research_log_dir, exist_ok=True)
        os.makedirs(self._llm_log_dir, exist_ok=True)

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

    async def _call_llm_json(self, *, agent_id: str, action_name: str, prompt: str) -> Dict[str, Any]:
        tick = await self.controller.run_system("timer", "get_tick")
        world_spec = await self.controller.run_environment("science", "get_world_spec")
        ts = datetime.utcnow().isoformat() + "Z"

        if not self._llm_ready(action_name):
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

    def _truncate(self, text: Any, limit: int = 280) -> str:
        value = str(text or "")
        if len(value) <= limit:
            return value
        return value[: limit - 3] + "..."

    def _safe_task_types(self, values: Any, *, fallback: Optional[List[str]] = None) -> List[str]:
        allowed = {"read", "hypothesize", "experiment", "write", "review", "replicate", "verify_strength", "verify_issue"}
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
            - Taskboard summary: {json.dumps(world_spec.get('taskboard') or {{}}, ensure_ascii=False)}

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
              "preferred_task_types": ["experiment", "hypothesize", "write", "verify_issue"],
              "primary_task_id": "Txxx",
              "selection_rationale": ["...", "..."],
              "risk_controls": ["...", "..."],
              "fallback_if_blocked": ["verify_issue", "verify_strength", "read", "hypothesize"]
            }}
            """
        ).strip()

    def _build_plan_prompt(self, world_spec: Dict[str, Any], cards: List[Dict[str, Any]], recent_runs: List[Dict[str, Any]]) -> str:
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

            Recent experimental traces:
            {json.dumps(runs_short, ensure_ascii=False)}

            Constraints:
            1) Explicitly align method with official metric and submission format.
            2) Include reproducibility safeguards and expected failure modes.
            3) Favor incremental, testable, falsifiable hypotheses.
            4) Keep strategy executable within limited budget.

            Return ONLY JSON with concise but technical content:
            {{
              "hypothesis_tags": ["..."],
              "strategy": "...",
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
        exp_count: int,
        budget: int,
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
        literature = await self.controller.run_environment("science", "read_literature", agent_id=agent_id, topic=topic)
        notes = await self._get_state(agent_id, "notes") or []
        notes.append(literature)
        await self._set_state(agent_id, "notes", notes)
        await self._log_evidence_cards(agent_id, literature, source="read")

        ar = ActionResult.success(
            method_name="read",
            message="Task cards retrieved.",
            data={
                "note": literature,
                "reward": 0.0,
                "effective_action": "read",
                "reward_components": {"learning_reward": 0.0, "read_reward": 0.0},
            },
        )
        await self._append_trace(agent_id, "read", 0.0, ar.data or {})
        return ar

    @AgentCall
    async def hypothesize(self, agent_id: str, hypothesis: Optional[List[str]] = None) -> ActionResult:
        notes = (await self._get_state(agent_id, "notes") or []) + (await self._get_state(agent_id, "shared_notes") or [])
        observations = (await self._get_state(agent_id, "observations") or []) + (
            await self._get_state(agent_id, "shared_observations") or []
        )
        world_spec = await self.controller.run_environment("science", "get_world_spec")
        existing_plan = await self._get_state(agent_id, "plan_spec") or {}
        plan_spec = self._merge_solver_plan(self._default_solver_plan(world_spec), existing_plan if isinstance(existing_plan, dict) else {})
        inferred = ["metric_alignment", "format_safety"]

        if hypothesis is None and self._llm_ready("hypothesize"):
            cards = []
            for n in notes[-4:]:
                cards.extend((n or {}).get("cards", [])[:3])
            prompt = self._build_plan_prompt(world_spec=world_spec, cards=cards, recent_runs=observations)
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

        ar = ActionResult.success(
            method_name="hypothesize",
            message="AIRS strategy plan updated.",
            data={
                "hypothesis": hypothesis,
                "plan_spec": plan_spec,
                "reward": 0.0,
                "effective_action": "hypothesize",
                "reward_components": {"learning_reward": 0.0, "hypothesize_reward": 0.0},
            },
        )
        await self._append_trace(agent_id, "hypothesize", 0.0, ar.data or {})
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
        run_config = dict(config or {})
        solver_spec = plan_spec.get("solver_spec") if isinstance(plan_spec.get("solver_spec"), dict) else {}
        run_config.setdefault("strategy", plan_spec.get("strategy", "iterative_solver_baseline"))
        if solver_spec:
            run_config.setdefault("solver_spec", json.loads(json.dumps(solver_spec)))
        llm_experiment_plan: Optional[Dict[str, Any]] = None

        if self._llm_ready("experiment"):
            prompt = self._build_experiment_prompt(
                world_spec=world_spec,
                hypothesis=hypothesis,
                plan_spec=plan_spec,
                notes=notes,
                observations=prior_observations,
                exp_count=exp_count,
                budget=budget,
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

        result = await self.controller.run_environment(
            "science",
            "run_experiment",
            config=run_config,
            agent_id=agent_id,
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
            "ok": bool((result or {}).get("ok", False)),
            "error": (result or {}).get("error"),
            "elapsed_s": (result or {}).get("elapsed_s"),
            "strategy": run_config.get("strategy"),
            "config": run_config.get("solver_spec") or run_config,
            "llm_experiment_plan": llm_experiment_plan,
        }
        observations.append(observation)
        await self._set_state(agent_id, "observations", observations)
        await self._set_state(agent_id, "run_history", observations)
        await self._set_state(agent_id, "exp_count", exp_count)

        score_norm = float((result or {}).get("score_norm", 0.0) or 0.0)
        dev_score_norm = (result or {}).get("dev_score_norm")
        dev_score_norm = float(dev_score_norm) if isinstance(dev_score_norm, (int, float)) else score_norm
        cost = float((result or {}).get("cost", 0.0) or 0.0)
        prev_best_dev = max(
            [float(o.get("dev_score_norm", 0.0) or 0.0) for o in prior_observations if bool(o.get("ok"))] or [0.0]
        )
        improvement = dev_score_norm - prev_best_dev
        reward = max(-0.06, min(0.12, 0.05 * dev_score_norm + 0.08 * improvement - 0.03 * cost))
        if not bool((result or {}).get("ok", False)):
            reward = -0.02

        # Keep an experiment -> review loop on taskboard for iterative refinement.
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
                },
                "run_config": run_config,
                "llm_experiment_plan": llm_experiment_plan,
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
        metric_name = str(best_run.get("metric_name") or world_spec.get("metric") or "score")
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
            prompt = self._build_write_prompt(
                world_spec=world_spec,
                best_run=best_run,
                hypothesis=hypothesis,
                plan_spec=plan_spec,
                citations=citations_for_prompt,
                observation_refs=obs_refs_for_prompt,
                notes=notes,
                observations=observations,
            )
            llm_result = await self._call_llm_json(agent_id=agent_id, action_name="write", prompt=prompt)
            if llm_result.get("ok") and isinstance(llm_result.get("data"), dict):
                llm_write_spec = llm_result.get("data") or {}

        paper = self._build_paper_payload(
            task_name=task_name,
            metric_name=metric_name,
            best_run=best_run,
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
        ar = ActionResult.success(
            method_name="write",
            message="AIRS submission written and evaluated.",
            data={
                "metrics": metrics,
                "paper_id": paper_id,
                "paper": paper,
                "llm_write_spec": llm_write_spec,
                "credit_by_agent": contribution_credit,
                "reward": reward,
                "effective_action": "write",
                "reward_components": {
                    "terminal_quality_reward": float(reward),
                    "learning_reward": float(reward),
                    "paper_write_reward": float(reward),
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
                reward = max(-0.08, min(0.12, 0.12 * review_score - 0.03))

                validation_tasks = await self._spawn_review_validation_tasks(
                    paper_id=str(paper_id),
                    reviewer_id=agent_id,
                    review_note=review_note,
                    critique_quality=critique_quality,
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
                        },
                        "review_note": review_note,
                        "critique_quality": critique_quality,
                        "validation_tasks": validation_tasks,
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
                },
            )
            await self._append_trace(agent_id, "review", reward, ar.data or {})
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
        if bool((replication_submit or {}).get("ok")):
            support = (replication_submit or {}).get("support") or {}
            support_ratio = self._clamp01(support.get("support_ratio", 0.0))
            replication_signal = abs(support_ratio - 0.5) * 2.0
            contradiction_bonus = max(0.0, 0.5 - support_ratio) / 0.5
            confirmation_bonus = max(0.0, support_ratio - 0.5) / 0.5
            # Reward informative replication in both directions; detecting failure gets slightly higher credit.
            reward = 0.01 + 0.02 * replication_signal + 0.02 * contradiction_bonus + 0.01 * confirmation_bonus
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
                                fallback=["verify_issue", "verify_strength", "read", "hypothesize"],
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

        ar = ActionResult.success(
            method_name="claim_task",
            message="Task claimed." if ok else f"Task claim failed: {(claim_res or {}).get('reason')}",
            data={
                "task_id": selected_task_id,
                "task": task,
                "ok": ok,
                "reward": reward,
                "llm_selection": llm_selection,
                "effective_action": "claim_task",
                "claim_result": claim_res,
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
        action_name = task_action or (task or {}).get("task_type")
        payload = dict((task or {}).get("payload") or {})
        payload.update(task_payload or {})

        action_result = ActionResult.success(method_name="noop", message="No action executed.", data={"reward": 0.0})
        if action_name:
            action_result = await self._execute_task_action(agent_id=agent_id, task_action=action_name, task_payload=payload)

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
