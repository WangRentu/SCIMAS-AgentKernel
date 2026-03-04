import json
from datetime import datetime
from typing import Any, Dict, List, Optional

from agentkernel_standalone.toolkit.logger import get_logger

logger = get_logger(__name__)

class AuditService:
    """Centralized log/trace adapters used by operators."""

    def __init__(self, plugin: Any):
        self.plugin = plugin

    async def append_trace(self, agent_id: str, action: str, reward: float, data: Dict[str, Any]) -> None:
        if not self.plugin._trace_enabled:
            return
        try:
            tick = await self.plugin.controller.run_system("timer", "get_tick")
            world_spec = await self.plugin.controller.run_environment("science", "get_world_spec")
            exp_count = int((await self.plugin._get_state(agent_id, "exp_count")) or 0)
            hypothesis = await self.plugin._get_state(agent_id, "hypothesis") or []
            fitness = await self.plugin._get_state(agent_id, "fitness")
            detail = dict(data or {})
            detail = self.plugin._sanitize_for_log(detail)
            detail = self.plugin._truncate_strings_in_obj(detail, 400)
            if self.plugin._log_mode == "compact" and not self.plugin._verbose_action_logs:
                keys = ["task_id", "run_id", "paper_id", "score_norm", "ok", "effective_action", "message", "reason"]
                detail = {k: detail.get(k) for k in keys if k in detail}

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
            with open(self.plugin._trace_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.error(f"Failed to write trace: {e}")

    async def log_precondition_gate(
        self,
        agent_id: str,
        action: str,
        phase: str,
        failures: List[str],
        summary: Optional[Dict[str, Any]] = None,
    ) -> None:
        tick = await self.plugin.controller.run_system("timer", "get_tick")
        world_spec = await self.plugin.controller.run_environment("science", "get_world_spec")
        await self.plugin._append_jsonl(
            self.plugin._precondition_gate_log_path,
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

    async def log_vdh_gate(
        self,
        *,
        agent_id: str,
        vdh_report: Dict[str, Any],
        reward_components: Optional[Dict[str, Any]] = None,
    ) -> None:
        tick = await self.plugin.controller.run_system("timer", "get_tick")
        world_spec = await self.plugin.controller.run_environment("science", "get_world_spec")
        await self.plugin._append_jsonl(
            self.plugin._vdh_gate_log_path,
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

    async def log_review_gate(
        self,
        *,
        agent_id: str,
        paper_id: Optional[str],
        run_id: Optional[str],
        gate: Dict[str, Any],
    ) -> None:
        tick = await self.plugin.controller.run_system("timer", "get_tick")
        world_spec = await self.plugin.controller.run_environment("science", "get_world_spec")
        await self.plugin._append_jsonl(
            self.plugin._review_gate_log_path,
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

    async def log_code_loop(
        self,
        agent_id: str,
        attempts: List[Dict[str, Any]],
        best_dev_score_norm: Optional[float],
    ) -> None:
        tick = await self.plugin.controller.run_system("timer", "get_tick")
        world_spec = await self.plugin.controller.run_environment("science", "get_world_spec")
        compact_attempts: List[Dict[str, Any]] = []
        for a in (attempts or []):
            r = (a or {}).get("result") if isinstance((a or {}).get("result"), dict) else {}
            compact_attempts.append(
                {
                    "round": a.get("round"),
                    "phase": a.get("phase"),
                    "llm_ok": bool(a.get("llm_ok", False)),
                    "llm_reason": self.plugin._truncate(a.get("llm_reason"), 180),
                    "run_id": r.get("run_id"),
                    "ok": bool(r.get("ok", False)) if isinstance(r, dict) else False,
                    "error": self.plugin._truncate((r or {}).get("error"), 220) if isinstance(r, dict) else "",
                    "dev_score_norm": (r or {}).get("dev_score_norm") if isinstance(r, dict) else None,
                    "score_norm": (r or {}).get("score_norm") if isinstance(r, dict) else None,
                    "solver_mode": (r or {}).get("solver_mode") if isinstance(r, dict) else None,
                    "error_class": ((a.get("diagnosis") or {}).get("error_class") if isinstance(a.get("diagnosis"), dict) else None),
                    "error_codes": ((a.get("diagnosis") or {}).get("error_codes") if isinstance(a.get("diagnosis"), dict) else []),
                    "template_rules": ((a.get("template_fix") or {}).get("rules_hit") if isinstance(a.get("template_fix"), dict) else []),
                }
            )
        await self.plugin._append_jsonl(
            self.plugin._code_loop_log_path,
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

    async def log_code_diagnosis(
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
        tick = await self.plugin.controller.run_system("timer", "get_tick")
        world_spec = await self.plugin.controller.run_environment("science", "get_world_spec")
        await self.plugin._append_jsonl(
            self.plugin._code_diagnosis_log_path,
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
                "code_agent_ok": code_agent_ok,
                "fallback_solver_used": fallback_solver_used,
                "execution_path": execution_path,
                "diagnosis": diagnosis or {},
                "template_fix": template_fix or {},
            },
        )

    async def log_evidence_cards(self, agent_id: str, literature: Dict[str, Any], source: str = "read") -> None:
        if not self.plugin._cards_log_enabled:
            return
        tick = await self.plugin.controller.run_system("timer", "get_tick")
        world_spec = await self.plugin.controller.run_environment("science", "get_world_spec")
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
        await self.plugin._append_jsonl(self.plugin._cards_log_path, record)

    async def log_paper_result(
        self,
        agent_id: str,
        paper_id: Optional[str],
        paper: Dict[str, Any],
        metrics: Dict[str, Any],
        source: str,
    ) -> None:
        if not self.plugin._papers_log_enabled:
            return
        tick = await self.plugin.controller.run_system("timer", "get_tick")
        world_spec = await self.plugin.controller.run_environment("science", "get_world_spec")
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
        await self.plugin._append_jsonl(self.plugin._papers_log_path, record)
