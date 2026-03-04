import json
import textwrap
from typing import Any, Dict, List, Optional


class PromptService:
    """Centralized prompt builders for research operators."""

    def __init__(self, plugin: Any):
        self.plugin = plugin

    def build_task_role_prompt(
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

    def build_plan_prompt(
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
                "text": self.plugin._truncate(c.get("text"), 180),
            }
            for c in cards[-self.plugin._llm_max_cards :]
        ]
        runs_short = [
            {
                "run_id": r.get("run_id"),
                "metric_name": r.get("metric_name"),
                "raw_score": r.get("raw_score"),
                "score_norm": r.get("score_norm"),
                "ok": r.get("ok"),
                "strategy": r.get("strategy"),
                "error": self.plugin._truncate(r.get("error"), 120),
            }
            for r in recent_runs[-self.plugin._llm_max_runs :]
        ]
        data_card_short = self.plugin._compact_data_card(data_card)
        method_card_short = self.plugin._compact_method_card(method_card)
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

            Return ONLY JSON with concise but technical content.
            Required output schema:
            {{
              "hypothesis": ["..."],
              "plan_spec": {{
                "strategy": "...",
                "target_cols": ["..."],
                "schema_assumptions": ["..."],
                "memory_safety": ["..."],
                "evidence_refs": ["..."],
                "solver_spec": {{
                  "model_family": "tfidf_logreg|linear_svc|tfidf_ridge|naive_series",
                  "seed": 42,
                  "target_cols": ["..."],
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
            }}
            """
        ).strip()

    def build_experiment_prompt(
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
                            "text": self.plugin._truncate(c.get("text"), 120),
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
                "error": self.plugin._truncate(o.get("error"), 120),
            }
            for o in observations[-self.plugin._llm_max_runs :]
        ]
        data_card_short = self.plugin._compact_data_card(data_card)
        method_card_short = self.plugin._compact_method_card(method_card)
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

    def build_retrieve_method_prompt(
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
                            "text": self.plugin._truncate(c.get("text"), 120),
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
                "error": self.plugin._truncate(o.get("error"), 180),
                "strategy": self.plugin._truncate(o.get("strategy"), 120),
            }
            for o in observations[-self.plugin._llm_max_runs :]
        ]
        data_card_short = self.plugin._compact_data_card(data_card)
        evidence_short = []
        for item in (evidence_pack or [])[:16]:
            if not isinstance(item, dict):
                continue
            evidence_short.append(
                {
                    "evidence_id": item.get("evidence_id"),
                    "source_collection": item.get("source_collection"),
                    "source_type": item.get("source_type"),
                    "source_id": item.get("source_id"),
                    "match_score": item.get("match_score"),
                    "tags": list(item.get("tags") or [])[:6],
                    "chunk_text": self.plugin._truncate(item.get("chunk_text"), 220),
                }
            )

        return textwrap.dedent(
            f"""
            You are building a machine-consumable MethodCard for AIRS code generation.
            The card must be executable, evidence-backed, and robust to failure modes.

            Task context:
            - task_name: {world_spec.get('task_name')}
            - metric: {world_spec.get('metric')}
            - category: {world_spec.get('category')}
            - research_problem: {world_spec.get('research_problem')}
            - rag_status: {rag_status}

            Data card (drives method choice):
            {json.dumps(data_card_short, ensure_ascii=False)}

            Existing notes:
            {json.dumps(notes_short, ensure_ascii=False)}

            Recent run history:
            {json.dumps(obs_short, ensure_ascii=False)}

            Query bundle:
            {json.dumps(query_bundle, ensure_ascii=False)}

            Evidence pack:
            {json.dumps(evidence_short, ensure_ascii=False)}

            Hard constraints:
            1) Output MUST be valid JSON only.
            2) Every key recommendation MUST include evidence_refs with evidence_id.
            3) If evidence is insufficient, still return structure and mark unknown/degraded explicitly.
            4) Provide at least one minimum runnable baseline path (data loading -> training -> dev eval -> submission checks).
            5) Do not fabricate citations/run references.

            Return ONLY JSON:
            {{
              "ok": true,
              "card_type": "method_card",
              "version": "v2",
              "task_name": "...",
              "topic": "task_baselines",
              "metric": "...",
              "category": "...",
              "research_problem": "...",
              "task_summary": {{
                "problem_type": "...",
                "data_signals": ["..."],
                "risk_flags": ["..."]
              }},
              "baseline_candidates": [
                {{
                  "name": "...",
                  "priority": 1,
                  "implementation_steps": ["..."],
                  "hyperparam_ranges": {{"...": "..."}},
                  "expected_gain": "...",
                  "risks": ["..."],
                  "evidence_refs": ["EVID-..."]
                }}
              ],
              "experiment_roadmap": {{
                "stage_1": ["..."],
                "stage_2": ["..."],
                "stop_conditions": ["..."]
              }},
              "failure_playbook": [
                {{
                  "error_type": "schema|io|deps|oom|eval",
                  "triage": ["..."],
                  "fix_actions": ["..."],
                  "evidence_refs": ["EVID-..."]
                }}
              ],
              "ablation_plan": [
                {{
                  "name": "...",
                  "control": "...",
                  "treatment": "...",
                  "metric_expectation": "...",
                  "evidence_refs": ["EVID-..."]
                }}
              ],
              "citation_map": {{
                "baseline:primary": ["EVID-..."],
                "playbook:oom": ["EVID-..."]
              }},
              "quality": {{
                "schema_valid": true,
                "citation_coverage": 0.0,
                "executable_minimum": true,
                "level": "L0|L1|L2",
                "degraded": false,
                "fail_reasons": []
              }},
              "source": "retrieve_v2_llm"
            }}
            """
        ).strip()

    def build_write_prompt(
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
                        "text": self.plugin._truncate(c.get("text"), 120),
                    }
                )
        obs_short = [
            {
                "run_id": o.get("run_id"),
                "score_norm": o.get("score_norm"),
                "ok": o.get("ok"),
                "strategy": o.get("strategy"),
            }
            for o in observations[-self.plugin._llm_max_runs :]
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

    def build_review_prompt(self, *, paper: Dict[str, Any], metrics: Dict[str, Any]) -> str:
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

    def build_replication_prompt(
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

    def build_code_experiment_prompt(
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
                            "text": self.plugin._truncate(c.get("text"), 120),
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
                "error": self.plugin._truncate(o.get("error"), 180),
            }
            for o in observations[-self.plugin._llm_max_runs :]
        ]
        data_card_short = self.plugin._compact_data_card(data_card)
        method_card_short = self.plugin._compact_method_card(method_card)
        observed_columns = data_columns if isinstance(data_columns, dict) else {}
        train_columns = [str(x) for x in (observed_columns.get("train") or []) if str(x).strip()][:128]
        test_columns = [str(x) for x in (observed_columns.get("test") or []) if str(x).strip()][:128]
        enforce_columns = bool(getattr(self.plugin, "_experiment_prompt_enforce_columns", True))
        column_constraints = ""
        if enforce_columns:
            column_constraints = textwrap.dedent(
                """
                11) Use only columns observed in 'Observed train/test columns' unless guarded fallback checks prove existence at runtime.
                12) Never hardcode imaginary column names; validate columns before selecting features/targets.
                """
            ).strip()
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

            Observed train columns:
            {json.dumps(train_columns, ensure_ascii=False)}

            Observed test columns:
            {json.dumps(test_columns, ensure_ascii=False)}

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
            {column_constraints}

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
