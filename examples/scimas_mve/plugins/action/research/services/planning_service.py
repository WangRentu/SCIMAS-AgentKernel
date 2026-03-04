import json
import re
from typing import Any, Dict, List, Optional


class PlanningService:
    """Planning and compact-view helpers extracted from plugin orchestrator."""

    def __init__(self, plugin: Any):
        self.plugin = plugin

    def compact_data_card(self, data_card: Optional[Dict[str, Any]]) -> Dict[str, Any]:
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
                    "recommended_features": self.plugin._safe_text_list(item.get("recommended_features"), limit=4, item_limit=120),
                    "unit_checks": self.plugin._safe_text_list(item.get("unit_checks"), limit=3, item_limit=120),
                    "recommended_protocol": self.plugin._safe_text_list(item.get("recommended_protocol"), limit=4, item_limit=120),
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
            "risk_flags": self.plugin._safe_text_list(data_card.get("risk_flags"), limit=6, item_limit=120),
        }

    def compact_method_card(self, method_card: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if not isinstance(method_card, dict):
            return {}
        baseline_src = (
            method_card.get("baseline_candidates")
            if isinstance(method_card.get("baseline_candidates"), list)
            else method_card.get("recommended_baselines")
        )
        baselines = []
        for item in (baseline_src or [])[:4]:
            if not isinstance(item, dict):
                continue
            steps = item.get("implementation_steps") if isinstance(item.get("implementation_steps"), list) else item.get("key_steps")
            baselines.append(
                {
                    "name": item.get("name"),
                    "priority": item.get("priority"),
                    "use_when": self.plugin._truncate(item.get("use_when"), 120),
                    "key_steps": self.plugin._safe_text_list(steps, limit=4, item_limit=140),
                    "pitfalls": self.plugin._safe_text_list(item.get("pitfalls") or item.get("risks"), limit=4, item_limit=140),
                    "evidence_refs": self.plugin._safe_text_list(item.get("evidence_refs"), limit=4, item_limit=60),
                }
            )
        failure_playbook = []
        for item in (method_card.get("failure_playbook") or [])[:5]:
            if not isinstance(item, dict):
                continue
            failure_playbook.append(
                {
                    "error_type": item.get("error_type"),
                    "triage": self.plugin._safe_text_list(item.get("triage"), limit=3, item_limit=140),
                    "fix_actions": self.plugin._safe_text_list(item.get("fix_actions"), limit=3, item_limit=140),
                    "evidence_refs": self.plugin._safe_text_list(item.get("evidence_refs"), limit=4, item_limit=60),
                }
            )
        return {
            "version": method_card.get("version"),
            "topic": method_card.get("topic"),
            "metric": method_card.get("metric"),
            "category": method_card.get("category"),
            "task_summary": method_card.get("task_summary") if isinstance(method_card.get("task_summary"), dict) else {},
            "recommended_baselines": baselines,
            "experiment_roadmap": method_card.get("experiment_roadmap") if isinstance(method_card.get("experiment_roadmap"), dict) else {},
            "failure_playbook": failure_playbook,
            "citation_map": method_card.get("citation_map") if isinstance(method_card.get("citation_map"), dict) else {},
            "evaluation_protocol": self.plugin._safe_text_list(method_card.get("evaluation_protocol"), limit=5, item_limit=140),
            "common_pitfalls": self.plugin._safe_text_list(method_card.get("common_pitfalls"), limit=6, item_limit=140),
            "quality": method_card.get("quality") if isinstance(method_card.get("quality"), dict) else {},
        }

    def default_solver_plan(self, world_spec: Dict[str, Any]) -> Dict[str, Any]:
        metric = str(world_spec.get("metric") or "").lower()
        category = str(world_spec.get("category") or "").lower()
        model_family = "tfidf_logreg"
        target_cols: List[str] = []
        scoring = world_spec.get("scoring_column")
        if isinstance(scoring, list):
            target_cols = [str(x).strip() for x in scoring if str(x).strip()][:4]
        elif isinstance(scoring, str) and scoring.strip():
            target_cols = [scoring.strip()]
        if any(tok in metric for tok in ("mae", "mase", "meanabsoluteerror", "spearman")):
            model_family = "tfidf_ridge"
        if "time series" in category:
            model_family = "naive_series"
        return {
            "strategy": "iterative_solver_baseline",
            "target_cols": target_cols,
            "solver_spec": {
                "model_family": model_family,
                "seed": 42,
                "preprocess": {"max_features": 50000, "ngram_range": [1, 2], "min_df": 1},
                "hyperparams": {"C": 1.0, "max_iter": 2000, "class_weight": "balanced", "alpha": 1.0},
                "input_columns": [],
                "target_cols": target_cols,
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

    def merge_solver_plan(self, base_plan: Dict[str, Any], candidate: Dict[str, Any]) -> Dict[str, Any]:
        merged = dict(base_plan or {})
        if not isinstance(candidate, dict):
            return merged
        if isinstance(candidate.get("target_cols"), list):
            merged["target_cols"] = [str(c) for c in candidate.get("target_cols") if str(c).strip()][:16]
        for key in ("schema_assumptions", "memory_safety", "evidence_refs"):
            if isinstance(candidate.get(key), list):
                merged[key] = self.plugin._safe_text_list(candidate.get(key), limit=12, item_limit=220)
        strategy = candidate.get("strategy")
        if isinstance(strategy, str) and strategy.strip():
            merged["strategy"] = strategy.strip()[:80]
        for key in ("rationale", "risk", "replication_plan"):
            if isinstance(candidate.get(key), list):
                merged[key] = self.plugin._safe_text_list(candidate.get(key), limit=6, item_limit=220)
        if isinstance(candidate.get("experiment_protocol"), dict):
            ep = candidate.get("experiment_protocol") or {}
            merged["experiment_protocol"] = {
                "primary_knob": self.plugin._truncate(ep.get("primary_knob"), 120),
                "ablation_axis": self.plugin._truncate(ep.get("ablation_axis"), 120),
                "format_checks": self.plugin._safe_text_list(ep.get("format_checks"), limit=6, item_limit=180),
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
            if isinstance(cand_solver.get("target_cols"), list):
                solver_spec["target_cols"] = [str(c) for c in cand_solver.get("target_cols") if str(c).strip()][:16]
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

    def derive_next_solver_plan_from_history(
        self,
        plan_spec: Dict[str, Any],
        run_history: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        current = self.merge_solver_plan(plan_spec or {}, {})
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
            if model_family == "tfidf_logreg":
                hp["C"] = max(0.1, float(hp.get("C", 1.0) or 1.0) * 0.7)
            elif model_family == "tfidf_ridge":
                hp["alpha"] = min(20.0, float(hp.get("alpha", 1.0) or 1.0) * 1.5)
            preprocess["max_features"] = max(8000, int(preprocess.get("max_features", 50000) or 50000) // 2)
            current["strategy"] = "stability_regularization"
        else:
            preprocess["max_features"] = min(120000, int(preprocess.get("max_features", 50000) or 50000) + 5000)
            if model_family == "tfidf_logreg":
                hp["C"] = min(5.0, float(hp.get("C", 1.0) or 1.0) * 1.2)
            current["strategy"] = "incremental_capacity_tuning"

        solver["preprocess"] = preprocess
        solver["hyperparams"] = hp
        current["solver_spec"] = solver
        return current

    def clamp01(self, value: Any) -> float:
        try:
            parsed = float(value)
        except Exception:
            return 0.0
        return max(0.0, min(1.0, parsed))

    def extract_evidence_refs(self, text: Any) -> List[str]:
        raw = str(text or "")
        if not raw:
            return []
        refs = set()
        for cid in re.findall(r"\bC\d{4}\b", raw):
            refs.add(cid)
        for rid in re.findall(r"\bRUN@[A-Za-z0-9\-_]+\b", raw):
            refs.add(rid)
        return sorted(refs)
