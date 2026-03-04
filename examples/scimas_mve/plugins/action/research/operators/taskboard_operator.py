from typing import Any, Dict, Optional

from agentkernel_standalone.toolkit.logger import get_logger
from agentkernel_standalone.types.schemas.action import ActionResult

logger = get_logger(__name__)


class TaskboardOperator:
    """Encapsulates taskboard claim/complete execution paths."""

    def __init__(self, plugin: Any):
        self.plugin = plugin

    async def _get_claimed_task(
        self,
        agent_id: str,
        task_id: str,
        current_tick: Optional[int] = None,
    ) -> Optional[Dict[str, Any]]:
        try:
            direct = await self.plugin.controller.run_environment(
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

        listed = await self.plugin.controller.run_environment(
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
        listed_running = await self.plugin.controller.run_environment(
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
            return await self.plugin.read(agent_id=agent_id, topic=payload.get("topic"))
        if task_action == "prepare_data":
            return await self.plugin.prepare_data(agent_id=agent_id, refresh=bool(payload.get("refresh")))
        if task_action == "profile_data":
            return await self.plugin.profile_data(
                agent_id=agent_id,
                focus_cols=payload.get("focus_cols"),
                refresh=bool(payload.get("refresh")),
            )
        if task_action == "retrieve_literature":
            return await self.plugin.retrieve_literature(
                agent_id=agent_id,
                topic=payload.get("topic"),
                refresh=bool(payload.get("refresh")),
            )
        if task_action == "hypothesize":
            result = await self.plugin.hypothesize(agent_id=agent_id)
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
            if self.plugin._vdh_enable and self.plugin._vdh_gate_policy == "hard_fail":
                last_vdh = await self.plugin._get_state(agent_id, "last_vdh_report")
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
            return await self.plugin.experiment(agent_id=agent_id, config=payload.get("config"))
        if task_action == "replicate":
            return await self.plugin.replicate(agent_id=agent_id, paper_id=payload.get("paper_id"))
        if task_action == "write":
            return await self.plugin.write(agent_id=agent_id)
        if task_action == "review":
            return await self.plugin.review(
                agent_id=agent_id,
                paper_id=payload.get("paper_id"),
                run_id=payload.get("run_id"),
            )
        if task_action == "verify_strength":
            return await self.plugin.verify_strength(
                agent_id=agent_id,
                paper_id=payload.get("paper_id"),
                reviewer_id=payload.get("reviewer_id"),
                strength=payload.get("strength"),
                test=payload.get("test"),
            )
        if task_action == "verify_issue":
            return await self.plugin.verify_issue(
                agent_id=agent_id,
                paper_id=payload.get("paper_id"),
                reviewer_id=payload.get("reviewer_id"),
                issue=payload.get("issue"),
                test=payload.get("test"),
            )
        if task_action == "share_evidence":
            return await self.plugin.share_evidence(agent_id=agent_id, to_agent_id=payload.get("to_agent_id"))
        if task_action == "share_observation":
            return await self.plugin.share_observation(agent_id=agent_id, to_agent_id=payload.get("to_agent_id"))
        return ActionResult.success(method_name=task_action, message="No-op task action.", data={"reward": 0.0})

    async def execute_claim_task(
        self,
        agent_id: str,
        task_id: Optional[str] = None,
        task_type: Optional[str] = None,
    ) -> ActionResult:
        current_tick = int(await self.plugin.controller.run_system("timer", "get_tick"))
        backoff_until = int((await self.plugin._get_state(agent_id, "claim_backoff_until_tick")) or 0)
        fail_streak = int((await self.plugin._get_state(agent_id, "claim_fail_streak")) or 0)
        if current_tick < backoff_until:
            wait_ticks = max(0, backoff_until - current_tick)
            ar = ActionResult.success(
                method_name="claim_task",
                message=f"Claim backoff active for {wait_ticks} tick(s).",
                data={
                    "ok": False,
                    "reason": "claim_backoff_active",
                    "wait_ticks": wait_ticks,
                    "reward": -self.plugin._claim_cost,
                    "effective_action": "claim_backoff",
                    "reward_components": {
                        "task_claim_reward": float(-self.plugin._claim_cost),
                        "task_claim_cost": float(self.plugin._claim_cost),
                        "learning_reward": float(-self.plugin._claim_cost),
                    },
                },
            )
            await self.plugin._append_trace(agent_id, "claim_task", -self.plugin._claim_cost, ar.data or {})
            return ar

        selected_task_id = task_id
        llm_selection: Optional[Dict[str, Any]] = None
        if not selected_task_id:
            listed = await self.plugin.controller.run_environment("science", "task_list", status="open", current_tick=current_tick)
            tasks = (listed or {}).get("tasks", []) if isinstance(listed, dict) else []
            if task_type:
                tasks = [t for t in tasks if t.get("task_type") == task_type]
            if tasks:
                world_spec = await self.plugin.controller.run_environment("science", "get_world_spec", current_tick=current_tick)
                role_plan = await self.plugin._get_state(agent_id, "llm_role_plan")
                active_task_name = str(world_spec.get("task_name") or "")
                role_plan_stale = not isinstance(role_plan, dict) or role_plan.get("task_name") != active_task_name
                if self.plugin._llm_ready("claim_task") and role_plan_stale:
                    hypothesis = await self.plugin._get_state(agent_id, "hypothesis") or []
                    notes = await self.plugin._get_state(agent_id, "notes") or []
                    observations = await self.plugin._get_state(agent_id, "observations") or []
                    prompt = self.plugin._build_task_role_prompt(
                        world_spec=world_spec,
                        open_tasks=tasks,
                        hypothesis=hypothesis,
                        notes_count=len(notes),
                        observations_count=len(observations),
                    )
                    llm_result = await self.plugin._call_llm_json(agent_id=agent_id, action_name="claim_task", prompt=prompt)
                    if llm_result.get("ok") and isinstance(llm_result.get("data"), dict):
                        llm_selection = llm_result.get("data") or {}
                        role_plan = {
                            "task_name": active_task_name,
                            "role_name": self.plugin._truncate(llm_selection.get("role_name"), 80),
                            "preferred_task_types": self.plugin._safe_task_types(
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
                            "fallback_if_blocked": self.plugin._safe_task_types(
                                llm_selection.get("fallback_if_blocked"),
                                fallback=["verify_issue", "verify_strength", "prepare_data", "profile_data", "read", "hypothesize"],
                            ),
                            "primary_task_id": str(llm_selection.get("primary_task_id") or ""),
                            "selection_rationale": self.plugin._safe_text_list(
                                llm_selection.get("selection_rationale"),
                                limit=5,
                                item_limit=220,
                            ),
                            "risk_controls": self.plugin._safe_text_list(
                                llm_selection.get("risk_controls"),
                                limit=5,
                                item_limit=220,
                            ),
                        }
                        await self.plugin._set_state(agent_id, "llm_role_plan", role_plan)

                preferred_types = []
                primary_task_id = ""
                if isinstance(role_plan, dict):
                    preferred_types = self.plugin._safe_task_types(role_plan.get("preferred_task_types"))
                    preferred_types.extend(
                        x for x in self.plugin._safe_task_types(role_plan.get("fallback_if_blocked")) if x not in preferred_types
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
            delay = min(self.plugin._claim_backoff_max, self.plugin._claim_backoff_base * (2 ** max(0, fail_streak - 1)))
            await self.plugin._set_state(agent_id, "claim_fail_streak", fail_streak)
            await self.plugin._set_state(agent_id, "claim_backoff_until_tick", current_tick + delay)
            ar = ActionResult.success(
                method_name="claim_task",
                message="No open task available.",
                data={
                    "ok": False,
                    "reason": "no_open_task",
                    "backoff_ticks": delay,
                    "reward": -self.plugin._claim_cost,
                    "effective_action": "claim_task",
                    "reward_components": {
                        "task_claim_reward": float(-self.plugin._claim_cost),
                        "task_claim_cost": float(self.plugin._claim_cost),
                        "learning_reward": float(-self.plugin._claim_cost),
                    },
                },
            )
            await self.plugin._append_trace(agent_id, "claim_task", -self.plugin._claim_cost, ar.data or {})
            return ar

        claim_res = await self.plugin.controller.run_environment(
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
            await self.plugin._set_state(agent_id, "claim_fail_streak", 0)
            await self.plugin._set_state(agent_id, "claim_backoff_until_tick", current_tick)
        else:
            fail_streak += 1
            reason = str((claim_res or {}).get("reason") or "")
            if reason == "not_active_worker":
                delay = self.plugin._claim_backoff_max
            else:
                delay = min(self.plugin._claim_backoff_max, self.plugin._claim_backoff_base * (2 ** max(0, fail_streak - 1)))
            await self.plugin._set_state(agent_id, "claim_fail_streak", fail_streak)
            await self.plugin._set_state(agent_id, "claim_backoff_until_tick", current_tick + delay)
        reward = (0.01 if ok else 0.0) - self.plugin._claim_cost
        if ok:
            await self.plugin._set_state(agent_id, "current_task_id", selected_task_id)
            await self.plugin._set_state(agent_id, "current_task_type", str((task or {}).get("task_type") or ""))

        auto_dispatch: Optional[ActionResult] = None
        task_type_lower = str((task or {}).get("task_type") or "").strip().lower()
        if ok and self.plugin._claim_dispatch_enabled and task_type_lower in self.plugin._claim_dispatch_task_types:
            try:
                auto_dispatch = await self.execute_complete_task(
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
                    "task_claim_cost": float(self.plugin._claim_cost),
                    "learning_reward": float(reward),
                },
            },
        )
        await self.plugin._append_trace(agent_id, "claim_task", reward, ar.data or {})
        return ar

    async def execute_complete_task(
        self,
        agent_id: str,
        task_id: str,
        task_action: Optional[str] = None,
        task_payload: Optional[Dict[str, Any]] = None,
    ) -> ActionResult:
        current_tick = int(await self.plugin.controller.run_system("timer", "get_tick"))
        task = await self._get_claimed_task(agent_id, task_id, current_tick=current_tick)
        if task is None:
            remembered_task_id = await self.plugin._get_state(agent_id, "current_task_id")
            if remembered_task_id and str(remembered_task_id) == str(task_id):
                try:
                    direct = await self.plugin.controller.run_environment(
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
            await self.plugin._append_trace(agent_id, "complete_task", 0.0, ar.data or {})
            return ar

        if self.plugin._task_heartbeat_enabled:
            try:
                await self.plugin.controller.run_environment(
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
                action_result = await self._execute_task_action(
                    agent_id=agent_id,
                    task_action=action_name,
                    task_payload=payload,
                )
            except Exception as e:
                action_result = ActionResult.error(
                    method_name=str(action_name),
                    message=f"Inner action raised exception: {self.plugin._truncate(str(e), 220)}",
                    data={
                        "ok": False,
                        "exception": self.plugin._truncate(str(e), 400),
                        "reward": 0.0,
                        "effective_action": action_name or "complete_task",
                        "reward_components": {"learning_reward": 0.0},
                    },
                )

        if self.plugin._task_heartbeat_enabled:
            try:
                await self.plugin.controller.run_environment(
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
            release_res = await self.plugin.controller.run_environment(
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
            await self.plugin._append_trace(agent_id, "complete_task", 0.0, ar.data or {})
            return ar

        # Deferred-path: inner action succeeded syntactically but explicitly reported
        # unmet preconditions. Do not mark task as completed.
        if isinstance(action_result, ActionResult) and isinstance(action_result.data, dict):
            inner_data = action_result.data or {}
            precondition_failed = bool(inner_data.get("precondition_failed", False))
            pending_prereq = bool(inner_data.get("pending_prereq", False))
            if precondition_failed or pending_prereq:
                release_reason = f"precondition_pending:{action_name or 'unknown'}"
                release_res = await self.plugin.controller.run_environment(
                    "science",
                    "task_release",
                    task_id=task_id,
                    agent_id=agent_id,
                    reason=release_reason,
                    current_tick=current_tick,
                )
                released = bool((release_res or {}).get("ok", False))
                if released:
                    await self.plugin._set_state(agent_id, "current_task_id", None)
                    await self.plugin._set_state(agent_id, "current_task_type", None)
                inner_reward = float(inner_data.get("reward", 0.0) or 0.0)
                ar = ActionResult.success(
                    method_name="complete_task",
                    message="Task deferred: prerequisites pending.",
                    data={
                        "task_id": task_id,
                        "task_action": action_name,
                        "ok": False,
                        "deferred": True,
                        "released": released,
                        "release_reason": release_reason,
                        "release_result": release_res,
                        "precondition_failed": precondition_failed,
                        "pending_prereq": pending_prereq,
                        "inner_action_status": action_result.status,
                        "inner_action_message": action_result.message,
                        "inner_action_data": inner_data,
                        "reward": inner_reward,
                        "effective_action": action_name or "complete_task",
                        "reward_components": {
                            "inner_reward": float(inner_reward),
                            "task_complete_bonus": 0.0,
                            "task_complete_total": float(inner_reward),
                            "learning_reward": float(inner_reward),
                        },
                    },
                )
                await self.plugin._append_trace(agent_id, "complete_task", inner_reward, ar.data or {})
                return ar

        completion_result = {
            "task_action": action_name,
            "action_status": action_result.status if isinstance(action_result, ActionResult) else None,
            "action_data": action_result.data if isinstance(action_result, ActionResult) else {},
        }
        complete_res = await self.plugin.controller.run_environment(
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
            await self.plugin._set_state(agent_id, "current_task_id", None)
            await self.plugin._set_state(agent_id, "current_task_type", None)

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
        await self.plugin._append_trace(agent_id, "complete_task", reward, ar.data or {})
        return ar
