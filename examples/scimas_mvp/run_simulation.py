"python -m examples.scimas_mvp.run_simulation"

import os
import asyncio
import yaml
import time
import json
import shutil
import hashlib
import subprocess
from datetime import datetime, timezone
from typing import Any, Optional, Dict

project_path = os.path.dirname(os.path.abspath(__file__))

os.environ["MAS_PROJECT_ABS_PATH"] = project_path
if "MAS_PROJECT_REL_PATH" not in os.environ:
    os.environ["MAS_PROJECT_REL_PATH"] = "examples.scimas_mvp"

from agentkernel_standalone.mas.builder import Builder
from examples.scimas_mvp.registry import RESOURCES_MAPS
from agentkernel_standalone.toolkit.logger import get_logger

from examples.scimas_mvp.custom_controller import CustomController

logger = get_logger(__name__)
VERBOSE_TICK_LOGS = os.getenv("SCIMAS_VERBOSE_TICK_LOGS", "0").lower() in {"1", "true", "yes"}


def _as_int(value: Any, default: int) -> int:
    try:
        parsed = int(value)
    except Exception:
        return default
    return parsed if parsed > 0 else default


def _read_num_episodes_from_yaml(project_root: str) -> Optional[int]:
    cfg_path = os.path.join(project_root, "configs", "simulation_config.yaml")
    if not os.path.exists(cfg_path):
        return None
    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        simulation = raw.get("simulation") or {}
        return _as_int(simulation.get("num_episodes"), default=1)
    except Exception as e:
        logger.warning(f"Failed to read num_episodes from yaml: {e}")
        return None


def _utc_ts() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _read_text(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return ""


def _git_commit(cwd: str) -> Optional[str]:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=cwd, text=True, stderr=subprocess.DEVNULL)
        return out.strip() or None
    except Exception:
        return None


def _config_snapshot(project_root: str) -> Dict[str, Any]:
    cfg_dir = os.path.join(project_root, "configs")
    snapshot: Dict[str, Any] = {}
    if not os.path.isdir(cfg_dir):
        return snapshot
    for name in sorted(os.listdir(cfg_dir)):
        if not name.endswith(".yaml"):
            continue
        path = os.path.join(cfg_dir, name)
        text = _read_text(path)
        snapshot[name] = {
            "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "text": text,
        }
    return snapshot


def _sim_dir(project_root: str) -> str:
    return os.path.join(project_root, "logs", "app", "simulation")


def _prepare_sim_logs(project_root: str) -> None:
    if os.getenv("SCIMAS_CLEAN_SIM_LOGS", "0").lower() not in {"1", "true", "yes"}:
        return
    sim_dir = _sim_dir(project_root)
    if not os.path.isdir(sim_dir):
        return
    for name in os.listdir(sim_dir):
        path = os.path.join(sim_dir, name)
        try:
            if os.path.isfile(path):
                os.remove(path)
        except Exception as e:
            logger.warning(f"Failed to clean simulation artifact {path}: {e}")


def _prepare_compact_logs(project_root: str) -> None:
    log_mode = (os.getenv("SCIMAS_LOG_MODE", "compact") or "compact").strip().lower()
    if log_mode not in {"compact", "minimal"}:
        return
    if os.getenv("SCIMAS_CLEAN_RAW_LOGS", "1").lower() in {"0", "false", "no"}:
        return
    candidates = [
        os.path.join(project_root, "logs", "app", "action", "trace.jsonl"),
        os.path.join(project_root, "logs", "app", "research", "evidence_cards.jsonl"),
        os.path.join(project_root, "logs", "app", "llm", "llm_calls.jsonl"),
        os.path.join(project_root, "logs", "app", "simulation", "leaderboard.csv"),
    ]
    for path in candidates:
        try:
            if os.path.isfile(path):
                os.remove(path)
        except Exception as e:
            logger.warning(f"Failed to remove compact-mode noisy log {path}: {e}")


def _write_run_manifest(
    project_root: str,
    *,
    max_ticks: int,
    num_episodes: int,
) -> Dict[str, Any]:
    sim_dir = _sim_dir(project_root)
    os.makedirs(sim_dir, exist_ok=True)
    run_tag = os.getenv("SCIMAS_RUN_TAG") or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    manifest = {
        "ts": _utc_ts(),
        "run_tag": run_tag,
        "experiment_name": os.getenv("SCIMAS_EXPERIMENT_NAME"),
        "variant_name": os.getenv("SCIMAS_VARIANT_NAME"),
        "world_seed": os.getenv("SCIMAS_WORLD_SEED"),
        "project_root": project_root,
        "git_commit": _git_commit(project_root),
        "simulation": {
            "max_ticks": max_ticks,
            "num_episodes": num_episodes,
        },
        "mechanism_flags": {
            "SCIMAS_EVOLVE_W_INDIV": os.getenv("SCIMAS_EVOLVE_W_INDIV"),
            "SCIMAS_EVOLVE_W_CONTRIB": os.getenv("SCIMAS_EVOLVE_W_CONTRIB"),
            "SCIMAS_EVOLVE_W_COLLAB": os.getenv("SCIMAS_EVOLVE_W_COLLAB"),
            "SCIMAS_TARGET_COLLAB_RATIO": os.getenv("SCIMAS_TARGET_COLLAB_RATIO"),
            "SCIMAS_LLM_ENABLE": os.getenv("SCIMAS_LLM_ENABLE"),
            "SCIMAS_LLM_HYPOTHESIZE": os.getenv("SCIMAS_LLM_HYPOTHESIZE"),
            "SCIMAS_LLM_WRITE": os.getenv("SCIMAS_LLM_WRITE"),
            "SCIMAS_LLM_REVIEW": os.getenv("SCIMAS_LLM_REVIEW"),
        },
        "config_snapshot": _config_snapshot(project_root),
    }
    path = os.path.join(sim_dir, "run_manifest.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
        f.write("\n")
    return manifest


def _archive_run(project_root: str, manifest: Dict[str, Any]) -> Optional[str]:
    archive_root = os.getenv("SCIMAS_EXPERIMENT_ARCHIVE_DIR")
    if not archive_root:
        return None
    variant = (manifest.get("variant_name") or "default").replace("/", "_")
    seed = manifest.get("world_seed") or "na"
    run_tag = manifest.get("run_tag") or "run"
    out_dir = os.path.join(archive_root, f"{variant}__seed_{seed}__{run_tag}")
    os.makedirs(out_dir, exist_ok=True)
    sim_dir = _sim_dir(project_root)
    if os.path.isdir(sim_dir):
        shutil.copytree(sim_dir, os.path.join(out_dir, "simulation"), dirs_exist_ok=True)
    flow_log = os.path.join(project_root, "logs", "app", "simulation_flow.log")
    if os.path.exists(flow_log):
        shutil.copy2(flow_log, os.path.join(out_dir, "simulation_flow.log"))
    return out_dir

"""Builder 装配环境/动作/agent/system，然后按 max_ticks 和 num_episodes 进入 episode 循环（配置在 simulation_config.yaml (line 7)）。"""
async def main():
    controller = None
    system = None
    run_manifest: Optional[Dict[str, Any]] = None
    try:
        logger.info(f"Project path set to: {project_path}")
        _prepare_sim_logs(project_path)
        _prepare_compact_logs(project_path)

        logger.info("Creating simulation builder...")
        sim_builder = Builder(project_path=project_path, resource_maps=RESOURCES_MAPS)

        logger.info("Assembling all simulation components...")
        controller, system = await sim_builder.init()

        # --- Simulation Loop ---
        max_ticks = _as_int(getattr(sim_builder.config.simulation, "max_ticks", 10), default=10)
        max_ticks = _as_int(os.getenv("SCIMAS_MAX_TICKS", max_ticks), default=max_ticks)
        episodes_from_model = getattr(sim_builder.config.simulation, "num_episodes", None)
        episodes_from_yaml = _read_num_episodes_from_yaml(project_path)
        num_episodes = _as_int(
            episodes_from_model if episodes_from_model is not None else episodes_from_yaml,
            default=1,
        )
        num_episodes = _as_int(os.getenv("SCIMAS_NUM_EPISODES", num_episodes), default=num_episodes)
        run_manifest = _write_run_manifest(project_path, max_ticks=max_ticks, num_episodes=num_episodes)
        logger.info(f"--- Starting Simulation Run: {num_episodes} episodes, {max_ticks} ticks/episode ---")

        if isinstance(controller, CustomController):
            await controller.reset_episode_state()

        for ep in range(num_episodes):
            logger.info(f"=== Episode {ep + 1}/{num_episodes} start ===")
            num_ticks_to_run = max_ticks
            total_duration = 0.0

            for _ in range(num_ticks_to_run):
                tick_start_time = time.time()

                # 程序主骨架：每个 tick 都调用 controller.step_agent() 和 system.run("messager", "dispatch_messages")
                await controller.step_agent()
                await system.run("messager", "dispatch_messages") 
                # 这会走 EasyCommunicationPlugin.send_message()，调用 system 的 messager.send_message 把 Message 放进系统消息队列（EasyCommunicationPlugin.py (line 41)、EasyCommunicationPlugin.py (line 53)）。
                # system.run("messager","dispatch_messages") 负责把队列里的消息按 to_id 分发，最终触发接收方的 EasyPerceivePlugin.add_message() 把它存到 received_messages（EasyPerceivePlugin.py (line 63)）。

                if isinstance(controller, CustomController):
                    await controller.update_agents_status()

                tick_end_time = time.time()
                actual_tick_duration = tick_end_time - tick_start_time
                total_duration += actual_tick_duration

                current_tick = await system.run("timer", "get_tick")
                await system.run("timer", "add_tick", duration_seconds=actual_tick_duration)
                if VERBOSE_TICK_LOGS:
                    logger.info(f"--- Tick {current_tick} finished in {actual_tick_duration:.4f} seconds ---")

            if num_ticks_to_run > 0:
                average_time_per_tick = total_duration / num_ticks_to_run
                logger.info(
                    f"--- Episode {ep + 1} finished: {num_ticks_to_run} ticks, avg {average_time_per_tick:.4f}s/tick ---"
                )

            if isinstance(controller, CustomController):
                final_info = await controller.finalize_episode(episode_index=ep)
                logger.info(f"Episode {ep + 1} summary: {final_info}")
                if ep < num_episodes - 1:
                    evolve_info = await controller.evolve_population()
                    top_donor = None
                    if isinstance(evolve_info, dict):
                        donors = evolve_info.get("top_donors") or []
                        if donors:
                            d0 = donors[0] or {}
                            top_donor = {
                                "agent_id": d0.get("agent_id"),
                                "selection_score": d0.get("selection_score"),
                                "individual_fitness": d0.get("individual_fitness"),
                                "contribution_credit_total": d0.get("contribution_credit_total"),
                                "collab_count": d0.get("collab_count"),
                            }
                    logger.info(
                        f"Episode {ep + 1} evolution: "
                        f"top_k={evolve_info.get('top_k') if isinstance(evolve_info, dict) else None}, "
                        f"selection_mode={evolve_info.get('selection_mode') if isinstance(evolve_info, dict) else None}, "
                        f"top_donor={top_donor}"
                    )
                    await controller.reset_episode_state()

        logger.info("\n--- Simulation Finished ---")
        try:
            from examples.scimas_mvp.analysis.analyze_metrics import analyze

            analysis = analyze(project_path)
            if run_manifest is not None:
                run_manifest["analysis"] = analysis
                manifest_path = os.path.join(_sim_dir(project_path), "run_manifest.json")
                with open(manifest_path, "w", encoding="utf-8") as f:
                    json.dump(run_manifest, f, ensure_ascii=False, indent=2)
                    f.write("\n")
                archived = _archive_run(project_path, run_manifest)
                if archived:
                    logger.info(f"Archived run artifacts to: {archived}")
            logger.info(f"Auto analysis completed: {analysis}")
        except Exception as e:
            logger.warning(f"Auto analysis failed: {e}")
    except Exception as e:
        logger.error(f"Simulation failed: {e}")
        logger.exception("An unhandled exception occurred during simulation.")
    finally:
        if controller:
            result = await controller.close()
            logger.info(f"Controller close result is {result}")
        if system:
            result = await system.close()
            logger.info(f"System close result is {result}")


if __name__ == "__main__":
    try:
        asyncio.run(main())

    except KeyboardInterrupt:
        logger.info("Simulation interrupted by user. Exiting.")
    finally:
        logger.info("Simulation ended.")
