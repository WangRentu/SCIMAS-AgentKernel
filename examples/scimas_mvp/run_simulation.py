"python -m examples.scimas_mvp.run_simulation"

import os
import asyncio
import yaml
import time
from typing import Any, Optional

project_path = os.path.dirname(os.path.abspath(__file__))

os.environ["MAS_PROJECT_ABS_PATH"] = project_path
if "MAS_PROJECT_REL_PATH" not in os.environ:
    os.environ["MAS_PROJECT_REL_PATH"] = "examples.scimas_mvp"

from agentkernel_standalone.mas.builder import Builder
from examples.scimas_mvp.registry import RESOURCES_MAPS
from agentkernel_standalone.toolkit.logger import get_logger

from examples.scimas_mvp.custom_controller import CustomController

logger = get_logger(__name__)


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


async def main():
    """Async main function to assemble and start the simulation"""
    controller = None
    system = None
    try:
        logger.info(f"Project path set to: {project_path}")

        logger.info("Creating simulation builder...")
        sim_builder = Builder(project_path=project_path, resource_maps=RESOURCES_MAPS)

        logger.info("Assembling all simulation components...")
        controller, system = await sim_builder.init()

        # --- Simulation Loop ---
        max_ticks = _as_int(getattr(sim_builder.config.simulation, "max_ticks", 10), default=10)
        episodes_from_model = getattr(sim_builder.config.simulation, "num_episodes", None)
        episodes_from_yaml = _read_num_episodes_from_yaml(project_path)
        num_episodes = _as_int(
            episodes_from_model if episodes_from_model is not None else episodes_from_yaml,
            default=1,
        )
        logger.info(f"--- Starting Simulation Run: {num_episodes} episodes, {max_ticks} ticks/episode ---")

        if isinstance(controller, CustomController):
            await controller.reset_episode_state()

        for ep in range(num_episodes):
            logger.info(f"=== Episode {ep + 1}/{num_episodes} start ===")
            num_ticks_to_run = max_ticks
            total_duration = 0.0

            for _ in range(num_ticks_to_run):
                tick_start_time = time.time()

                await controller.step_agent()
                await system.run("messager", "dispatch_messages")

                if isinstance(controller, CustomController):
                    await controller.update_agents_status()

                tick_end_time = time.time()
                actual_tick_duration = tick_end_time - tick_start_time
                total_duration += actual_tick_duration

                current_tick = await system.run("timer", "get_tick")
                await system.run("timer", "add_tick", duration_seconds=actual_tick_duration)
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
                    logger.info(f"Episode {ep + 1} evolution: {evolve_info}")
                    await controller.reset_episode_state()

        logger.info("\n--- Simulation Finished ---")
        try:
            from examples.scimas_mvp.analysis.analyze_metrics import analyze

            analysis = analyze(project_path)
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
