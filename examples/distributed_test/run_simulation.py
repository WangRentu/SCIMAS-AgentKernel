'python -m examples.distributed_test.run_simulation'
import os
project_path = os.path.dirname(os.path.abspath(__file__))
os.environ["MAS_PROJECT_ABS_PATH"] = project_path
os.environ["MAS_PROJECT_REL_PATH"] = "examples.distributed_test"
os.environ["MAS_EVENT_LOG_DIR"] = project_path
import asyncio
import ray
import time
from agentkernel_distributed.mas.builder import Builder
from .registry import RESOURCES_MAPS
from agentkernel_distributed.toolkit.logger import get_logger

logger = get_logger(__name__)

async def main():
    pod_manager = None
    system = None
    total_duration = 0
    try:
        logger.info(f'Project path set to {project_path}.')

        # ===== Step1 : Initialize Ray =====
        pythonpath_root = os.path.abspath(os.path.join(project_path, "..", ".."))
        current_pythonpath = os.environ.get("PYTHONPATH", "")
        new_pythonpath = f"{pythonpath_root}{os.pathsep}{current_pythonpath}"
        runtime_env = {
            'working_dir': project_path,
            'env_vars': {
                "MAS_EVENT_LOG_DIR": os.environ.get("MAS_EVENT_LOG_DIR", ""),
                "PYTHONPATH": new_pythonpath,
            },
            'excludes':[
                '*.pyc',
                '__pycache__',
                'docs',
                'info_extraction'
            ]
        }

        logger.info(f'Init Ray with runtime env: {runtime_env}.')

        ray.init(runtime_env = runtime_env)

        logger.info(f'Ray is initialized.')

        # ===== Step2 : initialize the bulder, start all the simulation components =====
        logger.info(f'Initialize the builder...')

        sim_builder = Builder(
            project_path = project_path, 
            resource_maps = RESOURCES_MAPS
        )

        logger.info(f'Start all the simulation components...')

        pod_manager, system = await sim_builder.init()

        # ===== Step3 : start the simulation =====
        start_tick = 0
        max_tick = sim_builder.config.simulation.max_ticks
        running_ticks = max_tick - start_tick
        for i in range(running_ticks):
            tick_start_time = time.time()
            phase_timestamps = {"start": tick_start_time}

            current_tick = await system.run('timer', 'get_tick')

            # ===== Agent Step =====
            await pod_manager.step_agent.remote()
            phase_timestamps[f'Agent_Step_{i}'] = time.time()

            # ===== Message Dispatch =====
            await system.run('messager', 'dispatch_messages')
            phase_timestamps[f'Message_Dispatch_{i}'] = time.time()

            # ===== Status Update =====
            await pod_manager.update_agents_status.remote()
            phase_timestamps[f'Status_Update_{i}'] = time.time()
            tick_end_time = time.time()
            
            tick_duration = tick_end_time - tick_start_time
            total_duration += tick_duration
            
            await system.run('timer', 'add_tick', duration_seconds = tick_duration)
            logger.info(f"--- Tick {current_tick} finished in {tick_duration:.4f} seconds ---")


        if running_ticks > 0:
            logger.info(f'Ran {running_ticks} ticks in total, average tick duration: {total_duration / running_ticks:.4f} seconds.')
            
        logger.info(f'Simulation finished.')
        

        
        
    except Exception as e:
        logger.error(f'Failed to run the simulation: {e}.')        
        
    # ===== Step4 : Stop the simulation =====
    
    finally:
        if "MAS_EVENT_LOG_DIR" in os.environ:
            del os.environ["MAS_EVENT_LOG_DIR"]
        if pod_manager:
            result = await pod_manager.close.remote()
            logger.info(f"Pod Manager close result is {result}")
        if system:
            result = await system.close()
            logger.info(f"System close result is {result}")
        if ray.is_initialized():
            ray.shutdown()
            
if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Simulation interrupted by user. Exiting.")
    finally:
        logger.info("Simulation ended.")