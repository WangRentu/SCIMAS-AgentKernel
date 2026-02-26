"""
API endpoints for plugin registry operations.
"""

from fastapi import APIRouter, HTTPException
from ..services.simulation_manager import simulation_manager
from ..services.registry_generator import registry_generator

router = APIRouter()


@router.get("/plugins")
async def get_available_plugins():
    """
    Scan workspace and return all available plugin information.

    Returns:
        dict: A dictionary containing plugin information organized by category.

    Raises:
        HTTPException: If failed to retrieve registry information.
    """
    try:
        info = await registry_generator.get_registry_info(simulation_manager.workspace_path)
        return info
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get registry info: {e}")


@router.post("/generate")
async def generate_registry_endpoint():
    """
    Manually trigger the regeneration of registry.py file.

    Returns:
        dict: A message indicating successful regeneration.

    Raises:
        HTTPException: If failed to generate registry.py.
    """
    try:
        await registry_generator.generate_registry_file(simulation_manager.workspace_path)
        return {"message": "registry.py has been successfully regenerated."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate registry.py: {e}")
