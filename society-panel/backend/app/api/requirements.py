"""
API endpoints for managing Python requirements.txt file.
"""

import os
from fastapi import APIRouter, HTTPException, Body
from ..services.simulation_manager import simulation_manager

router = APIRouter()
REQUIREMENTS_PATH = os.path.abspath(os.path.join(simulation_manager.workspace_path, "..", "requirements.txt"))


@router.get("/")
async def get_requirements():
    """
    Read the contents of requirements.txt file.

    Returns:
        dict: A dictionary containing the file content as a string.

    Raises:
        HTTPException: If an error occurs while reading the file.
    """
    if not os.path.exists(REQUIREMENTS_PATH):
        return {"content": ""}
    try:
        with open(REQUIREMENTS_PATH, "r", encoding="utf-8") as f:
            content = f.read()
        return {"content": content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading requirements.txt: {e}")


@router.post("/")
async def update_requirements(content: str = Body(..., embed=True)):
    """
    Update the requirements.txt file with new content.

    Args:
        content (str): The new content to write to requirements.txt.

    Returns:
        dict: A message indicating successful update.

    Raises:
        HTTPException: If an error occurs while writing the file.
    """
    try:
        with open(REQUIREMENTS_PATH, "w", encoding="utf-8") as f:
            f.write(content)

        pip_installed_marker = os.path.join(simulation_manager.workspace_path, "..", ".venv", "pip_installed_reqs")
        if os.path.exists(pip_installed_marker):
            os.remove(pip_installed_marker)

        return {"message": "requirements.txt updated successfully. Dependencies will be re-installed on next start."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error writing requirements.txt: {e}")
