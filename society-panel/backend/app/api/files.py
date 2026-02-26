"""
API endpoints for file upload, listing, and deletion operations.
"""

import os
import shutil
import zipfile
import io
import re
from typing import List, Tuple

from fastapi import APIRouter, UploadFile, File, HTTPException

from ..services.simulation_manager import simulation_manager
from ..services.registry_generator import registry_generator

router = APIRouter()


def _rewrite_imports_in_file(file_path: str) -> bool:
    """
    Scan and rewrite specific import paths in a Python file.

    Args:
        file_path (str): The path to the Python file to process.

    Returns:
        bool: True if the file was modified, False otherwise.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        pattern = r'(from|import)\s+examples\.\w+\.'
        replacement = r'\1 '

        new_content, num_subs = re.subn(pattern, replacement, content)

        if num_subs > 0:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
            print(f"Rewrote {num_subs} import(s) in: {os.path.basename(file_path)}")
            return True
        return False
    except Exception as e:
        print(f"Warning: Could not process imports for {file_path}. Error: {e}")
        return False


async def _unpack_zip_and_get_py_files(file: UploadFile, target_base_path: str) -> Tuple[int, List[str]]:
    """
    Safely unpack a ZIP file to the specified base path.

    Args:
        file (UploadFile): The uploaded ZIP file.
        target_base_path (str): The target directory to extract files to.

    Returns:
        Tuple[int, List[str]]: A tuple containing the count of saved files and list of Python file paths.

    Raises:
        HTTPException: If file is not a valid ZIP archive.
    """
    if not file.filename.endswith('.zip'):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a .zip file.")

    saved_files_count = 0
    py_file_paths = []

    try:
        zip_content = await file.read()

        with zipfile.ZipFile(io.BytesIO(zip_content)) as zf:
            for member in zf.infolist():
                if member.is_dir() or member.filename.startswith('__MACOSX/'):
                    continue

                target_path = os.path.join(target_base_path, member.filename)
                normalized_path = os.path.normpath(target_path)

                if not normalized_path.startswith(os.path.normpath(target_base_path)):
                    print(f"Skipping potentially malicious file path: {member.filename}")
                    continue

                destination_dir = os.path.dirname(normalized_path)
                os.makedirs(destination_dir, exist_ok=True)

                with zf.open(member, 'r') as source, open(normalized_path, 'wb') as target:
                    shutil.copyfileobj(source, target)

                saved_files_count += 1
                if normalized_path.endswith('.py'):
                    py_file_paths.append(normalized_path)

                print(f"Saved: {normalized_path}")

    except zipfile.BadZipFile:
        raise HTTPException(status_code=400, detail="The uploaded file is not a valid ZIP archive.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred while processing the ZIP file: {e}")
    finally:
        await file.close()

    return saved_files_count, py_file_paths


@router.post("/upload/package")
async def upload_package_zip(file: UploadFile = File(...)):
    """
    Upload and extract a MAS-Package ZIP file to workspace root.

    Args:
        file (UploadFile): The uploaded ZIP file containing the package.

    Returns:
        dict: A message with counts of saved and rewritten files.

    Raises:
        HTTPException: If registry regeneration fails after successful extraction.
    """
    workspace_path = simulation_manager.workspace_path

    saved_files_count, py_files = await _unpack_zip_and_get_py_files(file, workspace_path)

    rewritten_count = 0
    if py_files:
        print(f"Scanning {len(py_files)} Python files for import rewriting...")
        for py_file in py_files:
            if _rewrite_imports_in_file(py_file):
                rewritten_count += 1

    try:
        await registry_generator.generate_registry_file(workspace_path)
        message = (
            f"Successfully unpacked {saved_files_count} package files. "
            f"Rewrote imports in {rewritten_count} files. Registry has been regenerated."
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Files saved and imports rewritten, but failed to regenerate registry: {e}")

    return {"message": message, "saved_files_count": saved_files_count, "rewritten_files_count": rewritten_count}


@router.post("/upload/data")
async def upload_data_zip(file: UploadFile = File(...)):
    """
    Upload and extract a data ZIP file to workspace/data directory.

    Args:
        file (UploadFile): The uploaded ZIP file containing data files.

    Returns:
        dict: A message with the count of saved files.
    """
    target_path = os.path.join(simulation_manager.workspace_path, "data")
    os.makedirs(target_path, exist_ok=True)
    saved_files_count, _ = await _unpack_zip_and_get_py_files(file, target_path)
    return {"message": f"Successfully unpacked {saved_files_count} data files.", "saved_files_count": saved_files_count}


def _get_directory_contents(path: str, exclude: List[str] = None) -> List[str]:
    """
    Get a list of directory contents with optional exclusions.

    Args:
        path (str): The directory path to list.
        exclude (List[str], optional): Items to exclude from the listing.

    Returns:
        List[str]: A sorted list of item names in the directory.
    """
    if not os.path.exists(path) or not os.path.isdir(path):
        return []

    exclude_set = set(exclude) if exclude else set()
    # Always exclude auto-generated files
    exclude_set.add('registry.py')
    items = []
    for item in os.listdir(path):
        if item not in exclude_set and not item.startswith('.') and item != '__pycache__':
            items.append(item)
    return sorted(items)


@router.get("/list/package", response_model=List[str])
async def list_package_contents():
    """
    List contents of workspace root directory excluding 'data' folder.

    Returns:
        List[str]: A list of file and folder names.
    """
    workspace_path = simulation_manager.workspace_path
    return _get_directory_contents(workspace_path, exclude=['data'])


@router.get("/list/data", response_model=List[str])
async def list_data_files():
    """
    List all files and folders in workspace/data directory.

    Returns:
        List[str]: A list of file and folder names.
    """
    data_path = os.path.join(simulation_manager.workspace_path, "data")
    return _get_directory_contents(data_path)


@router.delete("/delete/package/{item_name}")
async def delete_package_item(item_name: str):
    """
    Delete a specified file or folder from workspace root.

    Args:
        item_name (str): The name of the item to delete.

    Returns:
        dict: A message indicating successful deletion.

    Raises:
        HTTPException: If item name is invalid, not found, or deletion fails.
    """
    if ".." in item_name or item_name.startswith("/"):
        raise HTTPException(status_code=400, detail="Invalid item name.")

    workspace_path = simulation_manager.workspace_path
    item_path = os.path.join(workspace_path, item_name)

    if not os.path.exists(item_path):
        raise HTTPException(status_code=404, detail=f"Item '{item_name}' not found.")

    try:
        if os.path.isdir(item_path):
            shutil.rmtree(item_path)
        else:
            os.remove(item_path)

        await registry_generator.generate_registry_file(workspace_path)

        return {"message": f"Successfully deleted '{item_name}'."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete '{item_name}': {e}")


@router.delete("/delete/data/{item_name}")
async def delete_data_item(item_name: str):
    """
    Delete a specified file or folder from workspace/data directory.

    Args:
        item_name (str): The name of the item to delete.

    Returns:
        dict: A message indicating successful deletion.

    Raises:
        HTTPException: If item name is invalid, not found, or deletion fails.
    """
    if ".." in item_name or item_name.startswith("/"):
        raise HTTPException(status_code=400, detail="Invalid item name.")

    data_path = os.path.join(simulation_manager.workspace_path, "data")
    item_path = os.path.join(data_path, item_name)

    if not os.path.exists(item_path):
        raise HTTPException(status_code=404, detail=f"Item '{item_name}' not found.")

    try:
        if os.path.isdir(item_path):
            shutil.rmtree(item_path)
        else:
            os.remove(item_path)
        return {"message": f"Successfully deleted '{item_name}'."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete '{item_name}': {e}")


@router.delete("/clear/package")
async def clear_all_package():
    """
    Clear all contents from workspace root while preserving 'data' directory.

    Returns:
        dict: A message indicating successful clearing.

    Raises:
        HTTPException: If clearing operation fails.
    """
    workspace_path = simulation_manager.workspace_path

    try:
        for item in os.listdir(workspace_path):
            if item == 'data' or item.startswith('.'):
                continue
            item_path = os.path.join(workspace_path, item)
            if os.path.isdir(item_path):
                shutil.rmtree(item_path)
            else:
                os.remove(item_path)

        await registry_generator.generate_registry_file(workspace_path)

        return {"message": "Successfully cleared all package files."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear package files: {e}")


@router.delete("/clear/data")
async def clear_all_data():
    """
    Clear all contents from workspace/data directory.

    Returns:
        dict: A message indicating successful clearing.

    Raises:
        HTTPException: If clearing operation fails.
    """
    data_path = os.path.join(simulation_manager.workspace_path, "data")

    if not os.path.exists(data_path):
        return {"message": "Data directory is already empty."}

    try:
        for item in os.listdir(data_path):
            if item.startswith('.'):
                continue
            item_path = os.path.join(data_path, item)
            if os.path.isdir(item_path):
                shutil.rmtree(item_path)
            else:
                os.remove(item_path)
        return {"message": "Successfully cleared all data files."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear data files: {e}")
