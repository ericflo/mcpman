# /// script
# requires-python = ">=3.10"  # Using pathlib features and tempfile.TemporaryDirectory
# dependencies = [
#     "mcp>=1.6.0",
# ]
# ///

import argparse
import os
import tempfile
from pathlib import Path
import shutil  # Will be used for cleanup if not using tempfile, but good to have
import stat  # For metadata
import time  # For metadata
from typing import Dict, List, Optional, Any  # Added for type hints

from mcp.server.fastmcp import FastMCP

app = FastMCP("FileSystemOps")

# This will hold the Path object for the base directory (either temp or specified)
BASE_DIR: Path | None = None
# Flag to indicate if we created a temporary directory
IS_TEMP_DIR: bool = False
# Reference to the TemporaryDirectory object if created
TEMP_DIR_CONTEXT: tempfile.TemporaryDirectory | None = None


# --- Sandboxing Helper ---
def _resolve_sandboxed_path(relative_path_str: str) -> Path:
    """Resolves a relative path against BASE_DIR and ensures it stays within.

    Args:
        relative_path_str (str): The user-provided path string, relative to BASE_DIR.

    Returns:
        Path: A resolved Path object guaranteed to be inside BASE_DIR.

    Raises:
        RuntimeError: If BASE_DIR is not initialized.
        PermissionError: If the resolved path attempts to escape BASE_DIR.
        FileNotFoundError: If the resolved path does not exist (optional, depends on usage).
        ValueError: If the provided path is absolute.
    """
    if BASE_DIR is None:
        raise RuntimeError("BASE_DIR has not been initialized.")

    # Prevent absolute paths from being provided by the user
    if Path(relative_path_str).is_absolute():
        raise ValueError("Paths must be relative to the base directory.")

    # Join and resolve the path
    # Using os.path.join initially to handle potential empty strings better
    # before converting to Path
    unsafe_path = Path(os.path.join(BASE_DIR, relative_path_str))
    resolved_path = unsafe_path.resolve()

    # Security check: Ensure the resolved path is still within BASE_DIR
    # Path.is_relative_to() is the key check here (Python 3.9+)
    if BASE_DIR not in resolved_path.parents and resolved_path != BASE_DIR:
        # Alternative check: Check common prefix (less robust)
        # if os.path.commonprefix([str(resolved_path), str(BASE_DIR)]) != str(BASE_DIR):
        raise PermissionError(
            f"Path escape attempt: '{relative_path_str}' resolves outside the allowed directory '{BASE_DIR}'"
        )

    return resolved_path


# --- Tools ---


@app.tool()
def list_directory(path: str = ".") -> List[str]:
    """Lists the contents (files and directories) of a specified directory.

    Args:
        path (str, optional): The directory path relative to the base directory. Defaults to the base directory itself (".").

    Returns:
        List[str]: A list of names of files and subdirectories within the specified path.

    Raises:
        PermissionError: If the path tries to escape the sandbox.
        FileNotFoundError: If the specified path does not exist.
        NotADirectoryError: If the specified path is not a directory.
    """
    resolved_path = _resolve_sandboxed_path(path)
    if not resolved_path.exists():
        raise FileNotFoundError(f"Directory not found: {path}")
    if not resolved_path.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {path}")

    return [item.name for item in resolved_path.iterdir()]


@app.tool()
def file_exists(path: str) -> bool:
    """Checks if a file or directory exists at the specified path.

    Args:
        path (str): The path relative to the base directory.

    Returns:
        bool: True if a file or directory exists at the path, False otherwise.

    Raises:
        PermissionError: If the path tries to escape the sandbox.
    """
    try:
        resolved_path = _resolve_sandboxed_path(path)
        return resolved_path.exists()
    except FileNotFoundError:
        # If resolve raises FileNotFoundError because intermediate components don't exist,
        # treat it as the final path not existing.
        return False
    # Let PermissionError propagate


@app.tool()
def get_metadata(path: str) -> Dict[str, Any]:
    """Gets metadata for a file or directory.

    Args:
        path (str): The path relative to the base directory.

    Returns:
        Dict[str, Any]: A dictionary containing metadata:
            - 'name' (str): The name of the file/directory.
            - 'path' (str): The relative path provided.
            - 'absolute_path' (str): The resolved absolute path (within sandbox).
            - 'type' (str): 'file', 'directory', or 'other'.
            - 'size_bytes' (Optional[int]): Size in bytes (for files), None otherwise.
            - 'modified_time' (str): Last modification timestamp (ISO 8601 format UTC).
            - 'access_time' (str): Last access timestamp (ISO 8601 format UTC).
            - 'creation_time' (str): Creation timestamp (ISO 8601 format UTC, platform dependent).

    Raises:
        PermissionError: If the path tries to escape the sandbox.
        FileNotFoundError: If the specified path does not exist.
    """
    resolved_path = _resolve_sandboxed_path(path)
    if not resolved_path.exists():
        raise FileNotFoundError(f"Path not found: {path}")

    stat_result = resolved_path.stat()

    file_type = "other"
    if stat.S_ISDIR(stat_result.st_mode):
        file_type = "directory"
    elif stat.S_ISREG(stat_result.st_mode):
        file_type = "file"

    metadata = {
        "name": resolved_path.name,
        "path": path,  # Return the user-provided relative path
        "absolute_path": str(resolved_path),
        "type": file_type,
        "size_bytes": stat_result.st_size if file_type == "file" else None,
        "modified_time": time.strftime(
            "%Y-%m-%dT%H:%M:%SZ", time.gmtime(stat_result.st_mtime)
        ),
        "access_time": time.strftime(
            "%Y-%m-%dT%H:%M:%SZ", time.gmtime(stat_result.st_atime)
        ),
        # Creation time might not be available on all systems
        "creation_time": time.strftime(
            "%Y-%m-%dT%H:%M:%SZ",
            time.gmtime(
                getattr(stat_result, "st_birthtime", stat_result.st_mtime)
            ),  # Fallback to mtime if birthtime not available
        ),
    }
    return metadata


@app.tool()
def read_file(path: str, encoding: Optional[str] = "utf-8") -> str:
    """Reads the content of a text file.

    Args:
        path (str): The path to the file relative to the base directory.
        encoding (Optional[str], optional): The encoding to use for reading the file (e.g., 'utf-8', 'latin-1'). Defaults to 'utf-8'.

    Returns:
        str: The content of the file as a string.

    Raises:
        PermissionError: If the path tries to escape the sandbox or read permissions are denied.
        FileNotFoundError: If the specified path does not exist.
        IsADirectoryError: If the specified path is a directory.
        ValueError: If the file exceeds the maximum allowed size or if the encoding is invalid.
        UnicodeDecodeError: If the file cannot be decoded with the specified encoding.
        RuntimeError: For other unexpected file reading errors.
    """
    global args  # Access the parsed command-line arguments
    resolved_path = _resolve_sandboxed_path(path)

    if not resolved_path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    if not resolved_path.is_file():
        raise IsADirectoryError(f"Path is a directory, not a file: {path}")

    # Check file size limit
    file_size = resolved_path.stat().st_size
    if file_size > args.max_file_size:
        raise ValueError(
            f"File size ({file_size} bytes) exceeds the maximum allowed limit ({args.max_file_size} bytes)."
        )

    try:
        # Specify encoding, handle errors during read
        content = resolved_path.read_text(encoding=encoding, errors="strict")
        return content
    except PermissionError as e:
        # Re-raise permission errors explicitly for clarity
        raise PermissionError(f"Permission denied when reading file '{path}': {e}")
    except UnicodeDecodeError as e:
        raise UnicodeDecodeError(
            encoding,
            e.object,
            e.start,
            e.end,
            f"Could not decode file '{path}' with encoding '{encoding}'. Try a different encoding or read as bytes (feature not implemented yet). {e.reason}",
        )
    except LookupError:
        raise ValueError(f"Invalid encoding specified: {encoding}")
    except Exception as e:
        # Catch other potential file reading errors
        raise RuntimeError(
            f"An unexpected error occurred while reading file '{path}': {e}"
        )


@app.tool()
def create_directory(
    path: str, exist_ok: bool = True, create_parents: bool = True
) -> Dict[str, str]:
    """Creates a new directory.

    Args:
        path (str): The path for the new directory, relative to the base directory.
        exist_ok (bool, optional): If True, do not raise an error if the directory already exists. Defaults to True.
        create_parents (bool, optional): If True, create any necessary parent directories. Defaults to True.

    Returns:
        Dict[str, str]: A dictionary confirming creation: {"message": "Directory created", "path": path}.

    Raises:
        PermissionError: If the path tries to escape the sandbox or write permissions are denied.
        FileExistsError: If the path already exists and `exist_ok` is False.
        FileNotFoundError: If a parent directory does not exist and `create_parents` is False.
        RuntimeError: For other unexpected errors during directory creation.
    """
    # Resolve the *intended* path first to check sandbox constraints
    # We don't require it to exist yet, so we can't use resolve(strict=True)
    intended_path = _resolve_sandboxed_path(path)

    try:
        intended_path.mkdir(parents=create_parents, exist_ok=exist_ok)
        return {
            "message": "Directory created successfully",
            "path": path,
            "absolute_path": str(intended_path),
        }
    except FileExistsError:
        # Should only happen if exist_ok=False
        raise FileExistsError(f"Directory already exists at path: {path}")
    except FileNotFoundError:
        # Should only happen if create_parents=False and a parent is missing
        raise FileNotFoundError(f"Parent directory does not exist for path: {path}")
    except PermissionError as e:
        raise PermissionError(
            f"Permission denied when creating directory '{path}': {e}"
        )
    except Exception as e:
        raise RuntimeError(
            f"An unexpected error occurred creating directory '{path}': {e}"
        )


@app.tool()
def write_file(
    path: str, content: str, encoding: Optional[str] = "utf-8", overwrite: bool = False
) -> Dict[str, Any]:  # Return type includes int
    """Writes content to a text file.

    Args:
        path (str): The path to the file relative to the base directory.
        content (str): The string content to write to the file.
        encoding (Optional[str], optional): The encoding to use for writing the file. Defaults to 'utf-8'.
        overwrite (bool, optional): If False, raise FileExistsError if the file already exists. If True, overwrite. Defaults to False.

    Returns:
        Dict[str, Any]: Confirmation: {"message": "File written", "path": path, "bytes_written": int}.

    Raises:
        PermissionError: If the path escapes the sandbox or write permissions are denied.
        FileExistsError: If the file exists and `overwrite` is False.
        IsADirectoryError: If the path points to an existing directory.
        ValueError: If the content size exceeds the maximum allowed limit or the encoding is invalid.
        UnicodeEncodeError: If the content cannot be encoded with the specified encoding.
        RuntimeError: For other unexpected file writing errors.
    """
    global args  # Access the parsed command-line arguments
    intended_path = _resolve_sandboxed_path(path)

    # Check if the path exists and handle overwrite logic
    if intended_path.exists():
        if not overwrite:
            raise FileExistsError(
                f"File already exists at path: {path}. Use overwrite=True to replace it."
            )
        if intended_path.is_dir():
            raise IsADirectoryError(f"Path exists but is a directory: {path}")
        # If overwriting a file, proceed

    # Check content size limit before encoding/writing
    # Estimate bytes based on encoding (UTF-8 can use 1-4 bytes per char)
    # A simple len(content) check is a decent proxy, though not exact for size limit.
    # For a precise check, we encode first, but that might be slow for large content.
    # Let's check encoded size.
    try:
        encoded_content = content.encode(encoding)
    except LookupError:
        raise ValueError(f"Invalid encoding specified: {encoding}")
    except UnicodeEncodeError as e:
        raise UnicodeEncodeError(
            encoding,
            e.object,
            e.start,
            e.end,
            f"Could not encode content for file '{path}' with encoding '{encoding}'. {e.reason}",
        )

    content_size = len(encoded_content)
    if content_size > args.max_file_size:
        raise ValueError(
            f"Content size ({content_size} bytes) exceeds the maximum allowed file write limit ({args.max_file_size} bytes)."
        )

    try:
        # Ensure parent directories exist if needed (mkdir logic handles sandbox check internally)
        # We don't need to call _resolve_sandboxed_path again here.
        intended_path.parent.mkdir(parents=True, exist_ok=True)

        # Write the encoded content as bytes
        bytes_written = intended_path.write_bytes(encoded_content)
        return {
            "message": "File written successfully",
            "path": path,
            "absolute_path": str(intended_path),
            "bytes_written": bytes_written,
        }
    except PermissionError as e:
        raise PermissionError(f"Permission denied when writing to file '{path}': {e}")
    except Exception as e:
        raise RuntimeError(
            f"An unexpected error occurred writing to file '{path}': {e}"
        )


@app.tool()
def delete_file(path: str) -> Dict[str, str]:
    """Deletes a file.

    Args:
        path (str): The path to the file relative to the base directory.

    Returns:
        Dict[str, str]: Confirmation: {"message": "File deleted", "path": path}.

    Raises:
        PermissionError: If the path tries to escape the sandbox or delete permissions are denied.
        FileNotFoundError: If the specified path does not exist.
        IsADirectoryError: If the path points to a directory.
        RuntimeError: For other unexpected errors during deletion.
    """
    resolved_path = _resolve_sandboxed_path(path)

    if not resolved_path.exists():
        raise FileNotFoundError(f"File not found at path: {path}")
    if not resolved_path.is_file():
        raise IsADirectoryError(f"Path points to a directory, not a file: {path}")

    try:
        resolved_path.unlink()  # missing_ok=False is default
        return {"message": "File deleted successfully", "path": path}
    except PermissionError as e:
        raise PermissionError(f"Permission denied when deleting file '{path}': {e}")
    except Exception as e:
        # Catch other potential errors during unlink
        raise RuntimeError(f"An unexpected error occurred deleting file '{path}': {e}")


@app.tool()
def delete_directory(path: str, recursive: bool = False) -> Dict[str, str]:
    """Deletes a directory.

    Args:
        path (str): The path to the directory relative to the base directory.
        recursive (bool, optional): If False, only delete empty directories. Raise OSError if not empty.
                              If True, recursively delete the directory and all its contents (USE WITH CAUTION).
                              Defaults to False.

    Returns:
        Dict[str, str]: Confirmation: {"message": "Directory deleted", "path": path}.

    Raises:
        PermissionError: If the path tries to escape the sandbox or delete permissions are denied.
        FileNotFoundError: If the specified path does not exist.
        NotADirectoryError: If the path points to a file.
        OSError: If the directory is not empty and `recursive` is False, or for other OS errors during deletion.
        RuntimeError: For other unexpected errors during deletion.
    """
    resolved_path = _resolve_sandboxed_path(path)

    if not resolved_path.exists():
        raise FileNotFoundError(f"Directory not found at path: {path}")
    if not resolved_path.is_dir():
        raise NotADirectoryError(f"Path points to a file, not a directory: {path}")

    # Crucial check: Ensure we don't accidentally delete the entire base directory
    if resolved_path == BASE_DIR:
        raise PermissionError("Deleting the root base directory is not allowed.")

    try:
        if recursive:
            # shutil.rmtree handles non-empty directories
            # The sandbox check in _resolve_sandboxed_path already ensures
            # we are operating within BASE_DIR.
            shutil.rmtree(resolved_path)
            return {
                "message": "Directory and its contents deleted successfully",
                "path": path,
            }
        else:
            # Path.rmdir() only works on empty directories
            resolved_path.rmdir()
            return {"message": "Empty directory deleted successfully", "path": path}
    except OSError as e:
        # Raised by rmdir if directory is not empty, or other OS-level issues
        if e.errno == 39:  # Directory not empty
            raise OSError(
                f"Directory '{path}' is not empty. Use recursive=True to delete anyway."
            )
        else:
            raise OSError(f"Could not delete directory '{path}': {e}")
    except PermissionError as e:
        raise PermissionError(
            f"Permission denied when deleting directory '{path}': {e}"
        )
    except Exception as e:
        raise RuntimeError(
            f"An unexpected error occurred deleting directory '{path}': {e}"
        )


@app.tool()
def move_item(source_path: str, destination_path: str) -> Dict[str, str]:
    """Moves or renames a file or directory.

    Args:
        source_path (str): The current path of the item relative to the base directory.
        destination_path (str): The new path for the item relative to the base directory.

    Returns:
        Dict[str, str]: Confirmation: {"message": "Item moved/renamed", "source": source_path, "destination": destination_path}.

    Raises:
        PermissionError: If either path tries to escape the sandbox or permissions are denied.
        FileNotFoundError: If the source path does not exist.
        FileExistsError: If the destination path already exists.
        OSError: For other OS-level errors during the move (e.g., moving directory across filesystems without shutil).
        ValueError: If source and destination paths resolve to the same location.
        RuntimeError: For other unexpected errors during move.
    """
    # Resolve and validate both source and destination paths within the sandbox
    resolved_source = _resolve_sandboxed_path(source_path)
    resolved_destination = _resolve_sandboxed_path(destination_path)

    if not resolved_source.exists():
        raise FileNotFoundError(f"Source path does not exist: {source_path}")

    if resolved_destination.exists():
        raise FileExistsError(f"Destination path already exists: {destination_path}")

    # Ensure we are not trying to move the base directory itself
    if resolved_source == BASE_DIR:
        raise PermissionError("Moving the root base directory is not allowed.")

    # Ensure the parent directory for the destination exists
    try:
        resolved_destination.parent.mkdir(parents=True, exist_ok=True)
    except PermissionError as e:
        raise PermissionError(
            f"Permission denied creating parent directory for destination '{destination_path}': {e}"
        )
    except Exception as e:
        raise RuntimeError(
            f"Error ensuring destination parent directory exists for '{destination_path}': {e}"
        )

    try:
        # Perform the rename/move operation
        resolved_source.rename(resolved_destination)
        return {
            "message": "Item moved/renamed successfully",
            "source": source_path,
            "destination": destination_path,
            "absolute_destination": str(resolved_destination),
        }
    except PermissionError as e:
        # This could be source read or destination write permissions
        raise PermissionError(
            f"Permission denied during move operation from '{source_path}' to '{destination_path}': {e}"
        )
    except OSError as e:
        # Catch various OS errors like moving across filesystems (less likely here) or other issues
        raise OSError(f"Could not move '{source_path}' to '{destination_path}': {e}")
    except Exception as e:
        raise RuntimeError(
            f"An unexpected error occurred moving '{source_path}' to '{destination_path}': {e}"
        )


@app.tool()
def replace_in_file(
    path: str,
    search_string: str,
    replace_string: str,
    encoding: Optional[str] = "utf-8",
) -> Dict[str, Any]:  # Return type includes int
    """Replaces exactly one occurrence of a search string within a text file.

    This tool will raise an error if the search string is not found or if it
    is found more than once, to prevent accidental mass replacements.

    Args:
        path (str): The path to the file relative to the base directory.
        search_string (str): The exact string to search for.
        replace_string (str): The string to replace the search string with.
        encoding (Optional[str], optional): The encoding to use for reading and writing the file. Defaults to 'utf-8'.

    Returns:
        Dict[str, Any]: Confirmation: {"message": "Replacement successful", "path": path, "bytes_written": int}.

    Raises:
        PermissionError: If the path escapes the sandbox or permissions are denied.
        FileNotFoundError: If the path does not exist.
        IsADirectoryError: If the path is a directory.
        ValueError: If the file exceeds size limits, encoding is invalid,
                    search_string is not found, or search_string is found more than once.
        UnicodeDecodeError: If the file cannot be read with the specified encoding.
        UnicodeEncodeError: If the new content cannot be written with the specified encoding.
        RuntimeError: For other unexpected errors during replacement.
    """
    global args  # Access command-line args for max_file_size
    resolved_path = _resolve_sandboxed_path(path)

    if not resolved_path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    if not resolved_path.is_file():
        raise IsADirectoryError(f"Path is a directory, not a file: {path}")

    # --- Read phase ---
    try:
        file_size = resolved_path.stat().st_size
        if file_size > args.max_file_size:
            raise ValueError(
                f"File size ({file_size} bytes) exceeds the maximum allowed read limit ({args.max_file_size} bytes). Cannot perform replacement."
            )
        original_content = resolved_path.read_text(encoding=encoding, errors="strict")
    except PermissionError as e:
        raise PermissionError(
            f"Permission denied reading file '{path}' for replacement: {e}"
        )
    except UnicodeDecodeError as e:
        raise UnicodeDecodeError(
            encoding,
            e.object,
            e.start,
            e.end,
            f"Could not decode file '{path}' with encoding '{encoding}' for replacement. {e.reason}",
        )
    except LookupError:
        raise ValueError(f"Invalid encoding specified for reading: {encoding}")
    except Exception as e:
        raise RuntimeError(f"Error reading file '{path}' before replacement: {e}")

    # --- Count occurrences ---
    occurrence_count = original_content.count(search_string)
    if occurrence_count == 0:
        raise ValueError(
            f"Search string not found in file '{path}'. No replacement performed."
        )
    elif occurrence_count > 1:
        raise ValueError(
            f"Found {occurrence_count} occurrences of the search string in '{path}'. "
            f"Replacement aborted to prevent unintended changes. Only single replacements are supported by this tool."
        )

    # --- Replacement and Write phase ---
    new_content = original_content.replace(search_string, replace_string, 1)
    try:
        encoded_new_content = new_content.encode(encoding, errors="strict")
    except LookupError:
        raise ValueError(f"Invalid encoding specified for writing: {encoding}")
    except UnicodeEncodeError as e:
        raise UnicodeEncodeError(
            encoding,
            e.object,
            e.start,
            e.end,
            f"Could not encode the modified content for file '{path}' with encoding '{encoding}'. {e.reason}",
        )

    new_content_size = len(encoded_new_content)
    if new_content_size > args.max_file_size:
        raise ValueError(
            f"Modified content size ({new_content_size} bytes) exceeds the maximum allowed file write limit ({args.max_file_size} bytes). Replacement aborted."
        )

    try:
        bytes_written = resolved_path.write_bytes(encoded_new_content)
        return {
            "message": "Replacement successful (1 occurrence replaced)",
            "path": path,
            "absolute_path": str(resolved_path),
            "bytes_written": bytes_written,
        }
    except PermissionError as e:
        raise PermissionError(
            f"Permission denied writing replacement to file '{path}': {e}"
        )
    except Exception as e:
        raise RuntimeError(
            f"An unexpected error occurred writing replacement to file '{path}': {e}"
        )


# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the FileSystemOps MCP server.")
    parser.add_argument(
        "--transport",
        type=str,
        choices=["sse", "stdio"],
        default="stdio",
        help="Transport method to use (sse or stdio).",
    )
    parser.add_argument(
        "--directory",
        type=str,
        default=None,
        help="Specify a directory to operate within. If not provided, a temporary directory will be created and automatically cleaned up.",
    )
    parser.add_argument(
        "--max-file-size",
        type=int,
        default=1024 * 1024,  # Default to 1MB
        help="Maximum file size in bytes allowed for reading. Defaults to 1MB.",
    )

    args = parser.parse_args()

    try:
        if args.directory:
            BASE_DIR = Path(args.directory).resolve(
                strict=True
            )  # Ensure it exists if specified
            if not BASE_DIR.is_dir():
                raise ValueError(f"Provided path is not a directory: {args.directory}")
            IS_TEMP_DIR = False
            print(f"Operating within specified directory: {BASE_DIR}")
        else:
            # Create a temporary directory
            TEMP_DIR_CONTEXT = tempfile.TemporaryDirectory()
            BASE_DIR = Path(TEMP_DIR_CONTEXT.name).resolve()
            IS_TEMP_DIR = True
            print(f"Created temporary directory: {BASE_DIR}")
            # TEMP_DIR_CONTEXT handles cleanup automatically when the script exits

        if BASE_DIR is None:
            raise RuntimeError(
                "Base directory was not initialized."
            )  # Should not happen

        # --- Register Tools Here (after BASE_DIR is set) ---
        # Example: @app.tool() def list_directory...

        print(
            f"Starting FileSystemOps server (Transport: {args.transport}, Base Dir: {BASE_DIR})"
        )
        # Keep the server running
        app.run(transport=args.transport)

    except Exception as e:
        print(f"Error during startup or execution: {e}")
    finally:
        # Cleanup is handled by TEMP_DIR_CONTEXT exiting scope if it was created
        if IS_TEMP_DIR and TEMP_DIR_CONTEXT:
            print(f"Temporary directory {TEMP_DIR_CONTEXT.name} will be cleaned up.")
        elif not IS_TEMP_DIR and args.directory:
            print(
                f"Specified directory {args.directory} will not be modified or deleted."
            )
        # Ensure cleanup happens even if app.run crashes?
        # tempfile.TemporaryDirectory cleanup is generally robust.
