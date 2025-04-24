# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "mcp>=1.6.0",
#     "mcpman>=0.3.3",
# ]
# ///

import argparse
import json
import os
import re
import subprocess
import tempfile
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Literal

from mcp.server.fastmcp import FastMCP

# AgentOps MCP Server
#
# This MCP server provides tools for orchestrating and analyzing LLM agents:
#
# - run_agent_on_task: Executes an agent with a specific MCP configuration on a given task.
# - inspect_agent_task_log: Retrieves and analyzes the logs from agent runs with varying levels of detail.
#
# These tools allow higher-level LLMs to delegate tasks to specialized agents and analyze
# their performance, enabling more complex workflows and meta-agent capabilities.

app = FastMCP("AgentOps")

# Constants for the mcpman command
MCPMAN_CMD = "mcpman"

# Default logs directory
LOGS_DIR = Path("logs")

# Timeout for subprocess execution (in seconds)
SUBPROCESS_TIMEOUT = 600  # 10 minutes

# Maximum number of tool calls allowed for delegated agents
MAX_AGENT_TURNS = 64  # Limiting tool calls to prevent runaway processes

# Constants for log detail levels
LOG_DETAIL_BASIC = "basic"  # Just task and result
LOG_DETAIL_STANDARD = "standard"  # Includes task, tools used, and result
LOG_DETAIL_VERBOSE = "verbose"  # Full log including all interaction details
LOG_DETAIL_RAW = "raw"  # Complete raw log output

# Pattern to extract run_id from mcpman output
RUN_ID_PATTERN = re.compile(r"run_id:\s*([a-f0-9-]+)", re.IGNORECASE)


def _validate_mcp_config(config: Dict[str, Any]) -> List[str]:
    """
    Validates an MCP configuration and returns a list of validation errors.
    Returns an empty list if the configuration is valid.

    Args:
        config (Dict[str, Any]): The MCP configuration to validate

    Returns:
        List[str]: List of validation error messages, empty if valid
    """
    errors = []

    # Check for mcpServers key
    if "mcpServers" not in config:
        errors.append(
            "Missing required 'mcpServers' key in MCP configuration. This is the root object containing server definitions."
        )
        return errors  # Can't validate further without this key

    if not isinstance(config["mcpServers"], dict):
        errors.append(
            "'mcpServers' must be an object containing server definitions with names as keys."
        )
        return errors

    # Validate each server definition
    for server_name, server_def in config["mcpServers"].items():
        # Server name validation
        if not isinstance(server_name, str) or not server_name.strip():
            errors.append(f"Server name must be a non-empty string, got: {server_name}")

        # Server definition validation
        if not isinstance(server_def, dict):
            errors.append(
                f"Server definition for '{server_name}' must be an object with command, args, and optional env."
            )
            continue

        # Required fields in server definition
        for field in ["command", "args"]:
            if field not in server_def:
                errors.append(
                    f"Missing required field '{field}' in server definition for '{server_name}'. MCP requires both command and args fields."
                )

        # Command validation
        if "command" in server_def and not isinstance(server_def["command"], str):
            errors.append(
                f"'command' in server '{server_name}' must be a string. This specifies the executable to run your MCP server."
            )

        # Args validation
        if "args" in server_def and not isinstance(server_def["args"], list):
            errors.append(
                f"'args' in server '{server_name}' must be a list of strings. This specifies arguments to pass to your MCP server."
            )
        elif "args" in server_def:
            for i, arg in enumerate(server_def["args"]):
                if not isinstance(arg, str):
                    errors.append(
                        f"Argument {i} in server '{server_name}' must be a string, got: {type(arg).__name__}"
                    )

        # Env validation (optional)
        if "env" in server_def and not isinstance(server_def["env"], dict):
            errors.append(
                f"'env' in server '{server_name}' must be an object with string key-value pairs for environment variables."
            )
        elif "env" in server_def:
            for env_key, env_val in server_def["env"].items():
                if not isinstance(env_key, str):
                    errors.append(
                        f"Environment variable key in server '{server_name}' must be a string."
                    )
                if not isinstance(env_val, str):
                    errors.append(
                        f"Environment variable value for '{env_key}' in server '{server_name}' must be a string."
                    )

    return errors


def _find_log_file(run_id: Optional[str] = None) -> Optional[str]:
    """
    Finds the appropriate log file based on run_id or returns the most recent log file.

    Args:
        run_id (Optional[str], optional): The run ID to search for.

    Returns:
        Optional[str]: Path to the log file if found, None otherwise.
    """
    # Ensure logs directory exists
    if not LOGS_DIR.exists() or not LOGS_DIR.is_dir():
        return None

    # List all log files
    log_files = list(LOGS_DIR.glob("mcpman_*.jsonl"))
    if not log_files:
        return None

    # If no run_id provided, return the most recent log file
    if run_id is None:
        return str(sorted(log_files, key=lambda x: x.stat().st_mtime, reverse=True)[0])

    # Search for the log file containing the run_id
    for log_file in sorted(log_files, key=lambda x: x.stat().st_mtime, reverse=True):
        try:
            with open(log_file, "r") as f:
                for line in f:
                    try:
                        entry = json.loads(line.strip())
                        if "run_id" in entry and entry["run_id"] == run_id:
                            return str(log_file)
                    except json.JSONDecodeError:
                        continue
        except Exception:
            continue

    # If run_id not found, return None
    return None


@app.tool()
def run_agent_on_task(
    task: str,
    mcp_config: Dict[str, Any],
    implementation: str = "openai_responses",
    model: str = "gpt-4.1-mini",
    user_prompt: Optional[str] = None,
    system_message: Optional[str] = None,
    validation_prompt: Optional[str] = None,
    temperature: Optional[float] = 0.7,
    base_url: Optional[str] = None,
) -> Dict[str, Any]:
    """Runs an agent on a specified task using the provided MCP server configuration.

    This tool allows running an LLM agent using the mcpman CLI with a specific MCP server configuration.
    The agent will execute the task and return the results when complete. It enables a higher-level
    agent to delegate tasks to specialized agents, creating a hierarchical agent architecture.

    Args:
        task (str): The task description or prompt for the agent to perform. This is the primary
                   instruction that guides the agent's behavior.

        mcp_config (Dict[str, Any]): The MCP server configuration in JSON format. This defines the
                                    servers and tools that will be available to the agent.

                                    The configuration must follow the MCP format with a 'mcpServers' key
                                    that maps to server definitions. Each server must have at least
                                    'command' and 'args' fields, and may optionally have an 'env' field.

                                    Example 1 - Simple calculator server:
                                    {
                                        "mcpServers": {
                                            "calculator": {
                                                "command": "uv",
                                                "args": ["run", "https://raw.githubusercontent.com/ericflo/mcpman/refs/heads/main/mcp_servers/calculator.py"],
                                            }
                                        }
                                    }

                                    Example 2 - Multiple servers configuration:
                                    {
                                        "mcpServers": {
                                            "calculator": {
                                                "command": "uv",
                                                "args": ["run", "https://raw.githubusercontent.com/ericflo/mcpman/refs/heads/main/mcp_servers/calculator.py"]
                                            },
                                            "datetime": {
                                                "command": "uv",
                                                "args": ["run", "https://raw.githubusercontent.com/ericflo/mcpman/refs/heads/main/mcp_servers/datetime_utils.py"]
                                            },
                                            "filesystem": {
                                                "command": "uv",
                                                "args": ["run", "https://raw.githubusercontent.com/ericflo/mcpman/refs/heads/main/mcp_servers/filesystem_ops.py"]
                                            }
                                        }
                                    }

                                    Example 3 - Using environment variables:
                                    {
                                        "mcpServers": {
                                            "web_search": {
                                                "command": "npx",
                                                "args": ["mcp-web-search-bing"],
                                                "env": {
                                                    "BING_API_KEY": "your_api_key_here"
                                                }
                                            }
                                        }
                                    }

                                    Example 4 - Local Python module:
                                    {
                                        "mcpServers": {
                                            "custom_tools": {
                                                "command": "python",
                                                "args": ["-m", "path.to.your.mcp_module"]
                                            }
                                        }
                                    }

                                    Important configuration notes:
                                    - Server names should be descriptive of their functionality
                                    - Each server requires 'command' (executable) and 'args' (list of string arguments)
                                    - 'env' is optional and contains environment variables as key-value string pairs
                                    - Common API keys (OPENAI_API_KEY, ANTHROPIC_API_KEY, etc.) are automatically propagated
                                    - Ensure server paths are valid and accessible
                                    - The configuration structure is validated before execution

        implementation (str, optional): The LLM implementation to use. Defaults to "openai_responses",
                                       which is recommended for most use cases due to its robust tool
                                       handling capabilities. Other options include "anthropic",
                                       "gemini", "ollama", etc.

        model (str, optional): The specific model to use. Defaults to "gpt-4.1-mini",
                             which provides a good balance of capability and cost. Other options
                             include "gpt-4-turbo", "claude-3-opus", "gemini-pro", etc.

        user_prompt (Optional[str], optional): An additional user prompt that supplements the task.
                                             This can be used to provide specific instructions or
                                             constraints that guide the agent's approach to the task.
                                             Defaults to None.

        system_message (Optional[str], optional): Optional system message to guide the agent's behavior.
                                               Useful for setting the overall behavior and capabilities
                                               of the agent. If not provided or empty, no system message
                                               will be passed to the CLI. Defaults to None.

        validation_prompt (Optional[str], optional): Optional validation prompt for the agent to verify
                                                  its responses. This can enforce output formats or
                                                  quality checks. If not provided or empty, no validation
                                                  prompt will be passed to the CLI. Defaults to None.

        temperature (Optional[float], optional): Sampling temperature for the LLM, controlling randomness.
                                               Higher values (e.g., 0.8) produce more creative outputs,
                                               while lower values (e.g., 0.2) produce more deterministic
                                               results. Defaults to 0.7.

        base_url (Optional[str], optional): Custom endpoint URL for the LLM API. Useful for
                                          self-hosted models or alternative endpoints. Defaults to None.

    Returns:
        Dict[str, Any]: A dictionary containing:
            - "task_id" (str): Unique identifier for the task.
            - "status" (str): The status of the task ("success" or "error").
            - "result" (Any): The result or output from the agent.
            - "error" (Optional[str]): Error message if the task failed.
            - "run_id" (Optional[str]): The run ID for the mcpman execution.
            - "log_path" (Optional[str]): Path to the log file if available.

    Raises:
        ValueError: If the MCP configuration is invalid or the command fails to execute.

    Example usage:
        # Basic calculator task
        result = run_agent_on_task(
            task="Calculate the square root of 144 and then add 10 to the result.",
            mcp_config={
                "mcpServers": {
                    "calculator": {
                        "command": "uv",
                        "args": ["run", "https://raw.githubusercontent.com/ericflo/mcpman/refs/heads/main/mcp_servers/calculator.py"]
                    }
                }
            }
        )

        # Multi-tool task with system message and validation
        result = run_agent_on_task(
            task="Find today's date, create a text file with the date, and calculate how many days until the end of the year.",
            mcp_config={
                "mcpServers": {
                    "datetime": {
                        "command": "uv",
                        "args": ["run", "https://raw.githubusercontent.com/ericflo/mcpman/refs/heads/main/mcp_servers/datetime_utils.py"]
                    },
                    "filesystem": {
                        "command": "uv",
                        "args": ["run", "https://raw.githubusercontent.com/ericflo/mcpman/refs/heads/main/mcp_servers/filesystem_ops.py"]
                    },
                    "calculator": {
                        "command": "uv",
                        "args": ["run", "https://raw.githubusercontent.com/ericflo/mcpman/refs/heads/main/mcp_servers/calculator.py"]
                    }
                }
            },
            system_message="You are a helpful assistant that specializes in time calculations and file operations.",
            validation_prompt="Verify that you have correctly calculated the days remaining and saved the information to a file."
        )
    """
    # Validate the MCP configuration
    validation_errors = _validate_mcp_config(mcp_config)
    if validation_errors:
        error_msg = "Invalid MCP configuration:\n" + "\n".join(
            f"- {err}" for err in validation_errors
        )
        return {
            "task_id": str(uuid.uuid4()),
            "status": "error",
            "result": None,
            "error": error_msg,
            "suggestions": [
                "Ensure your MCP configuration contains a 'mcpServers' object at the root",
                "Each server in 'mcpServers' must have 'command' and 'args' fields",
                "All commands and arguments must be strings",
                "Environment variables (if provided) must be key-value string pairs",
            ],
        }

    # Create a temporary file for the MCP configuration with environment variables propagated to servers
    try:
        # List of important API keys and environment variables that should be propagated
        important_env_vars = [
            "OPENAI_API_KEY",
            "ANTHROPIC_API_KEY",
            "GEMINI_API_KEY",
            "GOOGLE_API_KEY",
            "MISTRAL_API_KEY",
            "GROQ_API_KEY",
            "OPENROUTER_API_KEY",
            "OLLAMA_BASE_URL",
            "LMSTUDIO_API_KEY",
            "LMSTUDIO_BASE_URL",
            "MCPMAN_STRICT_TOOLS",
        ]

        # Make a deep copy of the config to avoid modifying the input parameter
        config_to_write = json.loads(json.dumps(mcp_config))

        # Propagate environment variables to each server configuration
        if "mcpServers" in config_to_write:
            current_env = os.environ.copy()

            for server_name, server_config in config_to_write["mcpServers"].items():
                # Ensure server config has an env section
                if "env" not in server_config:
                    server_config["env"] = {}

                # Propagate important environment variables if they exist in current environment
                for env_var in important_env_vars:
                    if env_var in current_env and env_var not in server_config["env"]:
                        server_config["env"][env_var] = current_env[env_var]

        # Write the updated configuration to a temporary file
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as config_file:
            json.dump(config_to_write, config_file, indent=2)
            config_path = config_file.name

    except Exception as e:
        return {
            "task_id": str(uuid.uuid4()),
            "status": "error",
            "result": None,
            "error": f"Failed to create temporary config file: {str(e)}",
        }

    # Generate a unique task ID
    task_id = str(uuid.uuid4())

    # Build the mcpman command
    cmd = [MCPMAN_CMD, "-c", config_path, "-i", implementation, "-m", model, "-p", task]

    # Add optional arguments if provided
    if user_prompt and user_prompt.strip():
        cmd.extend(["--user-prompt", user_prompt])
    if system_message and system_message.strip():
        cmd.extend(["-s", system_message])
    if validation_prompt and validation_prompt.strip():
        cmd.extend(["--validation-prompt", validation_prompt])
    if temperature is not None:
        cmd.extend(["--temperature", str(temperature)])
    if base_url:
        cmd.extend(["--base-url", base_url])

    # Always limit the maximum number of tool calls to prevent runaway processes
    cmd.extend(["--max-turns", str(MAX_AGENT_TURNS)])

    try:
        # Execute the command, inheriting the parent process's environment
        # This ensures API keys like OPENAI_API_KEY, ANTHROPIC_API_KEY, etc. are passed to the subprocess
        process = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
            env=os.environ.copy(),  # Pass through all environment variables including API keys
            timeout=SUBPROCESS_TIMEOUT,  # Use the constant defined at the top of the file
        )

        # Process the output
        output = process.stdout
        stderr = process.stderr

        # Try to extract run_id from output or logs
        run_id = None
        run_id_match = RUN_ID_PATTERN.search(output) or RUN_ID_PATTERN.search(stderr)
        if run_id_match:
            run_id = run_id_match.group(1)

        # Find the log file
        log_path = _find_log_file(run_id)

        return {
            "task_id": task_id,
            "status": "success",
            "result": output,
            "run_id": run_id,
            "log_path": log_path,
        }
    except subprocess.CalledProcessError as e:
        # Command execution failed
        return {
            "task_id": task_id,
            "status": "error",
            "result": None,
            "error": f"Command execution failed with exit code {e.returncode}",
            "stderr": e.stderr,
            "suggestions": [
                "Check that the implementation and model are valid and available",
                "Ensure your MCP server configuration is correct and the servers can be launched",
                "Verify that the mcpman command is installed and in your PATH",
            ],
        }
    except subprocess.TimeoutExpired as e:
        # Process timed out
        return {
            "task_id": task_id,
            "status": "error",
            "result": None,
            "error": f"Command execution timed out after {e.timeout} seconds",
            "command": " ".join(e.cmd),
            "partial_stdout": e.stdout,
            "partial_stderr": e.stderr,
            "suggestions": [
                "The agent task may be too complex or resource-intensive",
                "Consider breaking down the task into smaller steps",
                "Check that the MCP servers in the configuration can start properly",
                "Try a different model or implementation",
            ],
        }
    except Exception as e:
        # Other unexpected errors
        return {
            "task_id": task_id,
            "status": "error",
            "result": None,
            "error": f"An unexpected error occurred: {str(e)}",
        }
    finally:
        # Clean up the temporary config file
        try:
            os.unlink(config_path)
        except:
            pass  # Ignore cleanup errors


def _parse_log_file(
    log_path: str, detail_level: str, filters: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Parses a log file with the specified detail level and filters.

    Args:
        log_path (str): Path to the log file
        detail_level (str): One of "basic", "standard", "verbose", "raw"
        filters (Dict[str, Any], optional): Filters to apply to the log entries

    Returns:
        Dict[str, Any]: Parsed log data at the requested detail level
    """
    try:
        if not os.path.exists(log_path):
            return {"status": "error", "error": f"Log file not found: {log_path}"}

        # Initialize variables for tracking log information
        run_id = None
        start_time = None
        end_time = None
        prompt = None
        model = None
        implementation = None
        tools_available = []
        tools_used = {}  # Tool name -> count
        tool_calls = []
        tool_results = []
        final_response = None

        # For raw mode, we need to collect all entries
        raw_entries = [] if detail_level == LOG_DETAIL_RAW else None

        # If we're using filters, we may need to collect filtered entries
        filtered_entries = [] if filters else None

        # Process the log file line by line (more efficient for large logs)
        with open(log_path, "r") as f:
            for line in f:
                try:
                    entry = json.loads(line.strip())

                    # For raw mode, just collect all entries
                    if detail_level == LOG_DETAIL_RAW:
                        raw_entries.append(entry)
                        continue

                    # Apply filters if provided
                    if filters:
                        match = True
                        for key, value in filters.items():
                            if key not in entry or entry[key] != value:
                                match = False
                                break
                        if not match:
                            continue
                        filtered_entries.append(entry)

                    # Extract information based on entry type
                    if "run_id" in entry and run_id is None:
                        run_id = entry["run_id"]

                    if "timestamp" in entry:
                        timestamp = entry["timestamp"]
                        if start_time is None or timestamp < start_time:
                            start_time = timestamp
                        if end_time is None or timestamp > end_time:
                            end_time = timestamp

                    if "type" in entry:
                        entry_type = entry["type"]

                        if entry_type == "prompt" and "content" in entry:
                            prompt = entry["content"]

                        elif entry_type == "tool_registration" and "tools" in entry:
                            for tool in entry["tools"]:
                                if (
                                    "name" in tool
                                    and tool["name"] not in tools_available
                                ):
                                    tools_available.append(tool["name"])

                        elif entry_type == "tool_call" and "tool" in entry:
                            tool_name = entry["tool"]
                            tools_used[tool_name] = tools_used.get(tool_name, 0) + 1

                            if detail_level in [
                                LOG_DETAIL_STANDARD,
                                LOG_DETAIL_VERBOSE,
                            ]:
                                call_info = {
                                    "tool": tool_name,
                                    "input": entry.get("input", {}),
                                    "timestamp": entry.get("timestamp"),
                                }
                                if "call_id" in entry:
                                    call_info["call_id"] = entry["call_id"]
                                tool_calls.append(call_info)

                        elif entry_type == "tool_result" and "result" in entry:
                            if detail_level in [
                                LOG_DETAIL_STANDARD,
                                LOG_DETAIL_VERBOSE,
                            ]:
                                result_info = {
                                    "tool": entry.get("tool", "unknown"),
                                    "result": entry["result"],
                                    "timestamp": entry.get("timestamp"),
                                }
                                if "call_id" in entry:
                                    result_info["call_id"] = entry["call_id"]
                                tool_results.append(result_info)

                        elif entry_type == "response" and "content" in entry:
                            final_response = entry["content"]

                    if "model" in entry:
                        model = entry["model"]
                    if "implementation" in entry:
                        implementation = entry["implementation"]

                except json.JSONDecodeError:
                    continue  # Skip invalid JSON lines

        # For raw mode, just return the raw entries
        if detail_level == LOG_DETAIL_RAW:
            return {"status": "success", "entries": raw_entries}

        # Calculate total time if we have both start and end times
        total_time = None
        if start_time and end_time:
            try:
                start_dt = time.mktime(
                    time.strptime(start_time, "%Y-%m-%dT%H:%M:%S.%fZ")
                )
                end_dt = time.mktime(time.strptime(end_time, "%Y-%m-%dT%H:%M:%S.%fZ"))
                total_time = end_dt - start_dt
            except:
                total_time = None

        # Build the result based on detail level
        result = {"status": "success", "run_id": run_id, "log_path": log_path}

        # Basic timing and metadata info for all levels
        if start_time:
            result["start_time"] = start_time
        if end_time:
            result["end_time"] = end_time
        if total_time is not None:
            result["total_time_seconds"] = total_time

        if model:
            result["model"] = model
        if implementation:
            result["implementation"] = implementation

        if detail_level == LOG_DETAIL_BASIC:
            result.update(
                {
                    "prompt": prompt,
                    "tools_used_count": sum(tools_used.values()),
                    "tools_used": list(tools_used.keys()),
                    "response": final_response,
                }
            )
        elif detail_level == LOG_DETAIL_STANDARD:
            result.update(
                {
                    "prompt": prompt,
                    "tools_available": tools_available,
                    "tools_used_count": sum(tools_used.values()),
                    "tools_used": [
                        {"name": tool, "count": count}
                        for tool, count in tools_used.items()
                    ],
                    "tool_calls_summary": len(tool_calls),
                    "response": final_response,
                }
            )
        elif detail_level == LOG_DETAIL_VERBOSE:
            # Sort tool calls and results by timestamp if available
            if tool_calls and all("timestamp" in call for call in tool_calls):
                tool_calls.sort(key=lambda x: x.get("timestamp", ""))
            if tool_results and all("timestamp" in result for result in tool_results):
                tool_results.sort(key=lambda x: x.get("timestamp", ""))

            result.update(
                {
                    "prompt": prompt,
                    "tools_available": tools_available,
                    "tools_used_count": sum(tools_used.values()),
                    "tools_used": [
                        {"name": tool, "count": count}
                        for tool, count in tools_used.items()
                    ],
                    "tool_calls": tool_calls,
                    "tool_results": tool_results,
                    "response": final_response,
                }
            )

        return result

    except Exception as e:
        return {"status": "error", "error": f"Failed to parse log file: {str(e)}"}


@app.tool()
def inspect_agent_task_log(
    log_path: Optional[str] = None,
    run_id: Optional[str] = None,
    detail_level: Literal["basic", "standard", "verbose", "raw"] = "standard",
    filters: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Inspects the logs for a completed agent task with varying levels of detail.

    Retrieves and parses the logs for a specific agent run, allowing the caller to specify
    the level of detail needed and apply filters to focus on specific aspects of the logs.

    Args:
        log_path (Optional[str], optional): Direct path to a log file to inspect.
                                          If not provided, will attempt to find a log file based on run_id.
        run_id (Optional[str], optional): The run ID of the mcpman execution to inspect.
                                       If provided without log_path, will attempt to find the corresponding log file.
                                       If neither log_path nor run_id is provided, will use the most recent log file.
        detail_level (str, optional): The level of detail to include in the output. Options:
                                     - "basic": Just task and result summary with minimal tool usage info
                                     - "standard": Includes task, tools used summary, and result details
                                     - "verbose": Full details including all tool calls and results
                                     - "raw": Complete raw log entries for manual parsing
                                     Defaults to "standard".
        filters (Optional[Dict[str, Any]], optional): Optional filters to apply to log entries.
                                                   Entries that don't match ALL filters will be excluded.
                                                   Example: {"type": "tool_call", "tool": "calculator"}
                                                   Defaults to None.

    Returns:
        Dict[str, Any]: A dictionary containing the parsed log data at the requested detail level.
                      The structure varies based on the detail_level parameter.

    Raises:
        ValueError: If the specified log or run_id cannot be found or if the log file cannot be parsed.
    """
    # Find the log file based on the provided information
    if log_path is None:
        log_path = _find_log_file(run_id)

        if log_path is None:
            return {
                "status": "error",
                "error": f"Could not find a log file{' for run_id ' + run_id if run_id else ''}.",
                "suggestions": [
                    "Provide a direct log_path to a known log file",
                    "Check that the run_id is correct",
                    "Ensure the logs directory exists and contains log files",
                ],
            }

    # Validate log path exists
    if not os.path.exists(log_path):
        return {"status": "error", "error": f"Log file not found: {log_path}"}

    # Process and return the log file contents based on detail level
    return _parse_log_file(log_path, detail_level, filters)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the AgentOps MCP server.")
    parser.add_argument(
        "--transport",
        type=str,
        choices=["sse", "stdio"],
        default="stdio",
        help="Transport method to use (sse or stdio).",
    )

    args = parser.parse_args()

    print(f"Starting AgentOps server (Transport: {args.transport})")
    app.run(transport=args.transport)
