"""
MCPMan CLI interface

Copyright 2023-2025 Eric Florenzano

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import asyncio
import logging
import argparse
import sys
import os
import json
import datetime
import pathlib
from typing import Optional, Dict, Any, Union

# Import formatting utilities
from .formatting import (
    print_llm_config,
    BoxStyle,
    print_box,
    format_value,
    get_terminal_width,
    visible_length,
)

from .config import (
    DEFAULT_CONFIG_PATH,
    DEFAULT_SYSTEM_MESSAGE,
    DEFAULT_USER_PROMPT,
    get_llm_configuration,
    PROVIDERS,
)
from .llm_client import create_llm_client
from .orchestrator import initialize_and_run


class JsonlLogFormatter(logging.Formatter):
    """
    Enhanced formatter to output detailed, well-structured log records as JSON lines.
    """

    def __init__(self):
        super().__init__()
        # Generate a unique run ID for this execution
        self.run_id = (
            datetime.datetime.now().strftime("%Y%m%d%H%M%S") + "_" + str(os.getpid())
        )

    def format(self, record):
        """
        Format a log record as a rich, structured JSON line with consistent fields.
        """
        # Get timestamp with microsecond precision
        timestamp = datetime.datetime.fromtimestamp(record.created)
        formatted_time = timestamp.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"

        # Base log data with enhanced metadata
        log_data = {
            "timestamp": formatted_time,
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "run_id": self.run_id,  # Unique ID for this execution
            "source": f"{record.pathname}:{record.lineno}",
            "process_id": os.getpid(),
        }

        # Categorize the log entry if event_type is present
        if hasattr(record, "event_type"):
            log_data["event_type"] = record.event_type
            # Group related event types
            if record.event_type.startswith("tool_"):
                log_data["category"] = "tool_operation"
            elif record.event_type.startswith("llm_"):
                log_data["category"] = "llm_interaction"
            elif record.event_type.startswith("task_") or record.event_type.startswith(
                "execution_"
            ):
                log_data["category"] = "execution_flow"
            elif record.event_type.startswith("turn_"):
                log_data["category"] = "agent_turn"
            else:
                log_data["category"] = "general"
        else:
            log_data["category"] = "system"

        # Add timing data if available
        if hasattr(record, "response_time_seconds"):
            log_data["duration_ms"] = round(record.response_time_seconds * 1000)

        if hasattr(record, "turn_number"):
            log_data["turn_number"] = record.turn_number

        # Add exception info with clean formatting
        if record.exc_info:
            exception_info = self.formatException(record.exc_info)
            log_data["exception"] = {
                "type": (
                    record.exc_info[0].__name__ if record.exc_info[0] else "Unknown"
                ),
                "message": str(record.exc_info[1]) if record.exc_info[1] else "",
                "traceback": exception_info,
            }

        # List of attributes to exclude from extra data
        standard_attrs = {
            "args",
            "asctime",
            "created",
            "exc_info",
            "exc_text",
            "filename",
            "funcName",
            "id",
            "levelname",
            "levelno",
            "lineno",
            "module",
            "msecs",
            "message",
            "msg",
            "name",
            "pathname",
            "process",
            "processName",
            "relativeCreated",
            "stack_info",
            "thread",
            "threadName",
            "event_type",
            "category",
            "response_time_seconds",
            "turn_number",
        }

        # Extract detailed data into a payload section
        payload = {}
        if hasattr(record, "__dict__"):
            for key, value in record.__dict__.items():
                if key not in standard_attrs and key != "message":
                    try:
                        # Test if the value is JSON serializable
                        json.dumps({key: value})
                        payload[key] = value
                    except (TypeError, OverflowError):
                        # If not serializable, convert to string
                        payload[key] = str(value)

        # Add the payload if it contains data
        if payload:
            log_data["payload"] = payload

        # Add specific structured data based on event type
        if hasattr(record, "event_type"):
            # For tool calls, structure the data more clearly
            if (
                record.event_type == "tool_call"
                and hasattr(record, "tool_name")
                and hasattr(record, "arguments")
            ):
                log_data["tool"] = {
                    "name": getattr(record, "tool_name", "unknown"),
                    "original_name": getattr(record, "original_tool_name", "unknown"),
                    "server": getattr(record, "server_name", "unknown"),
                    "call_id": getattr(record, "tool_call_id", "unknown"),
                    "arguments": getattr(record, "arguments", {}),
                }

            # For tool responses, include the result
            elif record.event_type == "tool_response" and hasattr(record, "tool_name"):
                log_data["tool"] = {
                    "name": getattr(record, "tool_name", "unknown"),
                    "original_name": getattr(record, "original_tool_name", "unknown"),
                    "server": getattr(record, "server_name", "unknown"),
                    "call_id": getattr(record, "tool_call_id", "unknown"),
                    "response": getattr(record, "response", ""),
                    "success": getattr(record, "success", False),
                }

            # For LLM responses, structure the data more clearly
            elif record.event_type == "llm_response":
                llm_data = {
                    "has_tool_calls": getattr(record, "has_tool_calls", False),
                    "has_content": getattr(record, "has_content", False),
                    "duration_ms": round(
                        getattr(record, "response_time_seconds", 0) * 1000
                    ),
                }

                # Add content if present
                if hasattr(record, "assistant_content"):
                    llm_data["content"] = record.assistant_content or ""

                # Add tool calls if present
                if (
                    hasattr(record, "assistant_tool_calls")
                    and record.assistant_tool_calls
                ):
                    try:
                        tool_calls = json.loads(record.assistant_tool_calls)
                        llm_data["tool_calls"] = tool_calls
                    except (json.JSONDecodeError, TypeError):
                        llm_data["tool_calls_raw"] = str(record.assistant_tool_calls)

                log_data["llm_response"] = llm_data

            # For execution stats at completion
            elif record.event_type == "execution_complete" and hasattr(record, "model"):
                log_data["execution"] = {
                    "config_path": getattr(record, "config_path", "unknown"),
                    "provider": getattr(record, "provider", "unknown"),
                    "model": getattr(record, "model", "unknown"),
                    "temperature": getattr(record, "temperature", 0),
                    "max_turns": getattr(record, "max_turns", 0),
                    "verify_completion": getattr(record, "verify_completion", False),
                }

        # Ensure JSON serialization works
        try:
            return json.dumps(log_data)
        except (TypeError, OverflowError) as e:
            # Fallback - return a simplified log that contains essential info
            return json.dumps(
                {
                    "timestamp": formatted_time,
                    "level": record.levelname,
                    "logger": record.name,
                    "message": record.getMessage(),
                    "run_id": self.run_id,
                    "category": "error",
                    "error": f"Log serialization error: {str(e)}",
                }
            )


def setup_logging(
    debug: bool = False,
    log_to_file: bool = True,
    log_dir: str = "logs",
    output_only: bool = False,
) -> str:
    """
    Configure logging for the application.

    Args:
        debug: Whether to enable debug logging
        log_to_file: Whether to log to a file
        log_dir: Directory to store log files
        output_only: Whether to only print final output

    Returns:
        Path to the log file if created, otherwise None
    """
    # Set console level depending on mode
    if output_only:
        # In output-only mode, suppress all logging to console
        console_level = logging.CRITICAL
    else:
        # Normal mode - use debug level if requested
        console_level = logging.DEBUG if debug else logging.WARNING

    # Configure root logger - always use at least INFO level to capture all important events
    root_logger = logging.getLogger()
    # Set lowest level to capture all logs and let handlers filter
    root_logger.setLevel(logging.DEBUG)

    # Clear any existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Console handler with standard formatting
    console_handler = logging.StreamHandler()
    console_handler.setLevel(
        console_level
    )  # Use console_level (DEBUG or INFO) for console
    console_formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
    )
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    # File handler with JSON formatting (if enabled)
    log_file_path = None
    if log_to_file:
        # Create log directory if it doesn't exist
        pathlib.Path(log_dir).mkdir(exist_ok=True, parents=True)

        # Generate a unique filename with timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file_path = os.path.join(log_dir, f"mcpman_{timestamp}.jsonl")

        # Create and configure file handler - always use INFO level for file log
        file_handler = logging.FileHandler(log_file_path, mode="w")
        file_handler.setLevel(
            logging.INFO
        )  # Always use INFO level for file to capture important events
        file_handler.setFormatter(JsonlLogFormatter())

        # Enable immediate flushing after every emit
        original_emit = file_handler.emit

        def emit_and_flush(record):
            original_emit(record)
            file_handler.flush()

        file_handler.emit = emit_and_flush

        root_logger.addHandler(file_handler)

        # Log that we're starting with file logging
        root_logger.info(
            f"Logging to file: {log_file_path}", extra={"log_file": log_file_path}
        )

    return log_file_path


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments.

    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="MCPMan - Model Context Protocol Manager for agentic LLM workflows."
    )

    # Server configuration
    parser.add_argument(
        "-c", "--config", required=True, help="Path to the server config JSON file."
    )

    # LLM configuration
    parser.add_argument(
        "-m", "--model", help="Name of the LLM model to use (overrides environment)."
    )

    # Provider options
    parser.add_argument(
        "-i",
        "--impl",
        "--implementation",
        dest="impl",
        choices=PROVIDERS.keys(),
        help="Select a pre-configured LLM implementation (provider) to use (overrides environment).",
    )
    parser.add_argument(
        "--base-url",
        help="Custom LLM API URL (overrides environment, requires --api-key).",
    )

    # API key
    parser.add_argument(
        "--api-key",
        help="LLM API Key (overrides environment, use with --base-url or if provider requires it).",
    )

    # LLM parameters
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature for the LLM (default: 0.7).",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        help="Maximum number of tokens for the LLM response.",
    )

    # Agent parameters
    parser.add_argument(
        "--max-turns",
        type=int,
        default=2048,
        help="Maximum number of turns for the agent loop (default: 2048).",
    )

    parser.add_argument(
        "--timeout",
        type=float,
        default=180.0,
        help="Request timeout in seconds for LLM API calls (default: 180.0).",
    )
    parser.add_argument(
        "-s",
        "--system",
        default=DEFAULT_SYSTEM_MESSAGE,
        help="The system message to send to the LLM.",
    )
    parser.add_argument(
        "-p",
        "--prompt",
        required=True,
        help="The prompt to send to the LLM. If the value is a path to an existing file, the file contents will be used.",
    )

    # Task verification
    verification_group = parser.add_mutually_exclusive_group()
    verification_group.add_argument(
        "--no-verify",
        action="store_true",
        help="Disable task verification (verification is on by default).",
    )
    verification_group.add_argument(
        "--verify-prompt",
        dest="verification_prompt",
        help="Provide a custom verification prompt or path to a file containing the prompt.",
    )
    
    # Tool schema configuration
    strict_tools_group = parser.add_mutually_exclusive_group()
    strict_tools_group.add_argument(
        "--strict-tools",
        action="store_true", 
        dest="strict_tools",
        help="Enable strict mode for tool schemas (default if MCPMAN_STRICT_TOOLS=true).",
    )
    strict_tools_group.add_argument(
        "--no-strict-tools",
        action="store_false",
        dest="strict_tools", 
        help="Disable strict mode for tool schemas.",
    )

    # Logging options
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging.",
    )
    parser.add_argument(
        "--no-log-file",
        action="store_true",
        help="Disable logging to file (logging to file is enabled by default).",
    )
    parser.add_argument(
        "--log-dir",
        default="logs",
        help="Directory to store log files (default: logs).",
    )
    parser.add_argument(
        "--output-only",
        action="store_true",
        help="Only print the final validated output (useful for piping to files or ETL scripts).",
    )

    return parser.parse_args()


def read_file_if_exists(path_or_content: str) -> str:
    """
    If the path exists as a file, read and return its contents, otherwise return the original string.

    Args:
        path_or_content: Either a file path or a content string

    Returns:
        File contents if path exists, otherwise the original string
    """
    if os.path.exists(path_or_content) and os.path.isfile(path_or_content):
        try:
            with open(path_or_content, "r") as f:
                return f.read()
        except Exception as e:
            logging.warning(f"Failed to read file {path_or_content}: {e}")
            return path_or_content
    return path_or_content


async def main() -> None:
    """
    Main entry point for the application.

    In normal mode, this displays all the intermediate steps of the process.
    In output-only mode (--output-only flag), only the final LLM output is shown.

    Handles:
    - Argument parsing
    - Logging setup
    - LLM client creation
    - Server initialization
    - Agent execution
    """
    # Parse arguments first to get debug flag
    args = parse_args()

    # Setup logging
    log_to_file = not args.no_log_file

    # Configure logging levels for output-only mode
    # When in output-only mode, we don't want to suppress print statements,
    # just logging messages

    log_file_path = setup_logging(
        args.debug, log_to_file, args.log_dir, args.output_only
    )
    logger = logging.getLogger(__name__)

    if log_file_path:
        # Only print this if in debug mode
        if args.debug:
            print(f"Logging to: {log_file_path}")
        # Add timestamp to log for tracking execution
        logger.info("MCPMan execution started", extra={"event_type": "execution_start"})

    # Get LLM configuration
    provider_config = get_llm_configuration(
        provider_name=args.impl,
        api_url=args.base_url,
        api_key=args.api_key,
        model_name=args.model,
        timeout=args.timeout,
    )

    # Validate configuration
    if not provider_config["url"]:
        logger.error(
            "Could not determine LLM API URL. Please configure using -i/--impl, --base-url, or environment variables."
        )
        return

    if not provider_config["model"]:
        logger.error("No model name specified or found for provider.")
        return

    # Create LLM client
    llm_client = create_llm_client(provider_config, args.impl)

    # Print configuration (only if not in output-only mode)
    if not args.output_only:
        # Use the centralized LLM config display function
        config_data = {
            "impl": args.impl or "custom",
            "model": provider_config["model"],
            "url": provider_config["url"],
            "timeout": provider_config.get("timeout", 180.0),
            "strict_tools": "default" if args.strict_tools is None else str(args.strict_tools),
        }
        print_llm_config(config_data, args.config)

    # Process prompt and verification - check if they're file paths
    user_prompt = read_file_if_exists(args.prompt)

    # Process verification settings
    verify_completion = (
        not args.no_verify
    )  # Verification is on by default unless --no-verify is specified
    verification_prompt = None

    # Check if a custom verification prompt was provided
    if args.verification_prompt:
        verification_prompt = read_file_if_exists(args.verification_prompt)

    # Initialize servers and run the agent
    try:
        # Pass through the output_only flag and strict_tools to our implementation
        await initialize_and_run(
            config_path=args.config,
            user_prompt=user_prompt,
            system_message=args.system,
            llm_client=llm_client,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            max_turns=args.max_turns,
            verify_completion=verify_completion,
            verification_prompt=verification_prompt,
            provider_name=args.impl,
            output_only=args.output_only,
            strict_tools=args.strict_tools,
        )
    finally:
        # Log completion of execution even if there were exceptions
        logger.info(
            "MCPMan execution completed",
            extra={
                "event_type": "execution_complete",
                "config_path": args.config,
                "provider": args.impl or "custom",
                "model": provider_config.get("model", "unknown"),
                "temperature": args.temperature,
                "max_turns": args.max_turns,
                "verify_completion": verify_completion,
                "strict_tools": args.strict_tools,
            },
        )


def run() -> None:
    """
    Run the application.

    This function is the entry point for the console script.
    """
    logger = logging.getLogger("mcpman")
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        # Log the interruption
        if logger.isEnabledFor(logging.INFO):
            logger.info(
                "Operation cancelled by user",
                extra={
                    "event_type": "execution_interrupted",
                    "category": "execution_flow",
                    "reason": "keyboard_interrupt",
                    "timestamp": datetime.datetime.now().isoformat(),
                },
            )
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        # Log the error with details
        if logger.isEnabledFor(logging.ERROR):
            logger.error(
                f"Application error: {e}",
                exc_info=True,
                extra={
                    "event_type": "execution_error",
                    "category": "error",
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                    "timestamp": datetime.datetime.now().isoformat(),
                },
            )
        sys.exit(1)


if __name__ == "__main__":
    run()
