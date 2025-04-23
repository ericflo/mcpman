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
import glob
from typing import Dict, List, Any, Optional

# Import formatting utilities
from .formatting import (
    print_llm_config,
    format_tool_call,
    format_tool_response,
    format_llm_response,
    format_verification_result,
    format_processing_step,
)

from .config import (
    DEFAULT_SYSTEM_MESSAGE,
    get_llm_configuration,
    PROVIDERS,
)
from .llm_client import create_llm_client
from .orchestrator import initialize_and_run
from .logger import (
    setup_logging as enhanced_setup_logging,
    log_execution_start,
    log_execution_complete,
    get_logger,
)


# We now use the enhanced_setup_logging function directly from logger.py


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments.

    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="MCPMan - Model Context Protocol Manager for agentic LLM workflows."
    )

    # Create subparsers for run and replay modes
    subparsers = parser.add_subparsers(dest="mode", help="Operating mode")

    # Default run mode (no subcommand)
    run_parser = subparsers.add_parser("run", help="Run MCPMan in normal mode")
    replay_parser = subparsers.add_parser("replay", help="Replay a log file")

    # Add a --replay parameter to main parser as well (alternative to subcommand)
    parser.add_argument(
        "--replay",
        action="store_true",
        help="Run in replay mode to visualize a previous conversation log",
    )

    # Add replay-specific arguments to both replay parser and main parser
    parser.add_argument(
        "--replay-file",
        help="Path to the log file to replay (defaults to latest log)",
    )
    parser.add_argument(
        "--show-hidden",
        action="store_true",
        help="Show events that wouldn't normally be visible in replay mode",
    )
    replay_parser.add_argument(
        "--replay-file",
        help="Path to the log file to replay (defaults to latest log)",
    )
    replay_parser.add_argument(
        "--show-hidden",
        action="store_true",
        help="Show events that wouldn't normally be visible",
    )

    # Server configuration - for normal run mode
    parser.add_argument("-c", "--config", help="Path to the server config JSON file.")

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

    # Add all normal run mode arguments to the run subparser too
    run_parser.add_argument(
        "-c", "--config", required=True, help="Path to the server config JSON file."
    )
    run_parser.add_argument(
        "-m", "--model", help="Name of the LLM model to use (overrides environment)."
    )
    run_parser.add_argument(
        "-i",
        "--impl",
        "--implementation",
        dest="impl",
        choices=PROVIDERS.keys(),
        help="Select a pre-configured LLM implementation (provider) to use (overrides environment).",
    )
    run_parser.add_argument(
        "--base-url",
        help="Custom LLM API URL (overrides environment, requires --api-key).",
    )
    run_parser.add_argument(
        "--api-key",
        help="LLM API Key (overrides environment, use with --base-url or if provider requires it).",
    )
    run_parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature for the LLM (default: 0.7).",
    )
    run_parser.add_argument(
        "--max-tokens", type=int, help="Maximum number of tokens for the LLM response."
    )
    run_parser.add_argument(
        "--max-turns",
        type=int,
        default=2048,
        help="Maximum number of turns for the agent loop (default: 2048).",
    )
    run_parser.add_argument(
        "--timeout",
        type=float,
        default=180.0,
        help="Request timeout in seconds for LLM API calls (default: 180.0).",
    )
    run_parser.add_argument(
        "-s",
        "--system",
        default=DEFAULT_SYSTEM_MESSAGE,
        help="The system message to send to the LLM.",
    )
    run_parser.add_argument(
        "-p",
        "--prompt",
        required=True,
        help="The prompt to send to the LLM. If the value is a path to an existing file, the file contents will be used.",
    )

    run_verification_group = run_parser.add_mutually_exclusive_group()
    run_verification_group.add_argument(
        "--no-verify",
        action="store_true",
        help="Disable task verification (verification is on by default).",
    )
    run_verification_group.add_argument(
        "--verify-prompt",
        dest="verification_prompt",
        help="Provide a custom verification prompt or path to a file containing the prompt.",
    )

    run_strict_tools_group = run_parser.add_mutually_exclusive_group()
    run_strict_tools_group.add_argument(
        "--strict-tools",
        action="store_true",
        dest="strict_tools",
        help="Enable strict mode for tool schemas (default if MCPMAN_STRICT_TOOLS=true).",
    )
    run_strict_tools_group.add_argument(
        "--no-strict-tools",
        action="store_false",
        dest="strict_tools",
        help="Disable strict mode for tool schemas.",
    )

    run_parser.add_argument(
        "--debug", action="store_true", help="Enable debug logging."
    )
    run_parser.add_argument(
        "--no-log-file",
        action="store_true",
        help="Disable logging to file (logging to file is enabled by default).",
    )
    run_parser.add_argument(
        "--log-dir",
        default="logs",
        help="Directory to store log files (default: logs).",
    )
    run_parser.add_argument(
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
    - Mode selection (normal run or replay)
    - Logging setup
    - LLM client creation
    - Server initialization
    - Agent execution
    """
    # Parse arguments first to get debug flag
    args = parse_args()

    # Check if we're in replay mode
    # Replay mode can be triggered by:
    # 1. The "replay" subcommand: args.mode == "replay"
    # 2. The --replay flag: args.replay == True
    if getattr(args, "mode", None) == "replay" or getattr(args, "replay", False):
        # Run in replay mode - get log file path from args
        replay_file = getattr(args, "replay_file", None)
        log_dir = getattr(args, "log_dir", "logs")
        show_hidden = getattr(args, "show_hidden", False)

        # Run replay mode without async/await
        replay_mode(replay_file, log_dir, show_hidden)
        return

    # Regular run mode - validate required parameters
    missing_args = []
    if not args.config:
        missing_args.append("--config/-c")
    
    if not args.prompt:
        missing_args.append("--prompt/-p")
        
    if missing_args:
        print(f"Error: {', '.join(missing_args)} argument(s) required when not in replay mode.")
        logger = logging.getLogger(__name__)
        logger.error(f"Missing required arguments: {missing_args}")
        sys.exit(1)

    # Setup logging
    log_to_file = not args.no_log_file

    # Configure logging levels for output-only mode
    # When in output-only mode, we don't want to suppress print statements,
    # just logging messages

    # Set up enhanced logging with our new setup
    log_file_path = None
    if log_to_file:
        # Create log directory if it doesn't exist
        os.makedirs(args.log_dir, exist_ok=True)

        # Generate log filename with timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file_path = os.path.join(args.log_dir, f"mcpman_{timestamp}.jsonl")

    # Use enhanced logging setup
    # quiet_console = True means JSON logs only go to file, not console
    quiet_console = not args.debug
    log_level = logging.DEBUG if args.debug else logging.INFO

    # Setup the logging system
    enhanced_setup_logging(
        log_file=log_file_path,
        level=log_level,
        quiet_console=quiet_console,
        output_only=args.output_only,
    )
    logger = logging.getLogger(__name__)

    if log_file_path:
        # Only print the log file path if in debug mode and not in output_only mode
        if args.debug and not args.output_only:
            print(f"Logging to: {log_file_path}")

        # Use enhanced structured logging for execution start
        logger = get_logger()
        log_execution_start(
            logger, taskName=f"Task-{os.getpid()}", extra={"command_args": vars(args)}
        )

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
        # Provide more helpful error message for console
        print(f"Error: Could not determine LLM API URL for provider {args.impl or 'custom'}.")
        print("       Please configure using -i/--impl, --base-url, or environment variables.")
        print("       See documentation for required environment variables for each provider.")
        return

    if not provider_config["model"]:
        logger.error("No model name specified or found for provider.")
        # Provide helpful error message for console
        print(f"Error: No model name specified or found for provider {args.impl or 'custom'}.")
        print("       Specify with -m/--model or set appropriate environment variable.")
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
            "strict_tools": (
                "default" if args.strict_tools is None else str(args.strict_tools)
            ),
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
        # Get run ID from the earlier logger setup
        run_id = f"{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}_{os.getpid()}"

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
            run_id=run_id,
        )
    finally:
        # Log completion of execution with enhanced structured logging
        logger = get_logger()
        log_execution_complete(
            logger,
            config={
                "config_path": args.config,
                "provider": args.impl or "custom",
                "model": provider_config.get("model", "unknown"),
                "temperature": args.temperature,
                "max_turns": args.max_turns,
                "verify_completion": verify_completion,
                "strict_tools": args.strict_tools,
            },
            taskName=f"Task-{os.getpid()}",
            extra={"completion_status": "success", "command_args": vars(args)},
        )


# Functions for replay mode
def print_llm_config_box(config: Dict[str, Any]) -> None:
    """
    Print the LLM configuration box.

    Args:
        config: Dictionary with configuration data
    """
    # Create a custom formatted config box
    box_width = 80

    # Print box header
    print("╔" + "═" * (box_width - 2) + "╗")
    print("║" + "LLM CONFIGURATION".center(box_width - 2) + "║")
    print("╠" + "═" * (box_width - 2) + "╣")

    # Print config items
    for key, value in config.items():
        key_display = key.title() + ":"
        padding = box_width - len(key_display) - len(str(value)) - 4
        print(f"║{key_display}  {str(value)}{' ' * padding}║")

    # Print box footer
    print("╚" + "═" * (box_width - 2) + "╝")


def find_latest_log_file(log_dir="logs") -> str:
    """
    Find the most recent log file in the logs directory.

    Args:
        log_dir: Directory to search for log files

    Returns:
        Path to the most recent log file, or None if no log files found
    """
    # Ensure log directory exists
    if not os.path.exists(log_dir):
        print(f"Warning: Log directory '{log_dir}' not found.")
        return None

    # Find all jsonl files in the logs directory
    log_files = glob.glob(os.path.join(log_dir, "*.jsonl"))

    if not log_files:
        print(f"No log files found in '{log_dir}'.")
        return None

    # Sort by modification time (newest first)
    latest_log = max(log_files, key=os.path.getmtime)

    return latest_log


def extract_config_data(log_entries: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Extract configuration data from log entries.

    Args:
        log_entries: List of log entry dictionaries

    Returns:
        Dictionary with configuration data
    """
    config_data = {}

    # Look for execution_complete entry which contains full config
    for entry in log_entries:
        if entry.get("event_type") == "execution_complete" and "execution" in entry:
            execution = entry["execution"]
            config_data = {
                "implementation": entry.get("payload", {}).get("provider", "unknown"),
                "model": execution.get("model", "unknown"),
                "api_url": "https://api.openai.com/v1/chat/completions",  # Default, may need to extract from logs
                "timeout": f"{execution.get('timeout', 180.0)}s",
                "strict_tools": str(execution.get("strict_tools", False)),
                "config_path": execution.get("config_path", "unknown"),
            }
            break

    return config_data


def process_conversation_flow(
    log_entries: List[Dict[str, Any]], show_hidden: bool = False
) -> None:
    """
    Process and display the conversation flow in chronological order.

    Args:
        log_entries: List of log entry dictionaries
        show_hidden: Whether to show events that wouldn't normally be visible
    """
    prompt = ""
    tool_calls = []
    tool_responses = {}
    llm_responses = []
    verification_results = []

    # First pass - extract key events
    for entry in log_entries:
        # Extract user prompt
        if entry.get("message") == "Running prompt:":
            prompt = entry.get("payload", {}).get("taskName", "")

        # Extract prompt content
        if "prompt" in entry.get("payload", {}):
            prompt = entry.get("payload", {}).get("prompt", "")

        # Get tool calls
        if entry.get("event_type") == "tool_call":
            tool_name = entry.get("payload", {}).get("tool", "")
            parameters = entry.get("payload", {}).get("parameters", {})
            tool_call_id = entry.get("payload", {}).get("tool_call_id", "")

            if tool_name and parameters:
                tool_calls.append(
                    {"id": tool_call_id, "tool": tool_name, "parameters": parameters}
                )

        # Get tool responses
        if entry.get("event_type") == "tool_response":
            tool_name = entry.get("payload", {}).get("tool", "")
            response = entry.get("payload", {}).get("response", "")
            tool_call_id = entry.get("payload", {}).get("tool_call_id", "")

            if tool_name and response:
                tool_responses[tool_call_id] = {"tool": tool_name, "response": response}

        # Get LLM responses
        if entry.get("event_type") == "llm_response" and entry.get("payload", {}).get(
            "has_content", False
        ):
            response_content = (
                entry.get("payload", {}).get("response", {}).get("content", "")
            )
            if response_content:
                llm_responses.append(response_content)

        # Get verification results
        if entry.get("event_type") == "verification":
            is_complete = entry.get("payload", {}).get("is_complete", False)
            feedback = entry.get("payload", {}).get("feedback", "")
            verification_results.append(
                {"is_complete": is_complete, "feedback": feedback}
            )

    # Process request types
    process_requests = []
    for entry in log_entries:
        if entry.get("message", "").startswith("Processing request of type"):
            req_type = entry.get("message").replace("Processing request of type ", "")
            process_requests.append(req_type)

    # Now display the conversation in the correct order

    # Print initial prompt
    if prompt:
        print("┌─ Processing request:")
        print(f"└─► {prompt}")

    # Process tools in sequence
    for i, tool_call in enumerate(tool_calls):
        # Print tool call with proper formatting
        formatted_tool_call = format_tool_call(
            tool_call["tool"], json.dumps(tool_call["parameters"])
        )
        print(formatted_tool_call)

        # Show processing request if available
        if i < len(process_requests) and process_requests[i + 1] == "CallToolRequest":
            print("Processing request of type CallToolRequest")

        # Show tool response with proper formatting
        if tool_call["id"] in tool_responses:
            response = tool_responses[tool_call["id"]]
            formatted_tool_response = format_tool_response(
                response["tool"], response["response"]
            )
            print(formatted_tool_response)

    # Show final response (potential answer)
    if llm_responses:
        for i, response in enumerate(llm_responses):
            if i < len(verification_results):
                # This is a potential answer
                print(format_llm_response(response, is_final=False))

                # Show verification process
                print(format_processing_step("Verifying task completion"))

                if verification_results[i]["is_complete"]:
                    # Final answer (verification passed)
                    print(format_llm_response(response, is_final=True))
                    print(
                        format_verification_result(
                            True, verification_results[i]["feedback"]
                        )
                    )
                else:
                    # Verification failed
                    print(
                        format_verification_result(
                            False, verification_results[i]["feedback"]
                        )
                    )
            else:
                # Last response (final answer)
                print(format_llm_response(response, is_final=True))


def process_log_file(log_file_path: str, show_hidden: bool = False) -> None:
    """
    Process the log file and reproduce the original colorized output.

    Args:
        log_file_path: Path to the log file to process
        show_hidden: Whether to show events that wouldn't normally be visible
    """
    try:
        with open(log_file_path, "r") as f:
            log_lines = f.readlines()
    except FileNotFoundError:
        print(f"Error: Log file '{log_file_path}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading log file: {e}")
        sys.exit(1)

    # Parse log entries
    log_entries = []
    for line in log_lines:
        try:
            entry = json.loads(line)
            log_entries.append(entry)
        except json.JSONDecodeError:
            print(f"Warning: Could not parse log line: {line[:100]}...")
            continue

    # Extract configuration data
    config_data = extract_config_data(log_entries)

    # Print LLM configuration box
    if config_data:
        print_llm_config_box(config_data)

    # Track the conversation flow
    process_conversation_flow(log_entries, show_hidden)


def replay_mode(
    replay_file: Optional[str] = None, log_dir: str = "logs", show_hidden: bool = False
) -> None:
    """
    Run the application in replay mode.

    Args:
        replay_file: Path to the log file to replay
        log_dir: Directory containing log files
        show_hidden: Whether to show events that wouldn't normally be visible
    """
    # If no replay file specified, find the latest one
    if not replay_file:
        replay_file = find_latest_log_file(log_dir)
        if not replay_file:
            print(
                "Error: No replay file specified and no log files found in the logs directory."
            )
            sys.exit(1)

        print(f"Using latest log file: {replay_file}")

    # Process the log file
    process_log_file(replay_file, show_hidden)


def run() -> None:
    """
    Run the application.

    This function is the entry point for the console script.
    It handles both normal run mode and replay mode.
    """
    logger = logging.getLogger("mcpman")
    try:
        # This will handle both replay mode (non-async) and normal mode (async)
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
