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
from .logger import (
    setup_logging as enhanced_setup_logging,
    log_execution_start,
    log_execution_complete,
    get_logger
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
        output_only=args.output_only
    )
    logger = logging.getLogger(__name__)

    if log_file_path:
        # Only print the log file path if in debug mode and not in output_only mode
        if args.debug and not args.output_only:
            print(f"Logging to: {log_file_path}")
        
        # Use enhanced structured logging for execution start
        logger = get_logger()
        log_execution_start(
            logger, 
            taskName=f"Task-{os.getpid()}", 
            extra={"command_args": vars(args)}
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
            extra={"completion_status": "success", "command_args": vars(args)}
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
