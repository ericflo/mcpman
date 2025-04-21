import asyncio
import logging
import argparse
import sys
import os
from typing import Optional, Dict, Any

from .config import (
    DEFAULT_CONFIG_PATH,
    DEFAULT_SYSTEM_MESSAGE,
    DEFAULT_USER_PROMPT,
    get_llm_configuration,
    PROVIDERS,
)
from .llm_client import create_llm_client
from .orchestrator import initialize_and_run


def setup_logging(debug: bool = False) -> None:
    """
    Configure logging for the application.

    Args:
        debug: Whether to enable debug logging
    """
    log_level = logging.DEBUG if debug else logging.WARNING
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        force=True,  # Force re-configuration
    )


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments.

    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(description="Run MCPMan with configurable LLM.")

    # Server configuration
    parser.add_argument(
        "-c", "--config", required=True, help="Path to the server config JSON file."
    )

    # LLM configuration
    parser.add_argument(
        "-m", "--model", help="Name of the LLM model to use (overrides environment)."
    )

    # Provider options (mutually exclusive)
    provider_group = parser.add_mutually_exclusive_group()
    provider_group.add_argument(
        "-i",
        "--impl", "--implementation",
        dest="impl",
        choices=PROVIDERS.keys(),
        help="Select a pre-configured LLM implementation (provider) to use (overrides environment).",
    )
    provider_group.add_argument(
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
        default=10,
        help="Maximum number of turns for the agent loop (default: 10).",
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
    parser.add_argument(
        "--verify",
        nargs="?",
        const=True,
        help="Enable task verification to ensure the task is complete. Optionally provide a custom verification prompt or path to a file containing the prompt.",
    )

    # Debug logging
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging.",
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
    setup_logging(args.debug)
    logger = logging.getLogger(__name__)

    # Get LLM configuration
    provider_config = get_llm_configuration(
        provider_name=args.impl,
        api_url=args.base_url,
        api_key=args.api_key,
        model_name=args.model,
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

    # Print configuration
    print("--- LLM Configuration ---")
    print(f"  Implementation: {args.impl or 'custom'}")
    print(f"  Model: {provider_config['model']}")
    print(f"  API URL: {provider_config['url']}")
    print(f"  Server Config: {args.config}")
    print("-------------------------")

    # Process prompt and verification - check if they're file paths
    user_prompt = read_file_if_exists(args.prompt)

    # Process verification settings
    verify_completion = False
    verification_prompt = None

    if args.verify is not None:
        verify_completion = True
        # If args.verify is a string (not just True), check if it's a file path
        if isinstance(args.verify, str):
            verification_prompt = read_file_if_exists(args.verify)

    # Initialize servers and run the agent
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
    )


def run() -> None:
    """
    Run the application.

    This function is the entry point for the console script.
    """
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    run()
