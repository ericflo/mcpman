import logging
import os
import shutil
import json
import asyncio
import argparse
import contextlib
from typing import Any
import re

import httpx
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


# --- Helper Functions ---
def sanitize_name(name: str) -> str:
    """Sanitize a name to be a valid identifier (replace non-alphanumeric with _)."""
    # Replace non-alphanumeric characters with underscores
    sanitized = re.sub(r"[^a-zA-Z0-9_]", "_", name)
    # Ensure it doesn't start with a digit (optional, but good practice)
    if sanitized and sanitized[0].isdigit():
        sanitized = "_" + sanitized
    # Replace multiple consecutive underscores with a single one
    sanitized = re.sub(r"_+", "_", sanitized)
    # Remove leading/trailing underscores (optional)
    # sanitized = sanitized.strip('_')
    # Let's keep them for now to avoid potential empty names or collisions
    return sanitized


# ------------------------


# --- Configuration ---
# Set via environment variable or default
CONFIG_PATH = "server_configs/calculator_server_mcp.json"

OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_DEFAULT_MODEL = "gpt-4.1-nano"
LMSTUDIO_API_URL = "http://127.0.0.1:1234/v1/chat/completions"
LMSTUDIO_DEFAULT_MODEL = "qwen2.5-7b-instruct-1m"
OLLAMA_API_URL = "http://127.0.0.1:11434/api/generate"
OLLAMA_DEFAULT_MODEL = "qwen2.5:7b"
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_DEFAULT_MODEL = "deepseek/deepseek-chat-v3-0324"
TOGETHER_API_URL = "https://api.together.xyz/v1/chat/completions"
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY", "")
TOGETHER_DEFAULT_MODEL = "meta-llama/Llama-4-Scout-17B-16E-Instruct"
GEMINI_API_URL = (
    "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_DEFAULT_MODEL = "gemini-2.5-flash-preview-04-17"
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_DEFAULT_MODEL = "llama-3.3-70b-versatile"
HYPERBOLIC_API_URL = "https://api.hyperbolic.xyz/v1/chat/completions"
HYPERBOLIC_API_KEY = os.getenv("HYPERBOLIC_API_KEY", "")
HYPERBOLIC_DEFAULT_MODEL = "deepseek-ai/DeepSeek-V3-0324"
DEEPINFRA_API_URL = "https://api.deepinfra.com/v1/openai/chat/completions"
DEEPINFRA_API_KEY = os.getenv("DEEPINFRA_API_KEY", "")
DEEPINFRA_DEFAULT_MODEL = "meta-llama/Llama-4-Scout-17B-16E-Instruct"

# MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", "")
# ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

DEFAULT_SYSTEM_MESSAGE = """
You are an agent - please keep going until the user's query is completely resolved, before ending your turn and yielding back to the user. Only terminate your turn when you are sure that the problem is solved.
If you are not sure about file content or codebase structure pertaining to the user's request, use your tools to read files and gather the relevant information: do NOT guess or make up an answer.
You MUST plan extensively before each function call, and reflect extensively on the outcomes of the previous function calls. DO NOT do this entire process by making function calls only, as this can impair your ability to solve the problem and think insightfully.
""".strip()
DEFAULT_PROMPT = "What is 7 / 3 / 1.27?"

PROVIDERS = {
    "openai": {
        "url": OPENAI_API_URL,
        "key": OPENAI_API_KEY,
        "default_model": OPENAI_DEFAULT_MODEL,
    },
    "lmstudio": {
        "url": LMSTUDIO_API_URL,
        "key": "",
        "default_model": LMSTUDIO_DEFAULT_MODEL,
    },
    "ollama": {"url": OLLAMA_API_URL, "key": "", "default_model": OLLAMA_DEFAULT_MODEL},
    "openrouter": {
        "url": OPENROUTER_API_URL,
        "key": OPENROUTER_API_KEY,
        "default_model": OPENROUTER_DEFAULT_MODEL,
    },
    "together": {
        "url": TOGETHER_API_URL,
        "key": TOGETHER_API_KEY,
        "default_model": TOGETHER_DEFAULT_MODEL,
    },
    "gemini": {
        "url": GEMINI_API_URL,
        "key": GEMINI_API_KEY,
        "default_model": GEMINI_DEFAULT_MODEL,
    },
    "groq": {
        "url": GROQ_API_URL,
        "key": GROQ_API_KEY,
        "default_model": GROQ_DEFAULT_MODEL,
    },
    "hyperbolic": {
        "url": HYPERBOLIC_API_URL,
        "key": HYPERBOLIC_API_KEY,
        "default_model": HYPERBOLIC_DEFAULT_MODEL,
    },
    "deepinfra": {
        "url": DEEPINFRA_API_URL,
        "key": DEEPINFRA_API_KEY,
        "default_model": DEEPINFRA_DEFAULT_MODEL,
    },
    # "mistral": {"url": MISTRAL_API_URL, "key": MISTRAL_API_KEY, "default_model": MISTRAL_DEFAULT_MODEL},
    # "anthropic": {"url": ANTHROPIC_API_URL, "key": ANTHROPIC_API_KEY, "default_model": ANTHROPIC_DEFAULT_MODEL},
}

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")
LLM_API_KEY = os.getenv("LLM_API_KEY", OPENAI_API_KEY)
LLM_API_URL = os.getenv("LLM_API_URL", OPENAI_API_URL)
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "gpt-4.1-nano")


class Tool:
    """Represents a tool with its properties and formatting."""

    def __init__(
        self,
        name: str,
        description: str,
        input_schema: dict[str, Any],
        original_name: str,
    ) -> None:
        self.name: str = name  # This will be the prefixed name (e.g., calculator_add)
        self.description: str = description
        self.input_schema: dict[str, Any] = input_schema
        self.original_name: str = original_name  # Store the original name (e.g., add)

    def to_openai_schema(self) -> dict[str, Any]:
        """Format the tool definition for the OpenAI API 'tools' parameter."""
        parameters_schema = {"type": "object", "properties": {}, "required": []}
        if isinstance(self.input_schema, dict):
            input_props = self.input_schema.get("properties")
            if isinstance(input_props, dict):
                sanitized_props = {}
                for name, prop_info in input_props.items():
                    if isinstance(prop_info, dict) and "type" in prop_info:
                        sanitized_props[name] = prop_info
                    else:
                        sanitized_props[name] = {
                            "type": "string",
                            "description": str(
                                prop_info.get(
                                    "description", "Parameter without defined type"
                                )
                            ),
                        }
                        logging.warning(
                            f"Property '{name}' in tool '{self.name}' schema missing type, defaulting to string."
                        )
                parameters_schema["properties"] = sanitized_props
            input_required = self.input_schema.get("required")
            if isinstance(input_required, list):
                parameters_schema["required"] = [
                    req
                    for req in input_required
                    if req in parameters_schema["properties"]
                ]

        tool_schema = {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": parameters_schema,
            },
        }
        logging.debug(
            f"Generated OpenAI schema for tool '{self.name}': {json.dumps(tool_schema)}"
        )
        return tool_schema


class Server:
    """Manages MCP server connections and tool execution."""

    def __init__(self, name: str, config: dict[str, Any]) -> None:
        self.name: str = name
        self.config: dict[str, Any] = config
        self.session: ClientSession | None = None
        self.read: asyncio.StreamReader | None = None
        self.write: asyncio.StreamWriter | None = None
        self.tools: list[Tool] | None = None  # Added to cache tools/schemas

    async def initialize(
        self,
    ) -> Any:  # Return type is the stdio_client context manager
        """Prepare server parameters and return the stdio_client context manager."""
        command = (
            shutil.which("npx")
            if self.config["command"] == "npx"
            else self.config["command"]
        )
        if command is None:
            raise ValueError("The command must be a valid string and cannot be None.")

        server_params = StdioServerParameters(
            command=command,
            args=self.config["args"],
            env=(
                {**os.environ, **self.config["env"]} if self.config.get("env") else None
            ),
        )
        # Return the context manager instance, don't enter it here
        return stdio_client(server_params)

    async def list_tools(self) -> list[Tool]:
        """List available tools from the server and cache them."""
        if not self.session:
            raise RuntimeError(f"Server {self.name} session not initialized")
        # Return cached tools if already fetched
        if self.tools is not None:
            logging.debug(f"Returning cached tools for {self.name}.")
            return self.tools

        logging.debug(f"Fetching tools from {self.name}...")
        tools_response = await self.session.list_tools()
        fetched_tools = []
        for item in tools_response:
            if isinstance(item, tuple) and item[0] == "tools":
                sanitized_server_name = sanitize_name(self.name)
                fetched_tools.extend(
                    Tool(
                        name=f"{sanitized_server_name}_{tool.name}",  # Prefixed name
                        description=tool.description,
                        input_schema=tool.inputSchema,
                        original_name=tool.name,  # Pass original name explicitly
                    )
                    for tool in item[1]
                )
        self.tools = fetched_tools  # Cache the fetched tools
        logging.debug(f"Cached {len(self.tools)} tools for {self.name}.")
        return self.tools

    def get_tool_schema(self, original_tool_name: str) -> dict[str, Any] | None:
        """Get the input schema for a specific tool by its original name."""
        if self.tools is None:
            # This case should ideally not happen if list_tools is called first,
            # but adding a safeguard.
            logging.warning(
                f"Attempted to get schema for {original_tool_name} on {self.name}, but tools list is empty or not fetched yet."
            )
            # Potential improvement: Call self.list_tools() here if needed?
            # For now, assume list_tools was called during setup.
            return None

        for tool in self.tools:
            if tool.original_name == original_tool_name:
                return tool.input_schema

        logging.warning(
            f"Schema for tool '{original_tool_name}' not found in cached tools for server '{self.name}'."
        )
        return None

    async def execute_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> Any:
        """Execute a tool.
        Args:
            tool_name: Name of the tool to execute.
            arguments: Tool arguments.
        Returns:
            Tool execution result.
        Raises:
            RuntimeError: If server session is not initialized.
            Exception: From self.session.call_tool if execution fails.
        """
        if not self.session:
            raise RuntimeError(f"Server {self.name} session not initialized")

        logging.debug(f"Executing {tool_name} via MCP session...")
        result = await self.session.call_tool(tool_name, arguments)
        return result


class LLMClient:
    """Manages communication with the LLM provider."""

    def __init__(self, api_key: str, api_url: str, model_name: str) -> None:
        self.api_key = api_key
        self.api_url = api_url
        self.model_name = model_name

    def get_response(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None
    ) -> dict[str, Any]:
        """Get a response message object from the LLM."""
        logging.debug(
            f"Sending {len(messages)} messages to LLM. {'Including tools.' if tools else 'No tools.'}"
        )

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        payload = {
            "messages": messages,
            "model": self.model_name,
        }
        if tools:
            payload["tools"] = tools

        try:
            with httpx.Client(timeout=60.0) as client:
                # Only log full payload on DEBUG level
                if logging.getLogger().isEnabledFor(logging.DEBUG):
                    try:
                        payload_str = json.dumps(payload, indent=2)
                        logging.debug(f"Sending LLM Payload:\n{payload_str}")
                    except Exception as e:
                        logging.debug(
                            f"Could not serialize LLM payload for logging: {e}"
                        )
                else:
                    logging.info(
                        f"Sending request to LLM model {self.model_name} at {self.api_url}..."
                    )

                response = client.post(self.api_url, headers=headers, json=payload)
                logging.debug(
                    f"Received response from LLM: Status {response.status_code}"
                )
                response.raise_for_status()

                data = response.json()
                if not data or "choices" not in data or not data["choices"]:
                    raise KeyError("Invalid response: 'choices' missing or empty.")
                if "message" not in data["choices"][0]:
                    raise KeyError(
                        "Invalid response: 'message' missing in first choice."
                    )

                message = data["choices"][0]["message"]
                logging.debug(f"LLM response message object: {message}")
                return message

        except httpx.RequestError as e:
            logging.error(f"Error communicating with LLM at {self.api_url}: {e}")
            # Return an error-like message object
            return {"role": "assistant", "content": f"Error: Could not reach LLM: {e}"}
        except (KeyError, IndexError, json.JSONDecodeError) as e:
            logging.error(f"Error parsing LLM response: {e}")
            return {
                "role": "assistant",
                "content": f"Error: Invalid LLM response format: {e}",
            }


async def execute_tool_call(
    tool_call: dict[str, Any], servers: list[Server]
) -> dict[str, Any]:
    """Executes a single tool call and returns the result message."""
    prefixed_tool_name = tool_call["function"]["name"]
    tool_call_id = tool_call["id"]

    # --- Parse Prefixed Name (Improved) ---
    target_server_name = None
    original_tool_name = None

    sanitized_server_names = sorted(
        [sanitize_name(s.name) for s in servers], key=len, reverse=True
    )

    for s_name in sanitized_server_names:
        prefix = f"{s_name}_"
        if prefixed_tool_name.startswith(prefix):
            target_server_name = s_name
            original_tool_name = prefixed_tool_name[len(prefix) :]
            break

    if not target_server_name or not original_tool_name:
        logging.error(
            f"Could not parse server and tool name from '{prefixed_tool_name}'"
        )
        return {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "name": prefixed_tool_name,
            "content": f"Error: Invalid prefixed tool name format '{prefixed_tool_name}'",
        }

    # Find the target server
    target_server: Server | None = next(
        (s for s in servers if sanitize_name(s.name) == target_server_name), None
    )

    if not target_server:
        logging.warning(
            f"Target server '{target_server_name}' for tool '{prefixed_tool_name}' not found."
        )
        # Return error if server not found
        return {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "name": prefixed_tool_name,
            "content": f"Error: Server '{target_server_name}' (sanitized) not found.",
        }

    # --- Get Tool Schema --- REMOVED
    # --- Parse and Correct Arguments --- REMOVED

    # --- Simple Argument Parsing ---
    arguments: dict[str, Any] = {}
    try:
        arguments_str = tool_call["function"]["arguments"]
        arguments = json.loads(arguments_str)  # Simple parse

    except json.JSONDecodeError as e:
        logging.error(f"Error decoding arguments JSON for {prefixed_tool_name}: {e}")
        return {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "name": prefixed_tool_name,
            "content": f"Error: Invalid arguments JSON: {e}",
        }
    # ---------------------------

    print(
        f"-> Calling tool: {prefixed_tool_name}({arguments})", flush=True
    )  # Log parsed args

    execution_result_content = f"Error: Tool '{original_tool_name}' execution failed on server '{target_server.name}'."
    tool_found_on_target_server = False

    # Execute the tool on the target server
    try:
        logging.debug(
            f"Executing {original_tool_name} on server {target_server.name} (sanitized: {target_server_name}, prefixed: {prefixed_tool_name})..."
        )
        # Use original_tool_name and parsed arguments dictionary
        tool_output = await target_server.execute_tool(original_tool_name, arguments)
        tool_found_on_target_server = True

        # Simplify result
        if hasattr(tool_output, "isError") and tool_output.isError:
            error_detail = (
                tool_output.content
                if hasattr(tool_output, "content")
                else "Unknown tool error"
            )
            logging.warning(
                f"Tool '{prefixed_tool_name}' reported an error: {error_detail}"
            )
            # Check if it's an 'unknown tool' error - might indicate mismatch, but treat as error for now
            if "Unknown tool" in str(error_detail):
                execution_result_content = f"Error: Tool '{original_tool_name}' not found on server '{target_server_name}'."
            else:
                execution_result_content = (
                    f"Error: Tool execution failed: {error_detail}"
                )
        elif hasattr(tool_output, "content") and tool_output.content:
            text_parts = [c.text for c in tool_output.content if hasattr(c, "text")]
            if text_parts:
                execution_result_content = " ".join(text_parts)
            else:
                execution_result_content = json.dumps(tool_output.content)
        elif isinstance(tool_output, (str, int, float)):
            execution_result_content = str(tool_output)
        else:
            try:
                execution_result_content = json.dumps(tool_output)
            except Exception:
                execution_result_content = str(tool_output)

        logging.debug(f"Simplified Tool Result Text: {execution_result_content}")

    except RuntimeError as e:
        logging.warning(
            f"Runtime error executing {prefixed_tool_name} on {target_server.name}: {e}"
        )
        execution_result_content = (
            f"Error: Runtime error contacting server {target_server.name}: {e}"
        )
        tool_found_on_target_server = True  # Attempted but failed communication
    except Exception as e:
        logging.error(
            f"Exception executing tool '{prefixed_tool_name}' on {target_server.name}: {e}",
            exc_info=True,
        )
        execution_result_content = (
            f"Error: Tool execution failed unexpectedly on {target_server.name}."
        )
        tool_found_on_target_server = True  # Attempted but failed

    # --- Print the tool response ---
    # Use prefixed_tool_name for consistency with LLM's view
    print(
        f"<- Tool Response [{prefixed_tool_name}]: {execution_result_content}",
        flush=True,
    )
    # -----------------------------

    return {
        "role": "tool",
        "tool_call_id": tool_call_id,
        "name": prefixed_tool_name,
        "content": str(execution_result_content),
    }


async def run_prompt(
    prompt: str, servers: list[Server], llm_client: LLMClient, system_message: str
):
    """Runs a prompt, calls tools iteratively, and prints the final answer."""

    # Prepare tools for API
    all_tools = []
    for server in servers:
        try:
            server_tools = await server.list_tools()
            all_tools.extend(server_tools)
        except Exception as e:
            logging.warning(f"Failed to list tools for server {server.name}: {e}")

    openai_tools = [tool.to_openai_schema() for tool in all_tools]
    logging.debug(f"Prepared {len(openai_tools)} tools for the API.")

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": prompt},
    ]

    loop = asyncio.get_running_loop()
    max_turns = 2048  # Add a safety break

    for turn in range(max_turns):
        logging.debug(f"--- Turn {turn + 1} ---")
        assistant_message = await loop.run_in_executor(
            None, llm_client.get_response, messages, openai_tools
        )
        # Ensure assistant message content is string or None, tool_calls is list or None
        if "content" in assistant_message and not isinstance(
            assistant_message["content"], (str, type(None))
        ):
            assistant_message["content"] = str(assistant_message["content"])
        if "tool_calls" in assistant_message and not isinstance(
            assistant_message["tool_calls"], (list, type(None))
        ):
            logging.warning("Received non-list tool_calls, attempting to ignore.")
            del assistant_message["tool_calls"]  # Attempt recovery

        messages.append(assistant_message)
        logging.debug(
            f"Added assistant message: {json.dumps(assistant_message, indent=2)}"
        )

        tool_calls = assistant_message.get("tool_calls")

        if tool_calls:
            tool_results = []
            for tool_call in tool_calls:
                # Ensure arguments are strings before executing
                if isinstance(tool_call.get("function", {}).get("arguments"), dict):
                    tool_call["function"]["arguments"] = json.dumps(
                        tool_call["function"]["arguments"]
                    )

                if tool_call.get("type") == "function":
                    result_message = await execute_tool_call(tool_call, servers)
                    tool_results.append(result_message)
                else:
                    logging.warning(
                        f"Unsupported tool call type: {tool_call.get('type')}"
                    )
                    tool_results.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call["id"],
                            "name": tool_call.get("function", {}).get(
                                "name", "unknown"
                            ),
                            "content": f"Error: Unsupported tool type '{tool_call.get('type')}'",
                        }
                    )

            messages.extend(tool_results)
            logging.debug(f"Added {len(tool_results)} tool result message(s).")
            # --- Log messages before the next LLM call ---
            try:
                logging.debug(
                    f"Messages before next LLM call:\n{json.dumps(messages, indent=2)}"
                )
            except Exception as log_e:
                logging.error(f"Error logging messages: {log_e}")
            # Continue loop to call LLM again with results

        else:
            # No tool calls, assume final answer
            content = assistant_message.get("content")
            if content:
                print(f"\nFinal Answer:\n{content}", flush=True)
            else:
                print("\nFinal Answer: (LLM provided no content)", flush=True)
                logging.warning(
                    f"Final assistant message had no content: {assistant_message}"
                )
            break  # Exit loop
    else:
        print("\nWarning: Maximum turns reached without a final answer.", flush=True)


async def main() -> None:
    """Initialize components and run a single prompt."""
    # --- Configure Logging Early ---
    # Set initial level, may be overridden by args
    initial_log_level = logging.DEBUG if "--debug" in os.sys.argv else logging.WARNING
    logging.basicConfig(
        level=initial_log_level,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        force=True,  # Allow reconfiguring later
    )
    logger = logging.getLogger(__name__)

    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="Run MCPMan with configurable LLM.")
    parser.add_argument(
        "--config", default=CONFIG_PATH, help="Path to the server config JSON file."
    )
    parser.add_argument(
        "--model", help="Name of the LLM model to use (overrides environment)."
    )

    provider_group = parser.add_mutually_exclusive_group()
    provider_group.add_argument(
        "--provider",
        choices=PROVIDERS.keys(),
        help="Select a pre-configured LLM provider (overrides environment).",
    )
    provider_group.add_argument(
        "--api-url",
        help="Custom LLM API URL (overrides environment, requires --api-key).",
    )

    parser.add_argument(
        "--api-key",
        help="LLM API Key (overrides environment, use with --api-url or if provider requires it).",
    )

    # Add debug flag
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging.",
    )

    parser.add_argument(
        "--system-message",
        default=DEFAULT_SYSTEM_MESSAGE,
        help="The system message to send to the LLM.",
    )
    parser.add_argument(
        "--prompt",
        default=DEFAULT_PROMPT,
        help="The initial prompt to send to the LLM.",
    )

    args = parser.parse_args()

    # --- Re-configure Logging based on args ---
    log_level = logging.DEBUG if args.debug else logging.WARNING
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        force=True,  # Force re-configuration
    )
    logger = logging.getLogger(__name__)  # Get logger again after final configuration

    # --- Determine LLM Configuration ---
    # Precedence: CLI args > Environment Vars > Hardcoded defaults

    # 1. Model Name
    llm_model_name = args.model or os.getenv("LLM_MODEL_NAME", "gpt-4.1-nano")

    # 2. API URL and Key
    llm_api_url = None
    llm_api_key = None

    if args.provider:
        provider_info = PROVIDERS.get(args.provider)
        if provider_info:
            llm_api_url = provider_info["url"]
            # Prioritize explicit CLI key, then provider-specific env key, then general env key, then provider default (often empty)
            provider_env_key_name = f"{args.provider.upper()}_API_KEY"
            llm_api_key = (
                args.api_key
                or os.getenv(provider_env_key_name)
                or os.getenv("LLM_API_KEY")
                or provider_info["key"]
            )
            logger.info(f"Using provider '{args.provider}'. API URL: {llm_api_url}")
        else:
            # Should not happen due to choices in argparse, but safety check
            logger.error(f"Invalid provider selected: {args.provider}. Exiting.")
            return
    elif args.api_url:
        llm_api_url = args.api_url
        # Prioritize explicit CLI key, then general env key
        llm_api_key = args.api_key or os.getenv("LLM_API_KEY")
        if not llm_api_key:
            logger.warning("Using --api-url without --api-key or LLM_API_KEY env var.")
        logger.info(f"Using custom API URL: {llm_api_url}")
    else:
        # Fallback to environment variables or defaults for provider/URL/key
        default_provider_name = os.getenv("LLM_PROVIDER", "openai")
        provider_info = PROVIDERS.get(default_provider_name)
        if provider_info:
            llm_api_url = os.getenv("LLM_API_URL") or provider_info["url"]
            provider_env_key_name = f"{default_provider_name.upper()}_API_KEY"
            llm_api_key = (
                os.getenv("LLM_API_KEY")
                or os.getenv(provider_env_key_name)
                or provider_info["key"]
            )
            logger.info(
                f"Using default/env provider '{default_provider_name}'. API URL: {llm_api_url}"
            )
        else:
            # Fallback to generic URL/Key if default provider is invalid
            logger.warning(
                f"Default provider '{default_provider_name}' not found in config. Falling back to generic LLM_API_URL/LLM_API_KEY."
            )
            llm_api_url = os.getenv(
                "LLM_API_URL", OPENAI_API_URL
            )  # Provide a final fallback
            llm_api_key = os.getenv("LLM_API_KEY")

    if not llm_api_url:
        logger.error(
            "Could not determine LLM API URL. Please configure using --provider, --api-url, or environment variables. Exiting."
        )
        return

    # Ensure api_key is a string, default to empty if None
    llm_api_key = llm_api_key or ""

    # Print essential configuration regardless of log level
    print("--- LLM Configuration ---")
    print(f"  Model: {llm_model_name}")
    print(f"  API URL: {llm_api_url}")

    # --- Load Config ---
    config_path = args.config  # Use argument for config path
    print(f"  Server Config: {config_path}")
    print("-------------------------")
    # logger.info(f"Using server config file: {config_path}") # Keep as info log if needed elsewhere
    try:
        with open(config_path, "r") as f:
            server_config = json.load(f)
    except FileNotFoundError:
        logger.error(f"Server config file not found: {config_path}")
        return
    except json.JSONDecodeError:
        logger.error(f"Error decoding server config file: {config_path}")
        return

    servers_to_init = [
        Server(name, srv_config)
        for name, srv_config in server_config.get("mcpServers", {}).items()
    ]
    if not servers_to_init:
        logger.error("No mcpServers defined in the configuration file.")
        return

    # Instantiate LLM Client with determined config
    llm_client = LLMClient(
        api_key=llm_api_key, api_url=llm_api_url, model_name=llm_model_name
    )

    # --- Initialize Servers and Run Prompt using AsyncExitStack ---
    initialized_servers: list[Server] = []
    init_success = True

    try:
        async with contextlib.AsyncExitStack() as stack:
            # Initialize servers within the stack
            for server in servers_to_init:
                try:
                    logger.debug(f"Initializing server {server.name}...")
                    stdio_client_cm = await server.initialize()
                    # Enter stdio client context manager using the stack
                    read, write = await stack.enter_async_context(stdio_client_cm)
                    server.read = read
                    server.write = write
                    logger.debug(f"stdio client connected for {server.name}.")

                    # Create and enter session context manager using the stack
                    session = ClientSession(read, write)
                    server.session = await stack.enter_async_context(session)

                    # Initialize the session itself
                    await server.session.initialize()
                    logger.info(f"Server {server.name} initialized successfully.")
                    initialized_servers.append(server)

                    # Print server and its tools
                    try:
                        server_tools = await server.list_tools()
                        print(
                            f"  Server '{server.name}' initialized with tools:", end=""
                        )
                        if server_tools:
                            print(", ".join([tool.name for tool in server_tools]))
                        else:
                            print("(No tools found)")
                    except Exception as list_tools_e:
                        print(
                            f"  Server '{server.name}' initialized, but failed to list tools: {list_tools_e}"
                        )
                        logger.warning(
                            f"Could not list tools for {server.name} after init: {list_tools_e}"
                        )

                except Exception as e:
                    logger.error(
                        f"Failed to initialize server {server.name}: {e}", exc_info=True
                    )
                    init_success = False
                    # No need for manual cleanup here, AsyncExitStack handles partial failures
                    break

            # Proceed only if all initializations were successful
            if not init_success or not initialized_servers:
                logger.error("Server initialization failed. Exiting.")
                # Stack will clean up successfully entered contexts
                return

            # --- Main Execution Block (inside the stack) ---
            prompt = args.prompt
            system_message = args.system_message
            logger.info(f"System Message: {system_message}")
            logger.info(f"Running prompt: {prompt}")
            print(f"Running prompt: {prompt}")

            await run_prompt(prompt, initialized_servers, llm_client, system_message)

    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}", exc_info=True)
    finally:
        logger.info("Application finished.")  # Cleanup is now handled by AsyncExitStack


if __name__ == "__main__":
    asyncio.run(main())


# ------------------------
