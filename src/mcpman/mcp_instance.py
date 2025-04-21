import logging
import os
import shutil
import json
import asyncio
import re
from contextlib import AsyncExitStack
from typing import Any

import httpx
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# --- Configuration ---
# Set via environment variable or default
LLM_API_KEY = os.getenv("LLM_API_KEY", "api-key-placeholder")
LLM_API_URL = os.getenv("LLM_API_URL", "http://127.0.0.1:1234/v1/chat/completions")
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "qwen2.5-7b-instruct-1m")
LLM_TEMPERATURE = 0.7
LLM_MAX_TOKENS = 4096
CONFIG_PATH = "server_configs/calculator_server_mcp.json"


class Tool:
    """Represents a tool with its properties and formatting."""

    def __init__(
        self, name: str, description: str, input_schema: dict[str, Any]
    ) -> None:
        self.name: str = name
        self.description: str = description
        self.input_schema: dict[str, Any] = input_schema

    def format_for_llm(self) -> str:
        """Deprecated: Format tool information for old-style LLM prompt."""
        pass  # No longer needed for OpenAI tool format

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
        self.read: asyncio.StreamReader | None = None  # Store reader/writer
        self.write: asyncio.StreamWriter | None = None
        # self.exit_stack: AsyncExitStack = AsyncExitStack() # Removed internal exit stack

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

        # --- Removed old logic that used self.exit_stack ---
        # try:
        #     stdio_transport = await self.exit_stack.enter_async_context(
        #         stdio_client(server_params)
        #     )
        #     read, write = stdio_transport
        #     session = await self.exit_stack.enter_async_context(
        #         ClientSession(read, write)
        #     )
        #     await session.initialize()
        #     self.session = session
        # except Exception as e:
        #     logging.error(f"Error initializing server {self.name}: {e}")
        #     # await self.cleanup() # Don't call cleanup here
        #     raise

    async def list_tools(self) -> list[Tool]:  # Return type hint added
        """List available tools from the server."""
        if not self.session:
            raise RuntimeError(f"Server {self.name} session not initialized")

        tools_response = await self.session.list_tools()
        tools = []
        for item in tools_response:
            # Assuming item[1] are Tool objects from MCP response
            if isinstance(item, tuple) and item[0] == "tools":
                tools.extend(
                    Tool(tool.name, tool.description, tool.inputSchema)
                    for tool in item[1]
                )
        return tools

    async def execute_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        # retries: int = 2, # Retry logic might be better outside this low-level method
        # delay: float = 1.0,
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

        # Removed retry logic - keep it simple for now
        # attempt = 0
        # while attempt < retries:
        #     try:
        logging.debug(f"Executing {tool_name} via MCP session...")
        result = await self.session.call_tool(tool_name, arguments)
        return result
        # except Exception as e:
        #     attempt += 1
        #     logging.warning(
        #         f"Error executing tool: {e}. Attempt {attempt} of {retries}."
        #     )
        #     if attempt < retries:
        #         logging.info(f"Retrying in {delay} seconds...")
        #         await asyncio.sleep(delay)
        #     else:
        #         logging.error("Max retries reached. Failing.")
        #         raise

    # Removed cleanup method - managed by AsyncExitStack in main
    # async def cleanup(self) -> None:
    #    ...


class LLMClient:
    """Manages communication with the LLM provider."""

    def __init__(self, api_key: str) -> None:
        self.api_key = api_key

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
            "model": LLM_MODEL_NAME,  # Use constant
            "temperature": LLM_TEMPERATURE,  # Use constant
            "max_tokens": LLM_MAX_TOKENS,  # Use constant
            "top_p": 1,
            "stop": None,
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
                        f"Sending request to LLM model {LLM_MODEL_NAME}..."
                    )  # Less verbose info log

                response = client.post(
                    LLM_API_URL, headers=headers, json=payload
                )  # Use constant
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
            logging.error(f"Error communicating with LLM at {LLM_API_URL}: {e}")
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
    tool_name = tool_call["function"]["name"]
    tool_call_id = tool_call["id"]
    try:
        arguments = json.loads(tool_call["function"]["arguments"])
    except json.JSONDecodeError as e:
        logging.error(f"Error decoding arguments for {tool_name}: {e}")
        return {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "name": tool_name,
            "content": f"Error: Invalid arguments JSON: {e}",
        }

    print(f"-> Calling tool: {tool_name}({arguments})", flush=True)

    execution_result_content = f"Error: Tool '{tool_name}' not found on any server."
    tool_found = False
    for server in servers:
        try:
            logging.debug(f"Executing {tool_name} on server {server.name}...")
            tool_output = await server.execute_tool(tool_name, arguments)
            tool_found = True

            # Simplify result
            if hasattr(tool_output, "isError") and tool_output.isError:
                # Handle specific MCP error object
                error_detail = (
                    tool_output.content
                    if hasattr(tool_output, "content")
                    else "Unknown tool error"
                )
                logging.warning(f"Tool '{tool_name}' reported an error: {error_detail}")
                execution_result_content = (
                    f"Error: Tool execution failed: {error_detail}"
                )
            elif hasattr(tool_output, "content") and tool_output.content:
                # Handle successful result with content
                text_parts = [c.text for c in tool_output.content if hasattr(c, "text")]
                if text_parts:
                    execution_result_content = " ".join(text_parts)
                else:
                    execution_result_content = json.dumps(
                        tool_output.content
                    )  # Fallback to JSON dump of content
            elif isinstance(tool_output, (str, int, float)):
                # Handle simple return types
                execution_result_content = str(tool_output)
            else:
                # Fallback for unexpected successful result structure
                try:
                    execution_result_content = json.dumps(tool_output)
                except Exception:
                    execution_result_content = str(tool_output)

            logging.debug(f"Simplified Tool Result Text: {execution_result_content}")
            break  # Successful execution or handled error, exit server loop
        except RuntimeError as e:
            # Error connecting / communicating with the server process
            logging.warning(
                f"Runtime error executing {tool_name} on {server.name}: {e}"
            )
            # Keep searching on other servers if applicable
        except Exception as e:
            # Catch other exceptions during execute_tool call (e.g., validation within MCP client)
            logging.error(
                f"Exception executing tool '{tool_name}' on {server.name}: {e}",
                exc_info=True,
            )
            execution_result_content = f"Error: Tool execution failed unexpectedly."
            tool_found = True  # Mark as found but failed
            break  # Exit server loop after definite failure on this server

    if not tool_found:
        logging.warning(f"Tool '{tool_name}' not found on any server.")
        # Content already defaults to a not found error

    return {
        "role": "tool",
        "tool_call_id": tool_call_id,
        "name": tool_name,
        "content": str(execution_result_content),  # Ensure always string
    }


async def run_query(query: str, servers: list[Server], llm_client: LLMClient):
    """Runs a query, calls tools iteratively, and prints the final answer."""

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

    system_message = (
        "You are a helpful assistant that can use tools to answer questions. "
        "Break down complex problems into sequential tool calls if necessary. "
        "When you receive tool results, use them to make the next call or formulate the final answer."
    )
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": query},
    ]

    loop = asyncio.get_running_loop()
    max_turns = 10  # Add a safety break

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
    """Initialize components and run a single query."""
    logging.basicConfig(
        # level=logging.WARNING,
        level=logging.DEBUG,  # Keep DEBUG for inspecting cleanup
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )
    logger = logging.getLogger(__name__)

    # --- Load Config ---
    try:
        with open(CONFIG_PATH, "r") as f:
            server_config = json.load(f)
    except FileNotFoundError:
        logger.error(f"Server config file not found: {CONFIG_PATH}")
        return
    except json.JSONDecodeError:
        logger.error(f"Error decoding server config file: {CONFIG_PATH}")
        return

    servers_to_init = [
        Server(name, srv_config)
        for name, srv_config in server_config.get("mcpServers", {}).items()
    ]
    if not servers_to_init:
        logger.error("No servers defined in the configuration file.")
        return

    llm_api_key = LLM_API_KEY
    if llm_api_key == "api-key-placeholder":
        logger.warning(
            "LLM_API_KEY environment variable not set or is placeholder. Using dummy key."
        )
    llm_client = LLMClient(llm_api_key)

    # --- Initialize Servers Manually ---
    initialized_servers: list[Server] = []
    # Store context managers and sessions to clean up later
    entered_stdio_cms: list[Any] = []
    entered_sessions: list[ClientSession] = []

    init_success = True
    for server in servers_to_init:
        stdio_client_cm = None
        session = None
        try:
            logger.debug(f"Initializing server {server.name}...")
            # 1. Get the stdio_client context manager
            stdio_client_cm = await server.initialize()
            # 2. Enter its context MANUALLY
            read, write = await stdio_client_cm.__aenter__()
            entered_stdio_cms.append(stdio_client_cm)  # Track for cleanup
            server.read = read
            server.write = write
            logger.debug(f"stdio client connected for {server.name}.")

            # 3. Create the ClientSession
            session = ClientSession(read, write)
            # 4. Enter its context MANUALLY
            server.session = await session.__aenter__()
            entered_sessions.append(session)  # Track for cleanup

            # 5. Initialize the session itself
            await server.session.initialize()
            logger.info(f"Server {server.name} initialized successfully.")
            initialized_servers.append(server)
        except Exception as e:
            logger.error(
                f"Failed to initialize server {server.name}: {e}", exc_info=True
            )
            init_success = False
            # Need to clean up partially entered contexts for THIS server immediately
            if session in entered_sessions:
                try:
                    await session.__aexit__(None, None, None)
                    entered_sessions.remove(session)
                except Exception as cleanup_e:
                    logger.error(
                        f"Error cleaning up session during init failure for {server.name}: {cleanup_e}"
                    )
            if stdio_client_cm in entered_stdio_cms:
                try:
                    await stdio_client_cm.__aexit__(None, None, None)
                    entered_stdio_cms.remove(stdio_client_cm)
                except Exception as cleanup_e:
                    logger.error(
                        f"Error cleaning up stdio_cm during init failure for {server.name}: {cleanup_e}"
                    )
            # We won't proceed if any server fails, so break the loop
            break

    # --- Main Execution Block ---
    try:
        if not init_success or not initialized_servers:
            logger.error("Server initialization failed. Exiting.")
            # Cleanup will happen in finally block
            return

        # Run the query only if all servers initialized successfully
        query = "What is 7 / 3 / 1.27?"  # Using simple query for now
        logger.info(f"Running query: {query}")
        print(f"Running query: {query}")
        await run_query(query, initialized_servers, llm_client)

    except Exception as e:
        # Catch errors during run_query itself
        logger.error(f"An error occurred during query execution: {e}", exc_info=True)
    finally:
        # --- Explicit Cleanup ---
        logger.info("Starting manual cleanup...")
        # Cleanup sessions first, then stdio clients (reverse order of creation)
        while entered_sessions:
            session_to_clean = entered_sessions.pop()
            try:
                logger.debug(f"Cleaning up session {session_to_clean}...")
                await session_to_clean.__aexit__(None, None, None)
            except Exception as e:
                logger.error(f"Error during session cleanup: {e}", exc_info=True)

        while entered_stdio_cms:
            stdio_cm_to_clean = entered_stdio_cms.pop()
            try:
                logger.debug(f"Cleaning up stdio_client {stdio_cm_to_clean}...")
                await stdio_cm_to_clean.__aexit__(None, None, None)
            except Exception as e:
                logger.error(f"Error during stdio_client cleanup: {e}", exc_info=True)
        logger.info("Manual cleanup finished.")


if __name__ == "__main__":
    # Make sure to import necessary modules if not already done
    import logging
    import os
    import shutil
    import json
    import asyncio
    import re  # Needed for markdown stripping in LLMClient if kept
    from contextlib import AsyncExitStack
    from typing import Any
    import httpx  # Needed for LLMClient
    from mcp import (
        ClientSession,
        StdioServerParameters,
    )  # Assuming these are needed by Server
    from mcp.client.stdio import stdio_client  # Assuming needed by Server

    asyncio.run(main())
