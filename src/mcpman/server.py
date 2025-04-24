import os
import logging
import asyncio
import shutil
from typing import Dict, List, Any, Optional

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from .tools import Tool, sanitize_name


class Server:
    """
    Manages MCP server connections and tool execution.

    Handles:
    - Server initialization and connection
    - Tool discovery and caching
    - Tool execution
    """

    def __init__(self, name: str, config: Dict[str, Any]) -> None:
        """
        Initialize a Server instance.

        Args:
            name: Name of the server
            config: Server configuration dictionary
        """
        self.name: str = name
        self.config: Dict[str, Any] = config
        self.session: Optional[ClientSession] = None
        self.read: Optional[asyncio.StreamReader] = None
        self.write: Optional[asyncio.StreamWriter] = None
        self.tools: Optional[List[Tool]] = None  # Cache for discovered tools

    async def initialize(self, output_only=False) -> Any:
        """
        Prepare server parameters and return the stdio_client context manager.

        Args:
            output_only: When True, suppress server logging output

        Returns:
            The stdio_client context manager

        Raises:
            ValueError: If the command is invalid
        """
        # Handle special case for npx command
        command = self.config["command"]

        if command is None:
            raise ValueError("The command must be a valid string and cannot be None.")

        # Create server parameters
        env = os.environ.copy()
        if self.config.get("env"):
            env.update(self.config["env"])
        server_params = StdioServerParameters(
            command=command,
            args=self.config["args"],
            env=env,
        )

        # Return the context manager instance with appropriate error logging
        if output_only:
            # Redirect server output to /dev/null to suppress "Processing request" messages
            import sys

            null_file = open(os.devnull, "w")
            return stdio_client(server_params, errlog=null_file)
        else:
            # Normal operation with default error logging
            return stdio_client(server_params)

    async def list_tools(self) -> List[Tool]:
        """
        List available tools from the server and cache them.

        Returns:
            List of Tool instances

        Raises:
            RuntimeError: If the server session is not initialized
        """
        # Check if session is initialized
        if not self.session:
            raise RuntimeError(f"Server {self.name} session not initialized")

        # Return cached tools if already fetched
        if self.tools is not None:
            logging.debug(f"Returning cached tools for {self.name}.")
            return self.tools

        # Fetch tools from the server
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

        # Cache the fetched tools
        self.tools = fetched_tools
        logging.debug(f"Cached {len(self.tools)} tools for {self.name}.")
        return self.tools

    def get_tool_schema(self, original_tool_name: str) -> Optional[Dict[str, Any]]:
        """
        Get the input schema for a specific tool by its original name.

        Args:
            original_tool_name: Original name of the tool (without server prefix)

        Returns:
            Input schema dictionary or None if tool not found
        """
        if self.tools is None:
            logging.warning(
                f"Attempted to get schema for {original_tool_name} on {self.name}, but tools list is empty or not fetched yet."
            )
            return None

        # Find the tool by its original name
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
        arguments: Dict[str, Any],
    ) -> Any:
        """
        Execute a tool.

        Args:
            tool_name: Name of the tool to execute (original name without prefix)
            arguments: Tool arguments

        Returns:
            Tool execution result or error object with isError=True

        Notes:
            This method now catches all exceptions and returns them as error objects
            rather than raising exceptions, to improve resilience and allow the
            LLM to handle errors gracefully.
        """
        if not self.session:
            logging.error(f"Server {self.name} session not initialized")
            # Return an error object instead of raising an exception
            return type(
                "ErrorObject",
                (),
                {
                    "isError": True,
                    "content": f"Error: Server {self.name} session not initialized",
                },
            )()

        try:
            logging.debug(f"Executing {tool_name} via MCP session...")
            result = await self.session.call_tool(tool_name, arguments)
            return result
        except Exception as e:
            logging.error(
                f"Exception executing tool {tool_name} on server {self.name}: {e}",
                exc_info=True,
            )
            # Return error as an object with isError flag instead of raising
            return type(
                "ErrorObject",
                (),
                {
                    "isError": True,
                    "content": f"Error executing tool {tool_name}: {str(e)}",
                },
            )()


async def setup_servers(server_config: Dict[str, Any]) -> List[Server]:
    """
    Initialize all servers defined in the configuration.

    Args:
        server_config: Server configuration dictionary

    Returns:
        List of initialized Server instances
    """
    servers_to_init = [
        Server(name, srv_config)
        for name, srv_config in server_config.get("mcpServers", {}).items()
    ]

    if not servers_to_init:
        logging.error("No mcpServers defined in the configuration file.")
        return []

    initialized_servers: List[Server] = []

    # AsyncExitStack would be used in the calling code to manage contexts

    return servers_to_init  # Return servers to be initialized in the orchestrator
