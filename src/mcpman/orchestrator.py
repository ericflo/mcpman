"""
Orchestrator for MCPMan.

This module coordinates the agent loop, tool execution, and verification,
with a streamlined architecture that balances abstraction with simplicity.
"""

import json
import logging
import asyncio
import contextlib
import datetime
import time
import os
from typing import Dict, List, Any, Optional, Tuple
from colorama import Fore, Style

from .server import Server
from .llm_client import LLMClient
from .llm_clients.openai_compat import OpenAICompatClient
from .llm_clients.openai_responses import OpenAIResponsesClient
from .tools import sanitize_name
from .config import DEFAULT_VERIFICATION_MESSAGE
from .models import Conversation, Message, ToolCall, ToolResult
from .formatting import (
    format_tool_call,
    format_tool_response,
    format_llm_response,
    format_verification_result,
    format_processing_step,
    format_tool_list,
    ProgressSpinner,
    print_box,
    print_short_prompt,
    BoxStyle,
)
from .logger import (
    get_logger,
    log_tool_call,
    log_tool_response,
    log_verification,
)


class Orchestrator:
    """Main orchestrator for the agent loop."""

    def __init__(
        self, default_verification_message: str = DEFAULT_VERIFICATION_MESSAGE
    ):
        self.verification_message = default_verification_message

    async def _execute_tool(
        self, tool_call: ToolCall, servers: List[Server], output_only: bool = False
    ) -> ToolResult:
        """Execute a single tool call and return the result."""
        prefixed_tool_name = tool_call.function_name

        # Parse the tool name to extract server name and original tool name
        target_server_name = None
        original_tool_name = None

        # Sort server names by length (descending) to handle potential prefix conflicts
        sanitized_server_names = sorted(
            [sanitize_name(s.name) for s in servers], key=len, reverse=True
        )

        # Find the server prefix
        for s_name in sanitized_server_names:
            prefix = f"{s_name}_"
            if prefixed_tool_name.startswith(prefix):
                target_server_name = s_name
                original_tool_name = prefixed_tool_name[len(prefix) :]
                break

        # Update the tool call with parsed information
        tool_call.server_name = target_server_name
        tool_call.original_tool_name = original_tool_name

        # Handle parsing failures
        if not target_server_name or not original_tool_name:
            logging.error(
                f"Could not parse server and tool name from '{prefixed_tool_name}'"
            )
            return ToolResult(
                tool_call_id=tool_call.id,
                name=prefixed_tool_name,
                content=f"Error: Invalid prefixed tool name format '{prefixed_tool_name}'",
                success=False,
            )

        # Find the target server
        target_server = next(
            (s for s in servers if sanitize_name(s.name) == target_server_name), None
        )

        # Handle server not found
        if not target_server:
            logging.warning(
                f"Target server '{target_server_name}' for tool '{prefixed_tool_name}' not found."
            )
            return ToolResult(
                tool_call_id=tool_call.id,
                name=prefixed_tool_name,
                content=f"Error: Server '{target_server_name}' (sanitized) not found.",
                success=False,
            )

        # Log the tool call with enhanced logging
        if not output_only:
            formatted_tool_call = format_tool_call(
                prefixed_tool_name, str(tool_call.arguments)
            )
            print(formatted_tool_call, flush=True)

        # Get task name from the tool call if available
        task_name = getattr(tool_call, "task_name", f"Task-{os.getpid()}")
        run_id = getattr(tool_call, "run_id", None)

        # Log with enhanced structured logging
        logger = get_logger()
        log_tool_call(
            logger,
            tool_name=prefixed_tool_name,
            parameters=tool_call.arguments,
            taskName=task_name,
            extra={
                "run_id": run_id,
                "tool_call_id": tool_call.id,
                "server_name": target_server_name,
                "original_tool_name": original_tool_name,
                "full_arguments": tool_call.arguments,
            },
        )

        # Initialize execution tracking
        execution_result_content = f"Error: Tool '{original_tool_name}' execution failed on server '{target_server.name}'."
        execution_time_ms = 0

        # Execute the tool with timing measurements
        start_time = time.time()

        try:
            tool_output = await target_server.execute_tool(
                original_tool_name, tool_call.arguments
            )

            # Capture execution time
            execution_time_ms = (time.time() - start_time) * 1000

            # Format the result
            if hasattr(tool_output, "isError") and tool_output.isError:
                error_detail = getattr(tool_output, "content", "Unknown tool error")
                logging.warning(
                    f"Tool '{prefixed_tool_name}' reported an error: {error_detail}"
                )

                # Check if it's an 'unknown tool' error
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

        except Exception as e:
            logging.error(
                f"Exception executing tool '{prefixed_tool_name}': {e}", exc_info=True
            )
            execution_result_content = f"Error: Tool execution failed: {e}"
            execution_time_ms = (time.time() - start_time) * 1000

        # Log the tool response with enhanced logging
        if not output_only:
            formatted_response = format_tool_response(
                prefixed_tool_name, execution_result_content
            )
            print(formatted_response, flush=True)

        # Get logger and log the response with enhanced structured logging
        success = not execution_result_content.startswith("Error:")
        logger = get_logger()
        log_tool_response(
            logger,
            tool_name=prefixed_tool_name,
            response=execution_result_content,
            success=success,
            time_ms=round(execution_time_ms),
            taskName=task_name,
            extra={
                "run_id": run_id,
                "tool_call_id": tool_call.id,
                "server_name": target_server_name,
                "original_tool_name": original_tool_name,
                "full_response": execution_result_content,
                "execution_time_ms": execution_time_ms,
            },
        )

        # Create and return the tool result
        return ToolResult(
            tool_call_id=tool_call.id,
            name=prefixed_tool_name,
            content=str(execution_result_content),
            success=not execution_result_content.startswith("Error:"),
            execution_time_ms=execution_time_ms,
        )

    async def _execute_tools(
        self,
        tool_calls: List[ToolCall],
        servers: List[Server],
        output_only: bool = False,
    ) -> List[ToolResult]:
        """Execute multiple tool calls and return the results."""
        results = []
        for tool_call in tool_calls:
            if tool_call.type == "function":
                result = await self._execute_tool(tool_call, servers, output_only)
                results.append(result)
            else:
                logging.warning(f"Unsupported tool call type: {tool_call.type}")
                results.append(
                    ToolResult(
                        tool_call_id=tool_call.id,
                        name=tool_call.function_name,
                        content=f"Error: Unsupported tool type '{tool_call.type}'",
                        success=False,
                    )
                )
        return results

    def _create_verification_request(
        self, conversation: Conversation, custom_prompt: Optional[str] = None
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Create a verification request for the LLM."""
        verification_message = custom_prompt or self.verification_message

        # Define schema for verify_completion function
        verification_schema = [
            {
                "type": "function",
                "function": {
                    "name": "verify_completion",
                    "description": "Verify if the task has been fully completed and provide feedback",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "thoughts": {
                                "type": "string",
                                "description": "Detailed analysis of the conversation and task completion",
                            },
                            "is_complete": {
                                "type": "boolean",
                                "description": "Whether the task has been fully completed",
                            },
                            "summary": {
                                "type": "string",
                                "description": "Summary of what was accomplished",
                            },
                            "missing_steps": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "List of steps or aspects that are not yet complete",
                            },
                            "suggestions": {
                                "type": "string",
                                "description": "Constructive suggestions for the agent if the task is not complete",
                            },
                        },
                        "required": ["thoughts", "is_complete", "summary"],
                    },
                },
            }
        ]

        # Create serializable messages for verification
        serializable_messages = []
        for msg in conversation.messages:
            msg_dict = msg.to_dict()
            msg_copy = {}
            for key, value in msg_dict.items():
                if isinstance(value, (dict, list)):
                    try:
                        msg_copy[key] = json.dumps(value)
                    except:
                        msg_copy[key] = str(value)
                else:
                    msg_copy[key] = value
            serializable_messages.append(msg_copy)

        # Format the verification request
        verification_messages = [
            {"role": "system", "content": verification_message},
            {
                "role": "user",
                "content": "Below is a conversation between a user and an agent with tools. "
                "Evaluate if the agent has fully completed the user's request:\n\n"
                + json.dumps(serializable_messages, indent=2),
            },
        ]

        return verification_messages, verification_schema

    async def _verify_completion(
        self,
        conversation: Conversation,
        llm_client: LLMClient,
        verification_prompt: Optional[str] = None,
        temperature: float = 0.4,
        run_id: Optional[str] = None,
    ) -> Tuple[bool, str]:
        """Verify if the task has been completed successfully."""
        try:
            # Create verification request
            verification_messages, verification_schema = (
                self._create_verification_request(conversation, verification_prompt)
            )

            # Call the LLM with the verification tool
            tool_choice = None
            if isinstance(llm_client, OpenAICompatClient):
                tool_choice = "auto"
            elif isinstance(llm_client, OpenAIResponsesClient):
                tool_choice = {
                    "type": "function",
                    "function": {"name": "verify_completion"},
                }
            verification_response = llm_client.get_response(
                verification_messages,
                verification_schema,
                temperature=temperature,
                tool_choice=tool_choice,
                run_id=run_id,
            )

            # Extract the verification result
            verification_result = None
            try:
                if (
                    "tool_calls" in verification_response
                    and verification_response["tool_calls"]
                ):
                    tool_call = verification_response["tool_calls"][0]
                    if tool_call["function"]["name"] != "verify_completion":
                        return False, "Verification failed: Wrong function called."
                    try:
                        verification_result = json.loads(
                            tool_call["function"]["arguments"]
                        )
                    except json.JSONDecodeError as json_e:
                        logging.error(f"Error parsing verification arguments: {json_e}")
                        # Try to recover with a basic structure
                        verification_result = {
                            "is_complete": False,
                            "thoughts": f"Error parsing verification result: {json_e}",
                            "summary": "Verification could not be completed due to parsing error.",
                        }
            except Exception as e:
                logging.error(
                    f"Error extracting verification result: {e}", exc_info=True
                )
                # Create a default failure result
                verification_result = {
                    "is_complete": False,
                    "thoughts": f"Error during verification: {e}",
                    "summary": "Verification process encountered an unexpected error.",
                }

            # If no result found
            if not verification_result:
                return (
                    False,
                    "Verification failed: Could not determine if task is complete.",
                )

            # Check completion status
            is_complete = verification_result.get("is_complete", False)

            # Format feedback based on completion status
            if is_complete:
                feedback = verification_result.get(
                    "summary", "Task completed successfully."
                )
            else:
                missing_steps = verification_result.get("missing_steps", [])
                missing_steps_str = (
                    ", ".join(missing_steps)
                    if missing_steps
                    else "Unknown missing steps"
                )
                suggestions = verification_result.get("suggestions", "")
                feedback = f"The task is not yet complete. Missing: {missing_steps_str}. {suggestions}"

            # Log verification result with enhanced structured logging
            logger = get_logger()
            task_name = getattr(conversation, "task_name", f"Task-{os.getpid()}")
            run_id = getattr(conversation, "run_id", None)
            log_verification(
                logger,
                is_complete=is_complete,
                feedback=feedback,
                taskName=task_name,
                extra={
                    "run_id": run_id,
                    "verification_result": verification_result,
                    "verification_details": {
                        "missing_steps": verification_result.get("missing_steps", []),
                        "suggestions": verification_result.get("suggestions", ""),
                        "summary": verification_result.get("summary", ""),
                    },
                },
            )

            return is_complete, feedback

        except Exception as e:
            logging.error(f"Error during task verification: {e}", exc_info=True)
            return False, f"Verification error: {str(e)}"

    async def run_agent(
        self,
        prompt: str,
        servers: List[Server],
        llm_client: LLMClient,
        system_message: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        max_turns: int = 2048,
        verify_completion: bool = False,
        verification_prompt: Optional[str] = None,
        output_only: bool = False,
        strict_tools: Optional[bool] = None,
        run_id: Optional[str] = None,
    ):
        """Run the agent loop with tools and optional verification."""
        # Initialize conversation
        conversation = Conversation(system_message=system_message, user_prompt=prompt)

        # Prepare tools for the API
        all_tools = []
        for server in servers:
            try:
                server_tools = await server.list_tools()
                all_tools.extend(server_tools)
            except Exception as e:
                logging.warning(f"Failed to list tools for server {server.name}: {e}")

        # Convert tools to OpenAI schema with strict mode setting
        openai_tools = [
            tool.to_openai_schema(strict=strict_tools) for tool in all_tools
        ]
        logging.debug(
            f"Prepared {len(openai_tools)} tools for the API with strict={strict_tools} (None means use default)"
        )

        # Get the event loop
        loop = asyncio.get_running_loop()

        # Run the agent loop
        for turn in range(max_turns):
            logging.debug(f"--- Turn {turn + 1} ---")

            # Get LLM response
            start_time = datetime.datetime.now()

            # Call the LLM with a progress spinner
            spinner_message = "Thinking" if not output_only else ""

            async def call_llm_with_spinner():
                try:
                    if not output_only:
                        with ProgressSpinner(spinner_message):
                            return await loop.run_in_executor(
                                None,
                                lambda: llm_client.get_response(
                                    conversation.to_dict_list(),
                                    openai_tools,
                                    temperature=temperature,
                                    max_tokens=max_tokens,
                                    tool_choice=None,
                                    run_id=run_id,
                                ),
                            )
                    else:
                        # No spinner in output-only mode
                        return await loop.run_in_executor(
                            None,
                            lambda: llm_client.get_response(
                                conversation.to_dict_list(),
                                openai_tools,
                                temperature=temperature,
                                max_tokens=max_tokens,
                                tool_choice=None,
                                run_id=run_id,
                            ),
                        )
                except Exception as e:
                    logging.error(f"LLM response error: {e}", exc_info=True)
                    # Return an error message that looks like a valid response
                    # This allows the flow to continue and gives the LLM a chance to recover
                    return {
                        "role": "assistant",
                        "content": f"I encountered an error while processing your request: {e}. Please try again or rephrase your query.",
                    }

            try:
                assistant_response_dict = await call_llm_with_spinner()
            except Exception as e:
                logging.error(
                    f"Unexpected error in LLM response handling: {e}", exc_info=True
                )
                # Create a fallback response to prevent process termination
                assistant_response_dict = {
                    "role": "assistant",
                    "content": f"An unexpected error occurred: {e}. Let me try a different approach.",
                }

            # Calculate elapsed time
            elapsed_time = (datetime.datetime.now() - start_time).total_seconds()

            # Convert the response to a Message object
            assistant_message = Message.from_dict(assistant_response_dict)

            # Add the assistant message to the conversation
            conversation.add_message(assistant_message)

            # Log the LLM response (simplified)
            logging.info(
                f"LLM response received ({elapsed_time:.2f}s)",
                extra={
                    "event": "llm_response",
                    "has_tool_calls": assistant_message.has_tool_calls,
                    "has_content": assistant_message.content is not None,
                    "response_time": elapsed_time,
                },
            )

            # Process tool calls if any
            if assistant_message.has_tool_calls:
                # Execute the tool calls
                tool_results = await self._execute_tools(
                    assistant_message.tool_calls, servers, output_only
                )

                # Add tool results to conversation
                for result in tool_results:
                    conversation.add_message(
                        Message(
                            role=result.role,
                            content=result.content,
                            tool_call_id=result.tool_call_id,
                            name=result.name,
                        )
                    )

                # Continue to next turn
                continue
            else:
                # No tool calls, check for completion
                content = assistant_message.content or ""

                # If verification is enabled
                if verify_completion:
                    if not output_only:
                        # Format the potential answer with pretty formatting
                        formatted_content = format_llm_response(content, is_final=False)
                        print(formatted_content, flush=True)
                        print(
                            format_processing_step("Verifying task completion"),
                            flush=True,
                        )

                    # Run verification with spinner
                    async def verify_with_spinner():
                        if not output_only:
                            with ProgressSpinner("Verifying"):
                                return await self._verify_completion(
                                    conversation,
                                    llm_client,
                                    verification_prompt,
                                    run_id=run_id,
                                )
                        else:
                            return await self._verify_completion(
                                conversation,
                                llm_client,
                                verification_prompt,
                                run_id=run_id,
                            )

                    is_complete, feedback = await verify_with_spinner()

                    if is_complete:
                        # Task is complete
                        if output_only:
                            # In output-only mode, just print the clean content without headers
                            print(content.strip())
                        else:
                            # Show the final answer with nice formatting
                            formatted_final = format_llm_response(
                                content, is_final=True
                            )
                            print(formatted_final, flush=True)
                            print(
                                format_verification_result(True, feedback), flush=True
                            )
                        break
                    else:
                        # Task is not complete, continue with feedback
                        if not output_only:
                            print(
                                format_verification_result(False, feedback), flush=True
                            )
                        conversation.add_user_message(
                            f"Your response is incomplete. {feedback} Please continue working on the task."
                        )
                        continue
                else:
                    # No verification, assume final answer
                    if content:
                        if output_only:
                            # In output-only mode, just print the content
                            print(content.strip())
                        else:
                            # Show final answer with pretty formatting
                            formatted_final = format_llm_response(
                                content, is_final=True
                            )
                            print(formatted_final, flush=True)
                    else:
                        if not output_only:
                            print(
                                f"\n{Fore.RED}⚠ WARNING:{Style.RESET_ALL} LLM provided no content in response",
                                flush=True,
                            )
                        logging.warning(
                            f"Final assistant message had no content: {assistant_message.to_dict()}"
                        )
                    break
        else:
            # Max turns reached
            if not output_only:
                print(
                    f"\n{Fore.RED}⚠ WARNING:{Style.RESET_ALL} Maximum turns ({max_turns}) reached without a final answer.",
                    flush=True,
                )
            logging.warning(f"Maximum turns ({max_turns}) reached without completion")


async def initialize_and_run(
    config_path: str,
    user_prompt: str,
    system_message: str,
    llm_client: LLMClient,
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
    max_turns: int = 2048,
    verify_completion: bool = False,
    verification_prompt: Optional[str] = None,
    provider_name: Optional[str] = None,  # Kept for backward compatibility
    output_only: bool = False,
    strict_tools: Optional[bool] = None,
    run_id: Optional[str] = None,
):
    """
    Initialize servers and run the agent loop.

    Args:
        config_path: Path to the server configuration file
        user_prompt: User prompt to execute
        system_message: System message to guide the LLM
        llm_client: LLM client for getting responses
        temperature: Sampling temperature for the LLM
        max_tokens: Maximum number of tokens for LLM responses
        max_turns: Maximum number of turns for the agent loop
        verify_completion: Whether to verify task completion before finishing
        verification_prompt: Custom system message for verification
        provider_name: Provider name for backward compatibility
        output_only: Whether to suppress UI output and only show final result
        strict_tools: Whether to use strict mode for tool schemas (None = use default)
        run_id: Optional run identifier for logging
    """
    from .config import load_server_config

    # Load server configuration
    try:
        server_config = load_server_config(config_path)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger = get_logger()
        logger.error(f"Failed to load server configuration: {e}")
        return

    # Create server instances
    servers_to_init = [
        Server(name, srv_config)
        for name, srv_config in server_config.get("mcpServers", {}).items()
    ]

    if not servers_to_init:
        logger = get_logger()
        logger.error("No mcpServers defined in the configuration file.")
        return

    # Initialize servers
    initialized_servers: List[Server] = []

    try:
        async with contextlib.AsyncExitStack() as stack:
            # Initialize each server
            for server in servers_to_init:
                try:
                    logging.debug(f"Initializing server {server.name}...")
                    stdio_client_cm = await server.initialize(output_only=output_only)

                    # Enter stdio client context manager using the stack
                    read, write = await stack.enter_async_context(stdio_client_cm)
                    server.read = read
                    server.write = write

                    # Create and enter session context manager using the stack
                    from mcp import ClientSession

                    session = ClientSession(read, write)
                    server.session = await stack.enter_async_context(session)

                    # Initialize the session
                    await server.session.initialize()
                    logging.info(f"Server {server.name} initialized successfully.")
                    initialized_servers.append(server)

                    # Print server tools
                    try:
                        server_tools = await server.list_tools()
                        # Only log tool details at debug level
                        logging.debug(
                            f"Server '{server.name}' initialized with {len(server_tools)} tools"
                        )
                        if (
                            logging.getLogger().isEnabledFor(logging.DEBUG)
                            and not output_only
                        ):
                            if server_tools:
                                # Use the centralized formatting function to display the tools box
                                tool_box_lines = format_tool_list(
                                    server.name, server_tools, indent=2
                                )
                                for line in tool_box_lines:
                                    print(line)
                            else:
                                print(
                                    f"  Server '{server.name}' initialized with no tools"
                                )
                    except Exception as list_tools_e:
                        logging.warning(
                            f"Could not list tools for {server.name} after init: {list_tools_e}"
                        )

                except Exception as e:
                    logging.error(
                        f"Failed to initialize server {server.name}: {e}", exc_info=True
                    )
                    # Continue with other servers instead of exiting entirely
                    # This allows partial functionality if some servers fail

            # Continue even if some servers failed, as long as at least one initialized
            if not initialized_servers:
                logging.error("No servers were initialized successfully.")
                # Create a dummy server with basic echo functionality
                logging.warning("Creating a fallback server to maintain operation...")
                try:
                    # Create an echo server that just returns arguments as a string
                    class FallbackServer:
                        def __init__(self):
                            self.name = "fallback"
                            self.session = None

                        async def list_tools(self):
                            from .tools import Tool

                            # Create a simple echo tool
                            return [
                                Tool(
                                    name="fallback_echo",
                                    description="Echo back the input (fallback tool when regular servers fail)",
                                    input_schema={
                                        "type": "object",
                                        "properties": {"message": {"type": "string"}},
                                    },
                                    original_name="echo",
                                )
                            ]

                        async def execute_tool(self, tool_name, arguments):
                            if tool_name == "echo":
                                return f"FALLBACK SERVER: {json.dumps(arguments)}"
                            return f"Unknown tool: {tool_name}"

                    fallback = FallbackServer()
                    initialized_servers.append(fallback)
                    logging.info("Fallback server created successfully.")
                except Exception as fallback_e:
                    logging.error(
                        f"Failed to create fallback server: {fallback_e}", exc_info=True
                    )
                    # Now we really have no choice but to return
                    return

            # Create orchestrator
            orchestrator = Orchestrator(
                default_verification_message=DEFAULT_VERIFICATION_MESSAGE
            )

            # Run the agent
            logging.info(f"Running prompt: {user_prompt}")

            # Only print if not in output-only mode
            if not output_only:
                # Only print the full prompt in debug mode
                if logging.getLogger().isEnabledFor(logging.DEBUG):
                    # Use the consolidated formatting module to display the prompt in a nice box
                    print_box("Running prompt", user_prompt, style=BoxStyle.PROMPT)
                else:
                    # Use the centralized formatting for short prompt display
                    print_short_prompt(user_prompt)

            try:
                await orchestrator.run_agent(
                    user_prompt,
                    initialized_servers,
                    llm_client,
                    system_message,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    max_turns=max_turns,
                    verify_completion=verify_completion,
                    verification_prompt=verification_prompt,
                    output_only=output_only,
                    strict_tools=strict_tools,
                    run_id=run_id,
                )
            except Exception as e:
                logger = get_logger()
                logger.error(f"Error during agent execution: {e}", exc_info=True)
                # Print error message for the user if not in output_only mode
                if not output_only:
                    print(
                        f"\n{Fore.RED}An error occurred during execution: {e}{Style.RESET_ALL}"
                    )
                    print("The error has been logged. Check logs for details.")

    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}", exc_info=True)
    finally:
        logging.info("Application finished.")
