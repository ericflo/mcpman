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
import textwrap
import os
import threading
import itertools
import sys
from typing import Dict, List, Any, Optional, Tuple, Union

from colorama import Fore, Back, Style, init

from .server import Server
from .llm_client import LLMClient
from .tools import Tool, sanitize_name
from .config import DEFAULT_VERIFICATION_MESSAGE
from .models import Conversation, Message, ToolCall, ToolResult

# Initialize colorama
init(autoreset=True)

# Terminal width detection
def get_terminal_width():
    """Get terminal width or a reasonable default"""
    try:
        # Get actual terminal width but cap it to avoid extreme stretching
        width = os.get_terminal_size().columns
        return min(width, 120) # Cap at 120 chars for readability 
    except (OSError, AttributeError):
        return 80

# Output formatting helpers
def format_tool_call(name: str, args: str) -> str:
    """Format a tool call with colors and indentation"""
    terminal_width = get_terminal_width()
    indent = "  "
    
    # Truncate arguments if too long
    max_args_length = terminal_width - len(indent) - len(name) - 20
    if len(args) > max_args_length and max_args_length > 30:
        display_args = args[:max_args_length-20] + "..." + args[-15:]
    else:
        display_args = args
        
    return f"{Fore.CYAN}➤ {Fore.GREEN}{name}{Style.RESET_ALL}({Fore.YELLOW}{display_args}{Style.RESET_ALL})"

def format_tool_response(name: str, response: str) -> str:
    """Format a tool response with colors and indentation"""
    terminal_width = get_terminal_width()
    
    # Clean and normalize the response
    # Remove any control characters or weird whitespace
    import re
    clean_response = re.sub(r'[\x00-\x1F\x7F]', '', response)
    clean_response = re.sub(r'\s+', ' ', clean_response).strip()
    
    # Determine if response needs truncation
    is_error = clean_response.startswith("Error:")
    color = Fore.RED if is_error else Fore.WHITE
    
    # Calculate available width for response text
    prefix = f"{Fore.BLUE}← {Fore.GREEN}{name}{Style.RESET_ALL}: "
    prefix_visible_length = len(name) + 4  # Account for the arrow, colon and spaces
    
    # Calculate max response length to display
    max_resp_length = terminal_width - prefix_visible_length - 5  # Leave some margin
    
    # Apply intelligent truncation
    if len(clean_response) > max_resp_length:
        if max_resp_length > 60:
            # For longer terminals, show beginning and end
            start_len = int(max_resp_length * 0.6)  # Show more of the beginning
            end_len = max_resp_length - start_len - 5  # Space for ellipsis
            display_resp = clean_response[:start_len] + "..." + clean_response[-end_len:]
        else:
            # For narrow terminals, just show beginning with ellipsis
            display_resp = clean_response[:max_resp_length-5] + "..."
    else:
        display_resp = clean_response
    
    # Format the final response
    return f"{prefix}{color}{display_resp}{Style.RESET_ALL}"

def format_llm_response(content: str, is_final: bool = False) -> str:
    """Format an LLM response with proper indentation and wrapping"""
    terminal_width = get_terminal_width()
    indent = "  "
    
    # Calculate box width based on terminal size (but not too wide)
    box_width = min(terminal_width - 4, 80)  # Keep reasonable width for readability

    # First, clean and normalize the content
    import re
    # Remove control characters and normalize whitespace within lines
    clean_content = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', content)
    
    # Normalize line endings
    clean_content = re.sub(r'\r\n?', '\n', clean_content)
    
    # Remove excessive whitespace but preserve paragraph structure
    paragraphs = clean_content.split('\n\n')
    normalized_paragraphs = []
    for para in paragraphs:
        # Normalize internal whitespace in each paragraph
        norm_para = re.sub(r'\s+', ' ', para).strip()
        if norm_para:  # Only add non-empty paragraphs
            normalized_paragraphs.append(norm_para)
            
    # Rejoin with preserved paragraph structure
    normalized_content = '\n\n'.join(normalized_paragraphs)
    
    # Format box title
    title = "FINAL ANSWER" if is_final else "POTENTIAL ANSWER"
    
    # Calculate padding to center title precisely
    title_padding_left = (box_width - len(title) - 4) // 2
    title_padding_right = box_width - len(title) - 4 - title_padding_left
    
    # Ensure minimum padding
    title_padding_left = max(title_padding_left, 2)
    title_padding_right = max(title_padding_right, 2)
        
    # Format header, title, and footer with precisely calculated width
    if is_final:
        header = f"\n{Fore.GREEN}╔{'═' * (box_width - 2)}╗{Style.RESET_ALL}"
        title_line = f"{Fore.GREEN}║{' ' * title_padding_left}{Fore.WHITE}{title}{Style.RESET_ALL}{' ' * title_padding_right}{Fore.GREEN}║{Style.RESET_ALL}"
        separator = f"{Fore.GREEN}╠{'═' * (box_width - 2)}╣{Style.RESET_ALL}"
        footer = f"{Fore.GREEN}╚{'═' * (box_width - 2)}╝{Style.RESET_ALL}"
    else:
        header = f"\n{Fore.YELLOW}╭{'─' * (box_width - 2)}╮{Style.RESET_ALL}"
        title_line = f"{Fore.YELLOW}│{' ' * title_padding_left}{Fore.WHITE}{title}{Style.RESET_ALL}{' ' * title_padding_right}{Fore.YELLOW}│{Style.RESET_ALL}"
        separator = f"{Fore.YELLOW}├{'─' * (box_width - 2)}┤{Style.RESET_ALL}"
        footer = f"{Fore.YELLOW}╰{'─' * (box_width - 2)}╯{Style.RESET_ALL}"
    
    # Calculate visible length (excluding ANSI color codes)
    def visible_length(s):
        # Remove ANSI escape sequences
        import re
        ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
        return len(ansi_escape.sub('', s))
    
    # Adjust content width to fit within box
    content_width = box_width - 8  # Allow for borders and consistent padding
    
    # Wrap each paragraph separately to preserve structure
    all_wrapped_paragraphs = []
    for paragraph in normalized_content.split('\n\n'):
        if paragraph.strip():
            # Wrap the paragraph with consistent indentation
            wrapped_paragraph = textwrap.fill(
                paragraph,
                width=content_width,
                initial_indent=indent,
                subsequent_indent=indent
            )
            all_wrapped_paragraphs.append(wrapped_paragraph)
        else:
            # Add empty paragraph (just indentation) to preserve structure
            all_wrapped_paragraphs.append(indent)
    
    # Join paragraphs with blank lines between them
    wrapped_content = "\n\n".join(all_wrapped_paragraphs)
    
    # Format each line with proper border alignment
    content_lines = []
    
    # Process each line, including blank lines
    for line in wrapped_content.split('\n'):
        # Calculate padding precisely based on visible content length (without ANSI colors)
        line_visible_len = visible_length(line)
        
        # Calculate the padding needed for consistent alignment
        # We need to account for the difference between total and visible length
        padding_needed = max(0, box_width - line_visible_len - 4)  # -4 for border chars and spacing
        padding = ' ' * padding_needed
        
        # Add border characters with proper color based on answer type
        if is_final:
            content_lines.append(f"{Fore.GREEN}║{Style.RESET_ALL} {Fore.WHITE}{line.rstrip()}{Style.RESET_ALL}{padding} {Fore.GREEN}║{Style.RESET_ALL}")
        else:
            content_lines.append(f"{Fore.YELLOW}│{Style.RESET_ALL} {Fore.WHITE}{line.rstrip()}{Style.RESET_ALL}{padding} {Fore.YELLOW}│{Style.RESET_ALL}")
    
    formatted_content = "\n".join(content_lines)
    
    # Assemble the complete box with proper spacing
    return f"{header}\n{title_line}\n{separator}\n{formatted_content}\n{footer}"

def format_verification_result(passed: bool, feedback: str) -> str:
    """Format verification result with colors"""
    if passed:
        return f"\n{Fore.GREEN}✓ VERIFICATION PASSED:{Style.RESET_ALL} {feedback}"
    else:
        return f"\n{Fore.RED}✗ VERIFICATION FAILED:{Style.RESET_ALL} {feedback}"

def format_processing_step(step: str) -> str:
    """Format a processing step with subtle styling"""
    return f"{Fore.BLUE}■ {Fore.CYAN}{step}...{Style.RESET_ALL}"


class ProgressSpinner:
    """Displays an animated spinner in the console during long operations"""
    
    def __init__(self, message: str, colors: bool = True):
        self.message = message
        self.stop_event = threading.Event()
        self.spinner = itertools.cycle(['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'])
        self.spinner_thread = None
        self.colors = colors
        
    def spin(self):
        while not self.stop_event.is_set():
            prefix = f"{Fore.BLUE}" if self.colors else ""
            suffix = f"{Style.RESET_ALL}" if self.colors else ""
            sys.stdout.write(f"\r{prefix}{next(self.spinner)} {self.message}{suffix} ")
            sys.stdout.flush()
            time.sleep(0.1)
            
    def __enter__(self):
        self.spinner_thread = threading.Thread(target=self.spin)
        self.spinner_thread.daemon = True
        self.spinner_thread.start()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop_event.set()
        if self.spinner_thread:
            self.spinner_thread.join(timeout=0.5)
        sys.stdout.write("\r" + " " * (len(self.message) + 10) + "\r")  # Clear the line
        sys.stdout.flush()


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

        # Log the tool call
        if not output_only:
            formatted_tool_call = format_tool_call(prefixed_tool_name, str(tool_call.arguments))
            print(formatted_tool_call, flush=True)
        logging.info(
            f"Executing tool: {prefixed_tool_name}",
            extra={"event": "tool_call", "tool": prefixed_tool_name},
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

        # Log the tool response
        if not output_only:
            formatted_response = format_tool_response(prefixed_tool_name, execution_result_content)
            print(formatted_response, flush=True)

        logging.info(
            f"Tool response: {prefixed_tool_name}",
            extra={
                "event": "tool_response",
                "success": not execution_result_content.startswith("Error:"),
                "time_ms": round(execution_time_ms),
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
        self, tool_calls: List[ToolCall], servers: List[Server], output_only: bool = False
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
    ) -> Tuple[bool, str]:
        """Verify if the task has been completed successfully."""
        try:
            # Create verification request
            verification_messages, verification_schema = (
                self._create_verification_request(conversation, verification_prompt)
            )

            # Call the LLM with the verification tool
            verification_response = llm_client.get_response(
                verification_messages,
                verification_schema,
                temperature=temperature,
                tool_choice={
                    "type": "function",
                    "function": {"name": "verify_completion"},
                },
            )

            # Extract the verification result
            verification_result = None
            if (
                "tool_calls" in verification_response
                and verification_response["tool_calls"]
            ):
                tool_call = verification_response["tool_calls"][0]
                if tool_call["function"]["name"] != "verify_completion":
                    return False, "Verification failed: Wrong function called."
                verification_result = json.loads(tool_call["function"]["arguments"])

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

            # Log verification result (simplified)
            logging.info(
                "Task verification result",
                extra={
                    "event": "verification",
                    "is_complete": is_complete,
                    "feedback": feedback,
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

        # Convert tools to OpenAI schema
        openai_tools = [tool.to_openai_schema() for tool in all_tools]
        logging.debug(f"Prepared {len(openai_tools)} tools for the API.")

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
                        ),
                    )
            
            assistant_response_dict = await call_llm_with_spinner()
            
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
                        print(format_processing_step("Verifying task completion"), flush=True)

                    # Run verification with spinner
                    async def verify_with_spinner():
                        if not output_only:
                            with ProgressSpinner("Verifying"):
                                return await self._verify_completion(
                                    conversation, llm_client, verification_prompt
                                )
                        else:
                            return await self._verify_completion(
                                conversation, llm_client, verification_prompt
                            )
                    
                    is_complete, feedback = await verify_with_spinner()

                    if is_complete:
                        # Task is complete
                        if output_only:
                            # In output-only mode, just print the clean content without headers
                            print(content.strip())
                        else:
                            # Show the final answer with nice formatting
                            formatted_final = format_llm_response(content, is_final=True)
                            print(formatted_final, flush=True)
                            print(format_verification_result(True, feedback), flush=True)
                        break
                    else:
                        # Task is not complete, continue with feedback
                        if not output_only:
                            print(format_verification_result(False, feedback), flush=True)
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
                            formatted_final = format_llm_response(content, is_final=True)
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
                    flush=True
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
    output_only: bool = False
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
    """
    from .config import load_server_config

    # Load server configuration
    try:
        server_config = load_server_config(config_path)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logging.error(f"Failed to load server configuration: {e}")
        return

    # Create server instances
    servers_to_init = [
        Server(name, srv_config)
        for name, srv_config in server_config.get("mcpServers", {}).items()
    ]

    if not servers_to_init:
        logging.error("No mcpServers defined in the configuration file.")
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
                        if logging.getLogger().isEnabledFor(logging.DEBUG) and not output_only:
                            if server_tools:
                                # Group tools visually in a nice box
                                terminal_width = get_terminal_width()
                                # Calculate box width based on content and terminal width
                                # Base it on the longest tool name to ensure better alignment
                                tool_names = [tool.name for tool in server_tools]
                                max_name_len = max(len(name) for name in tool_names) if tool_names else 0
                                
                                # Determine box width - ensure it fits in terminal but is wide enough for content
                                # Add 15 for margins and decorations                               
                                min_width = max_name_len + 20
                                # Ensure box is wide enough to fit the title plus some padding
                                title_min_width = len(server.name) + 45
                                # Take the larger of tool width and title width, but cap to terminal size
                                box_width = min(terminal_width - 6, max(min_width, title_min_width))
                                
                                # Create the box borders
                                top_border = f"  {Fore.MAGENTA}╔{'═' * (box_width - 4)}╗{Style.RESET_ALL}"
                                bottom_border = f"  {Fore.MAGENTA}╚{'═' * (box_width - 4)}╝{Style.RESET_ALL}"
                                
                                # Create the title with properly calculated padding
                                title_text = f"{Fore.GREEN}Server '{server.name}'{Style.RESET_ALL} initialized with {Fore.CYAN}{len(server_tools)}{Style.RESET_ALL} tools:"
                                title_len = len(server.name) + len(str(len(server_tools))) + 26  # Estimate visible length
                                padding = ' ' * max(0, box_width - title_len - 7)  # -7 for the magenta borders and space
                                title = f"  {Fore.MAGENTA}║{Style.RESET_ALL} {title_text}{padding}{Fore.MAGENTA}║{Style.RESET_ALL}"
                                
                                # Show box with tools
                                print(top_border)
                                print(title)
                                print(f"  {Fore.MAGENTA}╠{'═' * (box_width - 4)}╣{Style.RESET_ALL}")
                                
                                # Always organize tools in columns for consistent display
                                # Calculate columns based on terminal width and tool name length
                                tool_names = [tool.name for tool in server_tools]
                                
                                # Add some padding between columns
                                column_padding = 4
                                # Each column needs space for the name plus arrow indicator
                                column_width = max_name_len + column_padding
                                                                
                                # Calculate optimal number of columns that can fit
                                # Box_width - 8 accounts for left/right borders and margins
                                usable_width = box_width - 8
                                cols = max(1, min(3, usable_width // column_width))
                                
                                # Recalculate actual column width to distribute space evenly
                                column_width = usable_width // cols
                                
                                # Prepare rows
                                rows = []
                                for i in range(0, len(tool_names), cols):
                                    chunk = tool_names[i:i+cols]
                                    row = []
                                    for name in chunk:
                                        # Truncate tool name if it's too long for the column
                                        display_name = name
                                        if len(name) > column_width - column_padding:
                                            display_name = name[:column_width - column_padding - 3] + "..."
                                        row.append(f"{Fore.CYAN}▸ {Fore.WHITE}{display_name:<{column_width - 3}}{Style.RESET_ALL}")
                                    rows.append(row)
                                
                                # Print rows
                                for row in rows:
                                    line = f"  {Fore.MAGENTA}║{Style.RESET_ALL} " + "".join(row)
                                    # Calculate remaining padding needed
                                    line_length = usable_width  # This is what we allocated for the columns
                                    line_padding = ' ' * max(0, usable_width - line_length)
                                    print(f"{line}{line_padding}{Fore.MAGENTA}║{Style.RESET_ALL}")
                                
                                print(bottom_border)
                            else:
                                print(f"  {Fore.MAGENTA}Server{Style.RESET_ALL} {Fore.GREEN}'{server.name}'{Style.RESET_ALL} initialized with {Fore.RED}no tools{Style.RESET_ALL}")
                    except Exception as list_tools_e:
                        logging.warning(
                            f"Could not list tools for {server.name} after init: {list_tools_e}"
                        )

                except Exception as e:
                    logging.error(
                        f"Failed to initialize server {server.name}: {e}", exc_info=True
                    )
                    return

            # Exit if initialization failed
            if not initialized_servers:
                logging.error("No servers were initialized successfully. Exiting.")
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
                    terminal_width = get_terminal_width()
                    # Keep box at a reasonable width for readability
                    box_width = min(terminal_width - 4, 100)
                    
                    # Format box borders with calculated width
                    header = f"{Fore.MAGENTA}┌{'─' * (box_width - 2)}┐{Style.RESET_ALL}"
                    footer = f"{Fore.MAGENTA}└{'─' * (box_width - 2)}┘{Style.RESET_ALL}"
                    
                    # Sanitize and normalize prompt text
                    # Remove any problematic characters and normalize whitespace
                    import re
                    
                    # First normalize all line endings and collapse multiple spaces
                    normalized_prompt = re.sub(r'\r\n?', '\n', user_prompt)
                    normalized_prompt = re.sub(r' +', ' ', normalized_prompt)
                    
                    # Replace any tab characters with spaces
                    normalized_prompt = normalized_prompt.replace('\t', '    ')
                    
                    # Collapse multiple newlines and strip
                    normalized_prompt = re.sub(r'\n\s*\n+', '\n\n', normalized_prompt).strip()
                    
                    # Clean text for display - treat as a single paragraph
                    clean_prompt = " ".join(normalized_prompt.split())
                    
                    # Now wrap as a single paragraph for clean display
                    wrapped_prompt = textwrap.fill(
                        clean_prompt,
                        width=box_width-6,  # Allow space for borders and padding
                        initial_indent="  ",
                        subsequent_indent="  "
                    )
                    
                    # Format the title line with proper indent
                    title_line = f"{Fore.MAGENTA}│{Style.RESET_ALL} {Fore.YELLOW}Running prompt:{Style.RESET_ALL}"
                    title_padding = ' ' * (box_width - 17)  # "Running prompt:" is 16 chars
                    title_line += f"{title_padding}{Fore.MAGENTA}│{Style.RESET_ALL}"
                    
                    # Print the completed box
                    print(header)
                    print(title_line)
                    print(f"{Fore.MAGENTA}├{'─' * (box_width - 2)}┤{Style.RESET_ALL}")
                    
                    # With the wrapped prompt as a single string, format with correct borders
                    lines = wrapped_prompt.split('\n')
                    
                    # Calculate visible length (excluding ANSI color codes)
                    def visible_length(s):
                        # Remove ANSI escape sequences
                        import re
                        ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
                        return len(ansi_escape.sub('', s))
                    
                    # Print each line with proper padding to align right border
                    for line in lines:
                        # Calculate required padding to align the right border
                        visible_len = visible_length(line)
                        padding_needed = max(0, box_width - visible_len - 4)  # -4 for border chars and spacing
                        padding = ' ' * padding_needed
                        
                        # Print line with consistent borders
                        print(f"{Fore.MAGENTA}│{Style.RESET_ALL} {Fore.WHITE}{line.strip()}{Style.RESET_ALL}{padding} {Fore.MAGENTA}│{Style.RESET_ALL}")
                    
                    print(footer)
                else:
                    # Format shorter version with same sanitization approach as in debug mode
                    import re
                    
                    # First normalize all line endings and collapse multiple spaces
                    normalized_prompt = re.sub(r'\r\n?', '\n', user_prompt)
                    normalized_prompt = re.sub(r' +', ' ', normalized_prompt)
                    
                    # Replace any tab characters with spaces
                    normalized_prompt = normalized_prompt.replace('\t', '    ')
                    
                    # Clean text for display - treat as a single paragraph
                    # This collapses all whitespace including newlines
                    clean_prompt = " ".join(normalized_prompt.split())
                    
                    # Create a truncated version for display
                    max_display_len = 70
                    if len(clean_prompt) > max_display_len:
                        short_prompt = clean_prompt[:max_display_len-3] + "..."
                    else:
                        short_prompt = clean_prompt
                    
                    # Print with nice formatting
                    print(f"{Fore.CYAN}┌─{Style.RESET_ALL} {Fore.YELLOW}Processing request:{Style.RESET_ALL}")
                    print(f"{Fore.CYAN}└─►{Style.RESET_ALL} {Fore.WHITE}{short_prompt}{Style.RESET_ALL}")

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
            )

    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}", exc_info=True)
    finally:
        logging.info("Application finished.")
