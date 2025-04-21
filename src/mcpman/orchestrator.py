import json
import logging
import asyncio
import contextlib
import datetime
import time
from typing import Dict, List, Any, Optional, Tuple, Union

from .server import Server
from .llm_client import LLMClient
from .tools import Tool, sanitize_name
from .config import DEFAULT_VERIFICATION_MESSAGE


async def execute_tool_call(
    tool_call: Dict[str, Any], servers: List[Server]
) -> Dict[str, Any]:
    """
    Execute a single tool call and return the result message.

    Args:
        tool_call: Tool call object from the LLM
        servers: List of available servers

    Returns:
        Tool result message object for the LLM
    """
    prefixed_tool_name = tool_call["function"]["name"]
    tool_call_id = tool_call["id"]

    # Parse the prefixed name
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

    # Handle parsing failures
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
    target_server: Optional[Server] = next(
        (s for s in servers if sanitize_name(s.name) == target_server_name), None
    )

    # Handle server not found
    if not target_server:
        logging.warning(
            f"Target server '{target_server_name}' for tool '{prefixed_tool_name}' not found."
        )
        return {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "name": prefixed_tool_name,
            "content": f"Error: Server '{target_server_name}' (sanitized) not found.",
        }

    # Parse arguments
    arguments: Dict[str, Any] = {}
    try:
        arguments_str = tool_call["function"]["arguments"]
        arguments = json.loads(arguments_str)
    except json.JSONDecodeError as e:
        logging.error(f"Error decoding arguments JSON for {prefixed_tool_name}: {e}")
        return {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "name": prefixed_tool_name,
            "content": f"Error: Invalid arguments JSON: {e}",
        }

    # Log the tool call
    print(f"-> Calling tool: {prefixed_tool_name}({arguments})", flush=True)

    # Log to file in structured format
    logging.info(
        f"Executing tool: {prefixed_tool_name}",
        extra={
            "event_type": "tool_call",
            "tool_name": prefixed_tool_name,
            "original_tool_name": original_tool_name,
            "server_name": target_server_name,
            "arguments": arguments,
            "tool_call_id": tool_call_id,
        },
    )

    # Initialize result content and execution status
    execution_result_content = f"Error: Tool '{original_tool_name}' execution failed on server '{target_server.name}'."
    tool_found_on_target_server = False

    # Execute the tool with timing measurements
    start_time = datetime.datetime.now()
    execution_start_time = time.time()  # More precise for timing

    try:
        logging.debug(
            f"Executing {original_tool_name} on server {target_server.name} (sanitized: {target_server_name}, prefixed: {prefixed_tool_name})..."
        )
        tool_output = await target_server.execute_tool(original_tool_name, arguments)

        # Capture execution time
        execution_end_time = time.time()
        execution_time_ms = (execution_end_time - execution_start_time) * 1000
        tool_found_on_target_server = True

        # Format the result
        if hasattr(tool_output, "isError") and tool_output.isError:
            error_detail = (
                tool_output.content
                if hasattr(tool_output, "content")
                else "Unknown tool error"
            )
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

    # Log the tool response
    print(
        f"<- Tool Response [{prefixed_tool_name}]: {execution_result_content}",
        flush=True,
    )

    # Calculate total execution time including processing overhead
    end_time = datetime.datetime.now()
    total_duration_ms = (end_time - start_time).total_seconds() * 1000

    # Get execution time (from the actual tool execution)
    tool_execution_ms = locals().get(
        "execution_time_ms", 0
    )  # Get it if defined, 0 otherwise

    # Log to file in structured format with performance metrics
    logging.info(
        f"Tool response received: {prefixed_tool_name}",
        extra={
            "event_type": "tool_response",
            "tool_name": prefixed_tool_name,
            "original_tool_name": original_tool_name,
            "server_name": target_server_name,
            "tool_call_id": tool_call_id,
            "response": execution_result_content,
            "success": not execution_result_content.startswith("Error:"),
            "performance": {
                "total_duration_ms": round(total_duration_ms),
                "tool_execution_ms": round(tool_execution_ms),
                "overhead_ms": round(total_duration_ms - tool_execution_ms),
            },
            "execution_timestamp": end_time.isoformat(),
        },
    )

    # Return the result message
    return {
        "role": "tool",
        "tool_call_id": tool_call_id,
        "name": prefixed_tool_name,
        "content": str(execution_result_content),
    }


async def verify_task_completion(
    messages: List[Dict[str, Any]],
    llm_client: LLMClient,
    verification_prompt: Optional[str] = None,
    temperature: float = 0.4,  # Lower temperature for verification
) -> Tuple[bool, str]:
    """
    Verify if the agent has completed the task successfully.

    Args:
        messages: Conversation history
        llm_client: LLM client for verification
        verification_prompt: Custom system message for verification
        temperature: Temperature for the verification LLM call

    Returns:
        Tuple of (is_complete, feedback_message)
    """
    # Use default verification message if none provided
    verification_message = verification_prompt or DEFAULT_VERIFICATION_MESSAGE

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

    # Create a simplified, serializable version of the messages for verification
    serializable_messages = []
    for msg in messages:
        # Make a shallow copy to avoid modifying the original
        msg_copy = {}
        for key, value in msg.items():
            # Convert complex values to strings for serialization
            if isinstance(value, (dict, list)):
                try:
                    msg_copy[key] = json.dumps(value)
                except:
                    msg_copy[key] = str(value)
            else:
                msg_copy[key] = value
        serializable_messages.append(msg_copy)

    # Format the user request for verification
    verification_messages = [
        {"role": "system", "content": verification_message},
        {
            "role": "user",
            "content": "Below is a conversation between a user and an agent with tools. "
            "Evaluate if the agent has fully completed the user's request:\n\n"
            + json.dumps(serializable_messages, indent=2),
        },
    ]

    try:
        # Call the LLM with the verification tool
        verification_response = llm_client.get_response(
            verification_messages,
            verification_schema,
            temperature=temperature,
            tool_choice={"type": "function", "function": {"name": "verify_completion"}},
        )

        # Extract the tool call with verification results
        if (
            "tool_calls" not in verification_response
            or not verification_response["tool_calls"]
        ):
            logging.warning("No tool calls in verification response")
            return (
                False,
                "Verification failed: Could not determine if task is complete.",
            )

        tool_call = verification_response["tool_calls"][0]
        if tool_call["function"]["name"] != "verify_completion":
            logging.warning(
                f"Unexpected function name: {tool_call['function']['name']}"
            )
            return False, "Verification failed: Wrong function called."

        # Parse the verification result
        verification_result = json.loads(tool_call["function"]["arguments"])

        # Create a safely loggable version with a preview of the thoughts
        thoughts_text = verification_result.get("thoughts", "")
        thoughts_preview = (
            thoughts_text[:100] + "..." if len(thoughts_text) > 100 else thoughts_text
        )

        # Log the verification analysis (with a shorter preview in the message)
        logging.info(
            f"Completion verification analysis",
            extra={
                "event_type": "verification_analysis",
                "category": "verification",
                "verification_result": {
                    "is_complete": verification_result.get("is_complete", False),
                    "summary": verification_result.get("summary", ""),
                    "thoughts_preview": thoughts_preview,
                    "missing_steps_count": len(
                        verification_result.get("missing_steps", [])
                    ),
                    "has_suggestions": bool(verification_result.get("suggestions")),
                },
            },
        )

        # Extract completion status and feedback
        is_complete = verification_result.get("is_complete", False)

        if is_complete:
            summary = verification_result.get("summary", "Task completed successfully.")
            logging.info(
                f"Task completion verified. Summary: {summary}",
                extra={
                    "event_type": "verification_success",
                    "category": "verification",
                    "summary": summary,
                },
            )
            return True, summary
        else:
            # Format feedback for the agent
            missing_steps = verification_result.get("missing_steps", [])
            missing_steps_str = (
                ", ".join(missing_steps) if missing_steps else "Unknown missing steps"
            )

            suggestions = verification_result.get("suggestions", "")

            # Log the verification failure details
            logging.info(
                f"Verification determined task is incomplete",
                extra={
                    "event_type": "verification_failure",
                    "category": "verification",
                    "missing_steps": missing_steps,
                    "missing_steps_str": missing_steps_str,
                    "has_suggestions": bool(suggestions),
                },
            )
            feedback = f"The task is not yet complete. Missing: {missing_steps_str}. {suggestions}"
            logging.info(f"Task is incomplete. Feedback: {feedback}")
            return False, feedback

    except Exception as e:
        logging.error(f"Error during task verification: {e}", exc_info=True)
        return False, f"Verification error: {str(e)}"


async def run_agent(
    prompt: str,
    servers: List[Server],
    llm_client: LLMClient,
    system_message: str,
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
    max_turns: int = 2048,
    verify_completion: bool = False,
    verification_prompt: Optional[str] = None,
):
    """
    Run an agent loop to execute a prompt with tools.

    Args:
        prompt: User prompt to execute
        servers: List of available servers
        llm_client: LLM client for getting responses
        system_message: System message to guide the LLM
        temperature: Sampling temperature for the LLM
        max_tokens: Maximum number of tokens for LLM responses
        max_turns: Maximum number of turns for the agent loop
        verify_completion: Whether to verify task completion before finishing
        verification_prompt: Custom system message for verification
    """
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

    # Initialize conversation
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": prompt},
    ]

    # Log the initial conversation setup
    logging.info(
        "Conversation initialized",
        extra={
            "event_type": "conversation_init",
            "category": "conversation",
            "conversation": {"system_message": system_message, "user_prompt": prompt},
            "message_count": len(messages),
        },
    )

    # Get the event loop
    loop = asyncio.get_running_loop()

    # Run the agent loop
    for turn in range(max_turns):
        logging.debug(f"--- Turn {turn + 1} ---")

        # Log turn start to file
        logging.info(
            f"Starting turn {turn + 1}/{max_turns}",
            extra={
                "event_type": "turn_start",
                "turn_number": turn + 1,
                "max_turns": max_turns,
                "messages_count": len(messages),
            },
        )

        # Get LLM response
        start_time = datetime.datetime.now()
        assistant_message = await loop.run_in_executor(
            None,
            lambda: llm_client.get_response(
                messages,
                openai_tools,
                temperature=temperature,
                max_tokens=max_tokens,
            ),
        )
        end_time = datetime.datetime.now()
        elapsed_time = (end_time - start_time).total_seconds()

        # Log LLM response to file
        has_tool_calls = (
            "tool_calls" in assistant_message
            and assistant_message.get("tool_calls") is not None
        )

        # Create a serializable version of the message for logging
        # Avoid including 'message' as it conflicts with LogRecord's built-in attribute
        content = assistant_message.get("content") or ""
        tool_calls_data = None
        if has_tool_calls:
            try:
                # Convert tool_calls to a serializable format
                tool_calls_data = json.dumps(assistant_message.get("tool_calls", []))
            except (TypeError, ValueError):
                tool_calls_data = str(assistant_message.get("tool_calls", []))

        logging.info(
            f"LLM response received (took {elapsed_time:.2f}s)",
            extra={
                "event_type": "llm_response",
                "turn_number": turn + 1,
                "has_tool_calls": has_tool_calls,
                "has_content": "content" in assistant_message
                and assistant_message.get("content") is not None,
                "response_time_seconds": elapsed_time,
                "assistant_content": content,
                "assistant_tool_calls": tool_calls_data,
            },
        )

        # Validate assistant message
        if "content" in assistant_message and not isinstance(
            assistant_message["content"], (str, type(None))
        ):
            assistant_message["content"] = str(assistant_message["content"])
        if "tool_calls" in assistant_message and not isinstance(
            assistant_message["tool_calls"], (list, type(None))
        ):
            logging.warning("Received non-list tool_calls, attempting to ignore.")
            del assistant_message["tool_calls"]  # Attempt recovery

        # Add assistant message to conversation
        messages.append(assistant_message)

        # Log the assistant message addition with detailed structure
        logging.info(
            f"Assistant message added (turn {turn + 1})",
            extra={
                "event_type": "message_added",
                "category": "conversation",
                "turn_number": turn + 1,
                "message_role": "assistant",
                "has_content": "content" in assistant_message
                and assistant_message.get("content") is not None,
                "has_tool_calls": "tool_calls" in assistant_message
                and assistant_message.get("tool_calls") is not None,
                "content_length": (
                    len(assistant_message.get("content") or "")
                    if "content" in assistant_message
                    else 0
                ),
                "tool_calls_count": (
                    len(assistant_message.get("tool_calls", []))
                    if "tool_calls" in assistant_message
                    else 0
                ),
                "message_index": len(messages)
                - 1,  # Index of this message in the conversation
            },
        )

        # Also keep the debug log for developers
        logging.debug(
            f"Added assistant message: {json.dumps(assistant_message, indent=2)}"
        )

        # Process tool calls if any
        tool_calls = assistant_message.get("tool_calls")

        if tool_calls:
            tool_results = []
            for tool_call in tool_calls:
                # Ensure arguments are strings before executing
                if isinstance(tool_call.get("function", {}).get("arguments"), dict):
                    tool_call["function"]["arguments"] = json.dumps(
                        tool_call["function"]["arguments"]
                    )

                # Execute the tool call
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

            # Add tool results to conversation
            starting_message_index = len(messages)
            messages.extend(tool_results)

            # Log the tool results addition with details
            logging.info(
                f"Tool result messages added (turn {turn + 1})",
                extra={
                    "event_type": "tool_results_added",
                    "category": "conversation",
                    "turn_number": turn + 1,
                    "tool_results_count": len(tool_results),
                    "message_indices": list(
                        range(
                            starting_message_index,
                            starting_message_index + len(tool_results),
                        )
                    ),
                    "tools_summary": [
                        {
                            "tool_name": result.get("name", "unknown"),
                            "tool_call_id": result.get("tool_call_id", "unknown"),
                            "success": (
                                not str(result.get("content", "")).startswith("Error:")
                                if "content" in result
                                else False
                            ),
                        }
                        for result in tool_results
                    ],
                },
            )

            # Also keep the debug log for developers
            logging.debug(f"Added {len(tool_results)} tool result message(s).")

            # Debug log messages before next LLM call
            try:
                logging.debug(
                    f"Messages before next LLM call:\n{json.dumps(messages, indent=2)}"
                )
            except Exception as log_e:
                logging.error(f"Error logging messages: {log_e}")

            # Continue to next turn (get next LLM response)
        else:
            # No tool calls, check for task completion
            content = assistant_message.get("content", "")

            # If verification is enabled, check if the task is complete
            if verify_completion:
                print(f"\nPotential Final Answer:\n{content}", flush=True)
                print("\nVerifying task completion...", flush=True)

                is_complete, feedback = await verify_task_completion(
                    messages, llm_client, verification_prompt
                )

                if is_complete:
                    # Task is complete, print the feedback as final result
                    print(f"\nVerification PASSED: {feedback}", flush=True)

                    # Create a simplified verification result copy if it's available
                    verification_summary = {}
                    if "verification_result" in locals():
                        # Extract only the key fields we care about for logging
                        verification_summary = {
                            "is_complete": verification_result.get("is_complete", True),
                            "summary": verification_result.get("summary", ""),
                            # Only include thoughts if they're not too long
                            "thoughts_preview": (
                                verification_result.get("thoughts", "")[:100] + "..."
                                if "thoughts" in verification_result
                                and len(verification_result["thoughts"]) > 100
                                else verification_result.get("thoughts", "")
                            ),
                        }

                    # Log successful verification and completion with full verification details
                    logging.info(
                        "Task verification passed",
                        extra={
                            "event_type": "task_verification",
                            "category": "verification",
                            "turn_number": turn + 1,
                            "verification": True,
                            "status": "verified_complete",
                            "feedback": feedback,
                            "final_content": content,
                            "verification_summary": verification_summary,  # Use the safe copy
                            "total_messages": len(messages),
                            "completion_time": datetime.datetime.now().isoformat(),
                            "conversation_summary": {
                                "total_turns": turn + 1,
                                "tool_usage_count": sum(
                                    1 for m in messages if m.get("role") == "tool"
                                ),
                                "assistant_messages": sum(
                                    1 for m in messages if m.get("role") == "assistant"
                                ),
                                "user_messages": sum(
                                    1 for m in messages if m.get("role") == "user"
                                ),
                            },
                        },
                    )
                    break
                else:
                    # Task is not complete, continue the conversation with feedback
                    print(f"\nVerification FAILED: {feedback}", flush=True)
                    # Create a simplified verification result copy if it's available
                    verification_summary = {}
                    missing_steps = []
                    suggestions = ""

                    if "verification_result" in locals():
                        # Extract only the key fields we care about for logging
                        verification_summary = {
                            "is_complete": verification_result.get(
                                "is_complete", False
                            ),
                            "summary": verification_result.get("summary", ""),
                        }
                        missing_steps = verification_result.get("missing_steps", [])
                        suggestions = verification_result.get("suggestions", "")

                    # Log failed verification with detailed diagnostic information
                    logging.info(
                        "Task verification failed",
                        extra={
                            "event_type": "task_verification",
                            "category": "verification",
                            "turn_number": turn + 1,
                            "verification": True,
                            "status": "verified_incomplete",
                            "feedback": feedback,
                            "verification_summary": verification_summary,
                            "missing_steps": missing_steps,
                            "suggestions": suggestions,
                            "total_messages": len(messages),
                            "conversation_state": {
                                "last_assistant_message_index": next(
                                    (
                                        i
                                        for i, m in enumerate(messages[::-1])
                                        if m.get("role") == "assistant"
                                    ),
                                    None,
                                ),
                                "last_tool_result_index": next(
                                    (
                                        i
                                        for i, m in enumerate(messages[::-1])
                                        if m.get("role") == "tool"
                                    ),
                                    None,
                                ),
                            },
                        },
                    )

                    # Add the feedback as a user message to continue the conversation
                    messages.append(
                        {
                            "role": "user",
                            "content": f"Your response is incomplete. {feedback} Please continue working on the task.",
                        }
                    )
                    # Continue to next turn
                    continue
            else:
                # No verification (explicitly disabled), assume final answer
                if content:
                    print(
                        f"\nFinal Answer (verification disabled):\n{content}",
                        flush=True,
                    )
                    # Log completion with detailed conversation summary
                    logging.info(
                        "Task completed (verification disabled)",
                        extra={
                            "event_type": "task_completion",
                            "category": "completion",
                            "turn_number": turn + 1,
                            "verification": False,
                            "status": "completed",
                            "final_content": content,
                            "total_messages": len(messages),
                            "completion_time": datetime.datetime.now().isoformat(),
                            "conversation_summary": {
                                "total_turns": turn + 1,
                                "tool_usage_count": sum(
                                    1 for m in messages if m.get("role") == "tool"
                                ),
                                "assistant_messages": sum(
                                    1 for m in messages if m.get("role") == "assistant"
                                ),
                                "user_messages": sum(
                                    1 for m in messages if m.get("role") == "user"
                                ),
                                "final_message_length": len(content),
                            },
                        },
                    )
                else:
                    print(
                        "\nFinal Answer (verification disabled): (LLM provided no content)",
                        flush=True,
                    )
                    logging.warning(
                        f"Final assistant message had no content: {assistant_message}"
                    )
                    # Log completion with warning and detailed diagnostics
                    logging.info(
                        "Task completed with empty response (verification disabled)",
                        extra={
                            "event_type": "task_completion",
                            "category": "completion",
                            "turn_number": turn + 1,
                            "verification": False,
                            "status": "completed_empty",
                            "final_content": "",
                            "total_messages": len(messages),
                            "completion_time": datetime.datetime.now().isoformat(),
                            "conversation_summary": {
                                "total_turns": turn + 1,
                                "tool_usage_count": sum(
                                    1 for m in messages if m.get("role") == "tool"
                                ),
                                "assistant_messages": sum(
                                    1 for m in messages if m.get("role") == "assistant"
                                ),
                                "user_messages": sum(
                                    1 for m in messages if m.get("role") == "user"
                                ),
                            },
                            "last_assistant_message": {
                                "role": "assistant",
                                "has_content": "content" in assistant_message,
                                "has_tool_calls": "tool_calls" in assistant_message
                                and bool(assistant_message.get("tool_calls")),
                                "raw_message": str(assistant_message),
                            },
                            "diagnostic_info": "Assistant returned an empty content field unexpectedly",
                        },
                    )
                break  # Exit loop
    else:
        # Loop completed without breaking (max turns reached)
        print("\nWarning: Maximum turns reached without a final answer.", flush=True)
        # Log max turns reached with detailed diagnostics
        logging.warning(
            "Maximum turns reached without completion",
            extra={
                "event_type": "max_turns_reached",
                "category": "timeout",
                "max_turns": max_turns,
                "status": "incomplete",
                "total_messages": len(messages),
                "completion_time": datetime.datetime.now().isoformat(),
                "conversation_summary": {
                    "total_messages": len(messages),
                    "tool_usage_count": sum(
                        1 for m in messages if m.get("role") == "tool"
                    ),
                    "assistant_messages": sum(
                        1 for m in messages if m.get("role") == "assistant"
                    ),
                    "user_messages": sum(
                        1 for m in messages if m.get("role") == "user"
                    ),
                },
                "last_messages": [
                    {
                        "index": len(messages) - i - 1,
                        "role": messages[len(messages) - i - 1].get("role", "unknown"),
                        "has_content": "content" in messages[len(messages) - i - 1]
                        and messages[len(messages) - i - 1].get("content") is not None,
                        "content_preview": (
                            (
                                messages[len(messages) - i - 1].get("content", "")[:100]
                                + "..."
                            )
                            if "content" in messages[len(messages) - i - 1]
                            and messages[len(messages) - i - 1].get("content")
                            and len(
                                messages[len(messages) - i - 1].get("content") or ""
                            )
                            > 100
                            else (messages[len(messages) - i - 1].get("content") or "")
                        ),
                    }
                    for i in range(min(3, len(messages)))  # Last 3 messages
                ],
                "diagnostic_info": "Agent loop reached maximum turns without producing a final answer or completing verification",
            },
        )


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
    init_success = True

    try:
        async with contextlib.AsyncExitStack() as stack:
            # Initialize each server
            for server in servers_to_init:
                try:
                    logging.debug(f"Initializing server {server.name}...")
                    stdio_client_cm = await server.initialize()

                    # Enter stdio client context manager using the stack
                    read, write = await stack.enter_async_context(stdio_client_cm)
                    server.read = read
                    server.write = write
                    logging.debug(f"stdio client connected for {server.name}.")

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
                        if logging.getLogger().isEnabledFor(logging.DEBUG):
                            print(
                                f"  Server '{server.name}' initialized with tools:",
                                end="",
                            )
                            if server_tools:
                                print(", ".join([tool.name for tool in server_tools]))
                            else:
                                print("(No tools found)")
                    except Exception as list_tools_e:
                        # Only print this in debug mode
                        if logging.getLogger().isEnabledFor(logging.DEBUG):
                            print(
                                f"  Server '{server.name}' initialized, but failed to list tools: {list_tools_e}"
                            )
                        logging.warning(
                            f"Could not list tools for {server.name} after init: {list_tools_e}"
                        )

                except Exception as e:
                    logging.error(
                        f"Failed to initialize server {server.name}: {e}", exc_info=True
                    )
                    init_success = False
                    break

            # Exit if initialization failed
            if not init_success or not initialized_servers:
                logging.error("Server initialization failed. Exiting.")
                return

            # Run the agent
            logging.info(f"Running prompt: {user_prompt}")
            # Only print the full prompt in debug mode
            if logging.getLogger().isEnabledFor(logging.DEBUG):
                print(f"Running prompt: {user_prompt}")
            else:
                # Print a shorter version for normal operation
                short_prompt = (
                    user_prompt[:50] + "..." if len(user_prompt) > 50 else user_prompt
                )
                print(f"Processing request: {short_prompt}")

            await run_agent(
                user_prompt,
                initialized_servers,
                llm_client,
                system_message,
                temperature=temperature,
                max_tokens=max_tokens,
                max_turns=max_turns,
                verify_completion=verify_completion,
                verification_prompt=verification_prompt,
            )

    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}", exc_info=True)
    finally:
        logging.info("Application finished.")
