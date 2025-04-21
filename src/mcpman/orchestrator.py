import json
import logging
import asyncio
import contextlib
from typing import Dict, List, Any, Optional, Tuple

from .server import Server
from .llm_client import LLMClient
from .tools import Tool, sanitize_name


async def execute_tool_call(
    tool_call: Dict[str, Any], 
    servers: List[Server]
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
            original_tool_name = prefixed_tool_name[len(prefix):]
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
    print(
        f"-> Calling tool: {prefixed_tool_name}({arguments})", 
        flush=True
    )
    
    # Initialize result content and execution status
    execution_result_content = f"Error: Tool '{original_tool_name}' execution failed on server '{target_server.name}'."
    tool_found_on_target_server = False
    
    # Execute the tool
    try:
        logging.debug(
            f"Executing {original_tool_name} on server {target_server.name} (sanitized: {target_server_name}, prefixed: {prefixed_tool_name})..."
        )
        tool_output = await target_server.execute_tool(original_tool_name, arguments)
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
                execution_result_content = f"Error: Tool execution failed: {error_detail}"
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
        execution_result_content = f"Error: Runtime error contacting server {target_server.name}: {e}"
        tool_found_on_target_server = True  # Attempted but failed communication
    except Exception as e:
        logging.error(
            f"Exception executing tool '{prefixed_tool_name}' on {target_server.name}: {e}",
            exc_info=True,
        )
        execution_result_content = f"Error: Tool execution failed unexpectedly on {target_server.name}."
        tool_found_on_target_server = True  # Attempted but failed
    
    # Log the tool response
    print(
        f"<- Tool Response [{prefixed_tool_name}]: {execution_result_content}",
        flush=True,
    )
    
    # Return the result message
    return {
        "role": "tool",
        "tool_call_id": tool_call_id,
        "name": prefixed_tool_name,
        "content": str(execution_result_content),
    }


async def run_agent(
    prompt: str, 
    servers: List[Server], 
    llm_client: LLMClient, 
    system_message: str,
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
    max_turns: int = 10
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
    
    # Get the event loop
    loop = asyncio.get_running_loop()
    
    # Run the agent loop
    for turn in range(max_turns):
        logging.debug(f"--- Turn {turn + 1} ---")
        
        # Get LLM response
        assistant_message = await loop.run_in_executor(
            None, 
            lambda: llm_client.get_response(
                messages, 
                openai_tools,
                temperature=temperature,
                max_tokens=max_tokens,
            )
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
        logging.debug(f"Added assistant message: {json.dumps(assistant_message, indent=2)}")
        
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
                    logging.warning(f"Unsupported tool call type: {tool_call.get('type')}")
                    tool_results.append({
                        "role": "tool",
                        "tool_call_id": tool_call["id"],
                        "name": tool_call.get("function", {}).get("name", "unknown"),
                        "content": f"Error: Unsupported tool type '{tool_call.get('type')}'",
                    })
            
            # Add tool results to conversation
            messages.extend(tool_results)
            logging.debug(f"Added {len(tool_results)} tool result message(s).")
            
            # Debug log messages before next LLM call
            try:
                logging.debug(f"Messages before next LLM call:\n{json.dumps(messages, indent=2)}")
            except Exception as log_e:
                logging.error(f"Error logging messages: {log_e}")
            
            # Continue to next turn (get next LLM response)
        else:
            # No tool calls, assume final answer
            content = assistant_message.get("content")
            if content:
                print(f"\nFinal Answer:\n{content}", flush=True)
            else:
                print("\nFinal Answer: (LLM provided no content)", flush=True)
                logging.warning(f"Final assistant message had no content: {assistant_message}")
            break  # Exit loop
    else:
        # Loop completed without breaking (max turns reached)
        print("\nWarning: Maximum turns reached without a final answer.", flush=True)


async def initialize_and_run(
    config_path: str,
    user_prompt: str,
    system_message: str,
    llm_client: LLMClient,
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
    max_turns: int = 10
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
                        print(f"  Server '{server.name}' initialized with tools:", end="")
                        if server_tools:
                            print(", ".join([tool.name for tool in server_tools]))
                        else:
                            print("(No tools found)")
                    except Exception as list_tools_e:
                        print(f"  Server '{server.name}' initialized, but failed to list tools: {list_tools_e}")
                        logging.warning(f"Could not list tools for {server.name} after init: {list_tools_e}")
                
                except Exception as e:
                    logging.error(f"Failed to initialize server {server.name}: {e}", exc_info=True)
                    init_success = False
                    break
            
            # Exit if initialization failed
            if not init_success or not initialized_servers:
                logging.error("Server initialization failed. Exiting.")
                return
            
            # Run the agent
            logging.info(f"Running prompt: {user_prompt}")
            print(f"Running prompt: {user_prompt}")
            
            await run_agent(
                user_prompt, 
                initialized_servers, 
                llm_client, 
                system_message,
                temperature=temperature,
                max_tokens=max_tokens,
                max_turns=max_turns
            )
    
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}", exc_info=True)
    finally:
        logging.info("Application finished.")