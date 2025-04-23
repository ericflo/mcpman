#!/usr/bin/env python3
"""
MCPMan Log Replay Tool

Reproduces the exact console output from an MCPMan log file, including
colorization, boxes, and other formatting elements.
"""

import argparse
import json
import os
import re
import sys
import glob
from collections import defaultdict
from typing import Dict, List, Any, Optional, Tuple

# Import the existing formatting module
from .formatting import (
    format_tool_call, 
    format_tool_response, 
    format_llm_response,
    format_verification_result, 
    format_processing_step, 
    print_box, 
    print_llm_config,
    BoxStyle,
    print_short_prompt
)


def process_log_file(log_file_path: str, show_hidden: bool = False) -> None:
    """
    Process the log file and reproduce the original colorized output.
    
    Args:
        log_file_path: Path to the log file to process
        show_hidden: Whether to show events that wouldn't normally be visible
    """
    try:
        with open(log_file_path, 'r') as f:
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


def print_llm_config_box(config: Dict[str, Any]) -> None:
    """
    Print the LLM configuration box.
    
    Args:
        config: Dictionary with configuration data
    """
    # Use the existing formatting function
    print_llm_config(config)


def process_conversation_flow(log_entries: List[Dict[str, Any]], show_hidden: bool = False) -> None:
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
                tool_calls.append({
                    "id": tool_call_id,
                    "tool": tool_name,
                    "parameters": parameters
                })
        
        # Get tool responses
        if entry.get("event_type") == "tool_response":
            tool_name = entry.get("payload", {}).get("tool", "")
            response = entry.get("payload", {}).get("response", "")
            tool_call_id = entry.get("payload", {}).get("tool_call_id", "")
            
            if tool_name and response:
                tool_responses[tool_call_id] = {
                    "tool": tool_name,
                    "response": response
                }
        
        # Get LLM responses
        if entry.get("event_type") == "llm_response" and entry.get("payload", {}).get("has_content", False):
            response_content = entry.get("payload", {}).get("response", {}).get("content", "")
            if response_content:
                llm_responses.append(response_content)
        
        # Get verification results
        if entry.get("event_type") == "verification":
            is_complete = entry.get("payload", {}).get("is_complete", False)
            feedback = entry.get("payload", {}).get("feedback", "")
            verification_results.append({
                "is_complete": is_complete,
                "feedback": feedback
            })
    
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
        formatted_tool_call = format_tool_call(tool_call['tool'], json.dumps(tool_call['parameters']))
        print(formatted_tool_call)
        
        # Show processing request if available
        if i < len(process_requests) and process_requests[i+1] == "CallToolRequest":
            print("Processing request of type CallToolRequest")
        
        # Show tool response with proper formatting
        if tool_call['id'] in tool_responses:
            response = tool_responses[tool_call['id']]
            formatted_tool_response = format_tool_response(response['tool'], response['response'])
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
                    print(format_verification_result(True, verification_results[i]['feedback']))
                else:
                    # Verification failed
                    print(format_verification_result(False, verification_results[i]['feedback']))
            else:
                # Last response (final answer)
                print(format_llm_response(response, is_final=True))


# Function removed - using format_llm_response directly


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


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Reproduce the exact console output from an MCPMan log file."
    )
    
    parser.add_argument(
        "log_file",
        nargs="?",  # Make the parameter optional
        help="Path to the MCPMan log file to process (defaults to latest log)",
        default=None
    )
    
    parser.add_argument(
        "--show-hidden",
        action="store_true",
        help="Show events that wouldn't normally be visible",
    )
    
    args = parser.parse_args()
    
    # If no log file specified, find the latest one
    log_file = args.log_file
    if not log_file:
        log_file = find_latest_log_file()
        if not log_file:
            print("Error: No log file specified and no log files found in the logs directory.")
            sys.exit(1)
        
        print(f"Using latest log file: {log_file}")
    
    # Process the log file
    process_log_file(log_file, args.show_hidden)


if __name__ == "__main__":
    main()