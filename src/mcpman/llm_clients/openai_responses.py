"""
Client for OpenAI's new Responses API.

This module provides a client implementation specifically for OpenAI's
Responses API, which offers different capabilities than the Chat Completions API.
"""

import json
import logging
import sys
from typing import Dict, List, Any, Optional, Union, Tuple

import httpx

from .base import BaseLLMClient


class OpenAIResponsesClient(BaseLLMClient):
    """
    Client for OpenAI's new Responses API.

    This client is designed to work exclusively with OpenAI's Responses API,
    which uses a different interface than the Chat Completions API.
    It defaults to using o4-mini and supports all the new format options.
    """

    def _convert_messages_to_responses_format(
        self, messages: List[Dict[str, Any]]
    ) -> Tuple[Optional[str], List[Dict[str, Any]]]:
        """
        Convert OpenAI-format messages to the format expected by the Responses API.

        Args:
            messages: List of message objects in OpenAI Chat Completions format

        Returns:
            Tuple of (system_content, message_list)
        """
        # Extract system message to use as instructions if present
        system_content = None
        message_list = []

        # Track tool call IDs to ensure responses are properly linked
        tool_calls_by_id = {}

        # First pass: collect all tool calls and their IDs
        for msg in messages:
            if msg.get("role") == "assistant" and "tool_calls" in msg:
                for tool_call in msg["tool_calls"]:
                    call_id = tool_call.get("id", "")
                    if call_id:
                        tool_calls_by_id[call_id] = tool_call

        # Second pass: build the message list
        for i, msg in enumerate(messages):
            if msg["role"] == "system":
                system_content = msg["content"]
            elif msg["role"] == "user" or msg["role"] == "assistant":
                # Only include content for non-empty messages
                if (
                    "content" in msg
                    and msg["content"] is not None
                    and msg["content"] != ""
                ):
                    message_list.append(
                        {"role": msg["role"], "content": msg["content"]}
                    )

                # Also include any tool calls from assistant messages
                if msg["role"] == "assistant" and "tool_calls" in msg:
                    for tool_call in msg["tool_calls"]:
                        if tool_call["type"] == "function":
                            function_call = {
                                "type": "function_call",
                                "call_id": tool_call["id"],
                                "name": tool_call["function"]["name"],
                                "arguments": tool_call["function"]["arguments"],
                            }
                            message_list.append(function_call)
            elif msg["role"] == "tool":
                # Handle tool response messages (convert to function_call_output)
                tool_call_id = msg.get("tool_call_id")

                # Skip invalid tool responses
                if not tool_call_id or tool_call_id not in tool_calls_by_id:
                    logging.warning(
                        f"Skipping tool response with invalid/missing call_id: {tool_call_id}"
                    )
                    continue

                # Create a properly formatted function_call_output
                tool_response = {
                    "type": "function_call_output",
                    "call_id": tool_call_id,
                    "output": msg.get("content", ""),
                }

                logging.info(
                    f"Adding tool response for call_id {tool_call_id}: {msg.get('content', '')[:100]}..."
                )
                message_list.append(tool_response)

        return system_content, message_list

    def _convert_tools_to_responses_format(
        self, tools: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Convert OpenAI-format tools to Responses API format.

        Args:
            tools: List of tool definitions in OpenAI Chat Completions format

        Returns:
            List of tools in Responses API format
        """
        if not tools:
            return []

        # Log the raw tools to debug conversion issues
        logging.debug(f"Raw tools before conversion: {json.dumps(tools, default=str)}")

        # The Responses API requires a different structure where 'name' must be
        # at the top level of each tool object, not nested in the 'function' property
        responses_tools = []
        for tool in tools:
            try:
                if tool["type"] == "function" and "function" in tool:
                    # Create a flattened tool with 'name' at the top level
                    function_name = tool["function"]["name"]
                    responses_tool = {
                        "type": "function",
                        "name": function_name,  # Move name to top level
                        "description": tool["function"].get(
                            "description", f"Function to execute {function_name}"
                        ),
                        "parameters": tool["function"].get("parameters", {}),
                    }

                    # Ensure parameters object is properly formatted
                    if "parameters" in responses_tool and isinstance(
                        responses_tool["parameters"], dict
                    ):
                        # Make sure 'type' field exists in parameters
                        if "type" not in responses_tool["parameters"]:
                            responses_tool["parameters"]["type"] = "object"

                        # Ensure properties exists if we have an object schema
                        if (
                            responses_tool["parameters"].get("type") == "object"
                            and "properties" not in responses_tool["parameters"]
                        ):
                            responses_tool["parameters"]["properties"] = {}

                    responses_tools.append(responses_tool)
                    logging.debug(f"Converted tool: {function_name}")
            except Exception as e:
                logging.error(f"Error converting tool: {e}", exc_info=True)

        logging.debug(f"Converted {len(responses_tools)} tools to Responses format")
        return responses_tools

    def _normalize_responses_response(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize Responses API response to Chat Completions format for compatibility.

        Args:
            data: Raw Responses API response

        Returns:
            Response in OpenAI Chat Completions format for the orchestrator
        """
        # Log comprehensive details about the data we received
        logging.info("=== RESPONSE NORMALIZATION START ===")
        logging.info(f"Response raw: {data}")
        logging.info(f"Response type: {type(data)}")
        logging.info(f"Response repr: {repr(data)}")
        logging.info(f"Response attributes: {dir(data)}")

        # Try to extract attributes from the response
        logging.info("Extracting all response attributes:")
        for attr in dir(data):
            if not attr.startswith("_") and not callable(getattr(data, attr, None)):
                try:
                    value = getattr(data, attr, None)
                    logging.info(f"  {attr}: {repr(value)}")
                except Exception as e:
                    logging.info(f"  {attr}: <Error accessing: {e}>")

        # Initialize normalized response
        normalized_response = {"role": "assistant", "content": ""}

        try:
            # Process response data based on type
            output_items = None

            # Check for output_text first (text content)
            if hasattr(data, "output_text") and data.output_text is not None:
                normalized_response["content"] = data.output_text
                logging.info(f"Found output_text: {data.output_text}")

            # Check for output array (tool calls)
            if hasattr(data, "output"):
                output_items = data.output
                logging.info(f"Found output array with {len(output_items)} items")
            elif isinstance(data, dict) and "output" in data:
                output_items = data["output"]
                logging.info(
                    f"Found output array in dict with {len(output_items)} items"
                )

            # Process tool calls if present
            if output_items:
                tool_calls = []

                # Log all output items for debugging
                for i, item in enumerate(output_items):
                    logging.info(
                        f"Output item {i} type: {getattr(item, 'type', type(item))}"
                    )
                    if hasattr(item, "__dict__"):
                        logging.info(f"Output item {i} attributes: {item.__dict__}")

                # Extract tool calls from output items
                for i, item in enumerate(output_items):
                    # Check for function_call items
                    if (hasattr(item, "type") and item.type == "function_call") or (
                        isinstance(item, dict) and item.get("type") == "function_call"
                    ):

                        # Extract fields based on object type
                        if hasattr(item, "type"):
                            # It's an object
                            call_id = getattr(item, "call_id", f"call_{i}")
                            name = getattr(item, "name", "unknown_function")
                            arguments = getattr(item, "arguments", "{}")
                        else:
                            # It's a dictionary
                            call_id = item.get("call_id", f"call_{i}")
                            name = item.get("name", "unknown_function")
                            arguments = item.get("arguments", "{}")

                        # Ensure arguments is a string (OpenAI format expects string)
                        if not isinstance(arguments, str):
                            try:
                                arguments = json.dumps(arguments)
                            except:
                                arguments = str(arguments)

                        # Create tool call in OpenAI format (exactly what orchestrator expects)
                        tool_call = {
                            "id": call_id,
                            "type": "function",
                            "function": {
                                "name": name,
                                "arguments": arguments,
                            },
                        }

                        tool_calls.append(tool_call)
                        logging.info(f"Added tool call: {name} with id {call_id}")

                # Add tool calls to response if found
                if tool_calls:
                    normalized_response["tool_calls"] = tool_calls

                    # Always set content to empty string when tool calls are present
                    # This is critical for compatibility with the orchestrator
                    normalized_response["content"] = ""
                    logging.info(
                        f"Set content to empty string with {len(tool_calls)} tool calls"
                    )

            # If no content or tool calls found, add fallback content
            if (
                "content" not in normalized_response
                and "tool_calls" not in normalized_response
            ):
                normalized_response["content"] = (
                    "I'm sorry, but I couldn't process your request properly."
                )
                logging.warning(
                    "No content or tool calls found in response, using fallback message"
                )

        except Exception as e:
            logging.error(
                f"Error normalizing Responses API response: {e}", exc_info=True
            )
            normalized_response["content"] = (
                f"Error processing model response: {str(e)}"
            )

        # Final safety check - ensure we never return None for content
        # The orchestrator expects empty string, not None
        if normalized_response.get("content") is None:
            normalized_response["content"] = ""
            logging.warning("Found null content, replaced with empty string")

        logging.info(
            f"Final normalized response: {json.dumps(normalized_response, indent=2)}"
        )
        return normalized_response

    def get_response(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        tool_choice: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Get a response from the OpenAI Responses API and convert to standard format.

        Args:
            messages: List of message objects in OpenAI Chat Completions format
            tools: Optional list of tool definitions
            temperature: Sampling temperature
            max_tokens: Maximum number of tokens for the response
            tool_choice: Optional specification for tool selection behavior

        Returns:
            Response message object in OpenAI Chat Completions format for compatibility
        """
        logging.debug(
            f"Sending request to OpenAI Responses API with model {self.model_name}. {'Including tools.' if tools else 'No tools.'}"
        )
        # Enhanced debug logging
        logging.debug(f"API URL: {self.api_url}")

        try:
            # Import the OpenAI client here to avoid dependencies for users who don't use this client
            try:
                from openai import OpenAI
            except ImportError:
                logging.error(
                    "OpenAI Python package is not installed. Please install it with: pip install openai>=1.0.0"
                )
                return {
                    "role": "assistant",
                    "content": "Error: OpenAI Python package is not installed. Please install it with: pip install openai>=1.0.0",
                }

            # Create OpenAI client with proper base URL
            openai_client = OpenAI(api_key=self.api_key, base_url=self.api_url)
            logging.debug(f"Using OpenAI client with base_url: {self.api_url}")

            # Log the input messages before conversion
            logging.info("Converting messages to Responses API format")
            logging.info(f"Input messages count: {len(messages)}")

            # Log message roles for debugging
            roles = [msg.get("role") for msg in messages]
            logging.info(f"Message roles: {roles}")

            # Log tool messages specifically
            tool_messages = [msg for msg in messages if msg.get("role") == "tool"]
            if tool_messages:
                logging.info(f"Found {len(tool_messages)} tool response messages")
                for i, tm in enumerate(tool_messages):
                    logging.info(f"Tool response {i+1}:")
                    logging.info(f"  tool_call_id: {tm.get('tool_call_id', 'missing')}")
                    logging.info(f"  content: {tm.get('content', 'empty')[:100]}...")

            # Log tool call messages specifically
            tool_call_messages = [
                msg
                for msg in messages
                if msg.get("role") == "assistant" and "tool_calls" in msg
            ]
            if tool_call_messages:
                logging.info(
                    f"Found {len(tool_call_messages)} messages with tool calls"
                )
                for i, tcm in enumerate(tool_call_messages):
                    if "tool_calls" in tcm:
                        logging.info(
                            f"Message {i+1} has {len(tcm['tool_calls'])} tool calls"
                        )
                        for j, tc in enumerate(tcm["tool_calls"]):
                            logging.info(f"  Tool call {j+1}:")
                            logging.info(f"    id: {tc.get('id', 'missing')}")
                            if "function" in tc:
                                logging.info(
                                    f"    function: {tc['function'].get('name', 'unnamed')}"
                                )

            # Convert messages to Responses API format
            instructions, input_messages = self._convert_messages_to_responses_format(
                messages
            )

            # Log the converted messages
            logging.info(
                f"Converted to {len(input_messages)} Responses API format messages"
            )
            for i, msg in enumerate(input_messages):
                msg_type = msg.get("type", "regular")
                if msg_type == "function_call_output":
                    logging.info(
                        f"Message {i+1}: function_call_output for call_id {msg.get('call_id', 'unknown')}"
                    )
                elif msg_type == "function_call":
                    logging.info(
                        f"Message {i+1}: function_call - {msg.get('name', 'unnamed')}"
                    )
                else:
                    logging.info(f"Message {i+1}: {msg.get('role', 'unknown')} message")

            # Prepare parameters for the API call
            params = {
                "model": self.model_name,
                # Enable parallel tool calling for more responsive tool execution
                "parallel_tool_calls": True,
            }

            # Only add temperature if not using o4-mini (which doesn't support it)
            if not self.model_name.startswith("o4-mini"):
                params["temperature"] = temperature

            # Add input based on format
            params["input"] = input_messages

            # Add instructions if present
            if instructions:
                # Include a strong instruction to use tools when available
                tool_instruction = "IMPORTANT: When specialized tools are available, you MUST use them instead of calculating or generating information yourself. Always prefer using tools for date/time operations, calculations, and specific data lookups."

                # Append our tool usage instruction
                if isinstance(instructions, str):
                    params["instructions"] = instructions + "\n\n" + tool_instruction
                else:
                    params["instructions"] = tool_instruction

                logging.info(
                    "Added specific tool usage instructions to encourage tool calling"
                )

            # Add tools if present
            if tools:
                converted_tools = self._convert_tools_to_responses_format(tools)
                params["tools"] = converted_tools
                logging.info(f"Adding {len(converted_tools)} tools to the request")
                for i, tool in enumerate(converted_tools):
                    logging.info(f"Tool {i+1}: {tool['name']}")

                # Force tool use by adding tool_choice parameter (similar to OpenAI Chat Completions API)
                params["tool_choice"] = "auto"
                logging.info("Added tool_choice=auto to force tool usage")

            # Add max tokens if specified
            if max_tokens:
                params["max_output_tokens"] = max_tokens

            # Only log full payload on DEBUG level
            if logging.getLogger().isEnabledFor(logging.DEBUG):
                try:
                    logging.debug(
                        f"Sending Responses API Payload:\n{json.dumps(params, indent=2)}"
                    )
                except Exception as e:
                    logging.debug(
                        f"Could not serialize Responses API payload for logging: {e}"
                    )
            else:
                logging.info(
                    f"Sending request to Responses API with model {self.model_name}..."
                )

            # Make the API call
            logging.debug(
                f"Making API call to Responses API with model {self.model_name}"
            )
            logging.info(
                f"API call params: {json.dumps(params, default=str)[:1000]}..."
            )
            response = openai_client.responses.create(**params)

            # Log detailed response information for debugging
            logging.debug(f"Received response from Responses API")
            logging.info(f"Response type: {type(response)}")
            logging.info(f"Response dir: {dir(response)}")

            # Try to extract and log model information
            if hasattr(response, "model"):
                logging.info(f"Response model: {response.model}")

            # Log detailed output information for debugging
            logging.info("--- OpenAI Responses API Output Details ---")

            # Log text output if present
            output_text = getattr(response, "output_text", None)
            if output_text is not None:
                logging.info(f"Response has output_text: {output_text}")
                logging.info(f"Output text type: {type(output_text)}")
            else:
                logging.info("Response has no output_text")

            # Log output array if present
            output_array = getattr(response, "output", None)
            if output_array is not None:
                logging.info(
                    f"Response has output array with {len(output_array)} items"
                )

                # Log details about each output item for debugging
                for i, item in enumerate(output_array):
                    item_type = getattr(item, "type", type(item).__name__)
                    logging.info(f"Output item {i} - Type: {item_type}")

                    # Detailed info for function calls
                    if hasattr(item, "type") and item.type == "function_call":
                        name = getattr(item, "name", "unknown")
                        call_id = getattr(item, "call_id", "unknown")
                        arguments = getattr(item, "arguments", "{}")
                        logging.info(f"  Function: {name}")
                        logging.info(f"  Call ID: {call_id}")
                        logging.info(f"  Arguments: {arguments}")
            else:
                logging.info("Response has no output array")

            # Normalize the response to the format expected by the orchestrator
            try:
                normalized_response = self._normalize_responses_response(response)

                # Ensure response has the required fields exactly matching what the orchestrator expects
                if "role" not in normalized_response:
                    normalized_response["role"] = "assistant"

                # Critical check: content must NEVER be None
                # The orchestrator cannot handle None values for content
                if (
                    "content" not in normalized_response
                    or normalized_response["content"] is None
                ):
                    normalized_response["content"] = ""

                # Detailed logging for tool calls
                if "tool_calls" in normalized_response:
                    logging.info(
                        f"Final response contains {len(normalized_response['tool_calls'])} tool calls"
                    )
                    for i, tool_call in enumerate(normalized_response["tool_calls"]):
                        function_name = tool_call.get("function", {}).get(
                            "name", "unknown"
                        )
                        function_args = tool_call.get("function", {}).get(
                            "arguments", "{}"
                        )
                        logging.info(f"  Tool call {i+1}: {function_name}")
                        logging.info(f"    with arguments: {function_args}")
                        logging.info(f"    with id: {tool_call.get('id', 'unknown')}")

                    # Double-check: content MUST be empty string (not None) with tool calls
                    # This is crucial for the orchestrator to function correctly
                    if normalized_response.get("content") is None:
                        logging.warning(
                            "CRITICAL: Found None content with tool calls - replacing with empty string"
                        )
                        normalized_response["content"] = ""
                return normalized_response
            except Exception as e:
                logging.error(f"Failed to normalize response: {e}", exc_info=True)
                error_response = {
                    "role": "assistant",
                    "content": f"Error processing model response from OpenAI Responses API: {str(e)}",
                }
                logging.debug(f"Error normalized response: {error_response}")
                return error_response

        except Exception as e:
            logging.error(f"Error communicating with OpenAI Responses API: {e}")
            # Return a standardized error response
            error_response = {
                "role": "assistant",
                "content": f"Error: Could not complete request to OpenAI Responses API: {str(e)}",
            }
            logging.debug(f"Global error response: {error_response}")
            return error_response
