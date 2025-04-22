"""
Client for OpenAI's Responses API.

Provides a client for OpenAI's Responses API with different input/output format
than their Chat Completions API. This client handles the conversion and normalization
needed for compatibility with mcpman's orchestrator.
"""

import json
import logging
from typing import Dict, List, Any, Optional, Tuple

from .base import BaseLLMClient


class OpenAIResponsesClient(BaseLLMClient):
    """
    Client for OpenAI's Responses API.

    Handles communication with OpenAI's Responses API, converting between
    the different formats required. Defaults to o4-mini model.
    """

    def _convert_messages_to_responses_format(
        self, messages: List[Dict[str, Any]]
    ) -> Tuple[Optional[str], List[Dict[str, Any]]]:
        """
        Convert messages from Chat Completions format to Responses API format.

        Args:
            messages: List of message objects in OpenAI Chat Completions format

        Returns:
            Tuple of (system_content, message_list)
        """
        system_content = None
        message_list = []

        # Track tool calls by ID to properly link tool responses
        tool_calls_by_id = {}
        for msg in messages:
            if msg.get("role") == "assistant" and "tool_calls" in msg:
                for tool_call in msg["tool_calls"]:
                    if "id" in tool_call:
                        tool_calls_by_id[tool_call["id"]] = tool_call

        # Process all messages
        for msg in messages:
            # Extract system message for instructions
            if msg["role"] == "system":
                system_content = msg["content"]

            # Process user and assistant messages
            elif msg["role"] in ["user", "assistant"]:
                # Add content if present and not empty
                if msg.get("content"):
                    message_list.append(
                        {"role": msg["role"], "content": msg["content"]}
                    )

                # Add tool calls from assistant messages
                if msg["role"] == "assistant" and "tool_calls" in msg:
                    for tool_call in msg["tool_calls"]:
                        if tool_call["type"] == "function":
                            message_list.append(
                                {
                                    "type": "function_call",
                                    "call_id": tool_call["id"],
                                    "name": tool_call["function"]["name"],
                                    "arguments": tool_call["function"]["arguments"],
                                }
                            )

            # Process tool response messages
            elif msg["role"] == "tool":
                tool_call_id = msg.get("tool_call_id")

                # Skip invalid tool responses
                if not tool_call_id or tool_call_id not in tool_calls_by_id:
                    logging.debug(
                        f"Skipping tool response with missing/invalid call_id: {tool_call_id}"
                    )
                    continue

                # Add properly formatted function_call_output
                message_list.append(
                    {
                        "type": "function_call_output",
                        "call_id": tool_call_id,
                        "output": msg.get("content", ""),
                    }
                )

        return system_content, message_list

    def _convert_tools_to_responses_format(
        self, tools: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Convert tools from Chat Completions format to Responses API format.

        Args:
            tools: List of tool definitions in OpenAI Chat Completions format

        Returns:
            List of tools in Responses API format
        """
        if not tools:
            return []

        responses_tools = []
        for tool in tools:
            if tool["type"] == "function" and "function" in tool:
                # Create a flattened tool with 'name' at the top level
                function_name = tool["function"]["name"]
                responses_tool = {
                    "type": "function",
                    "name": function_name,
                    "description": tool["function"].get("description", ""),
                    "parameters": tool["function"].get("parameters", {}),
                }

                # Ensure parameters is properly formatted
                if "parameters" in responses_tool and isinstance(
                    responses_tool["parameters"], dict
                ):
                    if "type" not in responses_tool["parameters"]:
                        responses_tool["parameters"]["type"] = "object"

                    if (
                        responses_tool["parameters"].get("type") == "object"
                        and "properties" not in responses_tool["parameters"]
                    ):
                        responses_tool["parameters"]["properties"] = {}

                responses_tools.append(responses_tool)

        return responses_tools

    def _normalize_responses_response(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize Responses API response to Chat Completions format.

        Args:
            data: Response from the Responses API

        Returns:
            Response in OpenAI Chat Completions format for the orchestrator
        """
        # Initialize normalized response with defaults
        normalized_response = {"role": "assistant", "content": ""}

        try:
            # Extract text content if present
            if hasattr(data, "output_text") and data.output_text is not None:
                normalized_response["content"] = data.output_text

            # Extract tool calls from output array
            output_items = getattr(data, "output", None)
            if output_items:
                tool_calls = []

                # Process each output item
                for i, item in enumerate(output_items):
                    # Check for function_call items
                    is_function_call = (
                        hasattr(item, "type") and item.type == "function_call"
                    ) or (
                        isinstance(item, dict) and item.get("type") == "function_call"
                    )

                    if is_function_call:
                        # Extract function call data
                        if hasattr(item, "type"):
                            call_id = getattr(item, "call_id", f"call_{i}")
                            name = getattr(item, "name", "unknown_function")
                            arguments = getattr(item, "arguments", "{}")
                        else:
                            call_id = item.get("call_id", f"call_{i}")
                            name = item.get("name", "unknown_function")
                            arguments = item.get("arguments", "{}")

                        # Ensure arguments is a string
                        if not isinstance(arguments, str):
                            try:
                                arguments = json.dumps(arguments)
                            except:
                                arguments = str(arguments)

                        # Add properly formatted tool call
                        tool_calls.append(
                            {
                                "id": call_id,
                                "type": "function",
                                "function": {
                                    "name": name,
                                    "arguments": arguments,
                                },
                            }
                        )

                # Add tool calls to response and ensure empty content
                if tool_calls:
                    normalized_response["tool_calls"] = tool_calls
                    normalized_response["content"] = ""

            # Ensure we have a valid fallback content if needed
            if (
                not normalized_response.get("content")
                and "tool_calls" not in normalized_response
            ):
                normalized_response["content"] = (
                    "I'm sorry, but I couldn't process your request properly."
                )

        except Exception as e:
            logging.error(
                f"Error normalizing Responses API response: {e}", exc_info=True
            )
            normalized_response["content"] = (
                f"Error processing model response: {str(e)}"
            )

        # Final safety check for content (critical for orchestrator compatibility)
        if normalized_response.get("content") is None:
            normalized_response["content"] = ""

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
        Get a response from the OpenAI Responses API.

        Args:
            messages: List of message objects in OpenAI Chat Completions format
            tools: Optional list of tool definitions
            temperature: Sampling temperature (not used with o4-mini)
            max_tokens: Maximum number of tokens for the response
            tool_choice: Optional specification for tool selection behavior

        Returns:
            Response message in OpenAI Chat Completions format for compatibility
        """
        logging.debug(f"Sending request to Responses API with model {self.model_name}")

        try:
            # Import the OpenAI client
            try:
                from openai import OpenAI
            except ImportError:
                logging.error(
                    "OpenAI Python package is not installed. Please install it with: pip install openai>=1.0.0"
                )
                return {
                    "role": "assistant",
                    "content": "Error: OpenAI Python package is not installed. Please install it with: uv pip install openai>=1.0.0",
                }

            # Create OpenAI client with proper base URL
            openai_client = OpenAI(api_key=self.api_key, base_url=self.api_url)

            # Convert messages to Responses API format
            instructions, input_messages = self._convert_messages_to_responses_format(
                messages
            )

            # Prepare API parameters
            params = {
                "model": self.model_name,
                "parallel_tool_calls": True,
                "input": input_messages,
            }

            # Add temperature if not using o4-mini (which doesn't support it)
            if not self.model_name.startswith("o4-mini"):
                params["temperature"] = temperature

            # Add instructions with tool usage guidance
            if instructions:
                tool_instruction = "IMPORTANT: When specialized tools are available, you MUST use them instead of calculating or generating information yourself."
                params["instructions"] = (
                    instructions + "\n\n" + tool_instruction
                    if isinstance(instructions, str)
                    else tool_instruction
                )

            # Add tools if present
            if tools:
                params["tools"] = self._convert_tools_to_responses_format(tools)
                params["tool_choice"] = "auto"  # Encourage tool usage

            # Add max tokens if specified
            if max_tokens:
                params["max_output_tokens"] = max_tokens

            # Make the API call
            response = openai_client.responses.create(**params)

            # Normalize and return the response
            try:
                normalized_response = self._normalize_responses_response(response)

                # Ensure required fields are present
                if "role" not in normalized_response:
                    normalized_response["role"] = "assistant"

                if normalized_response.get("content") is None:
                    normalized_response["content"] = ""

                return normalized_response

            except Exception as e:
                logging.error(f"Failed to normalize response: {e}", exc_info=True)
                return {
                    "role": "assistant",
                    "content": f"Error processing model response: {str(e)}",
                }

        except Exception as e:
            logging.error(f"Error communicating with OpenAI Responses API: {e}")
            return {
                "role": "assistant",
                "content": f"Error: Could not complete request to OpenAI Responses API: {str(e)}",
            }
