import json
import time
from typing import Dict, List, Any, Optional, Union

import httpx

from .base import BaseLLMClient
from ..logger import LLMClientLogger


class AnthropicClient(BaseLLMClient):
    """
    Client for Anthropic Claude API (v1/messages).

    Handles the conversion between OpenAI-compatible format and Anthropic's API.
    """

    def __init__(self, api_key, api_url, model_name, timeout=180.0):
        """Initialize the Anthropic client."""
        super().__init__(api_key, api_url, model_name, timeout)
        # Initialize the standardized logger
        self.logger = LLMClientLogger("anthropic", model_name)

    def _convert_openai_messages_to_anthropic(
        self, messages: List[Dict[str, Any]]
    ) -> Dict[str, Union[str, List[Dict[str, Any]]]]:
        """
        Convert OpenAI-format messages to Anthropic format.

        Args:
            messages: List of message objects in OpenAI format

        Returns:
            Dictionary with system prompt and Anthropic-formatted messages
        """
        # Extract system message
        system_prompt = None
        for msg in messages:
            if msg["role"] == "system":
                system_prompt = msg["content"]
                break

        # Convert messages to Anthropic format (excluding system)
        anthropic_messages = []
        for msg in messages:
            if msg["role"] == "system":
                continue
            elif msg["role"] == "tool":
                # Convert tool messages to assistant content with tool response
                anthropic_messages.append(
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "text",
                                "text": f"Tool '{msg.get('name', 'unknown')}' result: {msg.get('content', '')}",
                            }
                        ],
                    }
                )
            else:
                # Handle regular user/assistant messages
                content = msg.get("content")
                # If content is a string, convert to text block
                if isinstance(content, str):
                    anthropic_messages.append(
                        {
                            "role": msg["role"],
                            "content": [{"type": "text", "text": content}],
                        }
                    )
                elif isinstance(content, list):
                    # If it's already structured content, pass it through
                    anthropic_messages.append({"role": msg["role"], "content": content})
                else:
                    # Handle messages with no content or None content
                    anthropic_messages.append(
                        {"role": msg["role"], "content": []}  # Empty content
                    )

        return {"system": system_prompt, "messages": anthropic_messages}

    def _convert_tools_to_anthropic_format(
        self, tools: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Convert OpenAI-format tools to Anthropic format.

        Args:
            tools: List of tool definitions in OpenAI format

        Returns:
            List of tools in Anthropic format
        """
        if not tools:
            return []

        anthropic_tools = []
        for tool in tools:
            if tool["type"] == "function":
                func = tool["function"]
                anthropic_tools.append(
                    {
                        "name": func["name"],
                        "description": func.get("description", ""),
                        "input_schema": func.get("parameters", {}),
                    }
                )

        return anthropic_tools

    def _normalize_claude_response(
        self, data: Dict[str, Any], run_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Normalize Claude response to OpenAI format.

        Args:
            data: Raw Claude API response
            run_id: Optional run identifier for logging

        Returns:
            Response in OpenAI format
        """
        # Initialize normalized response
        normalized_response = {"role": "assistant"}

        # Extract content blocks
        content_blocks = data.get("content", [])

        # Extract text parts and tool calls
        text_parts = []
        tool_calls = []

        # Track unique tool calls based on name + arguments
        seen_tool_signatures = set()

        for idx, block in enumerate(content_blocks):
            if isinstance(block, dict):
                if block.get("type") == "text":
                    text_parts.append(block.get("text", ""))
                elif block.get("type") == "tool_use":
                    tool_name = block.get("name", "unknown_tool")
                    tool_input = block.get("input", {})

                    # Create a signature combining tool name and arguments
                    # Sort keys to ensure consistent ordering
                    tool_args_str = json.dumps(tool_input, sort_keys=True)
                    tool_signature = f"{tool_name}:{tool_args_str}"

                    # Skip duplicate tool calls with identical name AND arguments
                    # This prevents exact duplicates while allowing same tool with different args
                    if tool_signature in seen_tool_signatures:
                        self.logger.log_error(
                            error_type="duplicate_tool_call",
                            error_details=f"Skipping duplicate tool call to {tool_name} with identical arguments",
                            run_id=run_id,
                        )
                        continue

                    # Add to seen tool signatures
                    seen_tool_signatures.add(tool_signature)

                    # Create OpenAI-compatible tool call format
                    tool_calls.append(
                        {
                            "id": block.get("id", f"call_{idx}"),
                            "type": "function",
                            "function": {
                                "name": tool_name,
                                "arguments": json.dumps(tool_input),
                            },
                        }
                    )

        # Set content as plain text for standard handlers
        normalized_response["content"] = "".join(text_parts) if text_parts else None

        # Add tool calls if found
        if tool_calls:
            normalized_response["tool_calls"] = tool_calls
            self.logger.log_response(
                response={"tool_calls": tool_calls},
                response_time=None,
                run_id=run_id,
                extra={
                    "message": f"Normalized {len(tool_calls)} tool calls from Claude response"
                },
            )

        return normalized_response

    def get_response(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        tool_choice: Optional[Dict[str, Any]] = None,
        run_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Get a response from Claude API and convert to OpenAI-compatible format.

        Args:
            messages: List of message objects in OpenAI format
            tools: Optional list of tool definitions in OpenAI format
            temperature: Sampling temperature for the LLM
            max_tokens: Maximum number of tokens for the response
            tool_choice: Optional specification for tool selection behavior
            run_id: Optional run identifier for logging

        Returns:
            Response message object converted to OpenAI-compatible format
        """
        self.logger.log_request(
            messages=messages,
            tools=tools,
            temperature=temperature,
            run_id=run_id,
            extra={
                "message": f"Sending {len(messages)} messages to Claude. {'Including tools.' if tools else 'No tools.'}"
            },
        )

        # Convert messages to Anthropic format
        converted = self._convert_openai_messages_to_anthropic(messages)
        anthropic_messages = converted["messages"]
        system_prompt = converted["system"]

        # Prepare API payload
        payload = {
            "model": self.model_name,
            "messages": anthropic_messages,
            "temperature": temperature,
            "max_tokens": max_tokens or 4096,  # Anthropic requires max_tokens
        }

        # Add system prompt if present
        if system_prompt:
            payload["system"] = system_prompt

        # Convert tools to Anthropic format if provided
        if tools:
            anthropic_tools = self._convert_tools_to_anthropic_format(tools)
            if anthropic_tools:
                payload["tools"] = anthropic_tools

        # Prepare headers
        headers = {
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01",
            "x-api-key": self.api_key,
        }

        # Log the LLM request with our standardized logger
        self.logger.log_request(
            messages=messages, tools=tools, temperature=temperature, run_id=run_id
        )

        # Make API request and handle response
        try:
            # Use configurable timeout
            with httpx.Client(timeout=self.timeout) as client:
                # Log HTTP request with standardized logger
                self.logger.log_http_req(
                    url=self.api_url,
                    method="POST",
                    headers=headers,
                    body=payload,
                    run_id=run_id,
                )

                # Time the API call
                start_time = time.time()

                # Send the request
                response = client.post(self.api_url, headers=headers, json=payload)

                # Calculate response time
                response_time = time.time() - start_time

                # Log HTTP response with standardized logger
                self.logger.log_http_resp(
                    url=self.api_url,
                    status_code=response.status_code,
                    headers=dict(response.headers),
                    body=response.text,
                    response_time=response_time,
                    run_id=run_id,
                )

                # Handle error responses
                if response.status_code >= 400:
                    try:
                        error_json = response.json()
                        self.logger.log_error(
                            error_type="api_error",
                            status_code=response.status_code,
                            error_details=error_json,
                            run_id=run_id,
                        )
                    except Exception as e:
                        error_text = f"Raw error response: {response.text}"
                        self.logger.log_error(
                            error_type="parsing_error",
                            status_code=response.status_code,
                            error_details={
                                "error": str(e),
                                "raw_response": response.text,
                            },
                            run_id=run_id,
                        )

                response.raise_for_status()

                # Parse the Claude response
                data = response.json()

                # Normalize Claude response to OpenAI format
                openai_compatible_response = self._normalize_claude_response(
                    data, run_id=run_id
                )

                # Log LLM response with standardized logger
                self.logger.log_response(
                    response=openai_compatible_response,
                    response_time=response_time,
                    run_id=run_id,
                    extra={
                        "has_tool_calls": "tool_calls" in openai_compatible_response,
                        "has_content": openai_compatible_response.get("content")
                        is not None,
                        "raw_response": data,
                    },
                )

                return openai_compatible_response

        except httpx.RequestError as e:
            self.logger.log_error(
                error_type="connection_error", error_details=str(e), run_id=run_id
            )
            return {
                "role": "assistant",
                "content": f"Error: Could not reach Claude API: {e}",
            }
        except (KeyError, IndexError, json.JSONDecodeError) as e:
            self.logger.log_error(
                error_type="parsing_error", error_details=str(e), run_id=run_id
            )
            return {
                "role": "assistant",
                "content": f"Error: Invalid Claude response format: {e}",
            }
