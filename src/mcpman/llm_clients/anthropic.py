import json
import logging
import sys
from typing import Dict, List, Any, Optional, Union

import httpx

from .base import BaseLLMClient


class AnthropicClient(BaseLLMClient):
    """
    Client for Anthropic Claude API (v1/messages).

    Handles the conversion between OpenAI-compatible format and Anthropic's API.
    """

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

    def _normalize_claude_response(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize Claude response to OpenAI format.

        Args:
            data: Raw Claude API response

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
                        logging.warning(
                            f"Skipping duplicate tool call to {tool_name} with identical arguments"
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
            logging.debug(
                f"Normalized {len(tool_calls)} tool calls from Claude response"
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
        Get a response from Claude API and convert to OpenAI-compatible format.

        Args:
            messages: List of message objects in OpenAI format
            tools: Optional list of tool definitions in OpenAI format
            temperature: Sampling temperature for the LLM
            max_tokens: Maximum number of tokens for the response
            tool_choice: Optional specification for tool selection behavior

        Returns:
            Response message object converted to OpenAI-compatible format
        """
        logging.debug(
            f"Sending {len(messages)} messages to Claude. {'Including tools.' if tools else 'No tools.'}"
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

        # Make API request and handle response
        try:
            # Use configurable timeout
            with httpx.Client(timeout=self.timeout) as client:
                # Only log full payload on DEBUG level
                if logging.getLogger().isEnabledFor(logging.DEBUG):
                    try:
                        payload_str = json.dumps(payload, indent=2)
                        logging.debug(f"Sending Claude Payload:\n{payload_str}")
                    except Exception as e:
                        logging.debug(
                            f"Could not serialize Claude payload for logging: {e}"
                        )
                else:
                    logging.info(
                        f"Sending request to Claude model {self.model_name}..."
                    )

                # Send the request
                response = client.post(self.api_url, headers=headers, json=payload)
                logging.debug(
                    f"Received response from Claude: Status {response.status_code}"
                )

                # If there's an error, log the response body
                if response.status_code >= 400:
                    try:
                        error_json = response.json()
                        error_message = f"Claude API Error ({response.status_code}): {json.dumps(error_json, indent=2)}"
                        logging.error(error_message)
                        print("\n" + "=" * 80, file=sys.stderr)
                        print(error_message, file=sys.stderr)
                        print("=" * 80 + "\n", file=sys.stderr)
                    except Exception as e:
                        error_text = f"Raw error response: {response.text}"
                        logging.error(f"Error parsing Claude error response: {e}")
                        logging.error(error_text)
                        print("\n" + "=" * 80, file=sys.stderr)
                        print(
                            f"Failed to parse Claude error JSON: {e}", file=sys.stderr
                        )
                        print(error_text, file=sys.stderr)
                        print("=" * 80 + "\n", file=sys.stderr)

                response.raise_for_status()

                # Parse the Claude response
                data = response.json()

                # Log the full Claude response at debug level
                if logging.getLogger().isEnabledFor(logging.DEBUG):
                    try:
                        logging.debug(
                            f"Claude raw response: {json.dumps(data, indent=2)}"
                        )
                    except Exception as e:
                        logging.debug(
                            f"Couldn't serialize Claude response for logging: {e}"
                        )

                # Normalize Claude response to OpenAI format
                openai_compatible_response = self._normalize_claude_response(data)

                logging.debug(
                    f"Converted Claude response: {openai_compatible_response}"
                )
                return openai_compatible_response

        except httpx.RequestError as e:
            logging.error(f"Error communicating with Claude API: {e}")
            return {
                "role": "assistant",
                "content": f"Error: Could not reach Claude API: {e}",
            }
        except (KeyError, IndexError, json.JSONDecodeError) as e:
            logging.error(f"Error parsing Claude response: {e}")
            return {
                "role": "assistant",
                "content": f"Error: Invalid Claude response format: {e}",
            }
