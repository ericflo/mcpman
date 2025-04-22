import json
import logging
import sys
from typing import Dict, List, Any, Optional

import httpx

from .base import BaseLLMClient


class OpenAICompatClient(BaseLLMClient):
    """
    Client for OpenAI and OpenAI-compatible APIs (e.g., Together, DeepInfra, Groq).

    Works with any provider that implements the OpenAI Chat Completions API format.
    """

    def get_response(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        tool_choice: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Get a response message object from the LLM using OpenAI-compatible API.

        Args:
            messages: List of message objects to send to the LLM
            tools: Optional list of tool definitions
            temperature: Sampling temperature for the LLM
            max_tokens: Maximum number of tokens for the response
            tool_choice: Optional specification for tool selection behavior

        Returns:
            Response message object from the LLM in standard OpenAI format
        """
        logging.debug(
            f"Sending {len(messages)} messages to LLM. {'Including tools.' if tools else 'No tools.'}"
        )

        # Prepare headers and payload
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        payload = {
            "messages": messages,
            "model": self.model_name,
            "temperature": temperature,
            "parallel_tool_calls": False,
        }

        # Add optional parameters
        if max_tokens:
            payload["max_tokens"] = max_tokens

        if tools:
            payload["tools"] = tools

            # Add tool_choice if provided
            if tool_choice:
                payload["tool_choice"] = tool_choice

        try:
            # Make the API request with configurable timeout
            with httpx.Client(timeout=self.timeout) as client:
                # Only log full payload on DEBUG level
                if logging.getLogger().isEnabledFor(logging.DEBUG):
                    try:
                        payload_str = json.dumps(payload, indent=2)
                        logging.debug(f"Sending LLM Payload:\n{payload_str}")
                    except Exception as e:
                        logging.debug(
                            f"Could not serialize LLM payload for logging: {e}"
                        )
                else:
                    logging.info(
                        f"Sending request to LLM model {self.model_name} at {self.api_url}..."
                    )

                # Send the request
                response = client.post(self.api_url, headers=headers, json=payload)
                logging.debug(
                    f"Received response from LLM: Status {response.status_code}"
                )

                # If there's an error, log the response body
                if response.status_code >= 400:
                    try:
                        error_json = response.json()
                        error_message = f"API Error ({response.status_code}): {json.dumps(error_json, indent=2)}"
                        logging.error(error_message)
                        print("\n" + "=" * 80, file=sys.stderr)
                        print(error_message, file=sys.stderr)
                        print("=" * 80 + "\n", file=sys.stderr)
                    except Exception as e:
                        error_text = f"Raw error response: {response.text}"
                        logging.error(f"Error parsing error response: {e}")
                        logging.error(error_text)
                        print("\n" + "=" * 80, file=sys.stderr)
                        print(f"Failed to parse error JSON: {e}", file=sys.stderr)
                        print(error_text, file=sys.stderr)
                        print("=" * 80 + "\n", file=sys.stderr)

                response.raise_for_status()

                # Parse the response
                data = response.json()
                if not data or "choices" not in data or not data["choices"]:
                    raise KeyError("Invalid response: 'choices' missing or empty.")
                if "message" not in data["choices"][0]:
                    raise KeyError(
                        "Invalid response: 'message' missing in first choice."
                    )

                # Get the message in OpenAI format (already standardized)
                message = data["choices"][0]["message"]

                # Validate the message format has required fields
                if "role" not in message:
                    message["role"] = "assistant"
                if "content" not in message and "tool_calls" not in message:
                    message["content"] = ""

                logging.debug(f"OpenAI response object: {message}")
                return message

        except httpx.RequestError as e:
            logging.error(f"Error communicating with LLM at {self.api_url}: {e}")
            # Return an error-like message object
            return {"role": "assistant", "content": f"Error: Could not reach LLM: {e}"}
        except (KeyError, IndexError, json.JSONDecodeError) as e:
            logging.error(f"Error parsing LLM response: {e}")
            return {
                "role": "assistant",
                "content": f"Error: Invalid LLM response format: {e}",
            }
