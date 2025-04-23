import json
import time
from typing import Dict, List, Any, Optional

import httpx

from .base import BaseLLMClient
from ..logger import LLMClientLogger


class OpenAICompatClient(BaseLLMClient):
    """
    Client for OpenAI and OpenAI-compatible APIs (e.g., Together, DeepInfra, Groq).

    Works with any provider that implements the OpenAI Chat Completions API format.
    """

    def __init__(self, api_key, api_url, model_name, timeout=180.0):
        """Initialize the OpenAI-compatible client."""
        super().__init__(api_key, api_url, model_name, timeout)
        # Initialize the standardized logger
        self.logger = LLMClientLogger("openai_compat", model_name)

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
        Get a response message object from the LLM using OpenAI-compatible API.

        Args:
            messages: List of message objects to send to the LLM
            tools: Optional list of tool definitions
            temperature: Sampling temperature for the LLM
            max_tokens: Maximum number of tokens for the response
            tool_choice: Optional specification for tool selection behavior
            run_id: Optional run identifier for logging

        Returns:
            Response message object from the LLM in standard OpenAI format
        """
        # Log the LLM request with standardized logger
        self.logger.log_request(
            messages=messages, tools=tools, temperature=temperature, run_id=run_id
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
                # Log HTTP request with standardized logger
                self.logger.log_http_req(
                    url=self.api_url,
                    method="POST",
                    headers=headers,
                    body=payload,
                    run_id=run_id,
                )

                # Send the request and time it
                start_time = time.time()
                response = client.post(self.api_url, headers=headers, json=payload)
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

                # If there's an error, log it with standardized logger
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

                # Log the LLM response with standardized logger
                self.logger.log_response(
                    response=message,
                    response_time=response_time,
                    run_id=run_id,
                    extra={
                        "has_tool_calls": "tool_calls" in message,
                        "has_content": message.get("content") is not None,
                        "raw_response": data,
                    },
                )

                return message

        except httpx.RequestError as e:
            self.logger.log_error(
                error_type="connection_error", error_details=str(e), run_id=run_id
            )
            # Return an error-like message object
            return {"role": "assistant", "content": f"Error: Could not reach LLM: {e}"}
        except (KeyError, IndexError, json.JSONDecodeError) as e:
            self.logger.log_error(
                error_type="parsing_error", error_details=str(e), run_id=run_id
            )
            return {
                "role": "assistant",
                "content": f"Error: Invalid LLM response format: {e}",
            }
