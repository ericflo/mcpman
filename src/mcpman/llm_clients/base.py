from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional


class BaseLLMClient(ABC):
    """
    Abstract base class for all LLM client implementations.

    All LLM clients must normalize their native response formats to a standard format
    based on the OpenAI message structure. This allows the orchestrator to remain
    provider-agnostic.
    """

    def __init__(
        self, api_key: str, api_url: str, model_name: str, timeout: float = 180.0
    ):
        """
        Initialize base LLM client.

        Args:
            api_key: API key for the provider
            api_url: API URL for the provider
            model_name: Name of the model to use
            timeout: Request timeout in seconds (default: 180s)
        """
        self.api_key = api_key
        self.api_url = api_url
        self.model_name = model_name
        self.timeout = timeout

    @abstractmethod
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
        Get a response from the LLM and convert to standard format.

        Args:
            messages: List of message objects with role and content
            tools: Optional list of tool definitions
            temperature: Sampling temperature for the LLM
            max_tokens: Maximum number of tokens for the response
            tool_choice: Optional specification for tool selection behavior
            run_id: Optional run identifier for logging

        Returns:
            Response message object normalized to the OpenAI format:
            {
                "role": "assistant",
                "content": str | None,  # Text content (None if only tool calls)
                "tool_calls": [         # Only present if tools are used
                    {
                        "id": str,           # Unique ID for this tool call
                        "type": "function",  # Always "function" for now
                        "function": {
                            "name": str,        # Function name
                            "arguments": str    # JSON string of arguments
                        }
                    },
                    ...
                ]
            }
        """
        pass
