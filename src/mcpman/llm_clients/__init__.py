"""
LLM Clients for interfacing with various LLM providers.

This package provides a unified interface for working with different
LLM providers through a common API.
"""

import logging
from typing import Dict, Optional

from .base import BaseLLMClient
from .openai_compat import OpenAICompatClient
from .anthropic import AnthropicClient
from .openai_responses import OpenAIResponsesClient


def create_llm_client(
    provider_config: Dict[str, str],
    provider_name: Optional[str] = None,
) -> BaseLLMClient:
    """
    Factory function to create the appropriate LLM client based on provider.

    Args:
        provider_config: Provider configuration with url, key, and model
        provider_name: Optional provider name for explicit client selection

    Returns:
        Instance of the appropriate LLMClient implementation
    """
    # Map provider names to client classes
    provider_map = {
        "openai": OpenAICompatClient,
        "openai_responses": OpenAIResponsesClient,
        "anthropic": AnthropicClient,
        "lmstudio": OpenAICompatClient,
        "ollama": OpenAICompatClient,
        "openrouter": OpenAICompatClient,
        "together": OpenAICompatClient,
        "gemini": OpenAICompatClient,
        "groq": OpenAICompatClient,
        "hyperbolic": OpenAICompatClient,
        "deepinfra": OpenAICompatClient,
        # Add more providers here
    }

    # Default to OpenAI-compatible client
    client_class = OpenAICompatClient

    # If provider name is specified, use it
    if provider_name and provider_name in provider_map:
        client_class = provider_map[provider_name]
    # Otherwise try to auto-detect based on model name or URL
    elif provider_config.get("model"):
        model_name = provider_config["model"].lower()
        # Check if model name indicates Anthropic
        if "claude" in model_name:
            client_class = AnthropicClient
    elif provider_config.get("url") and "anthropic" in provider_config["url"].lower():
        # Check if URL indicates Anthropic
        client_class = AnthropicClient

    # Log the provider selection
    provider_str = provider_name or "auto-detected"
    logging.debug(f"Creating {client_class.__name__} for provider: {provider_str}")
    logging.debug(f"Model: {provider_config.get('model')}")
    logging.debug(f"API URL: {provider_config.get('url')}")

    # Extract client configuration
    api_key = provider_config.get("key", "")
    api_url = provider_config.get("url", "")
    model_name = provider_config.get("model", "")

    # Get timeout if provided, otherwise use default
    timeout = float(provider_config.get("timeout", 180.0))

    # Create and return the appropriate client
    return client_class(
        api_key=api_key,
        api_url=api_url,
        model_name=model_name,
        timeout=timeout,
    )


# Expose the public interface
__all__ = [
    "create_llm_client",
    "BaseLLMClient",
    "OpenAICompatClient",
    "AnthropicClient",
    "OpenAIResponsesClient",
]
