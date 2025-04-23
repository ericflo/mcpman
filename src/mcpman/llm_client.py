"""
This module provides the main interface for interacting with LLM providers.

It re-exports the factory function from the llm_clients package,
maintaining backward compatibility while providing enhanced functionality.
"""

from .llm_clients import create_llm_client, BaseLLMClient

# For backward compatibility
LLMClient = BaseLLMClient

# Export the factory function
__all__ = ["create_llm_client", "LLMClient"]
