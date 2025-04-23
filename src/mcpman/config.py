import os
import json
import logging
from typing import Dict, Any, Optional

# Default configuration values
DEFAULT_CONFIG_PATH = "server_configs/calculator_server_mcp.json"

# LLM provider configuration
OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_DEFAULT_MODEL = "gpt-4.1-nano"

# OpenAI Responses API configuration (new API)
OPENAI_RESPONSES_API_URL = (
    "https://api.openai.com/v1"  # Base URL, not endpoint-specific
)
OPENAI_RESPONSES_API_KEY = os.getenv("OPENAI_API_KEY", "")  # Same key as regular OpenAI
OPENAI_RESPONSES_DEFAULT_MODEL = "o4-mini"  # Default to o4-mini as requested

ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
ANTHROPIC_DEFAULT_MODEL = "claude-3-7-sonnet-20250219"

LMSTUDIO_API_URL = "http://127.0.0.1:1234/v1/chat/completions"
LMSTUDIO_DEFAULT_MODEL = "qwen2.5-7b-instruct-1m"

OLLAMA_API_URL = "http://127.0.0.1:11434/api/generate"
OLLAMA_DEFAULT_MODEL = "qwen2.5:7b"

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_DEFAULT_MODEL = "google/gemini-2.0-flash-001"

TOGETHER_API_URL = "https://api.together.xyz/v1/chat/completions"
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY", "")
TOGETHER_DEFAULT_MODEL = "meta-llama/Llama-4-Scout-17B-16E-Instruct"

GEMINI_API_URL = (
    "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_DEFAULT_MODEL = "gemini-2.0-flash-001"

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_DEFAULT_MODEL = "llama-3.3-70b-versatile"

HYPERBOLIC_API_URL = "https://api.hyperbolic.xyz/v1/chat/completions"
HYPERBOLIC_API_KEY = os.getenv("HYPERBOLIC_API_KEY", "")
HYPERBOLIC_DEFAULT_MODEL = "deepseek-ai/DeepSeek-V3-0324"

DEEPINFRA_API_URL = "https://api.deepinfra.com/v1/openai/chat/completions"
DEEPINFRA_API_KEY = os.getenv("DEEPINFRA_API_KEY", "")
DEEPINFRA_DEFAULT_MODEL = "meta-llama/Llama-4-Scout-17B-16E-Instruct"

# Define all supported providers
PROVIDERS = {
    "openai": {
        "url": OPENAI_API_URL,
        "key": OPENAI_API_KEY,
        "default_model": OPENAI_DEFAULT_MODEL,
    },
    "openai_responses": {
        "url": OPENAI_RESPONSES_API_URL,
        "key": OPENAI_RESPONSES_API_KEY,
        "default_model": OPENAI_RESPONSES_DEFAULT_MODEL,
    },
    "anthropic": {
        "url": ANTHROPIC_API_URL,
        "key": ANTHROPIC_API_KEY,
        "default_model": ANTHROPIC_DEFAULT_MODEL,
    },
    "lmstudio": {
        "url": LMSTUDIO_API_URL,
        "key": "dummy",
        "default_model": LMSTUDIO_DEFAULT_MODEL,
    },
    "ollama": {"url": OLLAMA_API_URL, "key": "", "default_model": OLLAMA_DEFAULT_MODEL},
    "openrouter": {
        "url": OPENROUTER_API_URL,
        "key": OPENROUTER_API_KEY,
        "default_model": OPENROUTER_DEFAULT_MODEL,
    },
    "together": {
        "url": TOGETHER_API_URL,
        "key": TOGETHER_API_KEY,
        "default_model": TOGETHER_DEFAULT_MODEL,
    },
    "gemini": {
        "url": GEMINI_API_URL,
        "key": GEMINI_API_KEY,
        "default_model": GEMINI_DEFAULT_MODEL,
    },
    "groq": {
        "url": GROQ_API_URL,
        "key": GROQ_API_KEY,
        "default_model": GROQ_DEFAULT_MODEL,
    },
    "hyperbolic": {
        "url": HYPERBOLIC_API_URL,
        "key": HYPERBOLIC_API_KEY,
        "default_model": HYPERBOLIC_DEFAULT_MODEL,
    },
    "deepinfra": {
        "url": DEEPINFRA_API_URL,
        "key": DEEPINFRA_API_KEY,
        "default_model": DEEPINFRA_DEFAULT_MODEL,
    },
}

# Default provider details
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")
LLM_API_KEY = os.getenv("LLM_API_KEY", OPENAI_API_KEY)
LLM_API_URL = os.getenv("LLM_API_URL", OPENAI_API_URL)
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "gpt-4.1-nano")

# Tool configuration settings
STRICT_TOOLS = os.getenv("MCPMAN_STRICT_TOOLS", "true").lower() in (
    "true",
    "1",
    "yes",
    "y",
)

# Default system message for agent behavior
DEFAULT_SYSTEM_MESSAGE = """
You are an agent - please keep going until the user's query is completely resolved, before ending your turn and yielding back to the user. Only terminate your turn when you are sure that the problem is solved.
If you are not sure about file content or codebase structure pertaining to the user's request, use your tools to read files and gather the relevant information: do NOT guess or make up an answer.
You MUST plan extensively before each function call, and reflect extensively on the outcomes of the previous function calls. DO NOT do this entire process by making function calls only, as this can impair your ability to solve the problem and think insightfully.
""".strip()

# Default verification system message for task completion checking
DEFAULT_VERIFICATION_MESSAGE = """
You are a verification assistant responsible for determining if a task has been fully completed.
Your job is to analyze the conversation history and determine if the agent has successfully completed the task requested by the user.
Be critical and thorough in your assessment. Only confirm completion if ALL aspects of the task have been addressed.
If the task is not complete, provide specific feedback on what remains to be done.
""".strip()

# Default user prompt - kept as fallback but now the parameter is required
DEFAULT_USER_PROMPT = "What is 7 / 3 / 1.27?"

# Whether to only print the final output
OUTPUT_ONLY_MODE = False


def load_server_config(config_path: str) -> Dict[str, Any]:
    """
    Load and parse the server configuration from the specified JSON file.

    Args:
        config_path: Path to the JSON configuration file

    Returns:
        Dictionary with server configurations

    Raises:
        FileNotFoundError: If the config file doesn't exist
        json.JSONDecodeError: If the config file isn't valid JSON
    """
    try:
        with open(config_path, "r") as f:
            server_config = json.load(f)

        # Basic validation that mcpServers key exists
        if "mcpServers" not in server_config:
            logging.warning("No 'mcpServers' section found in config file")
            server_config["mcpServers"] = {}

        return server_config
    except FileNotFoundError:
        logging.error(f"Server config file not found: {config_path}")
        raise
    except json.JSONDecodeError:
        logging.error(f"Error decoding server config file: {config_path}")
        raise


def get_llm_configuration(
    provider_name: Optional[str] = None,
    api_url: Optional[str] = None,
    api_key: Optional[str] = None,
    model_name: Optional[str] = None,
    timeout: Optional[float] = None,
) -> Dict[str, str]:
    """
    Determine the LLM configuration based on provided parameters and environment variables.

    Args:
        provider_name: Name of the LLM provider (e.g., "openai", "ollama")
        api_url: Custom API URL for the LLM
        api_key: API key for the LLM
        model_name: Name of the model to use
        timeout: Request timeout in seconds

    Returns:
        Dictionary with "url", "key", "model", and "timeout" keys
    """
    result = {
        "url": None,
        "key": None,
        "model": None,
        "timeout": timeout or 180.0,  # Default to 3 minutes
    }

    # Provider-based configuration
    if provider_name:
        provider_info = PROVIDERS.get(provider_name)
        if provider_info:
            result["url"] = provider_info["url"]
            result["key"] = provider_info["key"]
            result["model"] = provider_info["default_model"]

    # Custom URL override
    if api_url:
        result["url"] = api_url

    # API key override
    if api_key:
        result["key"] = api_key

    # Model override
    if model_name:
        result["model"] = model_name

    # Fallbacks to environment variables if values are still None
    if result["url"] is None:
        result["url"] = LLM_API_URL

    if result["key"] is None:
        result["key"] = LLM_API_KEY

    if result["model"] is None:
        result["model"] = LLM_MODEL_NAME

    return result
