import asyncio
import os
from src.mcpman.config import get_llm_configuration
from src.mcpman.llm_client import create_llm_client

async def test_llm_clients():
    # Test OpenAI client with gpt-4.1-nano
    print('
--- Testing OpenAI client with gpt-4.1-nano ---')
    openai_config = get_llm_configuration(provider_name='openai')
    openai_client = create_llm_client(openai_config, 'openai')
    
    # Print client type
    print(f'Client type: {type(openai_client).__name__}')
    print(f'Model: {openai_client.model_name}')
    
    # Test Anthropic client with claude
    print('
--- Testing Anthropic client with claude-3-7-sonnet-20250219 ---')
    anthropic_config = get_llm_configuration(provider_name='anthropic')
    anthropic_client = create_llm_client(anthropic_config, 'anthropic')
    
    # Print client type
    print(f'Client type: {type(anthropic_client).__name__}')
    print(f'Model: {anthropic_client.model_name}')
    
    # Test auto-detection with model name
    print('
--- Testing auto-detection with claude model ---')
    claude_config = {
        'url': 'https://api.anthropic.com/v1/messages',
        'key': os.environ.get('ANTHROPIC_API_KEY', ''),
        'model': 'claude-3-7-sonnet-20250219'
    }
    auto_client = create_llm_client(claude_config)
    
    # Print client type
    print(f'Client type: {type(auto_client).__name__}')
    print(f'Model: {auto_client.model_name}')

asyncio.run(test_llm_clients())

