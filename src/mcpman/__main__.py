"""
Entry point for running mcpman as a module.

Example:
    python -m mcpman --config config.json --provider openai --model gpt-4o --user "Hello"
"""

from .cli import run

if __name__ == "__main__":
    run()
