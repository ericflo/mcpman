# MCPMan (MCP Manager)

MCPMan orchestrates interactions between LLMs and Model Context Protocol (MCP) servers, making it easy to create powerful agentic workflows.

## Quick Start

Run MCPMan instantly without installing using `uvx`:

```bash
# Run with OpenAI
uvx mcpman -c server_configs/multi_server_mcp.json -i openai -m gpt-4o -p "Write a short poem about robots"

# Run with Claude
uvx mcpman -c server_configs/calculator_server_mcp.json -i anthropic -m claude-3-sonnet-20240229 -p "Calculate 245 * 378"

# Run with a local Ollama model
uvx mcpman -c server_configs/filesystem_server_mcp.json -i ollama -m llama3:8b -p "List files in this directory"
```

You can also use `uv run` for quick one-off executions:

```bash
uv run github.com/ericflo/mcpman -c server_configs/multi_server_mcp.json -i openai -m gpt-4o -p "What time is it in Tokyo?"
```

## Core Features

- **One-command setup**: Manage and launch MCP servers directly
- **Tool orchestration**: Automatically connect LLMs to any MCP-compatible tool
- **Detailed logging**: JSON structured logs for every interaction
- **Multiple LLM support**: Works with OpenAI, Anthropic, Google, Ollama, LMStudio and more
- **Flexible configuration**: Supports stdio and SSE server communication

## Installation

```bash
# Install with pip
pip install mcpman

# Install with uv
uv pip install mcpman

# Install from GitHub
uvx pip install git+https://github.com/ericflo/mcpman.git
```

## Basic Usage

```bash
mcpman -c <CONFIG_FILE> -i <IMPLEMENTATION> -m <MODEL> -p "<PROMPT>"
```

Examples:

```bash
# Use local models with Ollama
mcpman -c ./server_configs/filesystem_server_mcp.json \
       -i ollama \
       -m gemma3:4b-it-qat \
       -p "List files in the current directory and count the lines in README.md"

# Use OpenAI with system message
mcpman -c ./server_configs/multi_server_mcp.json \
       -i openai \
       -m gpt-4o \
       -s "You are a helpful assistant. Use tools effectively." \
       -p "What time is it in Tokyo right now and what's the weather like there?"
```

## Server Configuration

MCPMan uses JSON configuration files to define the MCP servers. Examples:

**Node.js stdio Server**:
```json
{
  "mcpServers": {
    "calculator": {
      "command": "npx",
      "args": ["-y", "mcp-server"],
      "env": { "API_KEY": "value" }
    }
  }
}
```

**Python stdio Server**:
```json
{
  "mcpServers": {
    "datetime": {
      "command": "python",
      "args": ["-m", "mcp_servers.datetime_utils"],
      "env": { "TIMEZONE_API_KEY": "abc123" }
    }
  }
}
```

**SSE Server** (manually managed):
```json
{
  "mcpServers": {
    "filesystem": {
      "url": "http://localhost:3000/sse"
    }
  }
}
```

## Key Options

| Option | Description |
|--------|-------------|
| `-c, --config <PATH>` | Path to MCP server config file |
| `-i, --implementation <IMPL>` | LLM implementation (openai, anthropic, google, ollama, lmstudio) |
| `-m, --model <MODEL>` | Model name (gpt-4o, claude-3-opus-20240229, etc.) |
| `-p, --prompt <PROMPT>` | User prompt (text or file path) |
| `-s, --system <MESSAGE>` | Optional system message |
| `--base-url <URL>` | Custom endpoint URL |
| `--temperature <FLOAT>` | Sampling temperature (default: 0.7) |
| `--max-tokens <INT>` | Maximum response tokens |
| `--no-verify` | Disable task verification |

API keys are set via environment variables: `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, etc.

## Why MCPMan?

- **Standardized interaction**: Unified interface for diverse tools
- **Simplified development**: Abstract away LLM-specific tool call formats
- **Debugging support**: Detailed JSONL logs for every step in the agent process 
- **Local or cloud**: Works with local or cloud-based LLMs

## Supported LLMs

- OpenAI (GPT models)
- Anthropic (Claude models)
- Google Gemini
- OpenRouter
- Ollama (local models)
- LM Studio (local models)

## Development Setup

```bash
# Clone and setup
git clone https://github.com/ericflo/mcpman.git
cd mcpman

# Create environment and install deps
uvx venv
source .venv/bin/activate  # Linux/macOS
# or .venv\Scripts\activate  # Windows
uvx pip install -e ".[dev]"

# Run tests
pytest tests/
```

## Project Structure

- `src/mcpman/`: Core source code
- `mcp_servers/`: Example MCP servers for testing
- `server_configs/`: Example configuration files
- `logs/`: Auto-generated structured JSONL logs

## License

Licensed under the [Apache License 2.0](LICENSE).