# MCPMan (MCP Manager)

MCPMan orchestrates interactions between LLMs and Model Context Protocol (MCP) servers, making it easy to create powerful agentic workflows.

## Quick Start

Run MCPMan instantly without installing using `uvx`:

```bash
# Use the calculator server to perform math operations
uvx mcpman -c server_configs/calculator_server_mcp.json -i openai -m gpt-4.1-mini -p "What is 1567 * 329 and then divide by 58?"

# Use the datetime server to check time in different timezones
uvx mcpman -c server_configs/datetime_server_mcp.json -i gemini -m gemini-2.0-flash-001 -p "What time is it right now in Tokyo, London, and New York?"

# Use the filesystem server with Ollama for file operations
uvx mcpman -c server_configs/filesystem_server_mcp.json -i ollama -m llama3:8b -p "Create a file called example.txt with a sample Python function, then read it back to me"

# Use the filesystem server with LMStudio's local models
uvx mcpman -c server_configs/filesystem_server_mcp.json -i lmstudio -m qwen2.5-7b-instruct-1m -p "Create a simple JSON file with sample data and read it back to me"
```

You can also use `uv run` for quick one-off executions directly from GitHub:

```bash
uv run github.com/ericflo/mcpman -c server_configs/calculator_server_mcp.json -i openai -m gpt-4.1-mini -p "What is 256 * 432?"
```

## Core Features

- **One-command setup**: Manage and launch MCP servers directly
- **Tool orchestration**: Automatically connect LLMs to any MCP-compatible tool
- **Detailed logging**: Structured JSON logs for every interaction with run ID tracking
- **Log replay**: Visualize previous conversations with the mcpreplay tool
- **Multiple LLM support**: Works with OpenAI, Google Gemini, Ollama, LMStudio and more
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
# Run mode (default)
mcpman -c <CONFIG_FILE> -i <IMPLEMENTATION> -m <MODEL> -p "<PROMPT>"

# Replay mode
mcpman --replay [--replay-file <LOG_FILE>]
```

Examples:

```bash
# Use local models with Ollama for filesystem operations
mcpman -c ./server_configs/filesystem_server_mcp.json \
       -i ollama \
       -m codellama:13b \
       -p "Create a simple bash script that counts files in the current directory and save it as count.sh"

# Use OpenAI with multi-server config
mcpman -c ./server_configs/multi_server_mcp.json \
       -i openai \
       -m gpt-4.1-mini \
       -s "You are a helpful assistant. Use tools effectively." \
       -p "Calculate 753 * 219 and tell me what time it is in Sydney, Australia"

# Replay the most recent conversation
mcpman --replay

# Replay a specific log file
mcpman --replay --replay-file ./logs/mcpman_20250422_142536.jsonl
```

## Server Configuration

MCPMan uses JSON configuration files to define the MCP servers. Examples:

**Calculator Server**:
```json
{
  "mcpServers": {
    "calculator": {
      "command": "python",
      "args": ["-m", "mcp_servers.calculator"],
      "env": {}
    }
  }
}
```

**DateTime Server**:
```json
{
  "mcpServers": {
    "datetime": {
      "command": "python",
      "args": ["-m", "mcp_servers.datetime_utils"],
      "env": {}
    }
  }
}
```

**Filesystem Server**:
```json
{
  "mcpServers": {
    "filesystem": {
      "command": "python",
      "args": ["-m", "mcp_servers.filesystem_ops"],
      "env": {}
    }
  }
}
```

## Key Options

| Option | Description |
|--------|-------------|
| `-c, --config <PATH>` | Path to MCP server config file |
| `-i, --implementation <IMPL>` | LLM implementation (openai, gemini, ollama, lmstudio) |
| `-m, --model <MODEL>` | Model name (gpt-4.1-mini, gemini-2.0-flash-001, llama3:8b, qwen2.5-7b-instruct-1m, etc.) |
| `-p, --prompt <PROMPT>` | User prompt (text or file path) |
| `-s, --system <MESSAGE>` | Optional system message |
| `--base-url <URL>` | Custom endpoint URL |
| `--temperature <FLOAT>` | Sampling temperature (default: 0.7) |
| `--max-tokens <INT>` | Maximum response tokens |
| `--no-verify` | Disable task verification |
| `--strict-tools` | Enable strict mode for tool schemas (default) |
| `--no-strict-tools` | Disable strict mode for tool schemas |
| `--replay` | Run in replay mode to visualize a previous conversation log |
| `--replay-file <PATH>` | Path to the log file to replay (defaults to latest log) |

API keys are set via environment variables: `OPENAI_API_KEY`, `GEMINI_API_KEY`, etc.  
Tool schema behavior can be configured with the `MCPMAN_STRICT_TOOLS` environment variable.

## Why MCPMan?

- **Standardized interaction**: Unified interface for diverse tools
- **Simplified development**: Abstract away LLM-specific tool call formats
- **Debugging support**: Detailed JSONL logs for every step in the agent process 
- **Local or cloud**: Works with both local and cloud-based LLMs

## Currently Supported LLMs

- OpenAI (GPT-4.1, GPT-4.1-mini, GPT-4.1-nano)
- Anthropic Claude (claude-3-7-sonnet-20250219, etc.)
- Google Gemini (gemini-2.0-flash-001, etc.)
- OpenRouter
- Ollama (llama3, codellama, etc.)
- LM Studio (Qwen, Mistral, and other local models)

## Development Setup

```bash
# Clone and setup
git clone https://github.com/ericflo/mcpman.git
cd mcpman

# Create environment and install deps
uv venv
source .venv/bin/activate  # Linux/macOS
# or .venv\Scripts\activate  # Windows
uv pip install -e ".[dev]"

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