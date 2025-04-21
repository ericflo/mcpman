# MCPMan (MCP Manager)

MCPMan is a tool designed to manage and interact with Model Context Protocol (MCP) servers, enabling agentic workflows powered by various Large Language Models (LLMs).

## Installation

MCPMan uses [`uv`](https://github.com/astral-sh/uv) for dependency management and installation. `uv` is a very fast Python package installer and resolver, written in Rust, intended as a drop-in replacement for `pip` and `pip-tools`.

Ensure you have `uv` installed. You can usually install it via `pip`, `pipx`, or your system's package manager. See the [`uv` installation guide](https://github.com/astral-sh/uv?tab=readme-ov-file#installation) for details.

Once `uv` is installed, you can install MCPMan directly from the repository (replace `<TAG>` with the desired version or omit for the latest main branch):

```bash
uv pip install git+https://github.com/ericflo/mcpman.git@<TAG>
```

## Core Functionality

1.  **Server Management**: Spin up and manage multiple MCP servers defined in a configuration file, similar to how clients like Cursor handle MCP integrations. MCPMan supports servers communicating over `stdio` (managed by MCPMan) and `SSE` (Server-Sent Events, typically manually managed by the user).
2.  **Tool Integration**: Automatically connect to the managed MCP servers and extract the JSON schema for the tools they provide.
3.  **Agentic Loops & Logging**: Execute agentic tasks by interacting with LLMs. MCPMan orchestrates the communication between the LLM and the MCP tools, providing **detailed, incremental logging** of the agent's reasoning, tool calls, and results throughout the process.

## Why MCPMan?

Developers building LLM agents with tool use often face challenges:

- **Diverse Tooling APIs**: Interacting with different tools often requires writing bespoke client code for each protocol (`stdio`, `SSE`, REST, etc.).
- **Inconsistent LLM Integration**: Different LLMs have varying levels of support and formats for tool calling.
- **Debugging Agent Behavior**: Understanding _why_ an agent made a particular decision or tool call can be difficult without detailed, structured logs.

MCPMan addresses these challenges by:

- **Standardizing Interaction**: Leveraging the Model Context Protocol (MCP), it provides a unified way to manage and communicate with diverse tools, regardless of their underlying transport (`stdio` or `SSE`).
- **Simplifying LLM Integration**: It abstracts the complexities of tool schema formatting and interaction loops for compatible LLMs (initially focusing on OpenAI-compatible APIs).
- **Providing Granular Logging**: MCPMan captures each step of the agent's interaction—prompting, LLM reasoning (if available), tool selection, tool input/output, and final responses—creating a clear audit trail for analysis and debugging.

It acts as a central orchestrator, streamlining the development and debugging of agentic workflows that rely on external tools.

## How it Works

1.  **Configuration**: Define the MCP servers you want to use in a JSON configuration file. This file specifies how to start (`stdio`) or connect (`SSE`) to each server.
2.  **Initialization**: Provide an initial prompt (system and/or user message) to kick off the agentic process.
3.  **Execution & Logging**:
    - MCPMan sends the prompt and available tool schemas to the configured LLM.
    - **(Log)** Agent receives the request.
    - The LLM processes the prompt and may request to use one or more tools.
    - **(Log)** Agent decides to call a tool (or respond directly).
    - MCPMan receives the tool call request, executes the corresponding function via the appropriate MCP server, and sends the result back to the LLM.
    - **(Log)** Tool execution details (request, response/error).
    - This loop continues until the LLM indicates the task is complete or no further tool calls are needed, with each step being logged.

## Configuration Examples

MCPMan uses a JSON configuration file to define the servers to manage. This format is inspired by clients like Cursor and supports both `stdio` and `SSE` transports.

**Example: Node.js stdio Server (Managed by MCPMan)**

```json
{
  "mcpServers": {
    "server-name": {
      "command": "npx",
      "args": ["-y", "mcp-server"],
      "env": {
        "API_KEY": "value"
      }
    }
  }
}
```

**Example: Python stdio Server (Managed by MCPMan)**

```json
{
  "mcpServers": {
    "server-name": {
      "command": "python",
      "args": ["mcp-server.py"],
      "env": {
        "API_KEY": "value"
      }
    }
  }
}
```

**Example: SSE Server (Manually Managed)**

For SSE servers, you need to run the server process separately and provide the URL to MCPMan.

```json
{
  "mcpServers": {
    "server-name": {
      "url": "http://localhost:3000/sse",
      "env": {
        "API_KEY": "value" // Environment variables/secrets needed by the server
      }
    }
  }
}
```

## Usage

MCPMan is designed to be run from the command line. The minimum required arguments specify the MCP server configuration, the LLM implementation/model, and the initial prompt.

```bash
mcpman -c <PATH> -i <IMPLEMENTATION> -m <MODEL> -p "<PROMPT>"
```

**Example: Simple Invocation (Ollama - Local)**

This example uses a local Ollama model with just the required flags.

```bash
mcpman -c ./mcp-servers.json \
       -i ollama \
       -m gemma3:4b-it-qat \
       -p "Read the file 'README.md' and summarize its main points."
```

**Example: More Options (OpenAI)**

You can add more flags for finer control, like providing a system message or adjusting model parameters.

```bash
mcpman -c ./mcp-servers.json \
       -i openai \
       -m gpt-4o \
       -s "You are a helpful AI assistant. Use tools effectively to answer the user." \
       --temperature 0.5 \
       -p "Check the weather in London and find the top 3 news headlines from Associated Press."
```

**Key Options:**

- `-c, --config <PATH>`: (Required) Path to the JSON file containing MCP server configurations.
- `-i, --impl, --implementation <IMPLEMENTATION>`: (Required) Name of the LLM implementation to use (e.g., `openai`, `anthropic`, `google`, `ollama`, `lmstudio`).
- `-m, --model <MODEL>`: (Required) Model identifier for the chosen implementation (e.g., `gpt-4o`, `claude-3-opus-20240229`, `gemma3:12b-it-qat`).
- `-p, --prompt <PROMPT>`: (Required) The initial prompt to start the agentic loop. Can be a message or a path to a file containing the prompt.
- `-s, --system <MESSAGE>`: (Optional) An initial system message to guide the LLM's behavior.
- `--base-url <URL>`: (Optional) Custom base URL for implementations like Ollama, LMStudio, or other OpenAI-compatible endpoints. Defaults to implementation-specific standards (e.g., `http://localhost:11434` for Ollama).
- `--temperature <FLOAT>`: (Optional) Sampling temperature for the LLM (e.g., `0.7`).
- `--max-tokens <INT>`: (Optional) Maximum number of tokens for the LLM response.
- `--no-verify`: (Optional) Disable task verification. By default, verification is enabled to ensure the task is complete before finishing.
- `--verify-prompt <PROMPT>`: (Optional) Provide a custom verification prompt or path to a verification prompt file. Cannot be used with `--no-verify`.

_(Note: API keys are typically expected to be configured via environment variables like `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, etc.)_

## Supported LLMs

MCPMan aims for broad compatibility. Initially, it targets LLMs supporting the OpenAI `/chat/completions` endpoint format with tool calling capabilities. This includes providers such as:

- OpenAI (GPT models)
- Google Gemini (via compatible endpoints)
- Anthropic Claude (via compatible endpoints)
- OpenRouter
- LM Studio (local models)
- Ollama (local models)

## Future Goals

- Support for additional LLM provider APIs (like Anthropic's native API for models such as Claude 3.7).
- More sophisticated agentic control flows.
- Enhanced configuration options.

## Development Setup

Interested in contributing? Here's how to set up your development environment using `uv`:

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/ericflo/mcpman.git
    cd mcpman
    ```

2.  **Create and activate a virtual environment:**

    ```bash
    uv venv # Creates a .venv directory
    source .venv/bin/activate # Linux/macOS
    # .venv\Scripts\activate # Windows
    ```

3.  **Install dependencies (including development tools):**

    ```bash
    uv pip install -e ".[dev]"
    ```

    This installs the package in editable mode (`-e`) along with the development dependencies defined in `pyproject.toml`.

4.  **Run tests (Example):**

    ```bash
    pytest tests/
    ```

5.  **(Optional) Pre-commit Hooks:** If the project uses pre-commit hooks for linting/formatting, install them:
    ```bash
    pre-commit install
    ```

Now you're ready to start developing!

## Project Structure

The project follows a standard Python project layout:

- `src/mcpman/`: Contains the core source code for the MCPMan application.
  - `cli.py`: Entry point for the command-line interface (using Typer/Click).
  - `mcp/`: Modules related to MCP server management and communication.
  - `llm/`: Modules for interacting with different LLM providers.
  - `logging/`: Custom logging setup.
  - `config.py`: Configuration loading and validation.
- `tests/`: Contains unit and integration tests.
- `pyproject.toml`: Defines project metadata, dependencies (including optional `[dev]` dependencies), and build system configuration (managed by `uv`).
- `README.md`: This file.
- `.gitignore`: Specifies intentionally untracked files that Git should ignore.
- `(Optional) .pre-commit-config.yaml`: Configuration for pre-commit hooks.

Understanding this structure will help you navigate the codebase and locate relevant files for your contributions.
