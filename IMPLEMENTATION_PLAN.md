# MCPMan Implementation Plan

This document outlines the steps to build the MCPMan tool, prioritizing an incremental approach with runnable milestones.

## Phase 1: Core Foundation & Minimal Viable Product (MVP)

Goal: Create a basic CLI application that can load configuration, start _one_ stdio MCP server, send a prompt to an LLM (without tools), get a response, and shut down cleanly.

- [ ] **Project Setup:**
  - [ ] Initialize project structure (`src/mcpman`, `tests/`).
  - [ ] Create `pyproject.toml` with basic metadata and dependencies (`typer`, `httpx`, `python-dotenv`, potentially `mcp` SDK early on if helpful for types).
  - [ ] Set up basic `.gitignore`.
  - [ ] Create this `IMPLEMENTATION_PLAN.md` file.
- [ ] **CLI Interface (Basic):** (`src/mcpman/cli.py`)
  - [ ] Implement basic CLI using Typer.
  - [ ] Add required arguments: `--config <PATH>`, `--provider <NAME>`, `--model <ID>`, `--user <MESSAGE>`.
  - [ ] Add optional arguments: `--base-url <URL>`.
- [ ] **Configuration Loading:** (`src/mcpman/config.py`)
  - [ ] Implement loading of `mcpServers` from the specified JSON config file.
  - [ ] Add basic validation for the config structure (initially just checking for `mcpServers` key).
  - [ ] Create a sample `servers_config.json` for testing (e.g., with a simple echo server command).
- [ ] **Basic Logging:** (`src/mcpman/logging_setup.py` or similar)
  - [ ] Configure standard Python logging to output INFO level messages to the console.
- [ ] **MCP Server Management (Minimal):** (`src/mcpman/mcp/manager.py`)
  - [ ] Implement logic to parse the _first_ `stdio` server definition from the config.
  - [ ] Use `asyncio.create_subprocess_exec` or similar to start the server process based on `command` and `args`.
  - [ ] Implement basic process management (store process handle).
  - [ ] Implement graceful shutdown logic (terminate the process).
  - [ ] (Defer `mcp` SDK client integration for now).
- [ ] **LLM Client (Minimal):** (`src/mcpman/llm/openai_client.py`)
  - [ ] Create a basic LLM client structure (`openai_client.py`).
  - [ ] Implement a simple client for OpenAI-compatible APIs.
  - [ ] Use `httpx` to make the API call.
  - [ ] Load API key from environment variables (`OPENAI_API_KEY` or `LLM_API_KEY`).
  - [ ] Handle `--base-url` if provided.
  - [ ] Send only the user message.
  - [ ] Parse and return the content of the first choice.
- [ ] **Core Loop (No Tools):** (`src/mcpman/cli.py`)
  - [ ] Orchestrate the flow: Load config -> Start server process -> Prepare LLM message -> Call LLM -> Print response -> Stop server process.
  - [ ] Use `asyncio` for managing server process and potential future async operations.

**Milestone 1:** Run `mcpman --config <path> --provider openai --model <id> --user "Hello"` (or similar for another provider). It should start the defined server process, call the LLM, print the LLM's text response, and shut down the server process cleanly. **[COMPLETED]**

## Phase 2: Basic Tool Integration (stdio)

Goal: Connect to the stdio MCP server using the `mcp` SDK, retrieve its tool schema, provide it to the LLM, and execute a tool call if requested by the LLM.

- [ ] **MCP Client Integration:** (`src/mcpman/mcp/manager.py`)
  - [ ] Add `mcp` SDK as a dependency.
  - [ ] Replace direct process management with `mcp.client.stdio.stdio_client` for the _first_ stdio server.
  - [ ] Establish a `ClientSession` with the server.
  - [ ] Implement fetching tool schemas (`session.get_tools_schema()`).
- [ ] **LLM Prompting with Tools:** (`src/mcpman/llm/client.py`, `src/mcpman/agent.py`)
  - [ ] Format the fetched tool schemas into a string suitable for the LLM system prompt (similar to the chatbot example).
  - [ ] Update the LLM client call to include the system prompt with tool definitions and instructions for tool calling (JSON format).
- [ ] **Tool Call Handling:** (`src/mcpman/agent.py`)
  - [ ] Parse the LLM response to detect if it's a JSON tool call request.
  - [ ] Validate the tool call request format.
  - [ ] Find the requested tool in the schemas retrieved earlier.
  - [ ] Use the `ClientSession` (`session.execute_tool(...)`) to execute the tool on the correct server.
- [ ] **Tool Result Processing:** (`src/mcpman/agent.py`)
  - [ ] Get the result from `execute_tool`.
  - [ ] Format the tool result into a message.
  - [ ] Send the tool result back to the LLM for summarization/final response.
  - [ ] Print the LLM's final response after the tool call cycle.
- [ ] **Refine Logging:**
  - [ ] Add log messages for: fetching tools, formatting prompt, detecting tool call, executing tool, sending result back.

**Milestone 2:** Run `mcpman` with a config pointing to a stdio server that offers a simple tool (e.g., add numbers). Provide a prompt like `"What is 5 + 7?"`. MCPMan should connect to the server, get the tool schema, prompt the LLM, receive a tool call request, execute the tool via MCP, send the result back, and print the LLM's final answer (e.g., "The sum is 12.").

## Phase 3: Multi-Server Support & SSE

Goal: Extend functionality to handle multiple servers (both stdio and SSE) defined in the configuration.

- [ ] **Multiple Stdio Servers:** (`src/mcpman/mcp/manager.py`, `src/mcpman/agent.py`)
  - [ ] Modify config loading and server management to handle a list/dict of servers.
  - [ ] Start/stop and manage connections (`ClientSession`) for all defined stdio servers concurrently.
  - [ ] Aggregate tool schemas from _all_ connected servers. Ensure tool names are unique or handled appropriately (e.g., prefixed with server name?).
  - [ ] Route `execute_tool` calls to the correct server based on the tool name/schema origin.
- [ ] **SSE Server Support:** (`src/mcpman/mcp/manager.py`)
  - [ ] Add logic to detect server type (`command` vs `url`) in the config.
  - [ ] Implement connection logic for SSE servers using `mcp.client.sse.sse_client` (requires `mcp[ws]` extras potentially).
  - [ ] Integrate SSE server tools into the aggregated tool schema list.
  - [ ] Ensure tool execution routes correctly to SSE servers.
  - [ ] Handle potential differences in connection management/lifecycle for SSE.
- [ ] **Configuration Enhancement:** (`src/mcpman/config.py`)
  - [ ] Add support for the `env` key within server configurations to pass environment variables. Apply these when starting stdio processes or potentially for SSE client setup if needed.

**Milestone 3:** Define a config with one stdio server and one SSE server (if a test SSE server is available). Run `mcpman`. It should connect to both, list tools from both, and be able to execute tools on either server based on LLM requests.

## Phase 4: LLM Provider Abstraction & Options

Goal: Refactor LLM interaction to support multiple providers easily and handle more LLM parameters.

- [ ] **LLM Client Abstraction:** (`src/mcpman/llm/base.py`, `src/mcpman/llm/*_client.py`)
  - [ ] Define a base class or protocol for LLM clients.
  - [ ] Refactor the existing OpenAI-compatible client to adhere to this interface.
  - [ ] Implement clients for other key providers (e.g., `OllamaClient`, potentially `AnthropicClient`, `GoogleClient` later). Provider selection should be based on the `--provider` flag.
  - [ ] Handle provider-specific API key environment variables (e.g., `OLLAMA_HOST`, `ANTHROPIC_API_KEY`, etc.).
- [ ] **Additional CLI Options:** (`src/mcpman/cli.py`)
  - [ ] Implement `--system <MESSAGE>` argument and pass it to the LLM client.
  - [ ] Implement `--temperature <FLOAT>` and pass it to the LLM client.
  - [ ] Implement `--max-tokens <INT>` and pass it to the LLM client.
  - [ ] Ensure these options are passed correctly in the API calls for each supported provider.

**Milestone 4:** Run `mcpman` specifying `--provider ollama` (assuming Ollama is running locally) or another implemented provider. Use `--system`, `--temperature` flags. Verify the correct LLM is called with the specified parameters and system message.

## Phase 5: Enhanced Logging & Debugging

Goal: Implement the detailed, incremental logging described in the README for better debugging and observability.

- [ ] **Structured Logging:** (`src/mcpman/logging_setup.py`, throughout the code)
  - [ ] Refine logging setup. Consider using structured logging (e.g., outputting JSON) for easier parsing.
  - [ ] Add distinct log messages for each step mentioned in the README:
    - [ ] Agent receives request (initial prompt).
    - [ ] Agent decides to call a tool (log the parsed tool call JSON).
    - [ ] Agent decides to respond directly (log the direct response).
    - [ ] Tool execution request details (tool name, arguments sent to MCP server).
    - [ ] Tool execution response/error details (result received from MCP server).
    - [ ] Final response generation (response sent back to user after tool cycle).
  - [ ] Include timestamps and potentially turn numbers or unique request IDs in logs.

**Milestone 5:** Run a multi-turn interaction involving tool use. Inspect the logs. Verify that the detailed steps are logged clearly and provide a good trace of the agent's execution flow.

## Phase 6: Packaging, Testing & Documentation

Goal: Prepare the project for distribution and ensure robustness.

- [ ] **Packaging:** (`pyproject.toml`)
  - [ ] Finalize dependencies, including optional dependencies (e.g., `mcp[ws]` if needed for SSE).
  - [ ] Define `[project.scripts]` entry point for the `mcpman` command.
  - [ ] Add `[dev]` extra for development tools (`pytest`, `ruff`, `pre-commit`, etc.).
- [ ] **Testing:** (`tests/`)
  - [ ] Add unit tests for configuration loading, argument parsing, tool formatting.
  - [ ] Add integration tests (may require mocking MCP servers and LLM responses) for the core agentic loop and tool execution flow.
  - [ ] Configure `pytest`.
- [ ] **Linting/Formatting:**
  - [ ] Configure and run linters/formatters (e.g., `ruff`, `black`).
  - [ ] (Optional) Set up pre-commit hooks (`.pre-commit-config.yaml`).
- [ ] **Documentation:**
  - [ ] Add docstrings to public functions and classes.
  - [ ] Update `README.md` with comprehensive installation, usage, configuration details, and examples based on the final implementation.
- [ ] **Installation:**
  - [ ] Test installation using `uv pip install .` and `uv pip install -e ".[dev]"`.

**Milestone 6:** The project can be installed via `uv`, the command runs correctly, tests pass, code is well-formatted, and the `README.md` is accurate and complete.

## Phase 7: Cleanup & Future Goals

Goal: Final review and removal of the implementation plan.

- [ ] **Review:** Perform a final code review.
- [ ] **Remove Plan:** Delete `IMPLEMENTATION_PLAN.md`.
- [ ] **Future Goals:** Consider initial steps or stubs for future goals mentioned in the README (e.g., different LLM API formats).
