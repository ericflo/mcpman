# MCPMan Implementation Plan

This document outlines the steps to build the MCPMan tool, prioritizing an incremental approach with runnable milestones.

## Phase 1: Core Foundation & Minimal Viable Product (MVP)

Goal: Create a basic CLI application that can load configuration, start _one_ stdio MCP server, send a prompt to an LLM (without tools), get a response, and shut down cleanly.

- [x] **Project Setup:**
  - [x] Initialize project structure (`src/mcpman`, `tests/`).
  - [x] Create `pyproject.toml` with basic metadata and dependencies (`typer`, `httpx`, `python-dotenv`, potentially `mcp` SDK early on if helpful for types).
  - [x] Set up basic `.gitignore`.
  - [x] Create this `IMPLEMENTATION_PLAN.md` file.
- [x] **CLI Interface (Basic):** (`src/mcpman/cli.py`)
  - [x] Implement basic CLI using Typer.
  - [x] Add required arguments: `--config <PATH>`, `--provider <n>`, `--model <ID>`, `--user <MESSAGE>`.
  - [x] Add optional arguments: `--base-url <URL>`.
- [x] **Configuration Loading:** (`src/mcpman/config.py`)
  - [x] Implement loading of `mcpServers` from the specified JSON config file.
  - [x] Add basic validation for the config structure (initially just checking for `mcpServers` key).
  - [x] Create a sample `servers_config.json` for testing (e.g., with a simple echo server command).
- [x] **Basic Logging:**
  - [x] Configure standard Python logging to output INFO level messages to the console.
- [x] **MCP Server Management (Minimal):** (`src/mcpman/server.py`)
  - [x] Implement logic to parse the _first_ `stdio` server definition from the config.
  - [x] Use `asyncio` to start the server process based on `command` and `args`.
  - [x] Implement basic process management (store process handle).
  - [x] Implement graceful shutdown logic (terminate the process).
  - [x] Integrate with `mcp` SDK.
- [x] **LLM Client (Minimal):** (`src/mcpman/llm_client.py`)
  - [x] Create a basic LLM client structure.
  - [x] Implement a simple client for OpenAI-compatible APIs.
  - [x] Use `httpx` to make the API call.
  - [x] Load API key from environment variables.
  - [x] Handle `--base-url` if provided.
  - [x] Send user message.
  - [x] Parse and return the content of the first choice.
- [x] **Core Loop (No Tools):** (`src/mcpman/cli.py`, `orchestrator.py`)
  - [x] Orchestrate the flow: Load config -> Start server process -> Prepare LLM message -> Call LLM -> Print response -> Stop server process.
  - [x] Use `asyncio` for managing server process and potential future async operations.

**Milestone 1:** Run `mcpman --config <path> --provider openai --model <id> --user "Hello"` (or similar for another provider). It should start the defined server process, call the LLM, print the LLM's text response, and shut down the server process cleanly. **[COMPLETED] ✓**

## Phase 2: Basic Tool Integration (stdio)

Goal: Connect to the stdio MCP server using the `mcp` SDK, retrieve its tool schema, provide it to the LLM, and execute a tool call if requested by the LLM.

- [x] **MCP Client Integration:** (`src/mcpman/server.py`)
  - [x] Add `mcp` SDK as a dependency.
  - [x] Use `mcp.client.stdio.stdio_client` for the stdio server.
  - [x] Establish a `ClientSession` with the server.
  - [x] Implement fetching tool schemas.
- [x] **LLM Prompting with Tools:** (`src/mcpman/llm_client.py`, `src/mcpman/tools.py`)
  - [x] Format the fetched tool schemas into the appropriate format for the LLM.
  - [x] Update the LLM client call to include tools parameter.
- [x] **Tool Call Handling:** (`src/mcpman/orchestrator.py`)
  - [x] Parse the LLM response to detect tool calls.
  - [x] Validate the tool call request format.
  - [x] Find the requested tool in the schemas retrieved earlier.
  - [x] Use the `ClientSession` to execute the tool on the correct server.
- [x] **Tool Result Processing:** (`src/mcpman/orchestrator.py`)
  - [x] Get the result from `execute_tool`.
  - [x] Format the tool result into a message.
  - [x] Send the tool result back to the LLM for summarization/final response.
  - [x] Print the LLM's final response after the tool call cycle.
- [x] **Refine Logging:**
  - [x] Add log messages for: fetching tools, formatting prompt, detecting tool call, executing tool, sending result back.

**Milestone 2:** Run `mcpman` with a config pointing to a stdio server that offers a simple tool (e.g., add numbers). Provide a prompt like `"What is 5 + 7?"`. MCPMan should connect to the server, get the tool schema, prompt the LLM, receive a tool call request, execute the tool via MCP, send the result back, and print the LLM's final answer (e.g., "The sum is 12."). **[COMPLETED] ✓**

## Phase 3: Multi-Server Support & SSE

Goal: Extend functionality to handle multiple servers (both stdio and SSE) defined in the configuration.

- [x] **Multiple Stdio Servers:** (`src/mcpman/server.py`, `src/mcpman/orchestrator.py`)
  - [x] Modify config loading and server management to handle a list/dict of servers.
  - [x] Start/stop and manage connections (`ClientSession`) for all defined stdio servers concurrently.
  - [x] Aggregate tool schemas from _all_ connected servers. Ensure tool names are unique or handled appropriately (e.g., prefixed with server name).
  - [x] Route `execute_tool` calls to the correct server based on the tool name/schema origin.
- [ ] **SSE Server Support:** (`src/mcpman/server.py`)
  - [ ] Add logic to detect server type (`command` vs `url`) in the config.
  - [ ] Implement connection logic for SSE servers using `mcp.client.sse.sse_client` (requires `mcp[ws]` extras potentially).
  - [ ] Integrate SSE server tools into the aggregated tool schema list.
  - [ ] Ensure tool execution routes correctly to SSE servers.
  - [ ] Handle potential differences in connection management/lifecycle for SSE.
- [x] **Configuration Enhancement:** (`src/mcpman/config.py`)
  - [x] Add support for the `env` key within server configurations to pass environment variables. Apply these when starting stdio processes or potentially for SSE client setup if needed.

**Milestone 3:** Define a config with one stdio server and one SSE server (if a test SSE server is available). Run `mcpman`. It should connect to both, list tools from both, and be able to execute tools on either server based on LLM requests. **[PARTIALLY COMPLETED] ⚠️**

## Phase 4: LLM Provider Abstraction & Options

Goal: Refactor LLM interaction to support multiple providers easily and handle more LLM parameters.

- [x] **LLM Client Abstraction:** (`src/mcpman/llm_client.py`, `src/mcpman/config.py`)
  - [x] Define a base class for LLM clients.
  - [x] Refactor the existing OpenAI-compatible client to adhere to this interface.
  - [x] Implement clients for other key providers. Provider selection based on the `--provider` flag.
  - [x] Handle provider-specific API key environment variables.
- [x] **Additional CLI Options:** (`src/mcpman/cli.py`)
  - [x] Implement `--system <MESSAGE>` argument and pass it to the LLM client.
  - [x] Implement `--temperature <FLOAT>` and pass it to the LLM client.
  - [x] Implement `--max-tokens <INT>` and pass it to the LLM client.
  - [x] Ensure these options are passed correctly in the API calls for each supported provider.

**Milestone 4:** Run `mcpman` specifying `--provider ollama` (assuming Ollama is running locally) or another implemented provider. Use `--system`, `--temperature` flags. Verify the correct LLM is called with the specified parameters and system message. **[COMPLETED] ✓**

## Phase 5: Enhanced Logging & Debugging

Goal: Implement the detailed, incremental logging described in the README for better debugging and observability.

- [x] **Structured Logging:** (throughout the code)
  - [x] Refine logging setup.
  - [x] Add distinct log messages for each step mentioned in the README:
    - [x] Agent receives request (initial prompt).
    - [x] Agent decides to call a tool (log the parsed tool call JSON).
    - [x] Agent decides to respond directly (log the direct response).
    - [x] Tool execution request details (tool name, arguments sent to MCP server).
    - [x] Tool execution response/error details (result received from MCP server).
    - [x] Final response generation (response sent back to user after tool cycle).
  - [x] Include timestamps in logs.

**Milestone 5:** Run a multi-turn interaction involving tool use. Inspect the logs. Verify that the detailed steps are logged clearly and provide a good trace of the agent's execution flow. **[COMPLETED] ✓**

## Phase 6: Packaging, Testing & Documentation

Goal: Prepare the project for distribution and ensure robustness.

- [x] **Packaging:** (`pyproject.toml`)
  - [x] Finalize dependencies, including optional dependencies.
  - [x] Define `[project.scripts]` entry point for the `mcpman` command.
  - [x] Add `[dev]` extra for development tools.
- [ ] **Testing:** (`tests/`)
  - [ ] Add unit tests for configuration loading, argument parsing, tool formatting.
  - [ ] Add integration tests for the core agentic loop and tool execution flow.
  - [ ] Configure `pytest`.
- [x] **Linting/Formatting:**
  - [x] Configure linters/formatters (e.g., `black`, `isort`).
  - [ ] Set up pre-commit hooks (`.pre-commit-config.yaml`).
- [x] **Documentation:**
  - [x] Add docstrings to public functions and classes.
  - [x] Create a comprehensive `README.md` with installation, usage, configuration details, and examples.
- [x] **Installation:**
  - [x] Test installation using `uv pip install .` and `uv pip install -e ".[dev]"`.

**Milestone 6:** The project can be installed via `uv`, the command runs correctly, tests pass, code is well-formatted, and the `README.md` is accurate and complete. **[PARTIALLY COMPLETED] ⚠️**

## Phase 7: Cleanup & Future Goals

Goal: Final review and removal of the implementation plan.

- [ ] **Review:** Perform a final code review.
- [ ] **Remove Plan:** Delete `IMPLEMENTATION_PLAN.md`.
- [ ] **Future Goals:** Consider initial steps or stubs for future goals mentioned in the README (e.g., different LLM API formats).

**Milestone 7:** Final review complete, implementation plan removed, future goals addressed. **[NOT STARTED] ❌**