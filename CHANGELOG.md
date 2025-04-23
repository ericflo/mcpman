# Changelog

All notable changes to MCPMan will be documented in this file.

## [Unreleased]

## [0.3.0] - 2025-04-22

### Added
- Added enhanced structured logging system with run ID tracking
- Added mcpreplay tool for visualizing previous conversations from log files
- Added replay mode to CLI with `--replay` and `--replay-file` parameters
- Added standardized logging across all LLM clients

## [0.2.3] - 2025-04-22

### Added
- Added OpenAI (v1.75.0) and Anthropic (v0.49.0) client libraries as explicit dependencies
- Improved dependency management for better user experience

## [0.2.2] - 2025-04-22

### Fixed
- Fixed critical bug in array schema validation for OpenAI tools without items type
- Fixed incorrect handling of tools with array parameters that didn't specify item types

## [0.2.1] - 2025-04-22

### Added
- Added colorama dependency explicitly to requirements
- Improved compatibility with Claude 3.5 Sonnet model
- Added standardized content handling across all models
- Added proper support for Anthropic's Claude models
- Added documentation for Claude model support in README

### Fixed
- Fixed missing dependency handling in packaging
- Fixed issue with Claude repeatedly calling the same tool by implementing intelligent deduplication that handles:
  - Removing exact duplicate tool calls (same name and arguments)
  - Allowing multiple calls to the same tool with different arguments
  - Properly handling JSON object key ordering in arguments comparison
- Fixed handling of messages without content in Anthropic client

### Changed
- Refactored codebase to use a domain model approach with cleaner abstractions
- Simplified orchestrator.py with better separation of concerns
- Improved error handling and logging

## [0.1.3] - 2025-04-21

### Added
- Improved max_turns handling with better help text

### Fixed
- Fixed version numbering

## [0.1.2] - 2025-04-20

### Fixed
- Fixed issues found in end-to-end tests