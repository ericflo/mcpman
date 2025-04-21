# Changelog

All notable changes to MCPMan will be documented in this file.

## [Unreleased]

### Added
- Added proper support for Anthropic's Claude models
- Added documentation for Claude model support in README

### Fixed
- Fixed issue with Claude repeatedly calling the same tool by implementing deduplication of identical tool calls
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