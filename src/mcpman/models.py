"""
Core models for MCPMan.

This module contains simplified data models that represent the core concepts
used throughout the application.
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
import json
import time


@dataclass
class ToolCall:
    """Represents a tool call request from the LLM."""

    id: str
    type: str
    function_name: str
    arguments: Dict[str, Any]
    server_name: Optional[str] = None
    original_tool_name: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        """Create a ToolCall from a dictionary (OpenAI format)."""
        import logging

        if not isinstance(data, dict):
            logging.error(
                f"Expected dictionary for ToolCall.from_dict, got {type(data)}: {data}"
            )
            # Generate a unique ID to prevent issues
            return cls(
                id=f"error_{time.time_ns()}",
                type="function",
                function_name="error_function",
                arguments={"error": f"Invalid tool call format: {data}"},
            )

        try:
            # Check for required fields
            if "id" not in data:
                logging.error(f"Missing required 'id' field in tool call: {data}")
                data["id"] = f"missing_id_{time.time_ns()}"

            if "function" not in data:
                logging.error(f"Missing required 'function' field in tool call: {data}")
                # Create a minimal function structure
                data["function"] = {"name": "error_function", "arguments": "{}"}
            elif not isinstance(data["function"], dict):
                logging.error(
                    f"Expected dict for 'function', got {type(data['function'])}: {data}"
                )
                data["function"] = {"name": "error_function", "arguments": "{}"}

            if "name" not in data["function"]:
                logging.error(
                    f"Missing required 'function.name' field in tool call: {data}"
                )
                data["function"]["name"] = "error_function"

            if "arguments" not in data["function"]:
                logging.error(
                    f"Missing required 'function.arguments' field in tool call: {data}"
                )
                data["function"]["arguments"] = "{}"

            # Process arguments safely
            try:
                if isinstance(data["function"]["arguments"], str):
                    arguments = json.loads(data["function"]["arguments"])
                else:
                    arguments = data["function"]["arguments"]
            except json.JSONDecodeError as e:
                logging.error(f"Invalid JSON in tool call arguments: {e}, data: {data}")
                arguments = {
                    "error": f"Invalid JSON in arguments: {data['function']['arguments']}"
                }

            return cls(
                id=data["id"],
                type=data.get("type", "function"),
                function_name=data["function"]["name"],
                arguments=arguments,
            )
        except Exception as e:
            logging.error(f"Error creating ToolCall from dict: {e}, data: {data}")
            # Return a fake tool call rather than raising an exception
            return cls(
                id=f"error_{time.time_ns()}",
                type="function",
                function_name="error_function",
                arguments={"error": f"Error processing tool call: {str(e)}"},
            )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (OpenAI format)."""
        import logging

        try:
            # Try to serialize arguments safely
            if isinstance(self.arguments, dict):
                try:
                    arguments = json.dumps(self.arguments)
                except (TypeError, ValueError, OverflowError) as e:
                    logging.error(
                        f"Error serializing arguments to JSON: {e}, args: {self.arguments}"
                    )
                    # Convert problematic dict to string representation
                    arguments = str(self.arguments)
            else:
                arguments = self.arguments

            return {
                "id": self.id,
                "type": self.type,
                "function": {
                    "name": self.function_name,
                    "arguments": arguments,
                },
            }
        except Exception as e:
            logging.error(f"Error in ToolCall.to_dict: {e}", exc_info=True)
            # Return a minimal valid structure
            return {
                "id": self.id if hasattr(self, "id") else f"error_{time.time_ns()}",
                "type": "function",
                "function": {
                    "name": "error_function",
                    "arguments": "{}",
                },
            }


@dataclass
class ToolResult:
    """Represents the result of a tool execution."""

    tool_call_id: str
    name: str
    content: str
    success: bool = True
    execution_time_ms: float = 0
    role: str = "tool"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (OpenAI format)."""
        return {
            "role": self.role,
            "tool_call_id": self.tool_call_id,
            "name": self.name,
            "content": self.content,
        }

    @property
    def is_error(self) -> bool:
        """Check if the result represents an error."""
        return self.content.startswith("Error:") or not self.success


@dataclass
class Message:
    """Represents a message in the conversation."""

    role: str
    content: Optional[str] = None
    tool_calls: List[ToolCall] = field(default_factory=list)
    tool_call_id: Optional[str] = None
    name: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        """Create a Message from a dictionary."""
        import logging

        if not isinstance(data, dict):
            logging.error(
                f"Expected dictionary for Message.from_dict, got {type(data)}: {data}"
            )
            # Return a default message rather than raising an exception
            return cls(
                role="assistant", content="Error: Invalid message format received"
            )

        tool_calls = []
        try:
            if "tool_calls" in data and data["tool_calls"]:
                if not isinstance(data["tool_calls"], list):
                    logging.error(
                        f"Expected list for tool_calls, got {type(data['tool_calls'])}"
                    )
                    # Continue without tool calls rather than crashing
                else:
                    # Try to parse each tool call, skipping any that fail
                    for tc in data["tool_calls"]:
                        try:
                            tool_calls.append(ToolCall.from_dict(tc))
                        except Exception as e:
                            logging.error(f"Error parsing tool call: {e}, data: {tc}")
                            # Skip this tool call but continue processing

            # Default to 'assistant' role if missing to prevent crashes
            role = data.get("role", "assistant")

            return cls(
                role=role,
                content=data.get("content"),
                tool_calls=tool_calls,
                tool_call_id=data.get("tool_call_id"),
                name=data.get("name"),
            )

        except Exception as e:
            logging.error(f"Error creating Message from dict: {e}, data: {data}")
            # Return a default message rather than crashing
            return cls(
                role="assistant", content=f"Error processing message format: {str(e)}"
            )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        import logging

        try:
            result = {"role": self.role if hasattr(self, "role") else "assistant"}

            # Handle content safely
            if hasattr(self, "content") and self.content is not None:
                result["content"] = self.content

            # Handle tool calls safely
            if hasattr(self, "tool_calls") and self.tool_calls:
                try:
                    # Process each tool call individually, skip any that fail
                    tool_calls = []
                    for tc in self.tool_calls:
                        try:
                            tool_calls.append(tc.to_dict())
                        except Exception as tc_err:
                            logging.error(
                                f"Error in processing tool call for to_dict: {tc_err}"
                            )
                            # Skip this tool call

                    if tool_calls:  # Only add if we have valid tool calls
                        result["tool_calls"] = tool_calls
                except Exception as tc_list_err:
                    logging.error(f"Error processing tool_calls list: {tc_list_err}")
                    # Omit tool_calls completely on error

            # Handle other optional fields
            if hasattr(self, "tool_call_id") and self.tool_call_id:
                result["tool_call_id"] = self.tool_call_id

            if hasattr(self, "name") and self.name:
                result["name"] = self.name

            return result
        except Exception as e:
            logging.error(f"Error in Message.to_dict: {e}", exc_info=True)
            # Return a minimal valid message
            return {
                "role": "assistant",
                "content": f"Error formatting message: {str(e)}",
            }

    @property
    def has_tool_calls(self) -> bool:
        """Check if the message has tool calls."""
        return bool(self.tool_calls)


@dataclass
class Conversation:
    """Manages the conversation state and message history."""

    messages: List[Message] = field(default_factory=list)

    def __init__(
        self, system_message: Optional[str] = None, user_prompt: Optional[str] = None
    ):
        self.messages = []

        if system_message:
            self.add_message(Message(role="system", content=system_message))

        if user_prompt:
            self.add_message(Message(role="user", content=user_prompt))

    def add_message(self, message: Message) -> None:
        """Add a message to the conversation."""
        self.messages.append(message)

    def add_user_message(self, content: str) -> None:
        """Add a user message to the conversation."""
        self.messages.append(Message(role="user", content=content))

    def to_dict_list(self) -> List[Dict[str, Any]]:
        """Convert the conversation to a list of dictionaries."""
        return [m.to_dict() for m in self.messages]
