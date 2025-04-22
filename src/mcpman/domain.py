"""
Domain models for MCPMan.

This module contains the core domain objects used throughout the application,
providing proper encapsulation and behavior for key concepts.
"""

from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
import json


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
        return cls(
            id=data["id"],
            type=data.get("type", "function"),
            function_name=data["function"]["name"],
            arguments=(
                json.loads(data["function"]["arguments"])
                if isinstance(data["function"]["arguments"], str)
                else data["function"]["arguments"]
            ),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (OpenAI format)."""
        return {
            "id": self.id,
            "type": self.type,
            "function": {
                "name": self.function_name,
                "arguments": (
                    json.dumps(self.arguments)
                    if isinstance(self.arguments, dict)
                    else self.arguments
                ),
            },
        }


@dataclass
class ToolResult:
    """Represents the result of a tool execution."""

    role: str = "tool"
    tool_call_id: str = ""
    name: str = ""
    content: str = ""
    success: bool = True
    execution_time_ms: float = 0

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
        tool_calls = []
        if "tool_calls" in data and data["tool_calls"]:
            tool_calls = [ToolCall.from_dict(tc) for tc in data["tool_calls"]]

        return cls(
            role=data["role"],
            content=data.get("content"),
            tool_calls=tool_calls,
            tool_call_id=data.get("tool_call_id"),
            name=data.get("name"),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {"role": self.role}

        if self.content is not None:
            result["content"] = self.content

        if self.tool_calls:
            result["tool_calls"] = [tc.to_dict() for tc in self.tool_calls]

        if self.tool_call_id:
            result["tool_call_id"] = self.tool_call_id

        if self.name:
            result["name"] = self.name

        return result

    @property
    def has_tool_calls(self) -> bool:
        """Check if the message has tool calls."""
        return bool(self.tool_calls)


@dataclass
class Conversation:
    """Manages the conversation state and message history."""

    messages: List[Message] = field(default_factory=list)
    system_message: Optional[str] = None

    def __init__(
        self, system_message: Optional[str] = None, user_prompt: Optional[str] = None
    ):
        self.messages = []

        if system_message:
            self.add_system_message(system_message)

        if user_prompt:
            self.add_user_message(user_prompt)

    def add_message(self, message: Message) -> None:
        """Add a message to the conversation."""
        self.messages.append(message)

    def add_system_message(self, content: str) -> None:
        """Add a system message to the conversation."""
        self.messages.append(Message(role="system", content=content))
        self.system_message = content

    def add_user_message(self, content: str) -> None:
        """Add a user message to the conversation."""
        self.messages.append(Message(role="user", content=content))

    def add_assistant_message(
        self, content: Optional[str] = None, tool_calls: List[ToolCall] = None
    ) -> None:
        """Add an assistant message to the conversation."""
        self.messages.append(
            Message(role="assistant", content=content, tool_calls=tool_calls or [])
        )

    def add_tool_result(self, result: ToolResult) -> None:
        """Add a tool result to the conversation."""
        self.messages.append(
            Message(
                role=result.role,
                content=result.content,
                tool_call_id=result.tool_call_id,
                name=result.name,
            )
        )

    def to_dict_list(self) -> List[Dict[str, Any]]:
        """Convert the conversation to a list of dictionaries."""
        return [m.to_dict() for m in self.messages]

    @classmethod
    def from_dict_list(cls, messages: List[Dict[str, Any]]):
        """Create a Conversation from a list of message dictionaries."""
        conversation = cls()
        for msg_dict in messages:
            conversation.add_message(Message.from_dict(msg_dict))
        return conversation
