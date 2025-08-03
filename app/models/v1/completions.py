# Request & Response models for chat-completions
from typing import Annotated, Any, Literal
from pydantic import BaseModel


class Message(BaseModel):
    """Represents a single message"""

    role: Literal["system", "user", "assistant", "developer"]
    """Valid values: 'system', 'user', 'assistant', 'developer'"""

    content: str | list[Any]
    """Message content. Usually a string but can be an array representing objs, such as images"""


class ChatCompletionsRequest(BaseModel):
    """
    Request payload for calls to `v1/chat/completions`.
    API reference: https://platform.openai.com/docs/api-reference/chat/create
    """

    messages: list[Message]
    """
    A list of `Message`s comprising the conversation so far, including the user's latest message.
    """

    model: str
    """Model ID to use."""

    n: int = 1
    """How many chat completion choices to generate for each input message"""

    stream: bool | None = False
    """Whether to stream the response or not."""

    def get_log_sanitized_str(self) -> str:
        """Logging-safe version that shows first 50 chars of content for last 3 messages"""
        last_messages = []
        for msg in self.messages[-3:]:
            content_str = str(msg.content)
            content_preview = content_str[:50] + ("..." if len(content_str) > 50 else "")
            last_messages.append(f"{msg.role}:{content_preview}")
        
        messages_summary = (
            last_messages
            if len(self.messages) <= 3
            else [f"({len(self.messages)} msgs)"] + last_messages
        )
        
        return (
            f"ChatCompletionsRequest("
            f"messages={messages_summary}, "
            f"model={self.model}, "
            f"n={self.n}, "
            f"stream={self.stream})"
        )


class ChatCompletionsChoice(BaseModel):
    index: int
    """Index of the current choice. Should be unique in the array"""

    message: Message
    """Message content of the choice."""

    finish_reason: (
        Literal["stop", "length", "tool_calls", "content_filter", "function_call", None] | None
    ) = "stop"


class ChatCompletionsResponse(BaseModel):
    """
    Response payload for the `v1/chat/completions` endpoint.
    """

    id: str
    """Unique response ID."""

    object: Literal["chat.completion"] = "chat.completion"

    created: int  # Unix timestamp, in seconds
    """Unix timestamp, in seconds"""

    model: str
    """Should match request model"""

    choices: list[ChatCompletionsChoice]
