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
    """Optional on this server."""

    n: int | None = 1
    """How many chat completion choices to generate for each input message"""

    stream: bool | None = False
    """Whether to stream the response or not."""


class ChatCompletionsChoice(BaseModel):
    index: int
    """Index of the current choice. Should be unique in the array"""

    message: Message
    """Message content of the choice."""

    finish_reason: Literal["stop", "length", "tool_calls", "content_filter", "function_call", None] | None = "stop"


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
