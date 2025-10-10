import pytest
from typing import List, Optional, Any
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice
from openai.types import CompletionUsage

from app.core.normalizer import normalize_chat_completion


class TestChatCompletionMessage(ChatCompletionMessage):
    """Test version that allows list content."""

    model_config = {"extra": "allow", "validate_assignment": False}
    content: Optional[Any] = None


class TestChoice(Choice):
    """Test version with relaxed message validation."""

    model_config = {"extra": "allow", "validate_assignment": False}
    message: TestChatCompletionMessage


class TestChatCompletion(ChatCompletion):
    """Test version that allows malformed content."""

    model_config = {"extra": "allow", "validate_assignment": False}
    choices: List[TestChoice]


def test_normalize_list_content_to_string():
    """Test converting list-based content to string."""
    response = TestChatCompletion(
        id="test-id",
        choices=[
            TestChoice(
                finish_reason="stop",
                index=0,
                message=TestChatCompletionMessage(
                    role="assistant",
                    content=[{"type": "text", "text": "Hello world"}],
                ),
            )
        ],
        created=1234567890,
        model="test-model",
        object="chat.completion",
    )

    normalized = normalize_chat_completion(response)
    assert normalized.choices[0].message.content == "Hello world"


def test_normalize_multiple_text_blocks():
    """Test concatenating multiple text blocks."""
    response = TestChatCompletion(
        id="test-id",
        choices=[
            TestChoice(
                finish_reason="stop",
                index=0,
                message=TestChatCompletionMessage(
                    role="assistant",
                    content=[
                        {"type": "text", "text": "Hello "},
                        {"type": "text", "text": "world"},
                    ],
                ),
            )
        ],
        created=1234567890,
        model="test-model",
        object="chat.completion",
    )

    normalized = normalize_chat_completion(response)
    assert normalized.choices[0].message.content == "Hello world"


def test_normalize_already_string_content():
    """Test that string content remains unchanged."""
    response = ChatCompletion(
        id="test-id",
        choices=[
            Choice(
                finish_reason="stop",
                index=0,
                message=ChatCompletionMessage(
                    role="assistant",
                    content="Already a string",
                ),
            )
        ],
        created=1234567890,
        model="test-model",
        object="chat.completion",
    )

    normalized = normalize_chat_completion(response)
    assert normalized.choices[0].message.content == "Already a string"


def test_normalize_none_content():
    """Test that None content remains None."""
    response = ChatCompletion(
        id="test-id",
        choices=[
            Choice(
                finish_reason="stop",
                index=0,
                message=ChatCompletionMessage(
                    role="assistant",
                    content=None,
                ),
            )
        ],
        created=1234567890,
        model="test-model",
        object="chat.completion",
    )

    normalized = normalize_chat_completion(response)
    assert normalized.choices[0].message.content is None


def test_normalize_empty_list_content():
    """Test that empty list content becomes empty string."""
    response = TestChatCompletion(
        id="test-id",
        choices=[
            TestChoice(
                finish_reason="stop",
                index=0,
                message=TestChatCompletionMessage(
                    role="assistant",
                    content=[],
                ),
            )
        ],
        created=1234567890,
        model="test-model",
        object="chat.completion",
    )

    normalized = normalize_chat_completion(response)
    assert normalized.choices[0].message.content == ""


def test_normalize_multiple_choices():
    """Test normalizing response with multiple choices."""
    response = TestChatCompletion(
        id="test-id",
        choices=[
            TestChoice(
                finish_reason="stop",
                index=0,
                message=TestChatCompletionMessage(
                    role="assistant",
                    content=[{"type": "text", "text": "First"}],
                ),
            ),
            TestChoice(
                finish_reason="stop",
                index=1,
                message=TestChatCompletionMessage(
                    role="assistant",
                    content=[{"type": "text", "text": "Second"}],
                ),
            ),
        ],
        created=1234567890,
        model="test-model",
        object="chat.completion",
    )

    normalized = normalize_chat_completion(response)
    assert normalized.choices[0].message.content == "First"
    assert normalized.choices[1].message.content == "Second"
