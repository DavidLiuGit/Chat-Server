import pytest
from app.models.v1.completions import ChatCompletionsRequest, Message


def test_str_single_message():
    """Test __str__ with single message"""
    request = ChatCompletionsRequest(
        messages=[Message(role="user", content="Hello")],
        model="gpt-4"
    )
    result = request.get_log_sanitized_str()
    assert "user:Hello" in result
    assert "(1 msgs)" not in result


def test_str_short_content():
    """Test __str__ with short message content"""
    request = ChatCompletionsRequest(
        messages=[Message(role="user", content="Hello")],
        model="gpt-4"
    )
    result = request.get_log_sanitized_str()
    assert "user:Hello" in result
    assert "model=gpt-4" in result


def test_str_long_content():
    """Test __str__ with long message content"""
    long_content = "x" * 60
    request = ChatCompletionsRequest(
        messages=[Message(role="user", content=long_content)],
        model="gpt-4"
    )
    result = request.get_log_sanitized_str()
    assert "user:" + "x" * 50 + "..." in result


def test_str_exactly_three_messages():
    """Test __str__ with exactly 3 messages"""
    request = ChatCompletionsRequest(
        messages=[
            Message(role="system", content="You are helpful"),
            Message(role="user", content="Hi"),
            Message(role="assistant", content="Hello!")
        ],
        model="gpt-4"
    )
    result = request.get_log_sanitized_str()
    assert "system:You are helpful" in result
    assert "user:Hi" in result
    assert "assistant:Hello!" in result
    assert "(3 msgs)" not in result


def test_str_more_than_three_messages():
    """Test __str__ with more than 3 messages shows count + last 3"""
    request = ChatCompletionsRequest(
        messages=[
            Message(role="system", content="System msg"),
            Message(role="user", content="User msg 1"),
            Message(role="assistant", content="Assistant msg 1"),
            Message(role="user", content="User msg 2"),
            Message(role="assistant", content="Assistant msg 2")
        ],
        model="gpt-4"
    )
    result = request.get_log_sanitized_str()
    assert "(5 msgs)" in result
    assert "assistant:Assistant msg 1" in result
    assert "user:User msg 2" in result
    assert "assistant:Assistant msg 2" in result


def test_str_all_parameters():
    """Test __str__ includes all request parameters"""
    request = ChatCompletionsRequest(
        messages=[Message(role="user", content="Test")],
        model="gpt-3.5-turbo",
        n=2,
        stream=True
    )
    result = request.get_log_sanitized_str()
    assert "user:Test" in result
    assert "model=gpt-3.5-turbo" in result
    assert "n=2" in result
    assert "stream=True" in result