import pytest
from unittest.mock import Mock, patch
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice
from openai.types import CompletionUsage
from openai.lib.streaming.chat import ChatCompletionStreamEvent

from app.plugins.logging import LoggingPlugin

pytestmark = pytest.mark.asyncio


@pytest.fixture
def plugin():
    return LoggingPlugin()


@pytest.fixture
def params():
    return {"model": "test-model", "messages": [{"role": "user", "content": "test"}]}


@pytest.fixture
def response():
    return ChatCompletion(
        id="test-id",
        choices=[
            Choice(
                finish_reason="stop",
                index=0,
                message=ChatCompletionMessage(role="assistant", content="test response"),
            )
        ],
        created=1234567890,
        model="test-model",
        object="chat.completion",
        usage=CompletionUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
    )


@patch("app.plugins.logging.logger")
async def test_before_request_logs_params(mock_logger, plugin, params):
    result = await plugin.before_request(params)

    mock_logger.info.assert_called_once()
    assert "Chat completion request" in mock_logger.info.call_args[0][0]
    assert result == params


@patch("app.plugins.logging.logger")
async def test_after_stream_async_with_events(mock_logger, plugin, params, response):
    events = [Mock(spec=ChatCompletionStreamEvent)]

    await plugin.after_stream_async(params, response, events)

    mock_logger.info.assert_called_once()
    log_msg = mock_logger.info.call_args[0][0]
    assert "Chat completion stream" in log_msg
    assert "events=1" in log_msg


@patch("app.plugins.logging.logger")
async def test_after_stream_async_without_events(mock_logger, plugin, params, response):
    await plugin.after_stream_async(params, response, [])

    mock_logger.info.assert_called_once()
    log_msg = mock_logger.info.call_args[0][0]
    assert "Chat completion response" in log_msg


@patch("app.plugins.logging.logger")
async def test_on_error_async_logs_error(mock_logger, plugin, params):
    error = Exception("test error")

    await plugin.on_error_async(params, error)

    mock_logger.error.assert_called_once_with("Chat completion error: test error", exc_info=True)


@patch("app.plugins.logging.logger")
async def test_before_request_missing_model(mock_logger, plugin):
    params = {"messages": [{"role": "user", "content": "test"}]}

    result = await plugin.before_request(params)

    mock_logger.info.assert_called_once()
    assert result == params


@patch("app.plugins.logging.logger")
async def test_after_stream_async_no_usage(mock_logger, plugin, params):
    response = ChatCompletion(
        id="test-id",
        choices=[
            Choice(
                finish_reason="stop",
                index=0,
                message=ChatCompletionMessage(role="assistant", content="test"),
            )
        ],
        created=1234567890,
        model="test-model",
        object="chat.completion",
        usage=None,
    )

    await plugin.after_stream_async(params, response, [])

    mock_logger.info.assert_called_once()
    log_msg = mock_logger.info.call_args[0][0]
    assert "usage=None" in log_msg
