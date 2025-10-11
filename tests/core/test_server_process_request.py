import pytest
from unittest.mock import Mock, patch, AsyncMock
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice
from openai.lib.streaming.chat import AsyncChatCompletionStreamManager

from chat_completion_server.core.server import ChatCompletionServer


@pytest.fixture
def server():
    return ChatCompletionServer(plugins=[])


@pytest.fixture
def mock_response():
    return ChatCompletion(
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
    )


# Helper method tests
@pytest.mark.asyncio
async def test_run_after_request_hooks():
    """Test _run_after_request_hooks calls all plugins."""
    plugin1 = Mock()
    plugin1.after_request_async = AsyncMock()
    plugin2 = Mock()
    plugin2.after_request_async = AsyncMock()

    server = ChatCompletionServer(plugins=[plugin1, plugin2])
    params = {"model": "test"}
    response = Mock()

    await server._run_after_request_hooks(params, response)

    plugin1.after_request_async.assert_called_once_with(params, response)
    plugin2.after_request_async.assert_called_once_with(params, response)


@pytest.mark.asyncio
@patch("chat_completion_server.core.server.logger")
async def test_run_after_request_hooks_with_error(mock_logger):
    """Test _run_after_request_hooks handles plugin errors."""
    plugin = Mock()
    plugin.after_request_async = AsyncMock(side_effect=Exception("test error"))

    server = ChatCompletionServer(plugins=[plugin])

    await server._run_after_request_hooks({}, Mock())

    mock_logger.exception.assert_called_once_with("Error in async hook")


@pytest.mark.asyncio
async def test_run_after_stream_hooks():
    """Test _run_after_stream_hooks calls all plugins."""
    plugin = Mock()
    plugin.after_stream_async = AsyncMock()

    server = ChatCompletionServer(plugins=[plugin])
    params = {"model": "test"}
    response = Mock()
    events = [Mock()]

    await server._run_after_stream_hooks(params, response, events)

    plugin.after_stream_async.assert_called_once_with(params, response, events)


@pytest.mark.asyncio
async def test_run_on_error_hooks():
    """Test _run_on_error_hooks calls all plugins."""
    plugin = Mock()
    plugin.on_error_async = AsyncMock()

    server = ChatCompletionServer(plugins=[plugin])
    params = {"model": "test"}
    error = Exception("test")

    await server._run_on_error_hooks(params, error)

    plugin.on_error_async.assert_called_once_with(params, error)


# process_request tests
@pytest.mark.asyncio
@patch("chat_completion_server.core.server.normalize_chat_completion")
async def test_process_request_non_streaming(mock_normalize, server, mock_response):
    """Test process_request for non-streaming response."""
    mock_normalize.return_value = mock_response
    server.handler.execute = AsyncMock(return_value=mock_response)

    params = {"model": "test", "messages": []}
    result = await server.process_request(params)

    assert result == mock_response
    mock_normalize.assert_called_once_with(mock_response)


@pytest.mark.asyncio
async def test_process_request_streaming(server):
    """Test process_request for streaming response."""
    mock_stream = Mock(spec=AsyncChatCompletionStreamManager)
    server.handler.execute = AsyncMock(return_value=mock_stream)

    params = {"model": "test", "stream": True, "messages": []}
    result = await server.process_request(params)

    assert result == mock_stream


@pytest.mark.asyncio
async def test_process_request_with_error(server):
    """Test process_request handles handler errors."""
    error = Exception("handler error")
    server.handler.execute = AsyncMock(side_effect=error)

    params = {"model": "test", "messages": []}

    with pytest.raises(Exception, match="handler error"):
        await server.process_request(params)


@pytest.mark.asyncio
async def test_process_request_calls_plugins(server, mock_response):
    """Test process_request calls plugin hooks."""
    plugin = Mock()
    plugin.before_request = AsyncMock(return_value={"model": "test"})
    server.plugins = [plugin]
    server.handler.execute = AsyncMock(return_value=mock_response)

    params = {"model": "test", "messages": []}
    await server.process_request(params)

    plugin.before_request.assert_called_once()
