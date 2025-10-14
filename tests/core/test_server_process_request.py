import pytest
from unittest.mock import Mock, patch, AsyncMock
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice
from openai.lib.streaming.chat import AsyncChatCompletionStreamManager, ChatCompletionStreamEvent

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
    server.proxy_handler.execute = AsyncMock(return_value=mock_response)

    params = {"model": "test", "messages": []}
    result = await server.process_request(params)

    assert result == mock_response
    mock_normalize.assert_called_once_with(mock_response)


@pytest.mark.asyncio
async def test_process_request_streaming(server):
    """Test process_request for streaming response."""
    mock_stream = Mock(spec=AsyncChatCompletionStreamManager)
    server.proxy_handler.execute = AsyncMock(return_value=mock_stream)

    params = {"model": "test", "stream": True, "messages": []}
    result = await server.process_request(params)

    assert result == mock_stream


@pytest.mark.asyncio
async def test_process_request_with_error(server):
    """Test process_request handles handler errors."""
    error = Exception("handler error")
    server.proxy_handler.execute = AsyncMock(side_effect=error)

    params = {"model": "test", "messages": []}

    with pytest.raises(Exception, match="handler error"):
        await server.process_request(params)


@pytest.mark.asyncio
async def test_process_request_calls_plugins(server, mock_response):
    """Test process_request calls plugin hooks."""
    plugin = Mock()
    plugin.before_request = AsyncMock(return_value={"model": "test"})
    server.plugins = [plugin]
    server.proxy_handler.execute = AsyncMock(return_value=mock_response)

    params = {"model": "test", "messages": []}
    await server.process_request(params)

    plugin.before_request.assert_called_once()


# _stream_with_hooks tests
@pytest.mark.asyncio
@patch("chat_completion_server.core.server.logger")
async def test_stream_with_hooks_content_delta(mock_logger, server):
    """Test _stream_with_hooks handles content.delta events."""
    # Mock stream manager and events
    mock_stream_manager = Mock(spec=AsyncChatCompletionStreamManager)
    mock_stream = Mock()
    mock_stream_manager.__aenter__ = AsyncMock(return_value=mock_stream)
    mock_stream_manager.__aexit__ = AsyncMock(return_value=None)

    # Create mock events
    event1 = Mock(spec=ChatCompletionStreamEvent)
    event1.type = "content.delta"
    event1.delta = "Hello"

    event2 = Mock(spec=ChatCompletionStreamEvent)
    event2.type = "content.delta"
    event2.delta = " world"

    async def mock_aiter():
        for event in [event1, event2]:
            yield event

    mock_stream.__aiter__ = lambda self: mock_aiter()
    mock_stream.get_final_completion = AsyncMock(return_value=Mock())

    params = {"model": "test"}

    # Collect streamed chunks
    chunks = []
    async for chunk in server._stream_with_hooks(mock_stream_manager, params):
        chunks.append(chunk)

    # Verify output
    assert chunks == ["data: Hello\n\n", "data:  world\n\n", "data: [DONE]\n\n"]
    mock_logger.info.assert_called()


@pytest.mark.asyncio
async def test_stream_with_hooks_chunk_events(server):
    """Test _stream_with_hooks handles chunk events."""
    mock_stream_manager = Mock(spec=AsyncChatCompletionStreamManager)
    mock_stream = Mock()
    mock_stream_manager.__aenter__ = AsyncMock(return_value=mock_stream)
    mock_stream_manager.__aexit__ = AsyncMock(return_value=None)

    # Mock chunk event
    mock_chunk = Mock()
    mock_chunk.model_dump_json = Mock(return_value='{"id":"test"}')

    event = Mock(spec=ChatCompletionStreamEvent)
    event.type = "chunk"
    event.chunk = mock_chunk

    async def mock_aiter():
        yield event

    mock_stream.__aiter__ = lambda self: mock_aiter()
    mock_stream.get_final_completion = AsyncMock(return_value=Mock())

    chunks = []
    async for chunk in server._stream_with_hooks(mock_stream_manager, {}):
        chunks.append(chunk)

    assert chunks == ['data: {"id":"test"}\n\n', "data: [DONE]\n\n"]
    mock_chunk.model_dump_json.assert_called_once_with(exclude_none=True)


@pytest.mark.asyncio
async def test_stream_with_hooks_refusal_delta(server):
    """Test _stream_with_hooks handles refusal.delta events."""
    mock_stream_manager = Mock(spec=AsyncChatCompletionStreamManager)
    mock_stream = Mock()
    mock_stream_manager.__aenter__ = AsyncMock(return_value=mock_stream)
    mock_stream_manager.__aexit__ = AsyncMock(return_value=None)

    event = Mock(spec=ChatCompletionStreamEvent)
    event.type = "refusal.delta"
    event.delta = "I cannot"

    async def mock_aiter():
        yield event

    mock_stream.__aiter__ = lambda self: mock_aiter()
    mock_stream.get_final_completion = AsyncMock(return_value=Mock())

    chunks = []
    async for chunk in server._stream_with_hooks(mock_stream_manager, {}):
        chunks.append(chunk)

    assert chunks == ["data: I cannot\n\n", "data: I cannot\n\n", "data: [DONE]\n\n"]


@pytest.mark.asyncio
async def test_stream_with_hooks_calls_after_stream_hooks(server):
    """Test _stream_with_hooks calls after_stream_async hooks."""
    plugin = Mock()
    plugin.after_stream_async = AsyncMock()
    server.plugins = [plugin]

    mock_stream_manager = Mock(spec=AsyncChatCompletionStreamManager)
    mock_stream = Mock()
    mock_stream_manager.__aenter__ = AsyncMock(return_value=mock_stream)
    mock_stream_manager.__aexit__ = AsyncMock(return_value=None)

    event = Mock(spec=ChatCompletionStreamEvent)
    event.type = "content.delta"
    event.delta = "test"

    async def mock_aiter():
        yield event

    mock_stream.__aiter__ = lambda self: mock_aiter()
    final_completion = Mock()
    mock_stream.get_final_completion = AsyncMock(return_value=final_completion)

    params = {"model": "test"}

    # Consume the stream
    async for _ in server._stream_with_hooks(mock_stream_manager, params):
        pass

    # Allow background task to complete
    import asyncio

    await asyncio.sleep(0.01)

    plugin.after_stream_async.assert_called_once_with(params, final_completion, [event])


@pytest.mark.asyncio
async def test_stream_with_hooks_tracks_event_types(server):
    """Test _stream_with_hooks tracks event type counts."""
    mock_stream_manager = Mock(spec=AsyncChatCompletionStreamManager)
    mock_stream = Mock()
    mock_stream_manager.__aenter__ = AsyncMock(return_value=mock_stream)
    mock_stream_manager.__aexit__ = AsyncMock(return_value=None)

    events = [
        Mock(spec=ChatCompletionStreamEvent, type="content.delta", delta="a"),
        Mock(spec=ChatCompletionStreamEvent, type="content.delta", delta="b"),
        Mock(
            spec=ChatCompletionStreamEvent,
            type="chunk",
            chunk=Mock(model_dump_json=Mock(return_value="{}")),
        ),
    ]

    async def mock_aiter():
        for event in events:
            yield event

    mock_stream.__aiter__ = lambda self: mock_aiter()
    mock_stream.get_final_completion = AsyncMock(return_value=Mock())

    with patch("chat_completion_server.core.server.logger") as mock_logger:
        async for _ in server._stream_with_hooks(mock_stream_manager, {}):
            pass

        # Check that event types were logged
        mock_logger.info.assert_any_call("\tevent_types={'content.delta': 2, 'chunk': 1}")


# Tool call loop tests
@pytest.mark.asyncio
@patch("chat_completion_server.core.server.normalize_chat_completion")
async def test_process_non_streaming_single_tool_call(mock_normalize, server):
    """Test single tool call execution."""
    from openai.types.chat.chat_completion_message_tool_call import ChatCompletionMessageToolCall
    from openai.types.chat.chat_completion_message_tool_call import Function

    # Mock tool call response
    tool_call = ChatCompletionMessageToolCall(
        id="call_123",
        function=Function(name="test_tool", arguments='{"arg": "value"}'),
        type="function",
    )

    # First response with tool call
    tool_response = ChatCompletion(
        id="test-id",
        choices=[
            Choice(
                finish_reason="tool_calls",
                index=0,
                message=ChatCompletionMessage(
                    role="assistant", content=None, tool_calls=[tool_call]
                ),
            )
        ],
        created=1234567890,
        model="test-model",
        object="chat.completion",
    )

    # Final response after tool execution
    final_response = ChatCompletion(
        id="test-id-2",
        choices=[
            Choice(
                finish_reason="stop",
                index=0,
                message=ChatCompletionMessage(role="assistant", content="Tool result processed"),
            )
        ],
        created=1234567890,
        model="test-model",
        object="chat.completion",
    )

    mock_normalize.side_effect = [tool_response, final_response]
    server.proxy_handler.execute = AsyncMock(return_value=tool_response)
    server.proxy_handler.execute_non_streaming = AsyncMock(return_value=final_response)
    server.proxy_tool_client.execute_tool = AsyncMock(
        return_value={"role": "tool", "content": "result"}
    )

    params = {"model": "test", "messages": [{"role": "user", "content": "test"}]}
    result = await server._process_non_streaming_response(params, tool_response)

    assert result == final_response
    server.proxy_tool_client.execute_tool.assert_called_once_with(tool_call)
    server.proxy_handler.execute_non_streaming.assert_called_once()


@pytest.mark.asyncio
@patch("chat_completion_server.core.server.normalize_chat_completion")
async def test_process_non_streaming_multiple_tool_calls(mock_normalize, server):
    """Test multiple consecutive tool calls."""
    from openai.types.chat.chat_completion_message_tool_call import ChatCompletionMessageToolCall
    from openai.types.chat.chat_completion_message_tool_call import Function

    tool_call1 = ChatCompletionMessageToolCall(
        id="call_1", function=Function(name="tool1", arguments="{}"), type="function"
    )
    tool_call2 = ChatCompletionMessageToolCall(
        id="call_2", function=Function(name="tool2", arguments="{}"), type="function"
    )

    # First response with tool call
    response1 = ChatCompletion(
        id="test-1",
        choices=[
            Choice(
                finish_reason="tool_calls",
                index=0,
                message=ChatCompletionMessage(role="assistant", tool_calls=[tool_call1]),
            )
        ],
        created=1234567890,
        model="test-model",
        object="chat.completion",
    )

    # Second response with another tool call
    response2 = ChatCompletion(
        id="test-2",
        choices=[
            Choice(
                finish_reason="tool_calls",
                index=0,
                message=ChatCompletionMessage(role="assistant", tool_calls=[tool_call2]),
            )
        ],
        created=1234567890,
        model="test-model",
        object="chat.completion",
    )

    # Final response
    final_response = ChatCompletion(
        id="test-3",
        choices=[
            Choice(
                finish_reason="stop",
                index=0,
                message=ChatCompletionMessage(role="assistant", content="Done"),
            )
        ],
        created=1234567890,
        model="test-model",
        object="chat.completion",
    )

    mock_normalize.side_effect = [response1, response2, final_response]
    server.proxy_handler.execute_non_streaming = AsyncMock(side_effect=[response2, final_response])
    server.proxy_tool_client.execute_tool = AsyncMock(
        return_value={"role": "tool", "content": "result"}
    )

    params = {"model": "test", "messages": [{"role": "user", "content": "test"}]}
    result = await server._process_non_streaming_response(params, response1)

    assert result == final_response
    assert server.proxy_tool_client.execute_tool.call_count == 2
    assert server.proxy_handler.execute_non_streaming.call_count == 2


@pytest.mark.asyncio
@patch("chat_completion_server.core.server.normalize_chat_completion")
@patch("chat_completion_server.core.server.MAX_TOOL_ROUNDS", 3)
async def test_process_non_streaming_max_tool_rounds(mock_normalize, server):
    """Test tool call loop respects MAX_TOOL_ROUNDS limit."""
    from openai.types.chat.chat_completion_message_tool_call import ChatCompletionMessageToolCall
    from openai.types.chat.chat_completion_message_tool_call import Function

    tool_call = ChatCompletionMessageToolCall(
        id="call_loop", function=Function(name="loop_tool", arguments="{}"), type="function"
    )

    # Response that always returns tool calls
    loop_response = ChatCompletion(
        id="test-loop",
        choices=[
            Choice(
                finish_reason="tool_calls",
                index=0,
                message=ChatCompletionMessage(role="assistant", tool_calls=[tool_call]),
            )
        ],
        created=1234567890,
        model="test-model",
        object="chat.completion",
    )

    mock_normalize.return_value = loop_response
    server.proxy_handler.execute_non_streaming = AsyncMock(return_value=loop_response)
    server.proxy_tool_client.execute_tool = AsyncMock(
        return_value={"role": "tool", "content": "result"}
    )

    params = {"model": "test", "messages": [{"role": "user", "content": "test"}]}
    result = await server._process_non_streaming_response(params, loop_response)

    # Should stop after MAX_TOOL_ROUNDS (3) iterations
    assert server.proxy_tool_client.execute_tool.call_count == 3
    assert server.proxy_handler.execute_non_streaming.call_count == 3
    assert result == loop_response


@pytest.mark.asyncio
@patch("chat_completion_server.core.server.normalize_chat_completion")
async def test_process_non_streaming_tool_calls_normalization(mock_normalize, server):
    """Test tool_calls finish_reason is normalized to tool_calls."""
    from openai.types.chat.chat_completion_message_tool_call import ChatCompletionMessageToolCall
    from openai.types.chat.chat_completion_message_tool_call import Function

    tool_call = ChatCompletionMessageToolCall(
        id="call_123", function=Function(name="test_tool", arguments="{}"), type="function"
    )

    # Response with tool_calls finish_reason
    tool_calls_response = ChatCompletion(
        id="test-id",
        choices=[
            Choice(
                finish_reason="tool_calls",
                index=0,
                message=ChatCompletionMessage(role="assistant", tool_calls=[tool_call]),
            )
        ],
        created=1234567890,
        model="test-model",
        object="chat.completion",
    )

    final_response = ChatCompletion(
        id="test-id-2",
        choices=[
            Choice(
                finish_reason="stop",
                index=0,
                message=ChatCompletionMessage(role="assistant", content="Done"),
            )
        ],
        created=1234567890,
        model="test-model",
        object="chat.completion",
    )

    mock_normalize.side_effect = [tool_calls_response, final_response]
    server.proxy_handler.execute_non_streaming = AsyncMock(return_value=final_response)
    server.proxy_tool_client.execute_tool = AsyncMock(
        return_value={"role": "tool", "content": "result"}
    )

    params = {"model": "test", "messages": [{"role": "user", "content": "test"}]}
    result = await server._process_non_streaming_response(params, tool_calls_response)

    # Verify tool_calls was normalized to tool_calls and tool was executed
    assert result == final_response
    server.proxy_tool_client.execute_tool.assert_called_once()


@pytest.mark.asyncio
@patch("chat_completion_server.core.server.normalize_chat_completion")
async def test_process_non_streaming_no_tool_calls(mock_normalize, server):
    """Test response without tool calls passes through unchanged."""
    normal_response = ChatCompletion(
        id="test-id",
        choices=[
            Choice(
                finish_reason="stop",
                index=0,
                message=ChatCompletionMessage(role="assistant", content="Normal response"),
            )
        ],
        created=1234567890,
        model="test-model",
        object="chat.completion",
    )

    mock_normalize.return_value = normal_response
    server.proxy_tool_client.execute_tool = AsyncMock()

    params = {"model": "test", "messages": [{"role": "user", "content": "test"}]}
    result = await server._process_non_streaming_response(params, normal_response)

    assert result == normal_response
    server.proxy_tool_client.execute_tool.assert_not_called()
