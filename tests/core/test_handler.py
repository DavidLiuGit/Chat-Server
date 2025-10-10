import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice
from openai.types.completion_usage import CompletionUsage

from app.core.handler import OpenAIProxyHandler
from app.models.config import ProxyConfig


@pytest.fixture
def config() -> ProxyConfig:
    return ProxyConfig(upstream_url="https://test.api.com/v1", upstream_api_key="test-key")


@pytest.fixture
def handler(config: ProxyConfig) -> OpenAIProxyHandler:
    return OpenAIProxyHandler(config)


@pytest.fixture
def mock_completion() -> ChatCompletion:
    return ChatCompletion(
        id="test-id",
        model="gpt-4",
        object="chat.completion",
        created=1234567890,
        choices=[
            Choice(
                index=0,
                message=ChatCompletionMessage(role="assistant", content="Test response"),
                finish_reason="stop",
            )
        ],
        usage=CompletionUsage(prompt_tokens=10, completion_tokens=20, total_tokens=30),
    )


@pytest.mark.asyncio
async def test_execute_non_streaming(
    handler: OpenAIProxyHandler, mock_completion: ChatCompletion
) -> None:
    with patch.object(
        handler.client.chat.completions, "create", new_callable=AsyncMock
    ) as mock_create:
        mock_create.return_value = mock_completion

        params = {"model": "gpt-4", "messages": [{"role": "user", "content": "Hello"}]}
        result = await handler.execute(params)

        assert result == mock_completion
        mock_create.assert_called_once_with(**params)


@pytest.mark.asyncio
async def test_execute_streaming(handler: OpenAIProxyHandler) -> None:
    mock_stream = MagicMock()

    with patch.object(handler.client.chat.completions, "stream") as mock_stream_method:
        mock_stream_method.return_value = mock_stream

        params = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": True,
        }
        result = await handler.execute(params)

        assert result == mock_stream
        mock_stream_method.assert_called_once_with(
            model="gpt-4", messages=[{"role": "user", "content": "Hello"}]
        )


def test_handler_initialization() -> None:
    config = ProxyConfig(upstream_url="https://test.api.com/v1", upstream_api_key="test-key")
    handler = OpenAIProxyHandler(config)

    assert handler.config == config
    assert config.upstream_url == "https://test.api.com/v1"
    assert handler.config.upstream_url == "https://test.api.com/v1"
    assert str(handler.client.api_key) == "test-key"
    assert str(handler.client.base_url).startswith(
        "https://test.api.com/v1/"
    )  # client may append /
    assert handler.client.timeout == 10.0
