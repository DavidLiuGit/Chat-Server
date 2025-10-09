from abc import ABC, abstractmethod
from typing import Any, Union

from openai import AsyncOpenAI, AsyncStream
from openai.types.chat import ChatCompletion, ChatCompletionChunk, CompletionCreateParams
from openai.lib.streaming.chat import AsyncChatCompletionStreamManager

from app.core.config import ProxyConfig


class ProxyHandler(ABC):
    """
    Abstract base class for handling chat completion requests.

    Override execute() to implement custom backends:
    - Claude API
    - MCP client integration
    - Custom LLM providers
    - Multi-provider routing
    """

    @abstractmethod
    async def execute(
        self, params: CompletionCreateParams
    ) -> ChatCompletion | AsyncChatCompletionStreamManager[Any]:
        """
        Execute the chat completion request.

        Args:
            params: The chat completion parameters

        Returns:
            ChatCompletion for non-streaming, AsyncStream[ChatCompletionChunk] for streaming
        """
        pass


class OpenAIProxyHandler(ProxyHandler):
    """Default handler that proxies to an OpenAI-compatible API."""

    def __init__(self, config: ProxyConfig):
        self.config = config
        self.client = AsyncOpenAI(
            base_url=config.upstream_url,
            api_key=config.upstream_api_key or "dummy",
        )

    async def execute(
        self, params: CompletionCreateParams
    ) -> ChatCompletion | AsyncChatCompletionStreamManager[Any]:
        """Forward request to upstream OpenAI-compatible API."""
        if params.get("stream"):
            # Use .stream() for streaming requests - returns AsyncChatCompletionStreamManager
            stream_params = params.copy()
            stream_params.pop("stream", None)
            return self.client.chat.completions.stream(**stream_params)  # type: ignore[return-value]
        else:
            # Use .create() for non-streaming requests
            return await self.client.chat.completions.create(**params)  # type: ignore[return-value]
