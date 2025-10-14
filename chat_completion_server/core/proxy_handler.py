from abc import ABC, abstractmethod
from typing import Any

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion, CompletionCreateParams
from openai.lib.streaming.chat import AsyncChatCompletionStreamManager

from chat_completion_server.models.config import ProxyConfig


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
        raise NotImplementedError("execute() shall be impl'd by child class")

    @abstractmethod
    async def execute_non_streaming(self, params: CompletionCreateParams) -> ChatCompletion:
        """Execute a non-streaming request. Return `ChatCompletion`."""
        raise NotImplementedError("execute_non_streaming() shall be impl'd by child class")

class OpenAIProxyHandler(ProxyHandler):
    """Default handler that proxies to an OpenAI-compatible API."""

    def __init__(self, config: ProxyConfig):
        self.config = config
        self.client = AsyncOpenAI(
            base_url=config.upstream_url,
            api_key=config.upstream_api_key or "dummy",
            timeout=20.0,  # in seconds
        )
        self.high_timeout_client = AsyncOpenAI(
            base_url=config.upstream_url,
            api_key=config.upstream_api_key or "dummy",
            timeout=60.0,  # in seconds
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
            return await self.execute_non_streaming(params)

    async def execute_non_streaming(self, params: CompletionCreateParams) -> ChatCompletion:
        """Execute a non-streaming request."""
        return await self.high_timeout_client.chat.completions.create(
            **params  # pyright: ignore[reportCallIssue,  reportArgumentType]
        )
