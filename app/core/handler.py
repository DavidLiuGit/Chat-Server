from abc import ABC, abstractmethod
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion, CompletionCreateParams

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
    async def execute(self, params: CompletionCreateParams) -> ChatCompletion:
        """
        Execute the chat completion request.
        
        Args:
            params: The chat completion parameters
            
        Returns:
            Chat completion response
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

    async def execute(self, params: CompletionCreateParams) -> ChatCompletion:
        """Forward request to upstream OpenAI-compatible API."""
        if self.config.default_model:
            params["model"] = self.config.default_model

        return await self.client.chat.completions.create(**params)  # type: ignore[return-value]
