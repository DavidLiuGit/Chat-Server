from abc import ABC
from typing import Any
from openai.types.chat import ChatCompletion, CompletionCreateParams
from openai.lib.streaming.chat import ChatCompletionStreamEvent


class ProxyPlugin(ABC):
    """
    Base class for proxy plugins that hook into the request lifecycle.
    
    Synchronous hooks (blocking - executed before request):
    - before_request: Modify/validate requests, inject MCP tools, etc.
    
    Asynchronous hooks (non-blocking - executed after response):
    - after_request_async: Log/telemetry without blocking response
    - on_error_async: Error logging/telemetry
    """

    async def before_request(
        self, params: CompletionCreateParams
    ) -> CompletionCreateParams:
        """
        Synchronous hook called before forwarding request.
        BLOCKS the request - use for critical modifications.
        
        Use cases:
        - Inject MCP tool descriptions into prompt
        - Sanitize/validate prompts (guardrails)
        - Modify model parameters
        
        Args:
            params: Chat completion parameters
            
        Returns:
            Modified parameters
        """
        return params

    async def after_request_async(
        self, params: CompletionCreateParams, response: ChatCompletion
    ) -> None:
        """
        Asynchronous hook called after response.
        DOES NOT BLOCK - runs in background.
        
        Use cases:
        - Log responses
        - Send telemetry
        - Analytics
        
        Args:
            params: Original request parameters
            response: Chat completion response
        """
        pass

    async def after_stream_async(
        self, params: CompletionCreateParams, response: ChatCompletion, events: list[ChatCompletionStreamEvent]
    ) -> None:
        """
        Asynchronous hook called after streaming response completes.
        DOES NOT BLOCK - runs in background.
        
        Use cases:
        - Log streaming responses
        - Send telemetry with full accumulated response
        - Analytics on streaming data
        
        Args:
            params: Original request parameters
            response: Final accumulated ChatCompletion
            events: All stream events received during streaming
        """
        pass

    async def on_error_async(
        self, params: CompletionCreateParams, error: Exception
    ) -> None:
        """
        Asynchronous hook for errors.
        DOES NOT BLOCK - runs in background.
        
        Use cases:
        - Error logging
        - Error telemetry
        
        Args:
            params: Original request parameters
            error: The exception that occurred
        """
        pass
