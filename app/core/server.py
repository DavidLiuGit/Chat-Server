import asyncio
from logging import getLogger

from typing import Any, AsyncIterator

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from openai.lib.streaming.chat import AsyncChatCompletionStreamManager, ChatCompletionStreamEvent
from openai.pagination import SyncPage
from openai.types import Model
from openai.types.chat import ChatCompletion, CompletionCreateParams

from app.models.config import ProxyConfig
from app.core.constants import SSE_DATA_PREFIX, SSE_LINE_ENDING, SSE_DONE_MESSAGE, STREAMING_HEADERS
from app.core.handler import OpenAIProxyHandler, ProxyHandler
from app.core.model_manager import ModelManager
from app.models.plugin import ProxyPlugin
from app.models import create_model_metadata, ModelConfig
from app.plugins.guardrails import GuardrailsPlugin
from app.plugins.logging import LoggingPlugin


logger = getLogger(__name__)


class ChatCompletionServer:
    """
    Extensible chat completion proxy server with REST API.

    Creates a FastAPI application with:
    - POST /v1/chat/completions
    - GET /v1/models
    - GET /v1/models/{model}

    Example usage:
        # Minimal setup
        server = ChatCompletionServer()
        app = server.app  # FastAPI app ready to run

        # With custom config and models
        config = ProxyConfig(upstream_url="https://custom.api")
        models = {"my-model": ModelConfig(id="my-model", ...)}
        server = ChatCompletionServer(config=config, models=models)

        # Run with uvicorn
        uvicorn.run(server.app, host="0.0.0.0", port=8765)

    Note: Future support planned for Responses API.
    """

    @property
    def app(self) -> FastAPI:
        """Get the FastAPI application."""
        return self._app

    def __init__(
        self,
        config: ProxyConfig | None = None,
        handler: ProxyHandler | None = None,
        plugins: list[ProxyPlugin] | None = None,
        models: dict[str, ModelConfig] | None = None,
    ):
        """
        Initialize the chat completion server.

        Args:
            config: Server configuration. Defaults to ProxyConfig()
            handler: Custom handler for executing requests. Defaults to OpenAIProxyHandler
            plugins: List of plugins. Defaults to [GuardrailsPlugin(), LoggingPlugin()]
            models: Custom model configurations. Defaults to {}
        """
        self.config = config or ProxyConfig()
        self.handler = handler or OpenAIProxyHandler(self.config)
        self.plugins = (
            plugins
            if plugins is not None
            else [
                GuardrailsPlugin(),
                LoggingPlugin(),
            ]
        )

        # Initialize models with default if none provided
        # TODO: can be moved to ModelManager
        if models is None:
            models = {"custom-model": ModelConfig(id="custom-model")}

        self.model_manager = ModelManager(models)
        self._app = self._create_app()

    async def process_request(
        self, params: CompletionCreateParams
    ) -> ChatCompletion | AsyncChatCompletionStreamManager[Any]:
        """
        Process a chat completion request through the plugin pipeline.

        Flow:
        1. Synchronous before_request hooks (blocking)
        2. Execute request via handler
        3. Fire async hooks in background (non-blocking) - only for non-streaming
        4. Return response immediately

        Args:
            params: Chat completion parameters

        Returns:
            ChatCompletion for non-streaming, AsyncStream (stream manager) for streaming

        Raises:
            Exception: Any error during processing
        """
        try:
            # Apply model-specific configuration
            params = self.model_manager.apply_model_config(params)

            # Synchronous before_request hooks (blocking)
            for plugin in self.plugins:
                params = await plugin.before_request(params)

            # Execute request
            response = self.handler.execute(params)

            # Fire async hooks only for non-streaming responses
            # Streaming returns AsyncStream manager, handled in SSE generator
            if not params.get("stream") and isinstance(response, ChatCompletion):
                asyncio.create_task(self._run_after_request_hooks(params, response))

            return await response

        except Exception as e:
            # Fire error hooks in background
            asyncio.create_task(self._run_on_error_hooks(params, e))
            raise

    async def _run_after_request_hooks(
        self, params: CompletionCreateParams, response: ChatCompletion
    ) -> None:
        """Run after_request_async hooks in background."""
        for plugin in self.plugins:
            try:
                await plugin.after_request_async(params, response)
            except Exception as e:
                logger.exception("Error in async hook")

    async def _run_after_stream_hooks(
        self,
        params: CompletionCreateParams,
        response: ChatCompletion,
        events: list[ChatCompletionStreamEvent],
    ) -> None:
        """Run after_stream_async hooks in background."""
        for plugin in self.plugins:
            try:
                await plugin.after_stream_async(params, response, events)
            except Exception as e:
                logger.exception("Error in stream hook")

    async def _run_on_error_hooks(self, params: CompletionCreateParams, error: Exception) -> None:
        """Run on_error_async hooks in background."""
        for plugin in self.plugins:
            try:
                await plugin.on_error_async(params, error)
            except Exception as e:
                logger.exception("Error in error hook")

    async def _stream_with_hooks(
        self, stream_manager: AsyncChatCompletionStreamManager[Any], params: CompletionCreateParams
    ) -> AsyncIterator[str]:
        """Stream chunks to client and run post-flight hooks."""
        events: list[ChatCompletionStreamEvent] = []

        # handle more types?
        # https://github.com/openai/openai-python/blob/main/examples/parsing_stream.py
        async with stream_manager as stream:
            async for event in stream:
                events.append(event)
                if event.type == "chunk":
                    yield f"{SSE_DATA_PREFIX}{event.chunk.model_dump_json(exclude_none=True)}{SSE_LINE_ENDING}"

            yield SSE_DONE_MESSAGE
            final_completion = await stream.get_final_completion()

        # debugging output
        logger.info(f"Final event: {events[-1]}")
        asyncio.create_task(self._run_after_stream_hooks(params, final_completion, events))

    def _create_app(self) -> FastAPI:
        """
        Create and configure the FastAPI application with core routes.

        Registers the following endpoints:
        - POST /v1/chat/completions - OpenAI-compatible chat completion
        - POST /chat/completions - Alias without /v1 prefix
        - GET /v1/models - List all registered models
        - GET /models - Alias without /v1 prefix
        - GET /v1/models/{model} - Retrieve specific model metadata
        - GET /models/{model} - Alias without /v1 prefix

        Consumers can add custom routes after instantiation:
            server = ChatCompletionServer()
            app = server.app

            @app.post("/custom/endpoint")
            async def my_endpoint():
                return {"custom": "response"}

        Returns:
            Configured FastAPI application with CORS enabled
        """
        app = FastAPI(title="Chat Completion Proxy Server")

        # Add CORS middleware
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        # Register routes inline
        @app.post("/v1/chat/completions", response_model=None)
        @app.post("/chat/completions", response_model=None)
        async def chat_completions(params: CompletionCreateParams):
            try:
                response = await self.process_request(params)

                if params.get("stream"):
                    assert isinstance(response, AsyncChatCompletionStreamManager)
                    return StreamingResponse(
                        self._stream_with_hooks(response, params),
                        media_type="text/event-stream",
                        headers=STREAMING_HEADERS,
                    )

                return response

            except Exception as e:
                logger.exception("Error in chat_completions")
                raise HTTPException(status_code=500, detail=str(e))

        @app.get("/v1/models", response_model_exclude_none=True)
        @app.get("/models", response_model_exclude_none=True)
        def list_models() -> SyncPage[Model]:
            models = [
                create_model_metadata(model_id) for model_id in self.model_manager.models.keys()
            ]
            return SyncPage(data=models, object="list")

        @app.get("/v1/models/{model}", response_model_exclude_none=True)
        @app.get("/models/{model}", response_model_exclude_none=True)
        def retrieve_model(model: str) -> Model | None:
            if model in self.model_manager.models:
                return create_model_metadata(model)
            return None

        return app
