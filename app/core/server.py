import asyncio
from logging import getLogger

from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from openai.lib.streaming.chat import AsyncChatCompletionStreamManager
from openai import AsyncStream
from openai.pagination import SyncPage
from openai.types import Model
from openai.types.chat import ChatCompletion, ChatCompletionChunk, CompletionCreateParams

from app.api.sse_utils import openai_sse_generator
from app.core.config import ProxyConfig
from app.core.handler import OpenAIProxyHandler, ProxyHandler
from app.core.model_manager import ModelManager
from app.core.plugin import ProxyPlugin
from app.models.model import ModelConfig
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
                asyncio.create_task(self._run_async_hooks(params, response))

            return await response

        except Exception as e:
            # Fire error hooks in background
            asyncio.create_task(self._run_error_hooks(params, e))
            raise

    async def _run_async_hooks(
        self, params: CompletionCreateParams, response: ChatCompletion
    ) -> None:
        """Run after_request_async hooks in background."""
        for plugin in self.plugins:
            try:
                await plugin.after_request_async(params, response)
            except Exception as e:
                logger.error(f"Error in async hook: {e}", exc_info=True)

    async def _run_error_hooks(self, params: CompletionCreateParams, error: Exception) -> None:
        """Run on_error_async hooks in background."""
        for plugin in self.plugins:
            try:
                await plugin.on_error_async(params, error)
            except Exception as e:
                logger.error(f"Error in error hook: {e}", exc_info=True)

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
                        openai_sse_generator(response, params, self.plugins),
                        media_type="text/event-stream",
                        headers={
                            "Cache-Control": "no-cache",
                            "Connection": "keep-alive",
                        },
                    )
                
                # Log non-streaming response
                # logger.info(f"Response: {response.model_dump_json(exclude_none=True)[:500]}...")
                return response
                
            except Exception as e:
                logger.error(f"Error in chat_completions: {e}", exc_info=True)
                raise HTTPException(status_code=500, detail=str(e))

        @app.get("/v1/models", response_model_exclude_none=True)
        @app.get("/models", response_model_exclude_none=True)
        def list_models() -> SyncPage[Model]:
            models = [
                Model(
                    id=model_id,
                    object="model",
                    created=1677610602,
                    owned_by="custom",
                )
                for model_id in self.model_manager.models.keys()
            ]
            return SyncPage(data=models, object="list")

        @app.get("/v1/models/{model}", response_model_exclude_none=True)
        @app.get("/models/{model}", response_model_exclude_none=True)
        def retrieve_model(model: str) -> Model | None:
            if model in self.model_manager.models:
                return Model(
                    id=model,
                    object="model",
                    created=1677610602,
                    owned_by="custom",
                )
            return None

        return app
