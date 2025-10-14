import asyncio
from logging import getLogger
from time import time

from typing import Any, AsyncIterator

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from openai.lib.streaming.chat import AsyncChatCompletionStreamManager, ChatCompletionStreamEvent
from openai.pagination import SyncPage
from openai.types import Model
from openai.types.chat import ChatCompletion, CompletionCreateParams

from chat_completion_server.models.config import ProxyConfig
from chat_completion_server.core.constants import (
    SSE_DATA_PREFIX,
    SSE_LINE_ENDING,
    SSE_DONE_MESSAGE,
    STREAMING_HEADERS,
)
from chat_completion_server.core.proxy_handler import OpenAIProxyHandler, ProxyHandler
from chat_completion_server.core.logging import generate_request_id, set_request_id
from chat_completion_server.core.model_manager import ModelManager
from chat_completion_server.core.normalizer import normalize_chat_completion
from chat_completion_server.core.tool_use import ProxyToolClient
from chat_completion_server.models.plugin import ProxyPlugin
from chat_completion_server.models import create_model_metadata, ModelConfig
from chat_completion_server.plugins.guardrails import GuardrailsPlugin
from chat_completion_server.plugins.logging import LoggingPlugin


logger = getLogger(__name__)

MAX_TOOL_ROUNDS = 5


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
        proxy_handler: ProxyHandler | None = None,
        proxy_tool_client: ProxyToolClient | None = None,
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
        self.proxy_handler = proxy_handler or OpenAIProxyHandler(self.config)
        self.proxy_tool_client = proxy_tool_client or ProxyToolClient(self.config)
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
        3. Split into streaming/non-streaming processing

        Args:
            params: Chat completion parameters

        Returns:
            ChatCompletion for non-streaming, AsyncStream (stream manager) for streaming

        Raises:
            Exception: Any error during processing
        """
        try:
            # Ensure messages is a list to prevent iterator consumption issues
            if "messages" in params:
                params["messages"] = list(params["messages"])

            # Apply model-specific configuration
            params = self.model_manager.apply_model_config(params)

            # Synchronous before_request hooks (blocking)
            for plugin in self.plugins:
                params = await plugin.before_request(params)

            # Execute initial user request
            response = await self.proxy_handler.execute(params)

            # Split processing based on streaming mode
            if params.get("stream"):
                return await self._process_streaming_response(params, response)
            else:
                return await self._process_non_streaming_response(params, response)

        except Exception as e:
            # Fire error hooks in background
            asyncio.create_task(self._run_on_error_hooks(params, e))
            raise

    async def _process_streaming_response(
        self, params: CompletionCreateParams, response: AsyncChatCompletionStreamManager[Any]
    ) -> AsyncChatCompletionStreamManager[Any]:
        """Process streaming response. Tool use support planned for future."""
        return response

    async def _process_non_streaming_response(
        self, params: CompletionCreateParams, response: ChatCompletion
    ) -> ChatCompletion:
        """Process non-streaming response with tool use support."""
        response = normalize_chat_completion(response)

        # TODO remove this after https://github.com/maximhq/bifrost/issues/617
        if response.choices[0].finish_reason == "tool_use":
            response.choices[0].finish_reason = "tool_calls"

        messages = list(params.get("messages", []))

        tool_round, tool_call_count = 0, 0

        # Handle tool calls
        while (
            response.choices[0].finish_reason == "tool_calls"
            and response.choices[0].message.tool_calls
            and tool_round < MAX_TOOL_ROUNDS
        ):
            tool_round += 1
            for tool_call in response.choices[0].message.tool_calls:
                tool_msg = await self.proxy_tool_client.execute_tool(tool_call)
                messages.append(ProxyToolClient.tool_call_to_msg(tool_call))
                messages.append(tool_msg)
                tool_call_count += 1

            params["messages"] = messages
            response = await self.proxy_handler.execute_non_streaming(params)
            response = normalize_chat_completion(response)

            # TODO remove this after https://github.com/maximhq/bifrost/issues/617
            if response.choices[0].finish_reason == "tool_use":
                response.choices[0].finish_reason = "tool_calls"

        if tool_round >= MAX_TOOL_ROUNDS:
            logger.warning(f"[ToolCalling] Max tool rounds reached: {MAX_TOOL_ROUNDS}")
            response.choices[0].finish_reason = "length"
        elif tool_round > 0:
            logger.info(f"[ToolCalling] Tool rounds: {tool_round}; tools called: {tool_call_count}")

        asyncio.create_task(self._run_after_request_hooks(params, response))
        return response

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
        event_types: dict[str, int] = {}

        # handle more types?
        # https://github.com/openai/openai-python/blob/main/examples/parsing_stream.py
        async with stream_manager as stream:
            async for event in stream:
                events.append(event)
                event_types[event.type] = event_types.get(event.type, 0) + 1
                if event.type == "content.delta" or event.type == "refusal.delta":
                    yield f"{SSE_DATA_PREFIX}{event.delta}{SSE_LINE_ENDING}"
                if event.type == "chunk":
                    yield f"{SSE_DATA_PREFIX}{event.chunk.model_dump_json(exclude_none=True)}{SSE_LINE_ENDING}"
                if event.type == "refusal.delta":
                    yield f"{SSE_DATA_PREFIX}{event.delta}{SSE_LINE_ENDING}"

            yield SSE_DONE_MESSAGE
            final_completion = await stream.get_final_completion()

        # debugging output
        logger.info(f"Final event: {events[-1]}")
        logger.info(f"\t{event_types=}")
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

        # Add CORS middleware, and header configs
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        @app.middleware("http")
        async def add_request_id_middleware(request: Request, call_next):
            request_id = generate_request_id()
            set_request_id(request_id)
            logger.info(f"Request started: {request.method} {request.url.path}")

            start_time = time()
            response = await call_next(request)
            process_time = time() - start_time

            logger.info(
                f"Request completed: {request.method} {request.url.path} - elapsed={process_time:.3f}s"
            )

            return response

        # Register routes inline
        @app.post("/v1/chat/completions", response_model=None)
        @app.post("/chat/completions", response_model=None)
        async def chat_completions(params: CompletionCreateParams):
            """
            OpenAI `/chat/completions` compatible endpoint.
            """
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
            """Return a list of all registered `Model`s."""
            models = [
                create_model_metadata(model_id) for model_id in self.model_manager.models.keys()
            ]
            return SyncPage(data=models, object="list")

        @app.get("/v1/models/{model}", response_model_exclude_none=True)
        @app.get("/models/{model}", response_model_exclude_none=True)
        def retrieve_model(model: str) -> Model | None:
            """Return a `Model`, if its ID is found."""
            if model in self.model_manager.models:
                return create_model_metadata(model)
            return None

        return app
