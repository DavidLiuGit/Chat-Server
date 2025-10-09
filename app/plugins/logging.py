from logging import getLogger
from typing import Any

from openai.types.chat import ChatCompletion, CompletionCreateParams
from openai.lib.streaming.chat import ChatCompletionStreamEvent

from app.core.plugin import ProxyPlugin

logger = getLogger(__name__)


class LoggingPlugin(ProxyPlugin):
    """Plugin that logs request and response details asynchronously."""

    async def before_request(
        self, params: CompletionCreateParams
    ) -> CompletionCreateParams:
        model = params.get("model", "unknown")
        stream = params.get("stream", False)
        logger.info(f"Chat completion request: model={model}, stream={stream}")
        return params

    async def after_request_async(
        self, params: CompletionCreateParams, response: ChatCompletion
    ) -> None:
        model = response.model
        usage = response.usage
        logger.info(f"Chat completion response: model={model}, usage={usage}")

    async def after_stream_async(
        self, params: CompletionCreateParams, response: ChatCompletion, events: list[ChatCompletionStreamEvent]
    ) -> None:
        model = response.model
        usage = response.usage
        event_count = len(events)
        logger.info(f"Chat completion stream: model={model}, usage={usage}, events={event_count}")

    async def on_error_async(
        self, params: CompletionCreateParams, error: Exception
    ) -> None:
        logger.error(f"Chat completion error: {error}", exc_info=True)
