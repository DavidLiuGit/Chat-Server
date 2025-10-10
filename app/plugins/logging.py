from logging import getLogger
from typing import Any

from openai.types.chat import ChatCompletion, CompletionCreateParams
from openai.lib.streaming.chat import ChatCompletionStreamEvent

from app.models.plugin import ProxyPlugin

logger = getLogger(__name__)


class LoggingPlugin(ProxyPlugin):
    """Plugin that logs request and response details asynchronously."""

    async def before_request(
        self, params: CompletionCreateParams
    ) -> CompletionCreateParams:
        model = params.get("model", "unknown")
        stream = params.get("stream", False)
        logger.info(f"Chat completion request: {params=}")
        return params

    async def after_stream_async(
        self, params: CompletionCreateParams, response: ChatCompletion, events: list[ChatCompletionStreamEvent]
    ) -> None:
        model = response.model
        usage = response.usage
        if events:
            logger.info(f"Chat completion stream: model={model}, usage={usage}, events={len(events)}")
        else:
            logger.info(f"Chat completion response: model={model}, usage={usage}")

    async def on_error_async(
        self, params: CompletionCreateParams, error: Exception
    ) -> None:
        logger.error(f"Chat completion error: {error}", exc_info=True)
