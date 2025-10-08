from logging import getLogger

from openai.types.chat import ChatCompletion, CompletionCreateParams

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

    async def on_error_async(
        self, params: CompletionCreateParams, error: Exception
    ) -> None:
        logger.error(f"Chat completion error: {error}", exc_info=True)
