from logging import getLogger

from openai.types.chat import CompletionCreateParams

from chat_completion_server.models.plugin import ProxyPlugin

logger = getLogger(__name__)


class GuardrailsPlugin(ProxyPlugin):
    """Default guardrails plugin for content validation."""

    async def before_request(
        self, params: CompletionCreateParams
    ) -> CompletionCreateParams:
        # Basic validation - extend as needed
        messages = params.get("messages", [])
        if not messages:
            raise ValueError("Messages cannot be empty")
        
        return params
