from logging import getLogger
from typing import Any

from openai.types.chat import ChatCompletion, CompletionCreateParams
from openai.lib.streaming.chat import ChatCompletionStreamEvent

from chat_completion_server.models.plugin import ProxyPlugin

logger = getLogger(__name__)


class LoggingPlugin(ProxyPlugin):
    """Plugin that logs request and response details asynchronously."""

    async def before_request(
        self, params: CompletionCreateParams
    ) -> CompletionCreateParams:
        log_data = {
            "model": params.get("model"),
            "messages": params.get("messages"),
        }
        
        if params.get("stream") is not None:
            log_data["stream"] = params.get("stream")
        
        if params.get("max_completion_tokens") is not None:
            log_data["max_completion_tokens"] = params.get("max_completion_tokens")
        
        messages = log_data.get("messages")
        if messages and isinstance(messages, (list, tuple)) and len(messages) > 4:
            messages = messages[-4:]
            
        if messages:
            truncated_messages = []
            for msg in messages:
                if isinstance(msg, dict) and "content" in msg:
                    truncated_msg = msg.copy()
                    if isinstance(msg["content"], str):
                        truncated_msg["content"] = self._truncate_message(msg["content"])
                    truncated_messages.append(truncated_msg)
                else:
                    truncated_messages.append(msg)
            log_data["messages"] = truncated_messages
        
        logger.info(f"Chat completion request: {log_data}")
        return params

    def _truncate_message(self, msg: str, start_len: int = 64, end_len: int = 128) -> str:
        if len(msg) <= start_len + end_len:
            return msg
        return f"{msg[:start_len]}...{msg[-end_len:]}"

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
