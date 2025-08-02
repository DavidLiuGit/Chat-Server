import asyncio
from logging import getLogger
import re
from time import time

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from app.services.chat import handle_chat
from app.models.v1.completions import (
    ChatCompletionsRequest,
    ChatCompletionsResponse,
    ChatCompletionsChoice,
)
from app.core.logging import get_request_id
from app.api.sse_utils import sse_chunk, create_chunk, sse_done


router = APIRouter()

logger = getLogger(__name__)


async def fake_stream(request_id: str, model: str, content: str):
    tokens = re.split(r"(\s+)", content)

    for token in tokens:
        if token:
            chunk = create_chunk(request_id, model, token)
            yield sse_chunk(chunk)
            await asyncio.sleep(0.05)

    # Required: final chunk with finish_reason
    yield sse_chunk(create_chunk(request_id, model, finish_reason="stop"))

    # Required: SSE terminator
    yield sse_done()


@router.post("/chat/completions")
async def chat_completions(request: ChatCompletionsRequest):
    request_id = get_request_id()
    logger.info(f"request: {request.get_log_sanitized_str()}")
    
    # generate n ChatCompletionsChoices
    message = handle_chat(request)

    if request.stream:
        return StreamingResponse(
            fake_stream(request_id, request.model, message.content), media_type="text/plain"
        )

    choices = [
        ChatCompletionsChoice(
            index=i,
            message=message,
            finish_reason="stop",
        )
        for i in range(request.n)
    ]

    return ChatCompletionsResponse(
        id=request_id,
        object="chat.completion",
        created=int(time()),
        model=request.model,
        choices=choices,
    )
