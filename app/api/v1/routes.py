from logging import getLogger
from time import time

from fastapi import APIRouter

from app.api.sse_utils import sse_chunk, create_chunk, sse_done
from app.services.chat import handle_chat, stream_chat
from app.models.v1.completions import ChatCompletionsRequest, ChatCompletionsResponse, ChatCompletionsChoice
from app.core.logging import get_request_id


router = APIRouter()

logger = getLogger(__name__)


@router.post(
    "/chat/completions",
    response_model=ChatCompletionsResponse,
)
async def chat_completions(request: ChatCompletionsRequest):
    request_id = get_request_id()
    
    # generate n ChatCompletionsChoices
    logger.info(f"Generating {request.n} choices. history_depth={len(request.messages)}")
    choices = [
        ChatCompletionsChoice(
            index=i,
            message=handle_chat(request),
            finish_reason="stop",
        )
        for i in range(request.n)
    ]

    response = ChatCompletionsResponse(
        id=request_id,
        object="chat.completion",
        created=int(time()),
        model=request.model,
        choices=choices,
    )
    return response
