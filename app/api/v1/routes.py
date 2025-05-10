from time import time
from uuid import uuid4

from fastapi import APIRouter, Request

from app.services.chat import handle_chat, stream_chat
from app.models.v1.completions import ChatCompletionsRequest, ChatCompletionsResponse, ChatCompletionsChoice

router = APIRouter()


@router.post(
    "/chat/completions",
    response_model=ChatCompletionsResponse,
)
async def chat_completions(request: ChatCompletionsRequest):
    # generate n ChatCompletionsChoices
    choices = [
        ChatCompletionsChoice(
            index=i,
            message=handle_chat(request),
            finish_reason="stop",
        )
        for i in range(request.n)
    ]

    response = ChatCompletionsResponse(
        id=str(uuid4()),
        object="chat.completion",
        created=int(time()),
        model=request.model,
        choices=choices,
    )
    return response
