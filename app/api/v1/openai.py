from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from logging import getLogger
from openai.types.chat import ChatCompletion, CompletionCreateParams
from typing import Union

from app.api.sse_utils import openai_sse_generator
from app.core.proxy import proxy_openai_chat_completion

logger = getLogger(__name__)

router = APIRouter()


@router.post("/chat/completions", response_model=None)
async def chat_completions(
    params: CompletionCreateParams,
):
    """
    OpenAI-compatible chat completions endpoint.
    Supports both streaming and non-streaming responses.
    """
    try:
        if params.get("stream"):
            response = await proxy_openai_chat_completion(params)
            return StreamingResponse(
                openai_sse_generator(response),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                },
            )

        return await proxy_openai_chat_completion(params)
    except Exception as e:
        logger.error(f"Error in chat_completions: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")
