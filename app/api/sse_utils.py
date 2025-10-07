import json
from time import time
from typing import Any


def sse_chunk(data: Any):
    """Format data as Server-Sent Event chunk."""
    return f"data: {json.dumps(data)}\n\n"


def create_chunk(request_id: str, model: str, content: str = "", finish_reason=None):
    """Create OpenAI-compatible streaming chunk."""
    return {
        "id": request_id,
        "object": "chat.completion.chunk",
        "created": int(time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": {"content": content} if content else {},
                "finish_reason": finish_reason,
            }
        ],
    }


def sse_done():
    """Return SSE stream terminator."""
    return "data: [DONE]\n\n"


async def openai_sse_generator(stream):
    """Convert OpenAI stream to SSE format with error handling."""
    try:
        async for chunk in stream:
            yield f"data: {chunk.model_dump_json()}\n\n"
        yield "data: [DONE]\n\n"
    except Exception as e:
        error_data = json.dumps({"error": str(e)})
        yield f"data: {error_data}\n\n"
