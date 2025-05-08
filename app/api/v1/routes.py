from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse, JSONResponse
from app.services.chat import handle_chat, stream_chat

router = APIRouter()

@router.post("/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    stream = body.get("stream", False)
    if stream:
        return StreamingResponse(stream_chat(body), media_type="text/event-stream")
    return JSONResponse(await handle_chat(body))
