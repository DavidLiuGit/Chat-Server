import json

async def openai_sse_generator(stream):
    """Convert OpenAI stream to SSE format with error handling."""
    from logging import getLogger
    logger = getLogger(__name__)
    
    try:
        async for chunk in stream:
            try:
                yield f"data: {chunk.model_dump_json()}\n\n"
            except Exception as e:
                logger.warning(f"Failed to serialize chunk: {e}")
                continue
        yield "data: [DONE]\n\n"
    except Exception as e:
        logger.error(f"Stream error: {e}", exc_info=True)
        error_data = json.dumps({"error": str(e)})
        yield f"data: {error_data}\n\n"
