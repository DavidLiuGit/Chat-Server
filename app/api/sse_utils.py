import json
from logging import getLogger

logger = getLogger(__name__)


async def openai_sse_generator(stream):
    """Convert OpenAI stream to SSE format with error handling."""
    chunk_count = 0
    
    try:
        async for chunk in stream:
            try:
                chunk_count += 1
                yield f"data: {chunk.model_dump_json()}\n\n"
            except Exception as e:
                logger.warning(f"Failed to serialize chunk: {e}")
                continue
        
        yield "data: [DONE]\n\n"
        logger.info(f"Stream completed: {chunk_count} chunks sent")
        
    except Exception as e:
        logger.error(f"Stream error after {chunk_count} chunks: {e}", exc_info=True)
        error_data = json.dumps({"error": str(e)})
        yield f"data: {error_data}\n\n"
