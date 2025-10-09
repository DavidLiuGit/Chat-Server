import json
from logging import getLogger
from typing import Any, AsyncIterator

from openai import AsyncStream
from openai.lib.streaming.chat import AsyncChatCompletionStreamManager

logger = getLogger(__name__)


async def openai_sse_generator(
    stream_manager: AsyncChatCompletionStreamManager[Any], params=None, plugins=None
) -> AsyncIterator[str]:
    """
    Convert OpenAI stream manager to SSE format with logging.
    
    Uses the stream manager's get_final_completion() to access the complete
    response after streaming finishes.
    """
    chunk_count = 0
    collected_content = []
    
    try:
        # Stream chunks to client
        async with stream_manager as stream:

            async for event in stream:
                try:
                    logger.info(f"Chunk_{chunk_count}: {event}")
                    chunk_count += 1
                    
                    # Collect content for logging
                    if hasattr(event, 'choices') and event.choices:
                        delta = event.choices[0].delta
                        if hasattr(delta, 'content') and delta.content:
                            collected_content.append(delta.content)
                    
                    yield f"data: {event.model_dump_json()}\n\n"
                except Exception as e:
                    logger.warning(f"Failed to serialize chunk {chunk_count}: {e}")
                    continue
            
            yield "data: [DONE]\n\n"
        
        # Get final completion from stream manager
        final_completion = None
        try:
            final_completion = await stream_manager.get_final_completion()
        except Exception as e:
            logger.warning(f"Could not get final completion: {e}")
        
        # Log complete response
        full_content = ''.join(collected_content)
        logger.info(
            f"Stream completed: {chunk_count} chunks, "
            f"{len(full_content)} chars: {full_content[:100]}..."
        )
        
        # Fire async hooks with final completion
        if plugins and params and final_completion:
            for plugin in plugins:
                try:
                    await plugin.after_request_async(params, final_completion)
                except Exception as e:
                    logger.error(f"Error in async hook: {e}", exc_info=True)
        
    except Exception as e:
        logger.error(f"Stream error after {chunk_count} chunks: {e}", exc_info=True)
        error_data = json.dumps({"error": str(e)})
        yield f"data: {error_data}\n\n"
