# Streaming Implementation

## Overview

The server now properly handles OpenAI-compatible streaming responses with post-flight analysis capabilities.

## Key Features

### 1. **Type Consistency with OpenAI SDK**
- Uses `AsyncChatCompletionStreamManager[Any]` for streaming responses
- Returns `ChatCompletion` for non-streaming responses
- All types match OpenAI SDK exactly: `ChatCompletionChunk`, `ChatCompletion`, `CompletionCreateParams`

### 2. **Streaming Flow**

```
Client Request (stream=true)
    ↓
before_request hooks (blocking)
    ↓
Handler.execute() → AsyncChatCompletionStreamManager
    ↓
openai_sse_generator() - streams chunks to client
    ├─→ Yields SSE-formatted chunks immediately
    └─→ Accumulates chunks in memory
    ↓
get_final_completion() - reconstructs full response
    ↓
after_stream_async hooks (non-blocking, background)
```

### 3. **Chunk Accumulation**

The `openai_sse_generator` function:
- Streams chunks to client in real-time (no buffering delay)
- Simultaneously accumulates chunks in a list
- After streaming completes, provides both:
  - `final_completion`: Full `ChatCompletion` object with usage stats
  - `chunks`: List of all `ChatCompletionChunk` objects

### 4. **Post-Flight Analysis**

Plugins can implement `after_stream_async` to:
- Log complete responses with usage statistics
- Send telemetry data
- Perform content analysis
- Calculate costs
- Store conversation history

Example:
```python
async def after_stream_async(
    self, 
    params: CompletionCreateParams, 
    response: ChatCompletion,
    chunks: list[ChatCompletionChunk]
) -> None:
    # Access final response
    print(f"Model: {response.model}")
    print(f"Usage: {response.usage}")
    print(f"Content: {response.choices[0].message.content}")
    
    # Access individual chunks
    print(f"Total chunks: {len(chunks)}")
```

## Implementation Details

### SSE Format
Chunks are streamed in Server-Sent Events format:
```
data: {"id":"chatcmpl-123","object":"chat.completion.chunk",...}

data: {"id":"chatcmpl-123","object":"chat.completion.chunk",...}

data: [DONE]
```

### Handler Contract
Handlers must return:
- `ChatCompletion` when `stream=False`
- `AsyncChatCompletionStreamManager[Any]` when `stream=True`

The `OpenAIProxyHandler` uses:
- `client.chat.completions.create()` for non-streaming
- `client.chat.completions.stream()` for streaming

## Testing

Run the test script:
```bash
# Start server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8765

# In another terminal
python test_streaming.py
```

The test demonstrates the same pattern as OpenAI's SDK examples.


## Proxying

Use Maxim bifrost as an LLM Gateway. Example setup:

```json
{
  "providers": {
    "bedrock": {
      "send_back_raw_response": true,
      "keys": [
        {
          "models": [
            "global.anthropic.claude-sonnet-4-20250514-v1:0",
            "anthropic.claude-sonnet-4-20250514-v1:0",
            "claude-sonnet-4"
          ],
          "weight": 1.0,
          "bedrock_key_config": {
            "access_key": "<access_key>",
            "secret_key": "<secret_key>",
            "deployments": {
              "global.anthropic.claude-sonnet-4-20250514-v1:0": "global.anthropic.claude-sonnet-4-20250514-v1:0",
              "anthropic.claude-sonnet-4-20250514-v1:0": "anthropic.claude-sonnet-4-20250514-v1:0",
              "claude-sonnet-4": "global.anthropic.claude-sonnet-4-20250514-v1:0"
            },
            "arn": "arn:aws:bedrock:us-east-1:533266982454:inference-profile"
          }
        }
      ]
    }
  }
}
```

Launch the server using Docker:
```bash
docker run -p 9090:8080 -v $(pwd)/data:/app/data maximhq/bifrost
```

