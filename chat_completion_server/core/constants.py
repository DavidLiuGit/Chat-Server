# Model metadata constants
MODEL_OBJECT_TYPE = "model"
MODEL_CREATED_TIMESTAMP = 1677610602
MODEL_OWNER = "custom"

# Streaming constants
SSE_DATA_PREFIX = "data: "
SSE_LINE_ENDING = "\n\n"
SSE_DONE_MESSAGE = "data: [DONE]\n\n"

# HTTP headers
STREAMING_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
}

# Message roles
ROLE_SYSTEM = "system"
