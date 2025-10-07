from logging import getLogger
from os import getenv as os_getenv
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion, CompletionCreateParams

logger = getLogger(__name__)

# Initialize client with custom base URL from environment
openai_like_client = AsyncOpenAI(
    base_url=os_getenv("OPENAI_API_URL"),
    api_key=os_getenv("OPENAI_API_KEY", "dummy"),
)


async def proxy_openai_chat_completion(params: CompletionCreateParams):
    """
    Proxy chat completion request to LLM gateway.

    TODO: Add guardrails
    TODO: Add logging
    TODO: Add telemetry
    """
    params["model"] = "bedrock/global.anthropic.claude-sonnet-4-20250514-v1:0"
    return await openai_like_client.chat.completions.create(**params)  # type: ignore[arg-type]
