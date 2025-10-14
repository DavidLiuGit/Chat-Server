import httpx
from typing import Any

from openai.types.chat import ChatCompletionMessageToolCallUnion, ChatCompletionToolMessageParam, ChatCompletionMessageParam, ChatCompletionAssistantMessageParam

from chat_completion_server.models.config import ProxyConfig


class ProxyToolClient:
    def __init__(self, config: ProxyConfig):
        self.config = config
        self.base_url = config.upstream_url.rstrip("/")
        self.tool_exec_url = f"{self.base_url}{config.tool_exec_path}"
        self.headers = {
            "Authorization": f"Bearer {config.upstream_api_key}",
            "Content-Type": "application/json",
        }
        self.client = httpx.AsyncClient()

    async def execute_tool(
        self, tool_call: ChatCompletionMessageToolCallUnion
    ) -> ChatCompletionToolMessageParam:
        """Execute a tool call via the upstream proxy."""
        response = await self.client.post(
            self.tool_exec_url, json=tool_call.model_dump(), headers=self.headers
        )
        response.raise_for_status()
        return ChatCompletionToolMessageParam(response.json())

    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()

    @staticmethod
    def tool_call_to_msg(tool_call: ChatCompletionMessageToolCallUnion, content: str | None = None) -> ChatCompletionAssistantMessageParam:
        """Convert a tool call to a message."""
        return ChatCompletionAssistantMessageParam(
            role="assistant",
            content=content,
            tool_calls=[tool_call],
        )

        
