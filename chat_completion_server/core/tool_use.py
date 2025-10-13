import requests
from typing import Dict, Any

from openai.types.chat import ChatCompletionMessageToolCall

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

    def execute_tool(self, tool_call: ChatCompletionMessageToolCall) -> Dict[str, Any]:
        """Execute a tool call via the upstream proxy."""
        response = requests.post(self.tool_exec_url, json=tool_call.model_dump(), headers=self.headers)
        response.raise_for_status()
        return response.json()
