"""
Example: Custom plugins for MCP, guardrails, and telemetry.

Demonstrates:
- Synchronous hooks (blocking) for prompt modification
- Asynchronous hooks (non-blocking) for logging/telemetry
"""

from openai.types.chat import ChatCompletion, CompletionCreateParams

from app import ChatCompletionServer, ProxyPlugin


class MCPPlugin(ProxyPlugin):
    """Plugin that injects MCP tool descriptions into prompts."""

    def __init__(self, mcp_client):
        self.mcp_client = mcp_client

    async def before_request(
        self, params: CompletionCreateParams
    ) -> CompletionCreateParams:
        # Synchronous - blocks request to inject tools
        # tools = await self.mcp_client.list_tools()
        # tool_descriptions = format_tools_for_prompt(tools)
        # 
        # messages = params.get("messages", [])
        # if messages:
        #     messages[0]["content"] = f"{tool_descriptions}\n\n{messages[0]['content']}"
        
        return params


class GuardrailsPlugin(ProxyPlugin):
    """Plugin for content validation (synchronous)."""

    async def before_request(
        self, params: CompletionCreateParams
    ) -> CompletionCreateParams:
        # Synchronous - blocks request for validation
        messages = params.get("messages", [])
        for msg in messages:
            content = msg.get("content", "")
            if "forbidden_word" in content:
                raise ValueError("Content violates guardrails")
        return params


class TelemetryPlugin(ProxyPlugin):
    """Plugin for metrics collection (asynchronous)."""

    async def after_request_async(
        self, params: CompletionCreateParams, response: ChatCompletion
    ) -> None:
        # Asynchronous - does not block response
        # metrics.increment("requests.total")
        # if response.usage:
        #     metrics.gauge("tokens.total", response.usage.total_tokens)
        pass

    async def on_error_async(
        self, params: CompletionCreateParams, error: Exception
    ) -> None:
        # Asynchronous - does not block error handling
        # metrics.increment("requests.errors")
        pass


# Usage
if __name__ == "__main__":
    # mcp_client = MyMCPClient()
    
    server = ChatCompletionServer(
        plugins=[
            # MCPPlugin(mcp_client),  # Injects tools into prompt
            GuardrailsPlugin(),       # Validates before sending
            TelemetryPlugin(),        # Logs async after response
        ]
    )
