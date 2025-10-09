"""
Example: Custom handler for alternative LLM providers.

This demonstrates how to override the default OpenAI proxy behavior
to use a different backend like Claude API, Gemini, etc.

Note: MCP integration should be done via plugins, not handlers.
See custom_plugin.py for MCP example.
"""

from openai.types.chat import ChatCompletion, CompletionCreateParams

from app import ChatCompletionServer, ProxyHandler


class ClaudeHandler(ProxyHandler):
    """Example handler for Claude API."""

    async def execute_streaming(self, params: CompletionCreateParams) -> ChatCompletion:
        # Implement Claude API integration here
        # from anthropic import AsyncAnthropic
        # client = AsyncAnthropic(api_key="...")
        # response = await client.messages.create(...)
        # return convert_to_openai_format(response)
        raise NotImplementedError("Claude API integration not implemented")


class MultiProviderHandler(ProxyHandler):
    """Example handler that routes to different providers based on model."""

    async def execute_streaming(self, params: CompletionCreateParams) -> ChatCompletion:
        model = params.get("model", "")
        
        # Route based on model prefix
        # if model.startswith("claude-"):
        #     return await self._call_claude(params)
        # elif model.startswith("gemini-"):
        #     return await self._call_gemini(params)
        # else:
        #     return await self._call_openai(params)
        
        raise NotImplementedError("Multi-provider routing not implemented")


# Usage
if __name__ == "__main__":
    # With Claude handler
    server = ChatCompletionServer(handler=ClaudeHandler())

    # With multi-provider routing
    # server = ChatCompletionServer(handler=MultiProviderHandler())
