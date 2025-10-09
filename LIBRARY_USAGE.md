# Library Usage Guide

## Overview

Chat-Server is an extensible OpenAI API-compatible proxy server built on FastAPI. It allows you to intercept, modify, and route chat completion requests with minimal code.

## Quick Start

### Minimal Setup

```python
import uvicorn
from app import ChatCompletionServer

# Create server - automatically exposes REST APIs
server = ChatCompletionServer()
app = server.app  # FastAPI application

# Run the server
uvicorn.run(app, host="0.0.0.0", port=8765)
```

This exposes:
- `POST /v1/chat/completions` - Chat completion endpoint
- `GET /v1/models` - List available models
- `GET /v1/models/{model}` - Get model details

### With Configuration

```python
from app import ChatCompletionServer, ProxyConfig, ModelConfig

config = ProxyConfig(
    upstream_url="https://api.openai.com/v1",
    upstream_api_key="sk-...",
)

models = {
    "my-model": ModelConfig(id="my-model", ...)
}

server = ChatCompletionServer(config=config, models=models)
app = server.app
```

### Environment Variables

Set `PROXY_*` prefixed environment variables:

```bash
PROXY_UPSTREAM_URL=https://custom.api/v1
PROXY_UPSTREAM_API_KEY=sk-...
PROXY_DEFAULT_MODEL=gpt-4
```

## Extensibility

### Custom Plugins

Plugins support both synchronous (blocking) and asynchronous (non-blocking) hooks:

```python
from app import ProxyPlugin, ChatCompletionServer

class MyPlugin(ProxyPlugin):
    async def before_request(self, params):
        # SYNCHRONOUS - blocks request
        # Use for: MCP tool injection, prompt modification, guardrails
        return params
    
    async def after_request_async(self, params, response):
        # ASYNCHRONOUS - runs in background, does not block response
        # Use for: logging, telemetry, analytics
        pass
    
    async def on_error_async(self, params, error):
        # ASYNCHRONOUS - runs in background
        # Use for: error logging, error telemetry
        pass

server = ChatCompletionServer(plugins=[MyPlugin()])
```

#### MCP Integration Example

```python
class MCPPlugin(ProxyPlugin):
    def __init__(self, mcp_client):
        self.mcp_client = mcp_client
    
    async def before_request(self, params):
        # Inject MCP tool descriptions into prompt
        tools = await self.mcp_client.list_tools()
        tool_text = format_tools(tools)
        
        messages = params.get("messages", [])
        if messages:
            messages[0]["content"] = f"{tool_text}\n\n{messages[0]['content']}"
        
        return params
```

### Custom Handlers

Override the proxy backend:

```python
from app import ProxyHandler, ChatCompletionServer

class ClaudeHandler(ProxyHandler):
    async def execute(self, params):
        # Call Claude API instead
        # Or integrate MCP client
        # Or route to multiple providers
        pass

server = ChatCompletionServer(handler=ClaudeHandler())
```

## Configuration Reference

See `ProxyConfig` in `app/core/config.py` for all available options:

- `upstream_url`: Base URL of upstream API
- `upstream_api_key`: API key for authentication
- `default_model`: Override model in all requests
- `host`: Server bind address
- `port`: Server port
- `enable_streaming`: Support streaming responses
- `enable_telemetry`: Enable built-in telemetry

## Examples

See the `examples/` directory for:
- `minimal_usage.py` - Simple setup patterns
- `custom_plugin.py` - Guardrails and telemetry plugins
- `custom_handler.py` - Claude API and MCP integration

## Future Support

- Responses API (OpenAI's new API)
- Additional built-in plugins
- More handler examples
