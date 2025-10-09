"""
Example: Minimal usage of ChatCompletionServer.

This demonstrates the simplest way to use the library with defaults.
"""

import uvicorn
from app import ChatCompletionServer, ProxyConfig

# Minimal - uses all defaults
server = ChatCompletionServer()
app = server.app  # FastAPI app with REST APIs

# With custom config
config = ProxyConfig(
    upstream_url="https://custom.api/v1",
    upstream_api_key="sk-...",
    default_model="gpt-4",
)
server = ChatCompletionServer(config=config)

# Config from environment variables (PROXY_* prefix)
# PROXY_UPSTREAM_URL=https://custom.api/v1
# PROXY_UPSTREAM_API_KEY=sk-...
# PROXY_DEFAULT_MODEL=gpt-4
server = ChatCompletionServer(config=ProxyConfig())

# Run the server
if __name__ == "__main__":
    uvicorn.run(server.app, host="0.0.0.0", port=8765)
