"""
Example: Custom model configurations with system prompt behaviors.

Demonstrates how to define models with different system prompt handling.
"""

from app import ChatCompletionServer, ModelConfig, SystemPromptBehavior

# Define custom models
models = {
    "custom-model": ModelConfig(
        id="custom-model",
        upstream_model="bedrock/global.anthropic.claude-sonnet-4-20250514-v1:0",
        system_prompt="You are a helpful assistant.",
        system_prompt_behavior=SystemPromptBehavior.DEFAULT,
    ),
    "assistant-strict": ModelConfig(
        id="assistant-strict",
        upstream_model="gpt-4",  # Maps to gpt-4 upstream
        system_prompt="You are a helpful assistant. Always be concise.",
        system_prompt_behavior=SystemPromptBehavior.OVERRIDE,
    ),
    "assistant-guided": ModelConfig(
        id="assistant-guided",
        system_prompt="Important: Follow all safety guidelines.",
        system_prompt_behavior=SystemPromptBehavior.PREPEND,
    ),
}

# Create server with custom models
server = ChatCompletionServer(models=models)

# FastAPI app with REST APIs
app = server.app

# Models are now available via:
# GET /v1/models - list all models
# GET /v1/models/{model} - get specific model
# POST /v1/chat/completions - chat with models

# Run with: uvicorn examples.custom_models:app --reload
