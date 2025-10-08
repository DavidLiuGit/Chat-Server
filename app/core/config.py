from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ProxyConfig(BaseSettings):
    """Configuration for the chat completion proxy server."""

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
    )

    # Upstream service configuration
    upstream_url: str = Field(
        default="https://api.openai.com/v1",
        description="Base URL of the upstream OpenAI-compatible API",
        validation_alias="OPENAI_API_URL",
    )
    upstream_api_key: str = Field(
        default="",
        description="API key for upstream service authentication",
        validation_alias="OPENAI_API_KEY",
    )

    # Model configuration
    default_model: str | None = Field(
        default=None,
        description="Override model name in all requests. None = use client's model",
    )

    # Server configuration
    host: str = Field(default="0.0.0.0", description="Server bind address")
    port: int = Field(default=8765, description="Server port")

    # Feature flags
    enable_streaming: bool = Field(
        default=True, description="Support streaming responses"
    )
    enable_telemetry: bool = Field(
        default=False, description="Enable built-in telemetry plugin"
    )
