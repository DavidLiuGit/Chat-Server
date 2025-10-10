from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ProxyConfig(BaseSettings):
    """Configuration for the chat completion proxy server."""

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
        use_attribute_docstrings=True,
    )

    upstream_url: str = Field(
        default="https://api.openai.com/v1"
        # default="https://api.openai.com/v1", validation_alias="OPENAI_API_URL"
    )
    """Base URL of the upstream OpenAI-compatible API"""

    upstream_api_key: str = Field(default="", )#validation_alias="OPENAI_API_KEY")
    """API key for upstream service authentication"""

    host: str = "0.0.0.0"
    """Server bind address"""

    port: int = 8765
    """Server port"""

    enable_streaming: bool = True
    """Support streaming responses"""

    enable_telemetry: bool = False
    """Enable built-in telemetry plugin"""
