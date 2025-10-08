from enum import Enum
from typing import Callable

from openai.types.chat import CompletionCreateParams
from pydantic import BaseModel, Field, ConfigDict


class SystemPromptBehavior(str, Enum):
    """Defines how system prompts are handled for a model."""

    PASSTHROUGH = "passthrough"
    """Use client's system prompt as-is"""
    OVERRIDE = "override"
    """Always replace with model's system prompt"""
    PREPEND = "prepend"
    """Prepend model's prompt to client's"""
    APPEND = "append"
    """Append model's prompt to client's"""
    DEFAULT = "default"
    """Use model's prompt only if client doesn't provide one"""


class ModelConfig(BaseModel):
    """Configuration for a custom LLM model."""

    id: str = Field(description="Model identifier exposed to API users")
    upstream_model: str | None = Field(
        default=None,
        description="Model name to use when forwarding to upstream. If None, uses id"
    )
    system_prompt: str | None = Field(default=None, description="System prompt for this model")
    system_prompt_behavior: SystemPromptBehavior = Field(
        default=SystemPromptBehavior.PASSTHROUGH, description="How to handle system prompts"
    )
    transform_params: Callable[[CompletionCreateParams], CompletionCreateParams] | None = Field(
        default=None, description="Optional function to transform request params"
    )

    model_config = ConfigDict(
        arbitrary_types_allowed = True
    )
