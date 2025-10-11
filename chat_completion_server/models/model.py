from enum import Enum
from typing import Callable

from openai.types.chat import CompletionCreateParams
from openai.types import Model
from pydantic import BaseModel, Field, ConfigDict

from chat_completion_server.core.constants import MODEL_OBJECT_TYPE, MODEL_CREATED_TIMESTAMP, MODEL_OWNER


def create_model_metadata(model_id: str) -> Model:
    """Create standard model metadata."""
    return Model(
        id=model_id,
        object=MODEL_OBJECT_TYPE,
        created=MODEL_CREATED_TIMESTAMP,
        owned_by=MODEL_OWNER,
    )


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

    id: str
    """Model identifier exposed to API users"""

    upstream_model: str | None = None
    """Model name to use when forwarding to upstream. If None, uses id"""

    system_prompt: str | None = None
    """System prompt for this model"""

    system_prompt_behavior: SystemPromptBehavior = SystemPromptBehavior.PASSTHROUGH
    """How to handle system prompts"""

    transform_params: Callable[[CompletionCreateParams], CompletionCreateParams] | None = None
    """Optional function to transform request params"""

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        use_attribute_docstrings=True,
    )
