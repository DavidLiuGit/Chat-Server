from logging import getLogger

from openai.types.chat import CompletionCreateParams

from chat_completion_server.core.constants import ROLE_SYSTEM
from chat_completion_server.models.model import ModelConfig, SystemPromptBehavior


logger = getLogger(__name__)


class ModelManager:
    """Manages model configurations and applies model-specific transformations."""

    def __init__(self, models: dict[str, ModelConfig] | None = None):
        self.models = models or {}

    def register_model(self, model: ModelConfig) -> None:
        """Register a custom model configuration."""
        self.models[model.id] = model

    def apply_model_config(self, params: CompletionCreateParams) -> CompletionCreateParams:
        """
        Apply model-specific configuration to request params.
        Replace the `model` field with the predefined upstream model.
        """
        model_id = params.get("model")
        if not model_id or model_id not in self.models:
            return params

        model = self.models[model_id]

        # Map to upstream model name if specified
        if model.upstream_model:
            params["model"] = model.upstream_model

        # Apply system prompt behavior
        if model.system_prompt:
            params = self._apply_system_prompt(params, model)

        # Apply custom transform
        if model.transform_params:
            params = model.transform_params(params)

        return params

    def _apply_system_prompt(
        self, params: CompletionCreateParams, model: ModelConfig
    ) -> CompletionCreateParams:
        """Apply system prompt based on model's behavior."""
        if (
            model.system_prompt_behavior == SystemPromptBehavior.PASSTHROUGH
            or not model.system_prompt
        ):
            return params

        messages = list(params.get("messages", []))
        if not messages:
            return params

        # attempt to find existing system message
        system_msg_idx = next(
            (i for i, m in enumerate(messages) if m.get("role") == ROLE_SYSTEM), None
        )
        try:
            if system_msg_idx is None:
                messages.insert(0, {"role": ROLE_SYSTEM, "content": model.system_prompt})
            elif model.system_prompt_behavior == SystemPromptBehavior.OVERRIDE:
                messages[system_msg_idx]["content"] = model.system_prompt
            elif model.system_prompt_behavior == SystemPromptBehavior.PREPEND:
                existing = messages[system_msg_idx][
                    "content"
                ]  # pyright: ignore[reportTypedDictNotRequiredAccess]
                messages[system_msg_idx]["content"] = f"{model.system_prompt}\n\n{existing}"
            elif model.system_prompt_behavior == SystemPromptBehavior.APPEND:
                existing = messages[system_msg_idx][
                    "content"
                ]  # pyright: ignore[reportTypedDictNotRequiredAccess]
                messages[system_msg_idx]["content"] = f"{existing}\n\n{model.system_prompt}"
        except Exception:
            logger.exception("[ModelManager] Error while applying ModelConfig.system_prompt")
            return params

        params["messages"] = messages
        return params
