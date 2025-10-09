from openai.types.chat import CompletionCreateParams

from app.models.model import ModelConfig, SystemPromptBehavior


class ModelManager:
    """Manages model configurations and applies model-specific transformations."""

    def __init__(self, models: dict[str, ModelConfig] | None = None):
        self.models = models or {}

    def register_model(self, model: ModelConfig) -> None:
        """Register a custom model configuration."""
        self.models[model.id] = model

    def apply_model_config(
        self, params: CompletionCreateParams
    ) -> CompletionCreateParams:
        """Apply model-specific configuration to request params."""
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
        messages = params.get("messages", [])
        if not messages:
            return params

        # Find existing system message
        system_idx = next(
            (i for i, m in enumerate(messages) if m.get("role") == "system"), None
        )

        behavior = model.system_prompt_behavior

        if behavior == SystemPromptBehavior.PASSTHROUGH:
            return params

        elif behavior == SystemPromptBehavior.OVERRIDE:
            if system_idx is not None:
                messages[system_idx]["content"] = model.system_prompt
            else:
                messages.insert(0, {"role": "system", "content": model.system_prompt})

        elif behavior == SystemPromptBehavior.PREPEND:
            if system_idx is not None:
                existing = messages[system_idx]["content"]
                messages[system_idx]["content"] = f"{model.system_prompt}\n\n{existing}"
            else:
                messages.insert(0, {"role": "system", "content": model.system_prompt})

        elif behavior == SystemPromptBehavior.APPEND:
            if system_idx is not None:
                existing = messages[system_idx]["content"]
                messages[system_idx]["content"] = f"{existing}\n\n{model.system_prompt}"
            else:
                messages.insert(0, {"role": "system", "content": model.system_prompt})

        elif behavior == SystemPromptBehavior.DEFAULT:
            if system_idx is None:
                messages.insert(0, {"role": "system", "content": model.system_prompt})

        params["messages"] = messages
        return params
