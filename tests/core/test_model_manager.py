import pytest
from chat_completion_server.core.model_manager import ModelManager
from chat_completion_server.models.model import ModelConfig, SystemPromptBehavior


@pytest.fixture
def manager():
    return ModelManager()


@pytest.fixture
def basic_model():
    return ModelConfig(id="test-model", upstream_model="gpt-4")


@pytest.fixture
def system_prompt_model():
    return ModelConfig(
        id="system-model",
        system_prompt="You are a helpful assistant.",
        system_prompt_behavior=SystemPromptBehavior.OVERRIDE,
    )


def test_register_model(manager, basic_model):
    manager.register_model(basic_model)
    assert "test-model" in manager.models
    assert manager.models["test-model"] == basic_model


def test_apply_model_config_unknown_model(manager):
    params = {"model": "unknown", "messages": []}
    result = manager.apply_model_config(params)
    assert result == params


def test_apply_model_config_upstream_mapping(manager, basic_model):
    manager.register_model(basic_model)
    params = {"model": "test-model", "messages": []}

    result = manager.apply_model_config(params)

    assert result["model"] == "gpt-4"


def test_apply_system_prompt_override(manager, system_prompt_model):
    manager.register_model(system_prompt_model)
    params = {
        "model": "system-model",
        "messages": [{"role": "system", "content": "Old prompt"}],
    }

    result = manager.apply_model_config(params)

    assert result["messages"][0]["content"] == "You are a helpful assistant."


def test_apply_system_prompt_default_no_existing(manager):
    model = ModelConfig(
        id="default-model",
        system_prompt="Default prompt",
        system_prompt_behavior=SystemPromptBehavior.DEFAULT,
    )
    manager.register_model(model)
    params = {"model": "default-model", "messages": [{"role": "user", "content": "Hi"}]}

    result = manager.apply_model_config(params)

    assert len(result["messages"]) == 2
    assert result["messages"][0]["role"] == "system"
    assert result["messages"][0]["content"] == "Default prompt"


def test_apply_system_prompt_prepend(manager):
    model = ModelConfig(
        id="prepend-model",
        system_prompt="New:",
        system_prompt_behavior=SystemPromptBehavior.PREPEND,
    )
    manager.register_model(model)
    params = {
        "model": "prepend-model",
        "messages": [{"role": "system", "content": "Old"}],
    }

    result = manager.apply_model_config(params)

    assert result["messages"][0]["content"] == "New:\n\nOld"


def test_apply_system_prompt_append(manager):
    model = ModelConfig(
        id="append-model",
        system_prompt="New",
        system_prompt_behavior=SystemPromptBehavior.APPEND,
    )
    manager.register_model(model)
    params = {
        "model": "append-model",
        "messages": [{"role": "system", "content": "Old"}],
    }

    result = manager.apply_model_config(params)

    assert result["messages"][0]["content"] == "Old\n\nNew"


def test_transform_params_custom_function(manager):
    def custom_transform(params):
        params["temperature"] = 0.5
        return params

    model = ModelConfig(id="custom-model", transform_params=custom_transform)
    manager.register_model(model)
    params = {"model": "custom-model", "messages": []}

    result = manager.apply_model_config(params)

    assert result["temperature"] == 0.5


def test_apply_model_config_no_model_param(manager):
    params = {"messages": []}
    result = manager.apply_model_config(params)
    assert result == params


def test_apply_model_config_empty_messages(manager, system_prompt_model):
    manager.register_model(system_prompt_model)
    params = {"model": "system-model", "messages": []}

    result = manager.apply_model_config(params)

    assert result == params


def test_apply_system_prompt_default_with_existing(manager):
    model = ModelConfig(
        id="default-model",
        system_prompt="Default prompt",
        system_prompt_behavior=SystemPromptBehavior.DEFAULT,
    )
    manager.register_model(model)
    params = {
        "model": "default-model",
        "messages": [{"role": "system", "content": "Existing"}],
    }

    result = manager.apply_model_config(params)

    assert result["messages"][0]["content"] == "Existing"


def test_apply_system_prompt_passthrough(manager):
    model = ModelConfig(
        id="passthrough-model",
        system_prompt="Ignored prompt",
        system_prompt_behavior=SystemPromptBehavior.PASSTHROUGH,
    )
    manager.register_model(model)
    params = {
        "model": "passthrough-model",
        "messages": [{"role": "system", "content": "Original"}],
    }

    result = manager.apply_model_config(params)

    assert result["messages"][0]["content"] == "Original"
