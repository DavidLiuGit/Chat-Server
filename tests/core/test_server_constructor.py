import pytest
from unittest.mock import Mock
from chat_completion_server.core.server import ChatCompletionServer
from chat_completion_server.models.config import ProxyConfig
from chat_completion_server.models.model import ModelConfig
from chat_completion_server.core.handler import OpenAIProxyHandler
from chat_completion_server.plugins.guardrails import GuardrailsPlugin
from chat_completion_server.plugins.logging import LoggingPlugin


def test_constructor_defaults():
    """Test constructor with all default values."""
    server = ChatCompletionServer()

    assert isinstance(server.config, ProxyConfig)
    assert isinstance(server.handler, OpenAIProxyHandler)
    assert len(server.plugins) == 2
    assert isinstance(server.plugins[0], GuardrailsPlugin)
    assert isinstance(server.plugins[1], LoggingPlugin)
    assert "custom-model" in server.model_manager.models


def test_constructor_with_config():
    """Test constructor with custom config."""
    config = ProxyConfig(upstream_url="https://test.api")
    server = ChatCompletionServer(config=config)

    assert server.config == config
    assert server.handler.config == config


def test_constructor_with_handler():
    """Test constructor with custom handler."""
    handler = Mock()
    server = ChatCompletionServer(handler=handler)

    assert server.handler == handler


def test_constructor_with_plugins():
    """Test constructor with custom plugins."""
    plugins = [Mock(), Mock()]
    server = ChatCompletionServer(plugins=plugins)

    assert server.plugins == plugins


def test_constructor_with_models():
    """Test constructor with custom models."""
    models = {"test-model": ModelConfig(id="test-model")}
    server = ChatCompletionServer(models=models)

    assert server.model_manager.models == models
    assert "custom-model" not in server.model_manager.models


def test_constructor_empty_plugins_list():
    """Test constructor with empty plugins list."""
    server = ChatCompletionServer(plugins=[])

    assert server.plugins == []


def test_constructor_empty_models_dict():
    """Test constructor with empty models dict."""
    server = ChatCompletionServer(models={})

    assert server.model_manager.models == {}


def test_app_property():
    """Test that app property returns FastAPI instance."""
    server = ChatCompletionServer()

    assert server.app is not None
    assert hasattr(server.app, "routes")
