import pytest
from unittest.mock import Mock, patch, AsyncMock
from fastapi.testclient import TestClient
from chat_completion_server.core.server import ChatCompletionServer
from chat_completion_server.core.logging import get_request_id


@pytest.fixture
def server():
    """Create a test server instance."""
    return ChatCompletionServer()


@pytest.fixture
def client(server):
    """Create a test client."""
    return TestClient(server.app)


def test_request_id_middleware_sets_request_id(client):
    """Test that middleware sets request ID in context."""
    with patch("chat_completion_server.core.server.generate_request_id") as mock_gen:
        mock_gen.return_value = "test-request-id"

        response = client.get("/models")

        assert response.status_code == 200
        mock_gen.assert_called_once()


def test_request_id_middleware_logs_request_start_and_completion(client, caplog):
    """Test that middleware logs request start and completion."""
    with caplog.at_level("INFO", logger="chat_completion_server.core.server"):
        with patch("chat_completion_server.core.server.generate_request_id") as mock_gen:
            mock_gen.return_value = "test-request-id"

            response = client.get("/models")

            assert response.status_code == 200

            # Check logs contain start and completion messages
            log_messages = [record.message for record in caplog.records]
            start_logged = any("Request started: GET /models" in msg for msg in log_messages)
            completion_logged = any(
                "Request completed: GET /models" in msg and "elapsed=" in msg for msg in log_messages
            )

            assert start_logged
            assert completion_logged


def test_request_id_middleware_measures_elapsed_time(client, caplog):
    """Test that middleware measures and logs elapsed time."""
    with caplog.at_level("INFO", logger="chat_completion_server.core.server"):
        with patch("chat_completion_server.core.server.generate_request_id") as mock_gen:
            mock_gen.return_value = "test-request-id"

            response = client.get("/models")

            assert response.status_code == 200

            # Find completion log message
            completion_logs = [
                record.message
                for record in caplog.records
                if "Request completed" in record.message and "elapsed=" in record.message
            ]

            assert len(completion_logs) > 0
            # Check that elapsed time is formatted correctly (e.g., "elapsed=0.001s")
            assert any("elapsed=" in log and "s" in log for log in completion_logs)


@pytest.mark.asyncio
async def test_request_id_middleware_with_async_endpoint(server):
    """Test middleware works with async endpoints."""
    app = server.app

    # Mock the handler to avoid actual API calls
    with patch.object(server, "process_request", new_callable=AsyncMock) as mock_process:
        mock_process.return_value = {"id": "test", "object": "chat.completion", "choices": []}

        with patch("chat_completion_server.core.server.generate_request_id") as mock_gen:
            mock_gen.return_value = "async-test-id"

            client = TestClient(app)
            response = client.post(
                "/v1/chat/completions",
                json={"model": "custom-model", "messages": [{"role": "user", "content": "test"}]},
            )

            assert response.status_code == 200
            mock_gen.assert_called_once()


def test_request_id_middleware_different_endpoints(client):
    """Test middleware works across different endpoints."""
    endpoints = ["/models", "/v1/models"]

    for endpoint in endpoints:
        with patch("chat_completion_server.core.server.generate_request_id") as mock_gen:
            mock_gen.return_value = f"test-id-{endpoint.replace('/', '-')}"

            response = client.get(endpoint)

            assert response.status_code == 200
            mock_gen.assert_called_once()


def test_request_id_middleware_with_different_http_methods(client, server):
    """Test middleware works with different HTTP methods."""
    # Test GET
    with patch("chat_completion_server.core.server.generate_request_id") as mock_gen:
        mock_gen.return_value = "get-test-id"
        response = client.get("/models")
        assert response.status_code == 200
        mock_gen.assert_called_once()

    # Test POST (mock the handler to avoid actual API calls)
    with patch.object(server, "process_request", new_callable=AsyncMock) as mock_process:
        mock_process.return_value = {"id": "test", "object": "chat.completion", "choices": []}

        with patch("chat_completion_server.core.server.generate_request_id") as mock_gen:
            mock_gen.return_value = "post-test-id"
            response = client.post(
                "/v1/chat/completions",
                json={"model": "custom-model", "messages": [{"role": "user", "content": "test"}]},
            )
            assert response.status_code == 200
            mock_gen.assert_called_once()
