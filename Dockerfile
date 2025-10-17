FROM python:3.12-slim

WORKDIR /chat_completion_server

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY pyproject.toml README.md ./
COPY chat_completion_server/ ./chat_completion_server/

# Install dependencies
RUN pip install --no-cache-dir -e .

# Create non-root user
RUN useradd --create-home --shell /bin/bash chat_completion_server \
    && chown -R chat_completion_server:chat_completion_server /chat_completion_server
USER chat_completion_server

EXPOSE 8765

CMD ["uvicorn", "chat_completion_server.main:app", "--host", "0.0.0.0", "--port", "8765", "--workers", "4"]
