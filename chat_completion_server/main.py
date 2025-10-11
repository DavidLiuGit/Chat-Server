from dotenv import load_dotenv

from chat_completion_server import ChatCompletionServer
from chat_completion_server.models.model import ModelConfig
from chat_completion_server.core.logging import setup_logging

load_dotenv()
logger = setup_logging()

# Initialize server and get FastAPI app
custom_model: ModelConfig = ModelConfig(
    id="custom-model",
    upstream_model="bedrock/global.anthropic.claude-sonnet-4-20250514-v1:0",
)
server = ChatCompletionServer(models={"custom-model": custom_model})
app = server.app
