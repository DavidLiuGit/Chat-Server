from dotenv import load_dotenv
from fastapi import Request
from time import time

load_dotenv()

from app import ChatCompletionServer
from app.models.model import ModelConfig
from app.core.logging import setup_logging, generate_request_id, set_request_id

logger = setup_logging()

# Initialize server and get FastAPI app
custom_model: ModelConfig = ModelConfig(
    id="custom-model",
    upstream_model="bedrock/global.anthropic.claude-sonnet-4-20250514-v1:0",
)
server = ChatCompletionServer(models={"custom-model": custom_model})
app = server.app


@app.middleware("http")
async def add_request_id_middleware(request: Request, call_next):
    request_id = generate_request_id()
    set_request_id(request_id)
    
    logger.info(f"Request started: {request.method} {request.url.path}")
    
    start_time = time()
    response = await call_next(request)
    process_time = time() - start_time
    
    logger.info(
        f"Request completed: {request.method} {request.url.path} - elapsed={process_time:.3f}s"
    )
    
    return response
