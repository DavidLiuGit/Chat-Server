# Load envvars that are used in the rest of the app
from dotenv import load_dotenv

load_dotenv()


# Import FastAPI and do some setup
from fastapi import FastAPI, Request
from app.api.v1.routes import router as api_router

app = FastAPI(title="LangChain RAG Chat Server")
app.include_router(api_router, prefix="/v1")


# Set up middleware
from app.core.logging import setup_logging, generate_request_id, set_request_id
from time import time

logger = setup_logging()

@app.middleware("http")
async def add_request_id_middleware(request: Request, call_next):
    # Generate a request ID
    request_id = generate_request_id()
    
    # Store it in the context
    set_request_id(request_id)
    
    # Log the incoming request
    logger.info(f"Request started: {request.method} {request.url.path}")
    
    # Process the request and measure time
    start_time = time()
    response = await call_next(request)
    process_time = time() - start_time
    
    # Add the request ID to the response headers
    response.headers["X-Request-ID"] = request_id
    
    # Log the completed request with timing
    logger.info(f"Request completed: {request.method} {request.url.path} - Took {process_time:.3f}s")
    
    return response


# warm up `build_chain()`
from contextlib import asynccontextmanager
from app.core.chain import build_chain


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup logic
    build_chain()  # ensures chain is built before serving requests

    # Let FastAPI do its thing
    yield

    # Shutdown logic, if needed
