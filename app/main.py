# Load envvars that are used in the rest of the app
from dotenv import load_dotenv

load_dotenv()


# Import FastAPI and do some setup
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

# Routes
from app.api.v1 import completions_router, models_router

app = FastAPI(title="LangChain RAG Chat Server")
app.include_router(completions_router, prefix="/v1")
app.include_router(models_router, prefix="/v1")
app.include_router(models_router)


# Allow CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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

    # Log the completed request with timing
    logger.info(
        f"Request completed: {request.method} {request.url.path} - elapsed={process_time:.3f}s"
    )

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
    

from dotenv import load_dotenv

load_dotenv()
