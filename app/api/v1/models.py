from fastapi import APIRouter
from openai.types import Model
from openai.pagination import SyncPage

models_router = APIRouter()


# Note: This router is legacy. Routes are now registered in ChatCompletionServer.
# Kept for backward compatibility if needed.
