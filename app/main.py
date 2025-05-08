from fastapi import FastAPI
from app.api.v1.routes import router as api_router

app = FastAPI(title="LangChain RAG Chat Server")
app.include_router(api_router, prefix="/v1")
