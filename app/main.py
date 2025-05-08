from fastapi import FastAPI
from app.api.v1.routes import router as api_router


app = FastAPI(title="LangChain RAG Chat Server")
app.include_router(api_router, prefix="/v1")


from contextlib import asynccontextmanager
from app.core.chain import build_chain

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup logic
    build_chain()  # ensures chain is built before serving requests
    
    # Let FastAPI do its thing
    yield
    
    # Shutdown logic, if needed
