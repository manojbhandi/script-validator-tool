from fastapi import FastAPI

from app.config import get_settings
from app.routers import ingest

app = FastAPI(title="Creative Script Validation Service")
app.include_router(ingest.router)


@app.get("/health")
def health():
    settings = get_settings()
    return {
        "status": "ok",
        "llm_provider": settings.llm_provider,
        "embedding_provider": settings.embedding_provider,
    }
