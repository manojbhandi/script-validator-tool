from fastapi import FastAPI
from app.routers import ingest, score
from app.utils.logging import configure_logging
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles



from app.config import get_settings

configure_logging()
app = FastAPI(title="Creative Script Validation Service")
app.include_router(ingest.router)
app.include_router(score.router)
app.mount("/static", StaticFiles(directory="app/static"), name="static")


@app.get("/", include_in_schema=False)
def index():
    return FileResponse("app/static/index.html")


@app.get("/health")
def health():
    settings = get_settings()
    return {
        "status": "ok",
        "llm_provider": settings.llm_provider,
        "embedding_provider": settings.embedding_provider,
    }
