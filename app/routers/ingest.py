from fastapi import APIRouter

from app.pipeline.ingestion import run_ingestion

router = APIRouter(prefix="/ingest", tags=["ingestion"])

MANUALS_DIR = "data/manuals"


@router.post("")
def ingest():
    return run_ingestion(MANUALS_DIR)
