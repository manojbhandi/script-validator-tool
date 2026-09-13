from fastapi import APIRouter

from app.models.schemas import Scorecard, ScoreRequest
from app.pipeline.orchestrator import run_pipeline

router = APIRouter(prefix="/score", tags=["scoring"])


@router.post("", response_model=Scorecard)
def score(request: ScoreRequest) -> Scorecard:
    return run_pipeline(request.brief, request.script)
