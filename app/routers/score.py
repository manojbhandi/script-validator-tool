from fastapi import APIRouter, HTTPException

from app.models.schemas import Scorecard, ScoreRequest
from app.pipeline.orchestrator import run_pipeline

router = APIRouter(prefix="/score", tags=["scoring"])


@router.post("", response_model=Scorecard)
def score(request: ScoreRequest) -> Scorecard:
    try:
        return run_pipeline(request.brief, request.script)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
