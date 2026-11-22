import logging

from app.config import get_settings
from app.models.schemas import Scorecard
from app.pipeline.brief_parser import parse_brief
from app.pipeline.claim_extractor import extract_claims
from app.pipeline.report import build_final_report
from app.pipeline.retrieval import compute_retrieval_confidence, retrieve_for_claims
from app.pipeline.scoring import score_brief_alignment, score_claim_validity, score_message_quality
from app.db.run_logs import log_run

logger = logging.getLogger(__name__)


def run_pipeline(brief_text: str, script_text: str) -> Scorecard:
    settings = get_settings()

    brief = parse_brief(brief_text)
    if not any([brief.target_audience, brief.key_message, brief.tone, brief.cta]):
        raise ValueError("The brief could not be parsed into any usable fields - provide a brief with an audience, key message, tone, or call to action.")

    claims = extract_claims(script_text)
    retrievals = retrieve_for_claims(claims, settings.retrieval_top_k)
    coverage = compute_retrieval_confidence(retrievals, settings.similarity_threshold)

    alignment = score_brief_alignment(brief, script_text)
    quality = score_message_quality(script_text)
    validity = score_claim_validity(claims, retrievals, coverage)

    scorecard = build_final_report(alignment, quality, validity, claims, coverage)

    try:
        log_run(scorecard, brief_text, script_text)
    except Exception:
        logger.exception("failed to persist run %s", scorecard.run_id)
    return scorecard

