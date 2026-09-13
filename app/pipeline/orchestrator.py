import logging

from app.config import get_settings
from app.models.schemas import Scorecard
from app.pipeline.brief_parser import parse_brief
from app.pipeline.claim_extractor import extract_claims
from app.pipeline.report import build_final_report
from app.pipeline.retrieval import compute_retrieval_confidence, retrieve_for_claims
from app.pipeline.scoring import score_brief_alignment, score_claim_validity, score_message_quality

logger = logging.getLogger(__name__)


def run_pipeline(brief_text: str, script_text: str) -> Scorecard:
    settings = get_settings()

    brief = parse_brief(brief_text)
    claims = extract_claims(script_text)
    retrievals = retrieve_for_claims(claims, settings.retrieval_top_k)
    coverage = compute_retrieval_confidence(retrievals, settings.similarity_threshold)

    alignment = score_brief_alignment(brief, script_text)
    quality = score_message_quality(script_text)
    validity = score_claim_validity(claims, retrievals, coverage)

    scorecard = build_final_report(alignment, quality, validity, claims, coverage)
    logger.info("run %s complete: overall %.1f", scorecard.run_id, scorecard.overall_score)
    return scorecard
