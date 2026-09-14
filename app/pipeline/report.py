import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional
from uuid import uuid4

from app.models.schemas import (
    BriefAlignmentScore,
    Claim,
    ClaimValidityScore,
    CoverageEntry,
    MessageQualityScore,
    OfflineEvalBaseline,
    Scorecard,
)

logger = logging.getLogger(__name__)

WEIGHTS = {"brief_alignment": 0.3, "message_quality": 0.3, "claim_validity": 0.4}
EVAL_REPORT_PATH = Path("data/eval/offline_eval_report.json")


def load_offline_baseline(path: Path = EVAL_REPORT_PATH) -> Optional[OfflineEvalBaseline]:
    if not path.exists():
        logger.warning("no offline eval report at %s", path)
        return None
    with open(path) as f:
        return OfflineEvalBaseline(**json.load(f))


def build_overall_feedback(
    alignment: BriefAlignmentScore,
    quality: MessageQualityScore,
    validity: ClaimValidityScore,
) -> str:
    issues = []
    if alignment.missing_mandatory_inclusions:
        issues.append("missing mandatory inclusions: " + ", ".join(alignment.missing_mandatory_inclusions))
    contradicted = [e.claim_id for e in validity.claim_verdicts if e.verdict.value == "contradicted"]
    unsupported = [e.claim_id for e in validity.claim_verdicts if e.verdict.value == "unsupported"]
    if contradicted:
        issues.append("claims contradicted by the manual: " + ", ".join(contradicted))
    if unsupported:
        issues.append("claims not backed by the manual: " + ", ".join(unsupported))
    if quality.score < 5:
        issues.append("copy quality is below par")

    if not issues:
        return "Script is on-brief, well-crafted and factually grounded - cleared for production."
    return "Revise before production - " + "; ".join(issues) + "."


def build_final_report(
    alignment: BriefAlignmentScore,
    quality: MessageQualityScore,
    validity: ClaimValidityScore,
    claims: List[Claim],
    coverage: List[CoverageEntry],
) -> Scorecard:
    axis_scores = {
        "brief_alignment": alignment.score,
        "message_quality": quality.score,
    }
    if claims:
        axis_scores["claim_validity"] = validity.score
    total_weight = sum(WEIGHTS[k] for k in axis_scores)
    overall = sum(WEIGHTS[k] * s for k, s in axis_scores.items()) / total_weight
    run_id = datetime.now(timezone.utc).strftime("run_%Y%m%d_%H%M%S_") + uuid4().hex[:6]
    return Scorecard(
        run_id=run_id,
        overall_score=round(overall, 1),
        overall_feedback=build_overall_feedback(alignment, quality, validity),
        brief_alignment=alignment,
        marketing_message_quality=quality,
        product_claim_validity=validity,
        claims=claims,
        retrieval_coverage_report=coverage,
        offline_eval_baseline=load_offline_baseline(),
    )
