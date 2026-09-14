import logging
from typing import List

from app.clients.llm_client import generate_structured
from app.models.schemas import (
    BriefAlignmentScore,
    Claim,
    ClaimRetrieval,
    ClaimValidityScore,
    ClaimVerdict,
    CoverageEntry,
    MessageQualityScore,
    RetrievalStatus,
    StructuredBrief,
)

logger = logging.getLogger(__name__)

BRIEF_ALIGNMENT_PROMPT = """Score how well this advertising script delivers the campaign brief, 0-10.

Judge ONLY brief fit, not writing quality or factual accuracy:
- Is it written for the target audience?
- Does it land the key message?
- Is every mandatory inclusion present? List any that are missing verbatim.
- Does the tone match?
- Does it end with the requested call to action?

A polished script that ignores the brief should score low.
If the brief is empty or states no usable requirements, score 0 and say so in the reasoning -
an unmeasurable brief is not a satisfied brief.

BRIEF (structured):
target_audience: {target_audience}
key_message: {key_message}
mandatory_inclusions: {mandatory_inclusions}
tone: {tone}
cta: {cta}

SCRIPT:
\"\"\"
{script}
\"\"\"
"""

MESSAGE_QUALITY_PROMPT = """Score this advertising script as a piece of marketing craft, 0-10.

Judge ONLY the writing, independent of any brief or of whether claims are true:
- Hook: does the opening earn attention?
- Clarity: is the offer instantly understandable?
- Persuasiveness: does it give a reason to believe and act?
- Tone consistency: does the voice hold from start to finish?
- CTA effectiveness: is the ask clear and compelling?

Be specific in the reasoning: quote the strongest and weakest lines.

SCRIPT:
\"\"\"
{script}
\"\"\"
"""

CLAIM_VALIDITY_PROMPT = """You are fact-checking advertising claims against official product manual excerpts.

For each claim, give a verdict using ONLY the excerpts provided - not your own knowledge:
- supported: an excerpt clearly states this
- contradicted: an excerpt states something that conflicts with it (quote both)
- unsupported: excerpts are about the right product but do not mention this
- unverifiable: no relevant excerpts were retrieved (marked LOW CONFIDENCE below)

For supported/contradicted, set source_chunk to the excerpt id and evidence to the quoted text.
Then give an overall 0-10 score for the script's factual reliability: contradictions weigh
far more than unsupported claims, and unverifiable claims should not be penalised as false.

{claims_block}
"""


def score_brief_alignment(brief: StructuredBrief, script_text: str) -> BriefAlignmentScore:
    prompt = BRIEF_ALIGNMENT_PROMPT.format(
        target_audience=brief.target_audience,
        key_message=brief.key_message,
        mandatory_inclusions=brief.mandatory_inclusions or "none",
        tone=brief.tone,
        cta=brief.cta,
        script=script_text,
    )
    return generate_structured(prompt, BriefAlignmentScore)


def score_message_quality(script_text: str) -> MessageQualityScore:
    return generate_structured(MESSAGE_QUALITY_PROMPT.format(script=script_text), MessageQualityScore)


def score_claim_validity(
    claims: List[Claim],
    retrievals: List[ClaimRetrieval],
    coverage: List[CoverageEntry],
) -> ClaimValidityScore:
    if not claims:
        return ClaimValidityScore(score=10, reasoning="The script makes no checkable product claims; this axis is excluded from the overall score.")

    retrieval_by_id = {r.claim_id: r for r in retrievals}
    status_by_id = {c.claim_id: c.status for c in coverage}

    blocks = []
    for claim in claims:
        chunks = retrieval_by_id[claim.claim_id].chunks
        low = status_by_id[claim.claim_id] == RetrievalStatus.LOW_CONFIDENCE
        header = f"CLAIM {claim.claim_id}: {claim.text}" + ("   [LOW CONFIDENCE RETRIEVAL]" if low else "")
        excerpts = "\n".join(
            f"  [{c.chunk_id}] (p.{c.page_number}, similarity {c.similarity:.2f})\n  {c.text}"
            for c in chunks
        )
        blocks.append(f"{header}\n{excerpts}")

    result = generate_structured(CLAIM_VALIDITY_PROMPT.format(claims_block="\n\n".join(blocks)), ClaimValidityScore)
    logger.info(
        "claim verdicts: %s",
        {v.value: sum(1 for e in result.claim_verdicts if e.verdict == v) for v in ClaimVerdict},
    )
    unverifiable = sum(1 for e in result.claim_verdicts if e.verdict == ClaimVerdict.UNVERIFIABLE)
    if claims and unverifiable / len(claims) > 0.5:
        result.score = min(result.score, 5)
        result.reasoning += " Score capped at 5: most claims could not be checked against the manuals."

    return result
