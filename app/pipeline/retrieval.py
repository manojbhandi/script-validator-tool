import logging
from typing import List

from app.clients.embedding_client import embed
from app.db.vector_store import similarity_search
from app.models.schemas import Claim, ClaimRetrieval, CoverageEntry, RetrievalStatus

logger = logging.getLogger(__name__)


def retrieve_for_claims(claims: List[Claim], top_k: int) -> List[ClaimRetrieval]:
    results: List[ClaimRetrieval] = []
    for claim in claims:
        query_vector = embed(claim.text, task_type="RETRIEVAL_QUERY")
        chunks = similarity_search(query_vector, top_k=top_k, product_name=claim.product_reference)
        logger.info(
            "claim %s: top match %s (%.3f)",
            claim.claim_id,
            chunks[0].chunk_id if chunks else None,
            chunks[0].similarity if chunks else 0.0,
        )
        results.append(ClaimRetrieval(claim_id=claim.claim_id, chunks=chunks))
    return results


def compute_retrieval_confidence(
    retrievals: List[ClaimRetrieval], threshold: float
) -> List[CoverageEntry]:
    report: List[CoverageEntry] = []
    for item in retrievals:
        best = max((c.similarity for c in item.chunks), default=0.0)
        status = RetrievalStatus.WELL_GROUNDED if best >= threshold else RetrievalStatus.LOW_CONFIDENCE
        report.append(CoverageEntry(claim_id=item.claim_id, best_similarity=round(best, 4), status=status))
    return report
