from enum import Enum
from pydantic import BaseModel, Field
from typing import Dict, List, Optional


class SectionType(str, Enum):
    SPEC = "spec"
    CLAIM = "claim"
    WARNING = "warning"
    USAGE = "usage"
    WARRANTY = "warranty"
    OTHER = "other"


class ClaimVerdict(str, Enum):
    SUPPORTED = "supported"
    UNSUPPORTED = "unsupported"
    CONTRADICTED = "contradicted"
    UNVERIFIABLE = "unverifiable"


class RetrievalStatus(str, Enum):
    WELL_GROUNDED = "well_grounded"
    LOW_CONFIDENCE = "low_confidence"


class RawPage(BaseModel):
    product_name:str
    page_number: int
    text: str


class Chunk(BaseModel):
    chunk_id: str
    product_name: str
    section_type: SectionType = SectionType.OTHER
    page_number: int
    text: str
    content_hash: str


class EmbeddedChunk(Chunk):
    vector: List[float]


class ScoredChunk(BaseModel):
    chunk_id: str
    product_name: str
    section_type: SectionType
    page_number: int
    text: str
    similarity: float = Field(ge=-1.0, le=1.0)


class Claim(BaseModel):
    claim_id: str
    text: str
    product_reference: Optional[str] = None


class ClaimRetrieval(BaseModel):
    claim_id: str
    chunks: List[ScoredChunk]


class CoverageEntry(BaseModel):
    claim_id: str
    best_similarity: float
    status: RetrievalStatus

class StructuredBrief(BaseModel):
    target_audience: str
    key_message: str
    mandatory_inclusions: List[str] = []
    tone: str
    cta: str


class BriefAlignmentScore(BaseModel):
    score: int = Field(ge=0, le=10)
    reasoning: str
    missing_mandatory_inclusions: List[str] = []


class MessageQualityScore(BaseModel):
    score: int = Field(ge=0, le=10)
    reasoning: str


class ClaimVerdictEntry(BaseModel):
    claim_id: str
    verdict: ClaimVerdict
    evidence: Optional[str] = None
    source_chunk: Optional[str] = None


class ClaimValidityScore(BaseModel):
    score: int = Field(ge=0, le=10)
    reasoning: str
    claim_verdicts: List[ClaimVerdictEntry] = []


class OfflineEvalBaseline(BaseModel):
    recall_at_5: float
    mrr: float
    num_queries: int


class Scorecard(BaseModel):
    run_id: str
    overall_score: float
    overall_feedback: str
    brief_alignment: BriefAlignmentScore
    marketing_message_quality: MessageQualityScore
    product_claim_validity: ClaimValidityScore
    claims: List[Claim]
    retrieval_coverage_report: List[CoverageEntry]
    offline_eval_baseline: Optional[OfflineEvalBaseline] = None


class ScoreRequest(BaseModel):
    brief: str = Field(min_length=20)
    script: str = Field(min_length=20)

class EvalQuery(BaseModel):
    question: str
    expected_chunk_id: str
    product_name: str


class EvalQueryResult(BaseModel):
    question: str
    expected_chunk_id: str
    rank: Optional[int] = None
    top_chunk_id: Optional[str] = None
    top_similarity: float
