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
