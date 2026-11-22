import logging
from typing import List

from pydantic import BaseModel

from app.clients.llm_client import generate_structured
from app.models.schemas import Claim

logger = logging.getLogger(__name__)


class _ClaimList(BaseModel):
    claims: List[Claim]


PROMPT = """You are reviewing an advertising script for a skincare brand before production.

Extract every discrete, factually checkable claim the script makes about a product.
A checkable claim is a statement that could be verified or refuted against the product's
manual: ingredients, percentages, certifications, clinical results, volumes, SPF ratings,
usage instructions, what the product is free from, awards, or guarantees.

Do NOT extract marketing flourish or subjective language ("glow like never before",
"your new favourite"). Do NOT paraphrase - keep the claim close to the script's wording.
Do NOT extract where the product is sold, its price, or availability - only facts about the
product itself that a manual could confirm.
If the script names a product line or product, set product_reference to it; otherwise null.

Number claims c1, c2, c3 ... in order of appearance.

SCRIPT:
\"\"\"
{script}
\"\"\"
"""


def extract_claims(script_text: str) -> List[Claim]:
    result = generate_structured(PROMPT.format(script=script_text), _ClaimList)
    logger.info("extracted %d claims", len(result.claims))
    return result.claims
