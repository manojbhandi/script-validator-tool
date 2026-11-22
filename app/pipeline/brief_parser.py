import logging

from app.clients.llm_client import generate_structured
from app.models.schemas import StructuredBrief

logger = logging.getLogger(__name__)

PROMPT = """Convert this campaign brief into structured fields.

- target_audience: who the campaign is for
- key_message: the single most important thing the audience should take away
- mandatory_inclusions: specific things the brief says MUST appear in the script
  (product names, claims, disclaimers, price, offers, hashtags). Empty list if none.
- tone: the intended voice (e.g. "playful", "clinical and reassuring")
- cta: the call to action the script must end with

Quote the brief's own wording where possible; do not invent requirements it does not state.

BRIEF:
\"\"\"
{brief}
\"\"\"
"""


def parse_brief(brief_text: str) -> StructuredBrief:
    result = generate_structured(PROMPT.format(brief=brief_text), StructuredBrief)
    logger.info("parsed brief: %d mandatory inclusions", len(result.mandatory_inclusions))
    return result
