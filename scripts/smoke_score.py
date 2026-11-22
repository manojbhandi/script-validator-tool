import logging

from app.config import get_settings
from app.pipeline.brief_parser import parse_brief
from app.pipeline.claim_extractor import extract_claims
from app.pipeline.retrieval import compute_retrieval_confidence, retrieve_for_claims
from app.pipeline.scoring import score_brief_alignment, score_claim_validity, score_message_quality

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

BRIEF = """Product: Alltimate PDRN Hyalu 7 Serum.
Audience: women 25-35 with dry, dull skin who already use serums.
Key message: deep hydration that shows as glow.
Must mention: 97% PDRN, 7 hyaluronic acids, vegan certification.
Tone: confident, science-backed, warm.
CTA: Shop now at THE FACE SHOP."""

SCRIPT = """Meet the Alltimate PDRN Hyalu 7 Serum. Formulated with 97% pure PDRN and seven types of
hyaluronic acid, it delivers moisture 1/210th the size of a pore - deep where it counts.
Clinically proven to boost glow in just two weeks. Vegan certified, and free from parabens.
Your skin will thank you. Available now at THE FACE SHOP - 30ml, for glow that lasts all day."""


def main() -> None:
    settings = get_settings()

    brief = parse_brief(BRIEF)
    print("\nBRIEF:", brief.model_dump())

    claims = extract_claims(SCRIPT)
    retrievals = retrieve_for_claims(claims, settings.retrieval_top_k)
    coverage = compute_retrieval_confidence(retrievals, settings.similarity_threshold)

    print("\nALIGN:", score_brief_alignment(brief, SCRIPT).model_dump())
    print("\nQUALITY:", score_message_quality(SCRIPT).model_dump())

    validity = score_claim_validity(claims, retrievals, coverage)
    print("\nVALIDITY", validity.score, "-", validity.reasoning)
    for entry in validity.claim_verdicts:
        print(" ", entry.claim_id, entry.verdict.value, "|", entry.source_chunk, "|", (entry.evidence or "")[:90])


if __name__ == "__main__":
    main()
