import json
import logging

from app.db.vector_store import get_connection
from app.models.schemas import Scorecard

logger = logging.getLogger(__name__)


def log_run(scorecard: Scorecard, brief_text: str, script_text: str) -> None:
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                insert into score_runs (run_id, overall_score, brief_text, script_text, scorecard)
                values (%s, %s, %s, %s, %s)
                """,
                (
                    scorecard.run_id,
                    scorecard.overall_score,
                    brief_text,
                    script_text,
                    json.dumps(scorecard.model_dump(mode="json")),
                ),
            )
    logger.info("persisted run %s", scorecard.run_id)
