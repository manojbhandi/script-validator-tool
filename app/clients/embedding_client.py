import logging
import time
from typing import List

from google import genai
from google.genai import errors, types

from app.config import get_settings

logger = logging.getLogger(__name__)

EMBED_BATCH_SIZE = 50
BATCH_PAUSE_SECONDS = 35
MAX_RETRIES = 3


def _client() -> genai.Client:
    settings = get_settings()
    return genai.Client(api_key=settings.gemini_api_key)


def _embed_with_retry(client: genai.Client, batch: List[str], task_type: str) -> List[List[float]]:
    settings = get_settings()
    for attempt in range(MAX_RETRIES):
        try:
            response = client.models.embed_content(
                model=settings.gemini_embedding_model,
                contents=batch,
                config=types.EmbedContentConfig(
                    task_type=task_type,
                    output_dimensionality=settings.embedding_dimensions,
                ),
            )
            return [e.values for e in response.embeddings]
        except errors.ClientError as exc:
            if exc.code != 429 or attempt == MAX_RETRIES - 1:
                raise
            logger.warning("rate limited, sleeping %ss (attempt %d/%d)", BATCH_PAUSE_SECONDS, attempt + 1, MAX_RETRIES)
            time.sleep(BATCH_PAUSE_SECONDS)
    return []


def embed_batch(texts: List[str], task_type: str = "RETRIEVAL_DOCUMENT") -> List[List[float]]:
    client = _client()
    vectors: List[List[float]] = []

    for start in range(0, len(texts), EMBED_BATCH_SIZE):
        batch = texts[start : start + EMBED_BATCH_SIZE]
        vectors.extend(_embed_with_retry(client, batch, task_type))
        logger.info("embedded %d/%d texts", len(vectors), len(texts))
        if start + EMBED_BATCH_SIZE < len(texts):
            time.sleep(BATCH_PAUSE_SECONDS)

    return vectors


def embed(text: str, task_type: str = "RETRIEVAL_QUERY") -> List[float]:
    return embed_batch([text], task_type=task_type)[0]
