import json
import logging
from typing import Type, TypeVar

from openai import OpenAI, RateLimitError
import time
from pydantic import BaseModel, ValidationError

from app.config import get_settings

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

MAX_ATTEMPTS = 2
RETRY_SLEEP_SECONDS = 20
MAX_429_RETRIES = 3

def _client() -> OpenAI:
    settings = get_settings()
    return OpenAI(api_key=settings.llm_api_key, base_url=settings.llm_base_url)


def generate(prompt: str, system: str = "You are a precise assistant.") -> str:
    settings = get_settings()
    for attempt in range(MAX_429_RETRIES):
        try:
            response = _client().chat.completions.create(
                model=settings.llm_model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
            )
            return response.choices[0].message.content or ""
        except RateLimitError:
            if attempt == MAX_429_RETRIES - 1:
                raise
            logger.warning("LLM rate limited, sleeping %ss (attempt %d)", RETRY_SLEEP_SECONDS, attempt + 1)
            time.sleep(RETRY_SLEEP_SECONDS)
    return ""


def generate_structured(prompt: str, schema: Type[T]) -> T:
    system = (
        "You are a precise assistant. Respond with a single JSON object only - "
        "no markdown, no code fences, no commentary. The object must match this JSON schema:\n"
        + json.dumps(schema.model_json_schema())
    )
    last_error: Exception = RuntimeError("no attempts made")
    for attempt in range(MAX_ATTEMPTS):
        raw = generate(prompt, system=system)
        try:
            return schema.model_validate_json(_strip_fences(raw))
        except (ValidationError, ValueError) as exc:
            last_error = exc
            logger.warning("structured output failed validation (attempt %d): %s", attempt + 1, str(exc)[:200])
            prompt = f"{prompt}\n\nYour previous reply was invalid: {str(exc)[:300]}\nReturn ONLY valid JSON matching the schema."
    raise last_error


def _strip_fences(raw: str) -> str:
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        if text.endswith("```"):
            text = text[:-3]
    return text.strip()
