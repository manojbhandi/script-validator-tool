import logging
from contextlib import contextmanager
from typing import Generator, List, Set

import psycopg2
from pgvector.psycopg2 import register_vector
from psycopg2.extras import execute_values

from app.config import get_settings
from app.models.schemas import EmbeddedChunk

logger = logging.getLogger(__name__)


@contextmanager
def get_connection() -> Generator[psycopg2.extensions.connection, None, None]:
    settings = get_settings()
    conn = psycopg2.connect(settings.supabase_db_url)
    register_vector(conn)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def existing_hashes(hashes: List[str]) -> Set[str]:
    if not hashes:
        return set()
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "select content_hash from manual_chunks where content_hash = any(%s)",
                (hashes,),
            )
            return {row[0] for row in cur.fetchall()}


def upsert_chunks(chunks: List[EmbeddedChunk]) -> int:
    if not chunks:
        return 0
    rows = [
        (
            c.chunk_id,
            c.product_name,
            c.section_type.value,
            c.page_number,
            c.text,
            c.content_hash,
            c.vector,
        )
        for c in chunks
    ]
    with get_connection() as conn:
        with conn.cursor() as cur:
            execute_values(
                cur,
                """
                insert into manual_chunks
                    (chunk_id, product_name, section_type, page_number, text, content_hash, embedding)
                values %s
                on conflict (chunk_id) do update set
                    product_name = excluded.product_name,
                    section_type = excluded.section_type,
                    page_number  = excluded.page_number,
                    text         = excluded.text,
                    content_hash = excluded.content_hash,
                    embedding    = excluded.embedding
                """,
                rows,
            )
    logger.info("upserted %d chunks", len(rows))
    return len(rows)
