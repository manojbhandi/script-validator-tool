import json
import logging
import random
import time
from pathlib import Path
from typing import List

from pydantic import BaseModel

from app.clients.embedding_client import embed
from app.clients.llm_client import generate_structured
from app.db.vector_store import fetch_all_chunks, similarity_search
from app.models.schemas import Chunk, EvalQuery, EvalQueryResult, OfflineEvalBaseline

logger = logging.getLogger(__name__)

EVAL_DIR = Path("data/eval")
DATASET_PATH = EVAL_DIR / "eval_dataset.json"
REPORT_PATH = EVAL_DIR / "offline_eval_report.json"

QUESTION_PROMPT = """Below is an excerpt from a skincare product training manual.

Write ONE question that an advertising copywriter might ask, whose answer is stated in this
excerpt and would be hard to answer from a different product's manual. Prefer questions about
specific facts: ingredients, percentages, certifications, clinical results, usage, volumes.
Do not mention the excerpt itself. Do not include the answer.

PRODUCT: {product_name}
EXCERPT:
\"\"\"
{text}
\"\"\"
"""


class _Question(BaseModel):
    question: str


def build_eval_dataset(sample_size: int, seed: int = 42, pause_seconds: float = 6.5) -> List[EvalQuery]:
    chunks = fetch_all_chunks()
    random.seed(seed)
    sampled = random.sample(chunks, min(sample_size, len(chunks)))

    queries: List[EvalQuery] = []
    for i, chunk in enumerate(sampled):
        result = generate_structured(
            QUESTION_PROMPT.format(product_name=chunk.product_name, text=chunk.text), _Question
        )
        queries.append(
            EvalQuery(question=result.question, expected_chunk_id=chunk.chunk_id, product_name=chunk.product_name)
        )
        logger.info("eval query %d/%d: %s", i + 1, len(sampled), result.question[:80])
        time.sleep(pause_seconds)

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    with open(DATASET_PATH, "w") as f:
        json.dump([q.model_dump() for q in queries], f, indent=2)
    return queries


def run_retrieval_eval(queries: List[EvalQuery], top_k: int = 5) -> OfflineEvalBaseline:
    results: List[EvalQueryResult] = []
    for query in queries:
        hits = similarity_search(embed(query.question, task_type="RETRIEVAL_QUERY"), top_k=top_k)
        ids = [h.chunk_id for h in hits]
        rank = ids.index(query.expected_chunk_id) + 1 if query.expected_chunk_id in ids else None
        results.append(
            EvalQueryResult(
                question=query.question,
                expected_chunk_id=query.expected_chunk_id,
                rank=rank,
                top_chunk_id=ids[0] if ids else None,
                top_similarity=hits[0].similarity if hits else 0.0,
            )
        )

    n = len(results)
    recall = sum(1 for r in results if r.rank is not None) / n
    mrr = sum(1 / r.rank for r in results if r.rank is not None) / n
    baseline = OfflineEvalBaseline(recall_at_5=round(recall, 3), mrr=round(mrr, 3), num_queries=n)

    with open(REPORT_PATH, "w") as f:
        json.dump({**baseline.model_dump(), "results": [r.model_dump() for r in results]}, f, indent=2)
    logger.info("offline eval: recall@%d=%.3f mrr=%.3f over %d queries", top_k, recall, mrr, n)
    return baseline
