import json
import logging
import sys

from app.models.schemas import EvalQuery
from app.pipeline.eval import DATASET_PATH, build_eval_dataset, run_retrieval_eval

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


def main() -> None:
    rebuild = "--rebuild" in sys.argv
    if rebuild or not DATASET_PATH.exists():
        queries = build_eval_dataset(sample_size=40)
    else:
        with open(DATASET_PATH) as f:
            queries = [EvalQuery(**q) for q in json.load(f)]
    baseline = run_retrieval_eval(queries)
    print(baseline.model_dump())


if __name__ == "__main__":
    main()
