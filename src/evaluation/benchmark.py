"""
Benchmark / evaluation report.  Run from the project root:

    python -m src.evaluation.benchmark

Produces ``benchmark_results.md`` containing
    1. BM25 vs TF-IDF retrieval quality on the sample corpus
       (precision@5 and reciprocal rank, using each document's ``topic`` as
       ground truth) plus average query latency;
    2. skip-pointer intersection vs the naive merge (comparison counts on
       synthetic postings lists).
"""

import random
import statistics
import time

from src.app import get_engine
from src.config import load_config, project_path
from src.data.sample_corpus import SAMPLE_DOCS
from src.query.set_ops import intersect_naive, intersect_with_skips

# (query, relevant topic)
JUDGED_QUERIES = [
    ("python programming language", "programming"),
    ("software developers code", "programming"),
    ("planets stars galaxy", "space"),
    ("rocket orbit gravity", "space"),
    ("bake bread flour oven", "cooking"),
    ("vegetables oil spices", "cooking"),
    ("football goals team", "sports"),
    ("runners race training", "sports"),
    ("doctor patient medicine", "health"),
    ("immune system vaccine sleep", "health"),
]
TOPIC_OF = {d["id"]: d["topic"] for d in SAMPLE_DOCS}


def evaluate_scorer(engine, scorer: str):
    precisions, reciprocal_ranks, latencies = [], [], []
    for query, topic in JUDGED_QUERIES:
        # OR-style query so partial matches are still ranked
        boolean_query = " OR ".join(query.split())
        started = time.perf_counter()
        for _ in range(20):
            result = engine.search(boolean_query, top_k=5, scorer=scorer)
        latencies.append((time.perf_counter() - started) / 20 * 1000)

        relevant = [TOPIC_OF[h.doc_id] == topic for h in result.hits]
        precisions.append(sum(relevant) / 5)
        reciprocal_ranks.append(next((1 / r for r, ok in enumerate(relevant, 1) if ok), 0.0))
    return (statistics.mean(precisions), statistics.mean(reciprocal_ranks), statistics.mean(latencies))


def evaluate_skips():
    rng = random.Random(42)
    rows = []
    for long_n, short_n in [(1_000, 10), (10_000, 10), (10_000, 100), (100_000, 50)]:
        long_list = sorted(rng.sample(range(long_n * 4), long_n))
        short_list = sorted(rng.sample(long_list, short_n // 2) + rng.sample(range(long_n * 4), short_n // 2))
        short_list = sorted(set(short_list))
        naive, skip = {}, {}
        a = intersect_naive(long_list, short_list, naive)
        b = intersect_with_skips(long_list, short_list, skip)
        assert a == b
        rows.append((long_n, len(short_list), naive["comparisons"], skip["comparisons"]))
    return rows


def main() -> None:
    config = load_config()
    config = {**config, "storage": {"backend": "memory", "json_path": "unused"}}
    config["dataset"] = {"source": "sample", "path": None, "max_docs": None}
    engine = get_engine(config, rebuild=True)

    lines = ["# Benchmark results", "", "## 1. BM25 vs TF-IDF (sample corpus)", "",
             "| Scorer | Precision@5 | MRR | Avg latency (ms) |", "|---|---|---|---|"]
    for scorer in ("bm25", "tfidf"):
        p, mrr, ms = evaluate_scorer(engine, scorer)
        lines.append(f"| {scorer.upper()} | {p:.3f} | {mrr:.3f} | {ms:.3f} |")

    lines += ["", "## 2. Skip pointers vs naive intersection", "",
              "| Long list | Short list | Naive comparisons | Skip comparisons | Saved |", "|---|---|---|---|---|"]
    for long_n, short_n, naive, skip in evaluate_skips():
        lines.append(f"| {long_n:,} | {short_n} | {naive:,} | {skip:,} | {1 - skip / naive:.0%} |")

    report = "\n".join(lines) + "\n"
    with open(project_path("benchmark_results.md"), "w", encoding="utf-8") as handle:
        handle.write(report)
    print(report)


if __name__ == "__main__":
    main()
