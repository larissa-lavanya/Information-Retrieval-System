"""
ALGORITHM 9 - RELEVANCE RANKING  (TF-IDF and Okapi BM25)
--------------------------------------------------------
Boolean retrieval only says *whether* a document matches.  A scorer orders
the matches by relevance to the query's (positive) terms.

TF-IDF
    score(q, d) = sum over terms t of  tf(t, d) * idf(t)
    idf(t)      = log(1 + N / df(t))
    tf is high when a term is frequent in the document; idf is high when the
    term is rare in the collection.  (The "1 +" smoothing keeps idf positive
    even for a term that occurs in every document.)

Okapi BM25   (default)
    score(q, d) = sum over t of  idf(t) * tf * (k1 + 1)
                                 -------------------------------------------
                                 tf + k1 * (1 - b + b * |d| / avgdl)
    idf(t)      = log(1 + (N - df + 0.5) / (df + 0.5))
    k1 -> term-frequency saturation: the 10th occurrence adds far less than the 2nd
    b  -> document-length normalisation (0 = none, 1 = full)
"""

import math
from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable

from src.index.inverted_index import InvertedIndex


class Scorer(ABC):
    name = "base"

    @abstractmethod
    def score(self, index: InvertedIndex, doc_id: int, terms: Iterable[str]) -> float: ...


class TFIDFScorer(Scorer):
    name = "tfidf"

    def score(self, index: InvertedIndex, doc_id: int, terms: Iterable[str]) -> float:
        total, n = 0.0, index.num_docs
        for term in terms:
            tf = index.tf(term, doc_id)
            if tf == 0:
                continue
            idf = math.log(1 + n / index.df(term))      # <-- IDF
            total += tf * idf                           # <-- TF x IDF
        return total


class BM25Scorer(Scorer):
    name = "bm25"

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b

    def score(self, index: InvertedIndex, doc_id: int, terms: Iterable[str]) -> float:
        total, n = 0.0, index.num_docs
        avgdl = index.avg_doc_length or 1.0
        doc_len = index.doc_lengths.get(doc_id, 0)
        for term in terms:
            tf = index.tf(term, doc_id)
            if tf == 0:
                continue
            df = index.df(term)
            idf = math.log(1 + (n - df + 0.5) / (df + 0.5))                    # <-- BM25 IDF
            norm = tf + self.k1 * (1 - self.b + self.b * doc_len / avgdl)      # <-- length normalisation
            total += idf * tf * (self.k1 + 1) / norm                           # <-- saturation
        return total


def create_scorer(config: Dict[str, Any], name: str = None) -> Scorer:
    """Build the scorer named ``name`` (default: ``ranking.scorer`` from config)."""
    ranking = config["ranking"]
    name = (name or ranking["scorer"]).lower()
    if name == "bm25":
        return BM25Scorer(ranking["bm25"]["k1"], ranking["bm25"]["b"])
    if name == "tfidf":
        return TFIDFScorer()
    raise ValueError(f"Unknown scorer {name!r} (use 'bm25' or 'tfidf')")
