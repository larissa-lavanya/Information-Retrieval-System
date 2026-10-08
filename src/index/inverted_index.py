"""
ALGORITHM 4 - POSITIONAL INVERTED INDEX
---------------------------------------
Maps every term (stem) to the documents containing it *and* the positions of
each occurrence inside the document:

    softwar -> { 3: [4, 19],  7: [2],  42: [8, 31, 90] }
                 doc id : positions

Used for:
    * fast lookup - a query touches only the postings of its own terms rather
      than scanning every document (O(postings) instead of O(corpus));
    * phrase queries - the stored positions let us check that two terms are
      adjacent;
    * ranking - term frequency is ``len(positions)`` and document length is
      stored for BM25 length-normalisation.

The index is storage-agnostic: ``to_dict`` / ``from_dict`` convert it to plain
JSON-friendly data, which a ``StorageAdapter`` (see storage.py) persists.
"""

from collections import Counter
from typing import Any, Dict, List, Optional

from src.index.skip_pointers import build_skip_pointers, skip_step


class InvertedIndex:
    def __init__(self) -> None:
        # term -> {doc_id -> [positions]}   (the core inverted index)
        self.postings: Dict[str, Dict[int, List[int]]] = {}
        # doc_id -> number of indexed tokens (needed by BM25 / TF-IDF)
        self.doc_lengths: Dict[int, int] = {}
        # doc_id -> {"title": ..., "text": ...}
        self.documents: Dict[int, Dict[str, str]] = {}
        # original (unstemmed) words with frequencies - spell-checker vocabulary
        self.surface_words: Counter = Counter()
        self._sorted_cache: Dict[str, List[int]] = {}

    # ------------------------------------------------------------------
    # Building
    # ------------------------------------------------------------------
    def add_document(
        self, doc_id: int, title: str, text: str, stems: List[str], surface: List[str]
    ) -> None:
        """Index one document given its already-processed stems."""
        self.documents[doc_id] = {"title": title, "text": text}
        self.doc_lengths[doc_id] = len(stems)
        for position, term in enumerate(stems):
            # --- inverted index construction: append (doc, position) to the term's list
            self.postings.setdefault(term, {}).setdefault(doc_id, []).append(position)
            self._sorted_cache.pop(term, None)
        self.surface_words.update(surface)

    # ------------------------------------------------------------------
    # Statistics (used by the rankers)
    # ------------------------------------------------------------------
    @property
    def num_docs(self) -> int:
        return len(self.documents)

    @property
    def avg_doc_length(self) -> float:
        return sum(self.doc_lengths.values()) / self.num_docs if self.num_docs else 0.0

    def df(self, term: str) -> int:
        """Document frequency: how many documents contain ``term``."""
        return len(self.postings.get(term, ()))

    def tf(self, term: str, doc_id: int) -> int:
        """Term frequency of ``term`` inside ``doc_id``."""
        return len(self.postings.get(term, {}).get(doc_id, ()))

    # ------------------------------------------------------------------
    # Lookups (used by the query engine)
    # ------------------------------------------------------------------
    def has_term(self, term: str) -> bool:
        return term in self.postings

    def sorted_docs(self, term: str) -> List[int]:
        """Sorted list of doc ids containing ``term`` (the postings list)."""
        cached = self._sorted_cache.get(term)
        if cached is None:
            cached = self._sorted_cache[term] = sorted(self.postings.get(term, ()))
        return cached

    def positions(self, term: str) -> Dict[int, List[int]]:
        """doc id -> positions for ``term`` (do not mutate the result)."""
        return self.postings.get(term, {})

    def all_doc_ids(self) -> List[int]:
        return sorted(self.documents)

    def vocabulary(self) -> List[str]:
        return sorted(self.postings)

    def skip_info(self, term: str) -> Dict[str, Any]:
        """Postings list plus its skip pointers (consumed by the visualiser)."""
        docs = self.sorted_docs(term)
        return {
            "term": term,
            "df": len(docs),
            "doc_ids": docs,
            "skip_step": skip_step(len(docs)),
            "skip_pointers": build_skip_pointers(len(docs)),
        }

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return {
            "postings": {
                term: {str(doc): list(pos) for doc, pos in docs.items()}
                for term, docs in self.postings.items()
            },
            "doc_lengths": {str(d): n for d, n in self.doc_lengths.items()},
            "documents": {str(d): dict(v) for d, v in self.documents.items()},
            "surface_words": dict(self.surface_words),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "InvertedIndex":
        index = cls()
        index.postings = {
            term: {int(doc): list(pos) for doc, pos in docs.items()}
            for term, docs in data["postings"].items()
        }
        index.doc_lengths = {int(d): n for d, n in data["doc_lengths"].items()}
        index.documents = {int(d): dict(v) for d, v in data["documents"].items()}
        index.surface_words = Counter(data.get("surface_words", {}))
        return index
