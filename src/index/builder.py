"""
Index construction pipeline:

    documents -> TextProcessor (tokenize / stopwords / stem) -> InvertedIndex
"""

from typing import Dict, Iterable

from src.index.inverted_index import InvertedIndex
from src.preprocessing.processor import TextProcessor


def build_index(documents: Iterable[Dict], processor: TextProcessor) -> InvertedIndex:
    """
    Build an index from dicts shaped ``{"id": int, "title": str, "text": str}``.

    Title and body are indexed together so title words are searchable too.
    """
    index = InvertedIndex()
    for doc in documents:
        # preprocessing pipeline: tokenization -> stopword removal -> Porter stemming
        stems, surface = processor.analyze(f"{doc['title']} {doc['text']}")
        # inverted index construction (positional postings) happens in add_document
        index.add_document(doc["id"], doc["title"], doc["text"], stems, surface)
    return index
