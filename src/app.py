"""
Application wiring: build or restore the index and hand back a ready
``SearchEngine``.  Used by search.py, api.py, main.py and the benchmark.
"""

from typing import Optional, Tuple

from src.config import load_config
from src.data.loaders import load_documents
from src.index.builder import build_index
from src.index.inverted_index import InvertedIndex
from src.index.storage import StorageAdapter, create_storage
from src.preprocessing.processor import TextProcessor
from src.query.engine import SearchEngine


def build_and_save(config: dict) -> Tuple[InvertedIndex, TextProcessor, StorageAdapter]:
    """Ingest the configured dataset, build the index and persist it."""
    processor = TextProcessor.from_config(config)
    data = config["dataset"]
    documents = load_documents(data["source"], data.get("path"), data.get("max_docs"))
    index = build_index(documents, processor)
    storage = create_storage(config)
    storage.save(index.to_dict())
    return index, processor, storage


def get_engine(config: Optional[dict] = None, rebuild: bool = False) -> SearchEngine:
    """Restore the index from storage if present, otherwise build it."""
    config = config or load_config()
    processor = TextProcessor.from_config(config)
    storage = create_storage(config)

    state = None if rebuild else storage.load()
    if state is None:
        index, processor, _ = build_and_save(config)
    else:
        index = InvertedIndex.from_dict(state)
    return SearchEngine(index, processor, config)
