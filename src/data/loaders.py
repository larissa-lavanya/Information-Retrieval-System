"""
Document loaders.  Every loader yields dicts shaped
``{"id": int, "title": str, "text": str}``.

    sample  - the built-in 30-document corpus
    jsonl   - one JSON object per line with "title" and "text" (and optional "id")
    dir     - every ``*.txt`` file in a folder (file name becomes the title)
"""

import json
import os
from typing import Dict, Iterator, Optional

from src.data.sample_corpus import SAMPLE_DOCS


def _load_jsonl(path: str) -> Iterator[Dict]:
    with open(path, "r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            yield {
                "id": int(record.get("id", line_no)),
                "title": str(record.get("title", f"Document {line_no}")),
                "text": str(record.get("text", "")),
            }


def _load_dir(path: str) -> Iterator[Dict]:
    names = sorted(n for n in os.listdir(path) if n.lower().endswith(".txt"))
    for doc_id, name in enumerate(names, start=1):
        with open(os.path.join(path, name), "r", encoding="utf-8", errors="replace") as handle:
            yield {"id": doc_id, "title": os.path.splitext(name)[0], "text": handle.read()}


def load_documents(source: str = "sample", path: Optional[str] = None,
                   max_docs: Optional[int] = None) -> Iterator[Dict]:
    """Yield up to ``max_docs`` documents from the chosen source."""
    if source == "sample":
        docs = iter(SAMPLE_DOCS)
    elif source == "jsonl":
        if not path:
            raise ValueError("source 'jsonl' needs a file path")
        docs = _load_jsonl(path)
    elif source == "dir":
        if not path:
            raise ValueError("source 'dir' needs a folder path")
        docs = _load_dir(path)
    else:
        raise ValueError(f"Unknown dataset source {source!r} (use sample | jsonl | dir)")

    for count, doc in enumerate(docs):
        if max_docs is not None and count >= max_docs:
            return
        yield doc
