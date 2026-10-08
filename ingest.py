"""
Ingestion pipeline: load documents -> preprocess -> build the inverted index
-> persist it through the configured storage backend.

    python ingest.py                              # dataset from config.yaml
    python ingest.py --source jsonl --path docs.jsonl --max-docs 500
    python ingest.py --source dir --path ./my_txt_files

Tip: with ``storage.backend: memory`` the index only lives as long as the
process, so use ``json`` if you want search.py / api.py to reuse what you
ingest here.
"""

import argparse
import time

from src.app import build_and_save
from src.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the search index")
    parser.add_argument("--source", choices=["sample", "jsonl", "dir"])
    parser.add_argument("--path", help="file (jsonl) or folder (dir)")
    parser.add_argument("--max-docs", type=int)
    parser.add_argument("--backend", choices=["memory", "json"])
    args = parser.parse_args()

    config = load_config()
    if args.source:
        config["dataset"]["source"] = args.source
    if args.path:
        config["dataset"]["path"] = args.path
    if args.max_docs:
        config["dataset"]["max_docs"] = args.max_docs
    if args.backend:
        config["storage"]["backend"] = args.backend

    started = time.perf_counter()
    index, _, storage = build_and_save(config)
    elapsed = time.perf_counter() - started

    print(f"Indexed {index.num_docs} documents in {elapsed:.2f}s")
    print(f"  vocabulary size : {len(index.postings)} stems")
    print(f"  avg doc length  : {index.avg_doc_length:.1f} tokens")
    print(f"  storage backend : {config['storage']['backend']} ({type(storage).__name__})")
    if config["storage"]["backend"] == "memory":
        print("  note: memory storage is not shared between processes (use --backend json).")


if __name__ == "__main__":
    main()
