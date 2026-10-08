"""
Sandbox / smoke test: build the index, save it, restore it from storage and
run a few representative queries.  Run:  python main.py
"""

from src.app import build_and_save, get_engine
from src.config import load_config
from src.index.inverted_index import InvertedIndex

DEMO_QUERIES = [
    "software",
    "computer AND science",
    '("computer" AND "science") OR ("software" AND NOT "hardware")',
    '"machine" PHRASE "learning"',
    "galaxy stars NOT telescope",
    "softwre",                       # misspelled on purpose
]


def main() -> None:
    config = load_config()

    # 1. build + save
    index, _, storage = build_and_save(config)
    print(f"Built index: {index.num_docs} docs, {len(index.postings)} terms")

    # 2. restore and verify the round-trip
    restored = InvertedIndex.from_dict(storage.load())
    assert restored.postings == index.postings, "restore mismatch!"
    print("Save / restore round-trip OK\n")

    # 3. query
    engine = get_engine(config)
    for query in DEMO_QUERIES:
        result = engine.search(query, top_k=3)
        print(f"Q: {query}")
        print(f"   postfix : {result.postfix}")
        if result.corrections:
            print(f"   did you mean: {result.corrections}")
        print(f"   matches : {result.total_matches}")
        for hit in result.hits:
            print(f"     {hit.score:7.3f}  {hit.title}")
        print()


if __name__ == "__main__":
    main()
