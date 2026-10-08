"""
Interactive command-line search.

    python search.py

Example queries
    ("computer" AND "science") OR ("software" AND NOT "hardware")
    "machine" PHRASE "learning"
    galaxy stars
Commands:  :bm25  :tfidf  :quit
"""

from src.app import get_engine
from src.query.parser import QuerySyntaxError

BANNER = __doc__


def main() -> None:
    engine = get_engine()
    scorer = None
    print(f"Index ready: {engine.index.num_docs} documents, {len(engine.index.postings)} terms.")
    print(BANNER)

    while True:
        try:
            query = input("search> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not query:
            continue
        if query in (":quit", ":q", "exit"):
            break
        if query in (":bm25", ":tfidf"):
            scorer = query[1:]
            print(f"scorer set to {scorer}")
            continue

        try:
            result = engine.search(query, scorer=scorer)
        except QuerySyntaxError as error:
            print(f"  query error: {error}")
            continue

        for word, suggestion in result.corrections.items():
            print(f"  did you mean '{suggestion}' instead of '{word}'?")
        print(f"  postfix: {result.postfix}")
        print(f"  {result.total_matches} match(es) in {result.elapsed_ms:.2f} ms")
        for rank, hit in enumerate(result.hits, start=1):
            print(f"  {rank:>2}. [{hit.score:7.3f}] #{hit.doc_id} {hit.title}")


if __name__ == "__main__":
    main()
