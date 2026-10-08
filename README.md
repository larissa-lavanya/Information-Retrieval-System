# Information Retrieval System (IRS)

A small search engine in pure Python: Boolean queries with precedence and parentheses, phrase search, spelling suggestions, and BM25 / TF-IDF ranking. Every algorithm is implemented from scratch in its own module (no NLTK download needed) and marked in the code with a header comment (`ALGORITHM n - ...`) and inline `# <--` markers at the exact line where it runs.

## Where each algorithm lives

| # | Algorithm | File | Used for |
|---|-----------|------|----------|
| 1 | Tokenization | `src/preprocessing/tokenizer.py` | split text into clean lower-case tokens |
| 2 | Stopword removal | `src/preprocessing/stopwords.py`, applied in `processor.py` | drop words like *the / is / of* |
| 3 | Porter stemming | `src/preprocessing/porter_stemmer.py` | map word forms to a common root |
| 4 | Positional inverted index | `src/index/inverted_index.py` | term -> {doc: positions} lookup |
| 5 | Postings intersection + skip pointers | `src/query/set_ops.py`, `src/index/skip_pointers.py` | AND / OR / NOT on postings lists |
| 6 | Phrase matching (positional intersect) | `src/query/phrase.py` | exact-phrase queries |
| 7 | Shunting-Yard (infix -> postfix) | `src/query/parser.py` | precedence and parentheses in queries |
| 8 | Levenshtein distance | `src/query/spell.py` | "did you mean" suggestions |
| 9 | BM25 and TF-IDF | `src/ranking/scorers.py` | rank the matching documents |

`src/query/engine.py` ties them together; `src/query/lexer.py` tokenises the query string.

## Flow

```
Index time:  documents -> tokenize -> stopwords -> stem -> inverted index -> storage
Query time:  query -> lexer -> spell check (Levenshtein) -> Shunting-Yard -> RPN
             -> stack evaluation (skip-pointer AND, OR, NOT, PHRASE) -> BM25/TF-IDF -> top-k
```

## Project layout

```
config.yaml            settings (dataset, storage backend, ranking, spelling)
ingest.py              build + persist the index
search.py              interactive CLI
api.py                 FastAPI backend (+ index.html, visualize.html)
main.py                smoke test: build -> save -> restore -> demo queries
src/config.py          config loader
src/app.py             wiring: build or restore index, return SearchEngine
src/data/              built-in sample corpus and loaders (sample | jsonl | dir)
src/preprocessing/     tokenizer, stopwords, Porter stemmer, pipeline
src/index/             inverted index, skip pointers, storage adapters, builder
src/query/             lexer, parser, set ops, phrase, spell, engine
src/ranking/           TF-IDF and BM25
src/evaluation/        benchmark -> benchmark_results.md
tests/                 unit tests (35)
```

## Setup and usage

```bash
pip install -r requirements.txt          # PyYAML is the only required package

python main.py                           # build, save, restore, run demo queries
python search.py                         # interactive CLI
python -m unittest discover -s tests -t .   # run the tests
python -m src.evaluation.benchmark       # writes benchmark_results.md

uvicorn api:app --reload                 # web UI at http://127.0.0.1:8000
                                         # visualiser at /visualize
```

### Query syntax

| Query | Meaning |
|-------|---------|
| `software` | single term (stemmed, so it also finds *softwares*) |
| `machine learning` | implicit AND |
| `a AND b`, `a OR b`, `NOT a` | Boolean operators (case-insensitive) |
| `"machine" PHRASE "learning"` or `"machine learning"` | exact adjacent phrase |
| `("computer" AND "science") OR ("software" AND NOT "hardware")` | parentheses override precedence |

Precedence: `PHRASE` > `NOT` > `AND` > `OR`.

### Using your own data

```bash
python ingest.py --source jsonl --path docs.jsonl --backend json   # {"title": ..., "text": ...} per line
python ingest.py --source dir   --path ./txt_files --backend json  # every *.txt file
```
Set `storage.backend: json` in `config.yaml` so `search.py` / `api.py` reuse the saved index.

## Design notes

* **Positions are counted after stopword removal**, so `"history of rome"` matches
  `history rome` adjacency. Stopword-only operands in a phrase are ignored.
* **Skip pointers are virtual**: they follow from the list length (step = sqrt(n)),
  so nothing extra is stored; `InvertedIndex.skip_info` materialises them for the visualiser.
* **Spell suggestions** are always reported; set `spelling.auto_correct: true` to apply them.
* **TF-IDF** uses `tf * log(1 + N/df)` (smoothed so common terms never score 0).
* **Storage** is behind `StorageAdapter`; `memory` and `json` backends are included,
  and new ones only need `save` / `load`.
