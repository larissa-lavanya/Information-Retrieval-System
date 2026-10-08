"""
FastAPI backend for the web UI and the skip-pointer visualiser.

    uvicorn api:app --reload        ->  http://127.0.0.1:8000

Endpoints
    GET /                  search page (index.html)
    GET /visualize         skip-pointer visualiser (visualize.html)
    GET /search?q=&k=&scorer=      ranked results (+ postfix, spelling suggestions)
    GET /suggest?prefix=           vocabulary terms starting with ``prefix``
    GET /postings/{term}           postings list + skip pointers for a term
"""

import os
from dataclasses import asdict

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse

from src.app import get_engine
from src.query.parser import QuerySyntaxError

ROOT = os.path.dirname(os.path.abspath(__file__))
app = FastAPI(title="Information Retrieval System")
engine = get_engine()


@app.get("/")
def home():
    return FileResponse(os.path.join(ROOT, "index.html"))


@app.get("/visualize")
def visualize():
    return FileResponse(os.path.join(ROOT, "visualize.html"))


@app.get("/search")
def search(q: str, k: int = Query(10, ge=1, le=100), scorer: str = "bm25"):
    try:
        result = engine.search(q, top_k=k, scorer=scorer)
    except QuerySyntaxError as error:
        raise HTTPException(status_code=400, detail=str(error))
    except ValueError as error:                      # unknown scorer name
        raise HTTPException(status_code=400, detail=str(error))
    payload = asdict(result)
    payload["scorer"] = scorer
    return payload


@app.get("/suggest")
def suggest(prefix: str, limit: int = Query(10, ge=1, le=50)):
    """Prefix-match vocabulary keys (stems)."""
    prefix = prefix.lower()
    matches = [t for t in engine.index.vocabulary() if t.startswith(prefix)]
    return {"prefix": prefix, "terms": matches[:limit]}


@app.get("/postings/{term}")
def postings(term: str):
    stem = engine.processor.stem_word(term.lower())
    if not engine.index.has_term(stem):
        raise HTTPException(status_code=404, detail=f"'{term}' (stem '{stem}') is not in the index")
    return engine.index.skip_info(stem)
