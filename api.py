"""FastAPI /ask endpoint for the CollegeMatch RAG backend.

Endpoints:
    GET  /health                    — liveness probe (Render, load balancers)
    GET  /colleges                  — list all indexed colleges (for UI dropdown)
    GET  /colleges/{college_id}     — metadata for one college
    POST /ask                       — ask a question (JSON body)
    GET  /ask                       — same, via query params (handy for dev/testing)

Environment:
    GROQ_API_KEY   — required (Groq LLM)
    PORT           — set automatically by Render (default 8000 locally)

Run locally:
    uvicorn api:app --reload --port 8000

Deploy to Render:
    Start command: uvicorn api:app --host 0.0.0.0 --port $PORT
"""
import json
import os
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import config
from rag import RAG, deep_link, snippet, pinecone_indexed_colleges


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="CollegeMatch RAG API",
    description="Admission Q&A powered by per-college RAG over official admission pages.",
    version="0.2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # restrict to your UI origin in production
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ── Lazy singletons ───────────────────────────────────────────────────────────
_rag: Optional[RAG] = None
_registry: Optional[dict] = None        # college_id -> college dict
_pinecone_ns: Optional[set] = None      # cached set of indexed college_ids


def get_rag() -> RAG:
    global _rag
    if _rag is None:
        _rag = RAG()
    return _rag


def get_registry() -> dict:
    global _registry
    if _registry is None:
        seeds_json = config.DATA_DIR / "registry_seeds.json"
        if seeds_json.exists():
            colleges = json.loads(seeds_json.read_text(encoding="utf-8"))
        else:
            import college_registry
            colleges = college_registry.load_local()
        _registry = {c["college_id"]: c for c in colleges}
    return _registry


def _college_or_404(college_id: str) -> dict:
    reg = get_registry()
    if college_id not in reg:
        raise HTTPException(status_code=404, detail=f"College '{college_id}' not found.")
    return reg[college_id]


def _get_pinecone_ns() -> set:
    """Fetch the set of indexed college_ids from Pinecone (cached per process)."""
    global _pinecone_ns
    if _pinecone_ns is None:
        _pinecone_ns = pinecone_indexed_colleges()
    return _pinecone_ns


def _has_index(college_id: str) -> bool:
    """True if a Pinecone namespace for this college has at least one vector."""
    try:
        return college_id in _get_pinecone_ns()
    except Exception:
        return False


# ── Pydantic models ───────────────────────────────────────────────────────────
class CollegeSummary(BaseModel):
    college_id: str
    display_name: str
    rank_numeric: Optional[int]
    location: Optional[str]
    indexed: bool


class CollegeDetail(CollegeSummary):
    college_name: Optional[str]
    business_school: Optional[str]
    admission_url: Optional[str]
    region: Optional[str]
    state: Optional[str]


class SourceItem(BaseModel):
    index: int
    url: str
    deep_link: str
    snippet: str


class AskRequest(BaseModel):
    college_id: str
    question: str


class AskResponse(BaseModel):
    college_id: str
    display_name: str
    question: str
    answer: str
    sources: list[SourceItem]


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/health", tags=["meta"])
def health():
    """Liveness probe — always returns 200 if the process is up."""
    return {"status": "ok", "version": app.version}


@app.get("/colleges", response_model=list[CollegeSummary], tags=["colleges"])
def list_colleges(indexed_only: bool = Query(False, description="Only return colleges with a Chroma index")):
    """List all colleges in the registry, optionally filtered to indexed-only."""
    reg = get_registry()
    result = []
    for c in reg.values():
        cid = c["college_id"]
        indexed = _has_index(cid)
        if indexed_only and not indexed:
            continue
        result.append(CollegeSummary(
            college_id=cid,
            display_name=c.get("display_name") or cid,
            rank_numeric=c.get("rank_numeric"),
            location=c.get("location"),
            indexed=indexed,
        ))
    # Sort by rank (None goes to end), then alphabetically
    result.sort(key=lambda x: (x.rank_numeric is None, x.rank_numeric or 0, x.college_id))
    return result


@app.get("/colleges/{college_id}", response_model=CollegeDetail, tags=["colleges"])
def get_college(college_id: str):
    """Return metadata for a single college."""
    c = _college_or_404(college_id)
    return CollegeDetail(
        college_id=college_id,
        display_name=c.get("display_name") or college_id,
        rank_numeric=c.get("rank_numeric"),
        location=c.get("location"),
        indexed=_has_index(college_id),
        college_name=c.get("college_name"),
        business_school=c.get("business_school"),
        admission_url=c.get("admission_url"),
        region=c.get("region"),
        state=c.get("state"),
    )


def _build_ask_response(college_id: str, question: str) -> AskResponse:
    """Shared logic for POST /ask and GET /ask."""
    college = _college_or_404(college_id)
    display = college.get("display_name") or college_id

    if not _has_index(college_id):
        raise HTTPException(
            status_code=503,
            detail=f"'{college_id}' has not been indexed yet. Run build_all.py first.",
        )

    rag = get_rag()
    try:
        result = rag.ask(college_id, question, college_display=display)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RAG error: {e}")

    sources = []
    for i, (doc, meta) in enumerate(result["sources"], start=1):
        url = meta.get("url", "")
        sources.append(SourceItem(
            index=i,
            url=url,
            deep_link=deep_link(url, doc),
            snippet=snippet(doc),
        ))

    return AskResponse(
        college_id=college_id,
        display_name=display,
        question=question,
        answer=result["answer"],
        sources=sources,
    )


@app.post("/ask", response_model=AskResponse, tags=["ask"])
def ask_post(req: AskRequest):
    """Ask a question about a college's admissions (JSON body).

    Example body:
        {"college_id": "upenn_wharton", "question": "What is the application deadline?"}
    """
    return _build_ask_response(req.college_id, req.question)


@app.get("/ask", response_model=AskResponse, tags=["ask"])
def ask_get(
    college_id: str = Query(..., description="College slug, e.g. upenn_wharton"),
    question: str  = Query(..., description="Admission question in plain English"),
):
    """Ask a question via query parameters (useful for quick browser/curl tests).

    Example:
        GET /ask?college_id=iu_kelley&question=What+is+the+application+deadline%3F
    """
    return _build_ask_response(college_id, question)


# ── Dev entrypoint ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("api:app", host="0.0.0.0", port=port, reload=True)
