"""Multi-college retrieval + LLM answer — Pinecone backend.

Pure function: ask(college_id, question) -> {answer, sources}.
The api.py FastAPI endpoint calls this; eval_college.py also calls this.

Refactor of rag_prototype/chat.py — same SYSTEM_PROMPT / USER_TEMPLATE,
same Groq backend, same deep-link / snippet helpers, but the vector store
is now Pinecone (one namespace per college) instead of local Chroma.
"""
import os
import re
import sys
import urllib.parse
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq
from embed import Embedder
from pinecone import Pinecone

import config


# ── Env loading ───────────────────────────────────────────────────────────────
# Try rag_backend/.env first, then fall back to rag_prototype/.env so both
# locations work without duplicating secrets.
_ENV_CANDIDATES = [
    Path(__file__).parent / ".env",
    Path(__file__).parent.parent / "rag_prototype" / ".env",
]
for _p in _ENV_CANDIDATES:
    if _p.exists():
        load_dotenv(_p)
        break


# ── Deep-link helpers (lifted from rag_prototype/chat.py) ────────────────────
def deep_link(url: str, chunk: str) -> str:
    """url with #:~:text=start,end so the browser scrolls to the matched passage."""
    words = re.findall(r"[^\s]+", chunk.strip())
    if len(words) < 4:
        return url
    start = " ".join(words[:4])
    end = " ".join(words[-4:])
    enc_start = urllib.parse.quote(start, safe="")
    enc_end = urllib.parse.quote(end, safe="")
    return f"{url}#:~:text={enc_start},{enc_end}"


def snippet(chunk: str, limit: int = 180) -> str:
    s = re.sub(r"\s+", " ", chunk.strip())
    return s if len(s) <= limit else s[:limit].rsplit(" ", 1)[0] + "..."


# ── Prompts ───────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are an admission research assistant for {college}.
Your primary source is the passages below from official admission pages.

RULES:
1. For any information found in the provided passages, cite inline as [S1], [S2], etc.
2. If the passages do not contain the answer, follow this two-step check:
   a. Is it a STATISTICAL FACT — acceptance rate, yield rate, average GPA, SAT/ACT ranges, class size, tuition, or enrollment?
      YES → Answer using your knowledge of Common Data Sets (CDS) and US News rankings. State the approximate figure and the year it is from if you know it (e.g. "approximately 20% as of the 2023 Common Data Set"). End with this exact line: "For current figures, search '[college name] Common Data Set' or visit https://www.usnews.com/best-colleges and search for this school." Only state figures you are highly confident about. Do NOT use [S#] citations for this training-data knowledge.
      NO  → State specifically what is not available: e.g. "{college}'s official admission pages do not include information about [topic]."
3. Be concise. 2–5 sentences is ideal. Use bullet points only when listing multiple items.
4. Never invent deadlines, policies, essay prompts, or program details not found in the passages.
5. Never cite [S#] for anything outside the provided passages.
"""

USER_TEMPLATE = """QUESTION:
{question}

SOURCES:
{sources}

Answer the question using only these sources, with inline [S#] citations."""


def _format_sources(docs: list[str], metas: list[dict]) -> str:
    out = []
    for i, (d, m) in enumerate(zip(docs, metas), start=1):
        out.append(f"[S{i}] (from {m['url']})\n{d}\n")
    return "\n".join(out)


# ── Module-level helper (used by api.py) ──────────────────────────────────────
def pinecone_indexed_colleges() -> set[str]:
    """Return the set of college_ids that have at least one vector in Pinecone.

    Makes a single describe_index_stats() call — O(1) regardless of how many
    colleges are indexed.  Used by api.py to build the indexed-only filter.
    """
    api_key = os.environ.get("PINECONE_API_KEY", "")
    if not api_key:
        return set()
    try:
        pc = Pinecone(api_key=api_key)
        stats = pc.Index(config.PINECONE_INDEX_NAME).describe_index_stats()
        return set((stats.namespaces or {}).keys())
    except Exception:
        return set()


# ── RAG (multi-college, lazy-init resources) ──────────────────────────────────
class RAG:
    _embedder: Embedder | None = None
    _pinecone_index = None      # shared Pinecone Index handle (stateless API)
    _groq: Groq | None = None   # shared Groq client

    def __init__(self):
        if RAG._groq is None:
            api_key = os.environ.get("GROQ_API_KEY")
            if not api_key:
                raise SystemExit("GROQ_API_KEY not set. Copy .env from rag_prototype/.env.")
            RAG._groq = Groq(api_key=api_key)
        if RAG._embedder is None:
            RAG._embedder = Embedder(config.EMBED_MODEL)
        if RAG._pinecone_index is None:
            pc_key = os.environ.get("PINECONE_API_KEY")
            if not pc_key:
                raise SystemExit("PINECONE_API_KEY not set. Add it to .env.")
            pc = Pinecone(api_key=pc_key)
            RAG._pinecone_index = pc.Index(config.PINECONE_INDEX_NAME)

    def retrieve(self, college_id: str, question: str, k: int = config.TOP_K):
        """Embed question, query Pinecone namespace, return (docs, metas)."""
        q_emb = RAG._embedder.embed_one(question)
        results = RAG._pinecone_index.query(
            vector=q_emb,
            top_k=k,
            namespace=college_id,
            include_metadata=True,
        )
        # Pinecone stores chunk text inside metadata["text"]; split it back out.
        docs  = [m.metadata.get("text", "") for m in results.matches]
        metas = [{k: v for k, v in m.metadata.items() if k != "text"}
                 for m in results.matches]
        return docs, metas

    def ask(self, college_id: str, question: str, college_display: str | None = None) -> dict:
        docs, metas = self.retrieve(college_id, question)
        sources_block = _format_sources(docs, metas)
        college_display = college_display or college_id
        completion = RAG._groq.chat.completions.create(
            model=config.GROQ_MODEL,
            temperature=config.GROQ_TEMPERATURE,
            max_tokens=config.GROQ_MAX_TOKENS,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT.format(college=college_display)},
                {"role": "user",   "content": USER_TEMPLATE.format(
                    question=question, sources=sources_block)},
            ],
        )
        answer = completion.choices[0].message.content
        return {
            "college_id": college_id,
            "question":   question,
            "answer":     answer,
            "sources":    list(zip(docs, metas)),
        }


# ── CLI for quick manual probes ───────────────────────────────────────────────
def _print_result(r: dict):
    print("\n" + "=" * 70)
    print(f"[{r['college_id']}] Q:", r["question"])
    print("-" * 70)
    print(r["answer"])
    print("-" * 70)
    print("Retrieved sources (deep-linked):")
    for i, (d, m) in enumerate(r["sources"], start=1):
        print(f"  [S{i}] {deep_link(m['url'], d)}")
        print(f"       \"{snippet(d)}\"")
    print("=" * 70)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python rag.py <college_id> <question...>")
        sys.exit(1)
    cid = sys.argv[1]
    q   = " ".join(sys.argv[2:])
    rag = RAG()
    _print_result(rag.ask(cid, q))
