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
2. If the passages do not contain the answer, do NOT include any [S#] citations.
   Instead, check whether it is a STATISTICAL FACT.

   STATISTICAL FACTS include ALL Common Data Set (CDS) metrics:
   acceptance rate, yield rate, average GPA, SAT/ACT score ranges,
   total applicants received, total applicants admitted, total students enrolled,
   class size, enrollment figures, tuition and fees, room and board,
   graduation rate, retention rate, student-to-faculty ratio,
   percentage receiving financial aid, average financial aid award,
   male/female ratio, in-state vs out-of-state enrollment, ethnicity breakdown.

   IF IT IS A STATISTICAL FACT:
   - You MUST provide the figure from your training knowledge of Common Data Sets (CDS) and US News rankings. Do not refuse or say you don't know — give the best approximate figure you have.
   - State the figure and the approximate year (e.g. "approximately 3,800 students admitted in the 2023–24 cycle").
   - End your answer with exactly this line: "Source: US News / Common Data Set (prior year data). For current figures, visit: {usnews_url}"
   - Do NOT use [S#] citations for training-knowledge figures.

   IF IT IS NOT A STATISTICAL FACT:
   - State specifically what is unavailable: e.g. "{college}'s official admission pages do not include details about [topic]."
   - Do NOT use any [S#] citations.

3. Be concise. 2–5 sentences is ideal. Use bullet points only when listing multiple items.
4. Never invent deadlines, policies, essay prompts, or program details not in the passages.
5. Never cite [S#] for anything outside the provided passages.
"""

# ── US News prompt — Tavily live-search second pass ──────────────────────────
USNEWS_SYSTEM = """You are an admission research assistant for {college}.

The official admission pages did not contain the answer.
The passages below are from US News & World Reports for this college.

RULES:
1. Start your answer with exactly this sentence:
   "Note: This information is not available on {college}'s official admission pages."
2. Answer using ONLY the US News passages provided.
3. Do NOT use any [S#] citation markers.
4. Be concise — 2–4 sentences.
"""

USNEWS_USER_TEMPLATE = """QUESTION:
{question}

US NEWS PASSAGES:
{sources}

Answer the question using these passages."""

# ── Fallback prompt — training-knowledge third pass ───────────────────────────
FALLBACK_SYSTEM = """You are an admission statistics expert for {college}.

The official admission website did not contain the requested information,
and a live search also returned no results.
Answer using your training knowledge of Common Data Sets (CDS) and US News rankings.

RULES:
1. Start your answer with exactly this sentence:
   "Note: This information is not available on {college}'s official admission pages."
2. Provide the specific figure with the approximate year
   (e.g. "approximately 3,800 students were admitted in the 2023–24 cycle").
3. You MUST give a number or fact — do not say you cannot help or that the data is unavailable.
4. Do NOT include any "Source:" line or URL in your answer — the source link is added separately.
5. Do NOT use any [S#] citations.
6. Be concise — 2–4 sentences.
"""

# ── Not-found / citation detection ───────────────────────────────────────────
_NOT_FOUND_RE = re.compile(
    r"do not include|does not include|not available|not mentioned|"
    r"does not contain|do not contain|pages do not|admission pages do not|"
    r"no information|cannot find|not found",
    re.IGNORECASE,
)

_HAS_CITATION_RE = re.compile(r"\[S\d+\]")

# Strip any inline "Source: US News / CDS ..." line the model might add
_CDS_INLINE_RE = re.compile(
    r"\n?Source:\s*US News\s*/\s*Common Data Set[^\n]*",
    re.IGNORECASE,
)


def _looks_like_not_found(answer: str) -> bool:
    """True when the LLM said the answer isn't in the passages."""
    return bool(_NOT_FOUND_RE.search(answer))


def _has_citations(answer: str) -> bool:
    """True when the LLM cited at least one [S#] source."""
    return bool(_HAS_CITATION_RE.search(answer))

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
    _tavily = None              # TavilyClient | None — optional live search

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
        if RAG._tavily is None:
            tavily_key = os.environ.get("TAVILY_API_KEY")
            if tavily_key:
                try:
                    from tavily import TavilyClient
                    RAG._tavily = TavilyClient(api_key=tavily_key)
                except Exception:
                    pass  # tavily-python not installed or key invalid — skip live search

    def _search_usnews(self, college_display: str, question: str) -> tuple[list[str], list[dict]]:
        """Search US News via Tavily. Returns (docs, metas) or ([], []) on failure."""
        if RAG._tavily is None:
            return [], []
        try:
            resp = RAG._tavily.search(
                query=f"{college_display} {question}",
                include_domains=["usnews.com"],
                max_results=3,
            )
            results = resp.get("results", [])
            docs  = [r.get("content", "") for r in results if r.get("content")]
            metas = [{"url": r.get("url", "")}  for r in results if r.get("content")]
            return docs, metas
        except Exception:
            return [], []

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

    def ask(self, college_id: str, question: str,
            college_display: str | None = None,
            usnews_url: str | None = None) -> dict:
        docs, metas = self.retrieve(college_id, question)
        sources_block = _format_sources(docs, metas)
        college_display = college_display or college_id
        usnews_url = usnews_url or "https://www.usnews.com/best-colleges"

        # ── Pass 1: official admission pages (Pinecone) ───────────────────────
        completion = RAG._groq.chat.completions.create(
            model=config.GROQ_MODEL,
            temperature=config.GROQ_TEMPERATURE,
            max_tokens=config.GROQ_MAX_TOKENS,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT.format(
                    college=college_display, usnews_url=usnews_url)},
                {"role": "user",   "content": USER_TEMPLATE.format(
                    question=question, sources=sources_block)},
            ],
        )
        answer = completion.choices[0].message.content

        # If pass 1 cited sources, we're done — answer came from official pages
        if _has_citations(answer):
            return {
                "college_id": college_id,
                "question":   question,
                "answer":     answer,
                "sources":    list(zip(docs, metas)),
            }

        # ── Pass 2: live US News search via Tavily ────────────────────────────
        usnews_docs, usnews_metas = self._search_usnews(college_display, question)
        if usnews_docs:
            usnews_block = "\n".join(
                f"[passage {i}] (from {m['url']})\n{d}\n"
                for i, (d, m) in enumerate(zip(usnews_docs, usnews_metas), 1)
            )
            usnews_completion = RAG._groq.chat.completions.create(
                model=config.GROQ_MODEL,
                temperature=config.GROQ_TEMPERATURE,
                max_tokens=config.GROQ_MAX_TOKENS,
                messages=[
                    {"role": "system", "content": USNEWS_SYSTEM.format(
                        college=college_display)},
                    {"role": "user",   "content": USNEWS_USER_TEMPLATE.format(
                        question=question, sources=usnews_block)},
                ],
            )
            usnews_answer = usnews_completion.choices[0].message.content
            if not _looks_like_not_found(usnews_answer):
                return {
                    "college_id": college_id,
                    "question":   question,
                    "answer":     usnews_answer,
                    "sources":    list(zip(usnews_docs, usnews_metas)),
                }

        # ── Pass 3: LLM training-knowledge fallback (CDS / US News) ──────────
        fallback = RAG._groq.chat.completions.create(
            model=config.GROQ_MODEL,
            temperature=0.1,
            max_tokens=config.GROQ_MAX_TOKENS,
            messages=[
                {"role": "system", "content": FALLBACK_SYSTEM.format(
                    college=college_display, usnews_url=usnews_url)},
                {"role": "user",   "content": question},
            ],
        )
        fallback_answer = fallback.choices[0].message.content

        if not _looks_like_not_found(fallback_answer):
            clean = _CDS_INLINE_RE.sub("", fallback_answer).strip()
            return {
                "college_id": college_id,
                "question":   question,
                "answer":     clean,
                "sources":    [("US News / Common Data Set — prior year admissions data",
                                {"url": usnews_url})],
            }

        # All three passes failed — return the not-found message, no sources
        return {
            "college_id": college_id,
            "question":   question,
            "answer":     fallback_answer,
            "sources":    [],
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
