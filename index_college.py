"""Per-college indexer — Pinecone backend.

Reads clean_text/<cid>/<slug>.txt files produced by scrape_college.py,
applies the same sliding-window chunker + content-hash dedup as before,
embeds with sentence-transformers, then upserts into a Pinecone serverless
index (one namespace per college, chunk text stored in vector metadata).

Usage:
    python index_college.py --pilot
    python index_college.py --only iu_kelley
"""
import argparse
import hashlib
import json
import os
import re
from pathlib import Path

from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer

import config

# Load .env for PINECONE_API_KEY
for _p in [Path(__file__).parent / ".env",
           Path(__file__).parent.parent / "rag_prototype" / ".env"]:
    if _p.exists():
        load_dotenv(_p)
        break

UPSERT_BATCH = 100   # Pinecone recommended max vectors per upsert call


# ── Helpers (same chunker as before) ─────────────────────────────────────────
def _hash(text: str) -> str:
    return hashlib.sha256(re.sub(r"\s+", " ", text.strip().lower()).encode()).hexdigest()


def chunk_text(text: str, size_tokens: int, overlap_tokens: int) -> list[str]:
    normalized = re.sub(r"\n{2,}", "\n\n", text).strip()
    size_chars    = size_tokens * 4
    overlap_chars = max(0, overlap_tokens * 4)
    chunks: list[str] = []
    start = 0
    n = len(normalized)
    while start < n:
        end = min(start + size_chars, n)
        if end < n:
            brk = normalized.rfind(" ", start, end)
            if brk > start + size_chars // 2:
                end = brk
        piece = normalized[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= n:
            break
        start = max(end - overlap_chars, start + 1)
    return [c for c in chunks if len(c) > 80]


# ── Lazy singletons ───────────────────────────────────────────────────────────
_embedder: SentenceTransformer | None = None
_pinecone_index = None


def _get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        print(f"[embed] loading {config.EMBED_MODEL}")
        _embedder = SentenceTransformer(config.EMBED_MODEL)
    return _embedder


def _get_pinecone_index():
    global _pinecone_index
    if _pinecone_index is None:
        api_key = os.environ.get("PINECONE_API_KEY")
        if not api_key:
            raise SystemExit("PINECONE_API_KEY not set. Add it to .env")
        pc = Pinecone(api_key=api_key)
        existing = [idx.name for idx in pc.list_indexes()]
        if config.PINECONE_INDEX_NAME not in existing:
            print(f"[pinecone] Creating index '{config.PINECONE_INDEX_NAME}' "
                  f"({config.EMBED_DIMENSION}-dim cosine, "
                  f"{config.PINECONE_CLOUD}/{config.PINECONE_REGION})...")
            pc.create_index(
                name=config.PINECONE_INDEX_NAME,
                dimension=config.EMBED_DIMENSION,
                metric="cosine",
                spec=ServerlessSpec(cloud=config.PINECONE_CLOUD,
                                    region=config.PINECONE_REGION),
            )
            print("[pinecone] Index created.")
        _pinecone_index = pc.Index(config.PINECONE_INDEX_NAME)
    return _pinecone_index


# ── Per-college index ─────────────────────────────────────────────────────────
def index_college(college: dict) -> dict:
    cid = college["college_id"]
    manifest_path = config.manifest_path(cid)
    if not manifest_path.exists():
        print(f"  SKIP {cid}: no manifest at {manifest_path}")
        return {"college_id": cid, "chunks": 0, "pages": 0, "skipped": True}

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not manifest:
        print(f"  SKIP {cid}: empty manifest")
        return {"college_id": cid, "chunks": 0, "pages": 0, "skipped": True}

    embedder = _get_embedder()
    index    = _get_pinecone_index()

    # Wipe the existing namespace so re-indexing is always clean.
    try:
        index.delete(delete_all=True, namespace=cid)
    except Exception:
        pass

    seen_pages:  set[str] = set()
    seen_chunks: set[str] = set()
    pages_indexed = pages_dup = chunks_dup = 0
    total_chunks  = 0
    pending: list[dict] = []   # buffered vectors waiting for batch upsert

    def _flush():
        nonlocal total_chunks
        if not pending:
            return
        index.upsert(vectors=pending, namespace=cid)
        total_chunks += len(pending)
        pending.clear()

    clean_dir = config.clean_text_dir(cid)
    for entry in manifest:
        slug  = entry["slug"]
        url   = entry["url"]
        title = entry.get("title") or ""
        text_path = clean_dir / f"{slug}.txt"
        if not text_path.exists():
            continue
        body = text_path.read_text(encoding="utf-8").split("\n\n", 1)[-1]

        page_hash = _hash(body)
        if page_hash in seen_pages:
            pages_dup += 1
            continue
        seen_pages.add(page_hash)

        raw_chunks = chunk_text(body, config.CHUNK_SIZE_TOKENS, config.CHUNK_OVERLAP_TOKENS)
        chunks: list[str] = []
        for c in raw_chunks:
            h = _hash(c)
            if h in seen_chunks:
                chunks_dup += 1
                continue
            seen_chunks.add(h)
            chunks.append(c)
        if not chunks:
            continue

        embeddings = embedder.encode(chunks, show_progress_bar=False).tolist()
        for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
            pending.append({
                "id":     f"{slug}__{i}",
                "values": emb,
                "metadata": {
                    "college_id": cid,
                    "url":        url,
                    "title":      title,
                    "slug":       slug,
                    "chunk_idx":  i,
                    "text":       chunk,   # Pinecone has no separate doc store
                },
            })
            if len(pending) >= UPSERT_BATCH:
                _flush()

        pages_indexed += 1

    _flush()   # upload any remaining vectors

    stats = {
        "college_id": cid,
        "namespace":  cid,
        "pages":      pages_indexed,
        "chunks":     total_chunks,
        "pages_dup":  pages_dup,
        "chunks_dup": chunks_dup,
    }
    print(f"  {cid}: {total_chunks} chunks from {pages_indexed} pages "
          f"(skipped {pages_dup} dup pages, {chunks_dup} dup chunks) "
          f"-> Pinecone ns={cid!r}")
    return stats


# ── Main ──────────────────────────────────────────────────────────────────────
PILOT_IDS = [
    "iu_kelley", "upenn_wharton", "nyu_stern", "uva_mcintire", "utaustin_mccombs",
]


def _load_seeds_registry() -> list[dict]:
    seeds_json = config.DATA_DIR / "registry_seeds.json"
    if seeds_json.exists():
        return json.loads(seeds_json.read_text(encoding="utf-8"))
    import college_registry
    return college_registry.load_local()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--only")
    args = ap.parse_args()

    registry = _load_seeds_registry()
    if args.only:
        registry = [c for c in registry if c["college_id"] == args.only]
    elif args.pilot:
        registry = [c for c in registry if c["college_id"] in PILOT_IDS]
    else:
        ap.error("specify --pilot or --only")

    if not registry:
        print("No colleges matched.")
        return

    print(f"Indexing {len(registry)} college(s) -> "
          f"Pinecone '{config.PINECONE_INDEX_NAME}'...\n")
    all_stats = []
    for c in registry:
        all_stats.append(index_college(c))

    print("\nSummary:")
    total_chunks = total_pages = 0
    for s in all_stats:
        if s.get("skipped"):
            print(f"  {s['college_id']}: SKIPPED")
            continue
        print(f"  {s['college_id']}: {s['chunks']} chunks ({s['pages']} pages)")
        total_chunks += s["chunks"]
        total_pages  += s["pages"]
    print(f"  TOTAL: {total_chunks} chunks across {total_pages} pages "
          f"-> Pinecone index '{config.PINECONE_INDEX_NAME}'")


if __name__ == "__main__":
    main()
