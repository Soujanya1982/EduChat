"""One-time migration: local Chroma -> Pinecone serverless.

Reads every existing Chroma collection (one per college), extracts the
stored vectors + documents + metadata, and upserts them into the Pinecone
index — one namespace per college.

No re-embedding needed: vectors are taken directly from Chroma.

Usage:
    python migrate_chroma_to_pinecone.py              # migrate all collections
    python migrate_chroma_to_pinecone.py --only iu_kelley
    python migrate_chroma_to_pinecone.py --dry-run    # list what would migrate

After this completes successfully you can delete data/chroma/ to free ~236 MB.
"""
import argparse
import os
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

import config

# Load .env for PINECONE_API_KEY
for _p in [Path(__file__).parent / ".env",
           Path(__file__).parent.parent / "rag_prototype" / ".env"]:
    if _p.exists():
        load_dotenv(_p)
        break

UPSERT_BATCH = 100   # Pinecone recommended max per upsert call


# ── Pinecone setup ────────────────────────────────────────────────────────────
def _get_pinecone_index():
    api_key = os.environ.get("PINECONE_API_KEY")
    if not api_key:
        raise SystemExit(
            "PINECONE_API_KEY not set.\n"
            "Add it to rag_backend/.env:\n"
            "  PINECONE_API_KEY=pc-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
        )
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
    else:
        print(f"[pinecone] Using existing index '{config.PINECONE_INDEX_NAME}'.")
    return pc.Index(config.PINECONE_INDEX_NAME)


# ── Per-college migration ─────────────────────────────────────────────────────
def migrate_college(cid: str,
                    chroma_client: chromadb.PersistentClient,
                    pinecone_index,
                    *,
                    dry_run: bool = False) -> int:
    """Migrate one Chroma collection -> Pinecone namespace.

    Returns the number of vectors uploaded (0 on skip/dry-run counts as found).
    """
    try:
        coll = chroma_client.get_collection(config.chroma_collection_name(cid))
    except Exception:
        print(f"  [{cid}] No Chroma collection found — skipping.")
        return 0

    count = coll.count()
    if count == 0:
        print(f"  [{cid}] Empty collection — skipping.")
        return 0

    if dry_run:
        print(f"  [{cid}] {count} vectors (dry-run, not uploading)")
        return count

    print(f"  [{cid}] {count} vectors -> Pinecone ns={cid!r}")

    # Fetch everything from Chroma in one shot.
    # (Chroma stores embeddings natively — no re-embedding needed.)
    result = coll.get(include=["documents", "embeddings", "metadatas"])
    ids        = result["ids"]
    documents  = result["documents"]
    embeddings = result["embeddings"]
    metadatas  = result["metadatas"]

    # Clear the target namespace for a clean upload.
    try:
        pinecone_index.delete(delete_all=True, namespace=cid)
    except Exception:
        pass

    # Build Pinecone vector dicts (add "text" to metadata — Pinecone has no
    # separate document store).
    vectors = []
    for vid, doc, emb, meta in zip(ids, documents, embeddings, metadatas):
        pm = dict(meta)
        pm["text"] = doc
        vectors.append({"id": vid, "values": emb, "metadata": pm})

    # Batch upsert.
    uploaded = 0
    for i in range(0, len(vectors), UPSERT_BATCH):
        batch = vectors[i : i + UPSERT_BATCH]
        pinecone_index.upsert(vectors=batch, namespace=cid)
        uploaded += len(batch)

    print(f"    {uploaded} vectors uploaded.")
    return uploaded


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(
        description="Migrate local Chroma collections to Pinecone (one-time)."
    )
    ap.add_argument("--only",    help="migrate a single college_id")
    ap.add_argument("--dry-run", action="store_true",
                    help="list collections without uploading")
    args = ap.parse_args()

    # Connect to local Chroma.
    chroma_client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    all_colls = chroma_client.list_collections()

    # Collect college_ids from _adm-suffixed collection names.
    college_ids = sorted(
        c.name.removesuffix("_adm")
        for c in all_colls
        if c.name.endswith("_adm")
    )

    if args.only:
        college_ids = [c for c in college_ids if c == args.only]
        if not college_ids:
            print(f"No Chroma collection found for '{args.only}'. Exiting.")
            return

    if not college_ids:
        print("No _adm collections found in Chroma. Nothing to migrate.")
        return

    print(f"Found {len(college_ids)} Chroma collections.")
    if args.dry_run:
        print("DRY-RUN mode — no data will be uploaded.\n")

    pinecone_index = None if args.dry_run else _get_pinecone_index()
    print()

    total_vectors = 0
    for i, cid in enumerate(college_ids, 1):
        print(f"[{i}/{len(college_ids)}] {cid}")
        n = migrate_college(cid, chroma_client, pinecone_index, dry_run=args.dry_run)
        total_vectors += n

    print(f"\n{'='*60}")
    if args.dry_run:
        print(f"DRY-RUN complete. {total_vectors} vectors would be migrated "
              f"across {len(college_ids)} colleges.")
    else:
        print(f"Migration complete. {total_vectors} vectors uploaded "
              f"across {len(college_ids)} colleges.")
        print(f"\nYou can now delete data/chroma/ to free ~236 MB:")
        print(f"  rmdir /s /q data\\chroma   (Windows)")
        print(f"  rm -rf data/chroma         (Linux/Mac/Render)")


if __name__ == "__main__":
    main()
