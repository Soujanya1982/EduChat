"""Driver script: discover seeds → scrape → index for all 115 colleges.

Runs the full pipeline for every college in the registry, with resumability
so a restart skips colleges that are already complete.

Usage:
    python build_all.py --resume           # skip already-indexed colleges (recommended)
    python build_all.py                    # process all (overwrites existing data)
    python build_all.py --pilot            # 5 pilot colleges only
    python build_all.py --only upenn_wharton
    python build_all.py --start-from nyu_stern   # resume alphabetically from a college
    python build_all.py --discover-only    # run seed discovery only, no scrape/index
    python build_all.py --index-only       # re-index from existing manifests (no scrape)

Failure handling:
    - Per-college errors are caught and logged; one failure never stops the run.
    - Errors are written to data/build_errors.jsonl.
    - A full build report is written to data/build_report.json on completion.
"""
import argparse
import json
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import config
import college_registry
import discover_seeds as _ds
from scrape_college import scrape_college
from index_college import index_college


# ── Pilot set (same as other scripts) ────────────────────────────────────────
PILOT_IDS = [
    "iu_kelley", "upenn_wharton", "nyu_stern", "uva_mcintire", "utaustin_mccombs",
]

# ── Paths ─────────────────────────────────────────────────────────────────────
BUILD_REPORT_PATH = config.DATA_DIR / "build_report.json"
BUILD_ERRORS_PATH = config.DATA_DIR / "build_errors.jsonl"


# ── Resume helpers ────────────────────────────────────────────────────────────
def _fetch_pinecone_counts() -> dict[str, int]:
    """Return {college_id: vector_count} for all Pinecone namespaces.

    Called once at the start of main() so we don't make 115 API calls.
    Returns {} if Pinecone is unreachable or the key is missing.
    """
    try:
        from pathlib import Path
        from dotenv import load_dotenv
        from pinecone import Pinecone
        for _p in [Path(__file__).parent / ".env",
                   Path(__file__).parent.parent / "rag_prototype" / ".env"]:
            if _p.exists():
                load_dotenv(_p)
                break
        api_key = os.environ.get("PINECONE_API_KEY", "")
        if not api_key:
            return {}
        pc = Pinecone(api_key=api_key)
        stats = pc.Index(config.PINECONE_INDEX_NAME).describe_index_stats()
        ns = stats.namespaces or {}
        return {cid: getattr(info, "vector_count", 0) for cid, info in ns.items()}
    except Exception:
        return {}


def _manifest_page_count(cid: str) -> int:
    """Return number of scraped pages in the manifest, 0 if missing."""
    p = config.manifest_path(cid)
    if not p.exists():
        return 0
    try:
        return len(json.loads(p.read_text(encoding="utf-8")))
    except Exception:
        return 0


def _is_indexed(cid: str, counts: dict[str, int]) -> bool:
    return counts.get(cid, 0) > 0


def _is_scraped(cid: str) -> bool:
    return _manifest_page_count(cid) > 0


# ── Seed loading ──────────────────────────────────────────────────────────────
def _load_registry_seeds() -> list[dict]:
    """Load full registry, overlaying any enriched seeds from registry_seeds.json.

    registry_seeds.json only contains colleges that have been through
    discover_seeds.py (currently just the 5 pilots). The remaining colleges
    come from the base xlsx registry and will get their seeds discovered on
    first run when do_discover=True.
    """
    # Start with the full base registry (all 115 colleges)
    base = college_registry.load_local()
    base_by_id = {c["college_id"]: c for c in base}

    # Overlay with any enriched seed data already discovered
    seeds_json = config.DATA_DIR / "registry_seeds.json"
    if seeds_json.exists():
        for enriched in json.loads(seeds_json.read_text(encoding="utf-8")):
            cid = enriched["college_id"]
            base_by_id[cid] = enriched  # enriched entry wins

    return list(base_by_id.values())


def _discover_one(college: dict, overrides: dict) -> dict:
    """Run seed discovery for a single college and merge into registry_seeds.json."""
    enriched = _ds.discover_for(college, overrides)
    # Merge back into registry_seeds.json
    seeds_json = config.DATA_DIR / "registry_seeds.json"
    if seeds_json.exists():
        registry = json.loads(seeds_json.read_text(encoding="utf-8"))
        registry = [c for c in registry if c["college_id"] != enriched["college_id"]]
        registry.append(enriched)
    else:
        registry = [enriched]
    seeds_json.write_text(
        json.dumps(registry, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return enriched


# ── Error logging ─────────────────────────────────────────────────────────────
def _log_error(cid: str, stage: str, exc: Exception):
    entry = {
        "college_id": cid,
        "stage": stage,
        "error": str(exc),
        "traceback": traceback.format_exc(),
        "ts": datetime.now(timezone.utc).isoformat(),
    }
    with BUILD_ERRORS_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


# ── Per-college pipeline ──────────────────────────────────────────────────────
def build_college(
    college: dict,
    overrides: dict,
    *,
    resume: bool,
    discover: bool,
    scrape: bool,
    index: bool,
    pinecone_counts: dict[str, int],
) -> dict:
    cid = college["college_id"]
    display = college.get("display_name") or cid
    stats = {
        "college_id": cid,
        "display_name": display,
        "skipped": False,
        "pages": 0,
        "chunks": 0,
        "error": None,
    }

    # ── Resume check ──────────────────────────────────────────────────────────
    if resume and index and _is_indexed(cid, pinecone_counts):
        chunks = pinecone_counts.get(cid, 0)
        pages = _manifest_page_count(cid)
        print(f"  [skip] {cid}: already indexed ({chunks} chunks, {pages} pages)")
        stats.update({"skipped": True, "pages": pages, "chunks": chunks})
        return stats

    print(f"\n  [{cid}] {display}")

    # ── Seed discovery ────────────────────────────────────────────────────────
    if discover:
        if cid not in overrides and college.get("seeds"):
            print(f"    seeds: already present ({len(college['seeds'])} seeds), skipping discovery")
        else:
            try:
                print(f"    [discover] ...")
                college = _discover_one(college, overrides)
                src = college.get("seeds_source", "unknown")
                print(f"    [discover] {len(college.get('seeds', []))} seeds ({src})")
            except Exception as e:
                _log_error(cid, "discover", e)
                print(f"    [discover ERR] {e}")
                # Fall back to admission_url as single seed
                college.setdefault("seeds", [college.get("admission_url", "")])
                college.setdefault("allowed_hosts", [])

    # ── Scrape ────────────────────────────────────────────────────────────────
    if scrape:
        if resume and _is_scraped(cid):
            pages = _manifest_page_count(cid)
            print(f"    [scrape] already done ({pages} pages) — skipping")
            stats["pages"] = pages
        else:
            try:
                scrape_stats = scrape_college(college)
                stats["pages"] = scrape_stats.get("pages", 0)
                print(f"    [scrape] {stats['pages']} pages")
            except Exception as e:
                _log_error(cid, "scrape", e)
                print(f"    [scrape ERR] {e}")
                stats["error"] = f"scrape: {e}"
                return stats

    # ── Index ─────────────────────────────────────────────────────────────────
    if index:
        try:
            idx_stats = index_college(college)
            stats["chunks"] = idx_stats.get("chunks", 0)
            if idx_stats.get("skipped"):
                print(f"    [index] skipped (no manifest)")
            else:
                print(f"    [index] {stats['chunks']} chunks")
        except Exception as e:
            _log_error(cid, "index", e)
            print(f"    [index ERR] {e}")
            stats["error"] = f"index: {e}"

    return stats


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(
        description="Build (discover → scrape → index) for all colleges."
    )
    ap.add_argument("--pilot", action="store_true", help="5 pilot colleges only")
    ap.add_argument("--only", help="single college_id")
    ap.add_argument(
        "--start-from",
        dest="start_from",
        help="skip all colleges before this college_id (alphabetical order of registry)",
    )
    ap.add_argument(
        "--resume",
        action="store_true",
        help="skip colleges that already have chunks in Chroma",
    )
    ap.add_argument(
        "--discover-only",
        action="store_true",
        help="run seed discovery only (no scrape/index)",
    )
    ap.add_argument(
        "--index-only",
        action="store_true",
        help="re-index from existing manifests, no scrape or discovery",
    )
    ap.add_argument(
        "--no-discover",
        action="store_true",
        help="skip seed discovery (use existing registry_seeds.json)",
    )
    args = ap.parse_args()

    # Derive pipeline stages from flags
    do_discover = not args.index_only and not args.no_discover
    do_scrape   = not args.discover_only and not args.index_only
    do_index    = not args.discover_only

    # Load overrides
    overrides = _ds._load_seed_overrides()

    # Load registry
    registry = _load_registry_seeds()
    print(f"Registry: {len(registry)} colleges loaded")

    # Filter
    if args.only:
        registry = [c for c in registry if c["college_id"] == args.only]
    elif args.pilot:
        registry = [c for c in registry if c["college_id"] in PILOT_IDS]

    if args.start_from:
        ids = [c["college_id"] for c in registry]
        if args.start_from in ids:
            idx = ids.index(args.start_from)
            registry = registry[idx:]
            print(f"Starting from {args.start_from} ({len(registry)} colleges remaining)")
        else:
            print(f"WARNING: --start-from {args.start_from!r} not found in registry")

    if not registry:
        print("No colleges matched. Exiting.")
        sys.exit(1)

    stages = []
    if do_discover: stages.append("discover")
    if do_scrape:   stages.append("scrape")
    if do_index:    stages.append("index")
    resume_note = " [resume mode]" if args.resume else ""
    print(f"Pipeline: {' -> '.join(stages)}{resume_note}")
    print(f"Colleges: {len(registry)}\n")

    started_at = datetime.now(timezone.utc)

    # Fetch Pinecone vector counts once (single API call) for resume checks.
    pinecone_counts: dict[str, int] = {}
    if args.resume and do_index:
        print("Fetching Pinecone index stats for resume check...")
        pinecone_counts = _fetch_pinecone_counts()
        already = sum(1 for c in registry if pinecone_counts.get(c["college_id"], 0) > 0)
        print(f"  {already}/{len(registry)} colleges already indexed in Pinecone\n")

    all_stats = []
    failed = []

    for i, college in enumerate(registry, 1):
        cid = college["college_id"]
        print(f"[{i}/{len(registry)}] {cid}")
        try:
            s = build_college(
                college,
                overrides,
                resume=args.resume,
                discover=do_discover,
                scrape=do_scrape,
                index=do_index,
                pinecone_counts=pinecone_counts,
            )
        except Exception as e:
            _log_error(cid, "build_college", e)
            print(f"  [FATAL ERR] {cid}: {e}")
            s = {"college_id": cid, "error": str(e), "pages": 0, "chunks": 0}
            failed.append(cid)
        all_stats.append(s)

    # ── Summary ───────────────────────────────────────────────────────────────
    elapsed = (datetime.now(timezone.utc) - started_at).total_seconds()
    total_pages  = sum(s.get("pages", 0)  for s in all_stats)
    total_chunks = sum(s.get("chunks", 0) for s in all_stats)
    skipped      = sum(1 for s in all_stats if s.get("skipped"))
    errored      = [s for s in all_stats if s.get("error")]

    print(f"\n{'='*70}")
    print("BUILD SUMMARY")
    print(f"{'='*70}")
    print(f"  Colleges processed : {len(all_stats)}")
    print(f"  Skipped (resume)   : {skipped}")
    print(f"  Errors             : {len(errored)}")
    print(f"  Total pages        : {total_pages}")
    print(f"  Total chunks       : {total_chunks}")
    print(f"  Elapsed            : {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"  Chroma dir         : {config.CHROMA_DIR}")
    if errored:
        print(f"\n  Failed colleges:")
        for s in errored:
            print(f"    {s['college_id']}: {s['error']}")
    if BUILD_ERRORS_PATH.exists():
        print(f"  Error details      : {BUILD_ERRORS_PATH}")

    # ── Write report ──────────────────────────────────────────────────────────
    report = {
        "built_at": started_at.isoformat(),
        "elapsed_sec": round(elapsed, 1),
        "total_colleges": len(all_stats),
        "skipped": skipped,
        "errors": len(errored),
        "total_pages": total_pages,
        "total_chunks": total_chunks,
        "colleges": all_stats,
    }
    BUILD_REPORT_PATH.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\n  Report -> {BUILD_REPORT_PATH}")


if __name__ == "__main__":
    main()
