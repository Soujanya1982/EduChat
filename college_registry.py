"""Load the 116 business schools from xlsx, build college records, optionally push to Firestore.

Usage:
    python college_registry.py --inspect        # show 3 sample records + summary stats, no writes
    python college_registry.py --dry-run        # write local JSON backup only
    python college_registry.py                  # write local JSON + upsert to Firestore
    python college_registry.py --only upenn_wharton  # just one college (for testing)

Each college record shape (shared with the prototype's config.py fields so
scrape_college.py / index_college.py can consume it directly):

    {
        "college_id":          "upenn_wharton",
        "college_name":        "University of Pennsylvania",
        "business_school":     "Wharton School",
        "display_name":        "Wharton School (University of Pennsylvania)",
        "location":            "Philadelphia, PA",
        "state":               "PA",
        "region":              "Northeast",
        "rank_raw":            "1",              # preserved from xlsx (handles "NR", "5 (tied)")
        "rank_numeric":        1,                # nullable int, for sorting / filtering
        "admission_url":       "https://...",
        "seeds":               ["https://..."],  # filled in later by discover_seeds.py
        "allowed_hosts":       ["admissions.wharton.upenn.edu"],
        "index_status":        "registered",     # registered -> scraped -> indexed -> evaluated
        "eval_score":          None,
        "last_scraped_at":     None,
        "last_indexed_at":     None,
    }
"""
import argparse
import json
import re
import sys
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path

import openpyxl
import yaml

import config


LOCAL_BACKUP = config.DATA_DIR / "registry.json"


# ── Slug / ID builders ───────────────────────────────────────────────────────
_COLLEGE_ALIAS = {
    "University of Pennsylvania": "upenn",
    "University of Virginia": "uva",
    "Cornell University": "cornell",
    "University of California-Berkeley": "ucberkeley",
    "University of California--Berkeley": "ucberkeley",
    "University of Michigan": "umich",
    "University of Michigan-Ann Arbor": "umich",
    "University of Texas at Austin": "utaustin",
    "University of Texas-Austin": "utaustin",
    "New York University": "nyu",
    "University of North Carolina": "unc",
    "University of North Carolina at Chapel Hill": "unc",
    "University of Southern California": "usc",
    "Carnegie Mellon University": "cmu",
    "Massachusetts Institute of Technology": "mit",
    "Georgia Institute of Technology": "gatech",
    "Indiana University": "iu",
    "Indiana University Bloomington": "iu",
}


def _short(name: str) -> str:
    """Short slug for a college name. Uses alias table when available."""
    if not name:
        return ""
    name = name.strip()
    if name in _COLLEGE_ALIAS:
        return _COLLEGE_ALIAS[name]
    # strip trailing " University", " College", descriptors
    s = re.sub(r"\s+(University|College|Institute|School)\b.*$", "", name, flags=re.I)
    s = re.sub(r"[^a-zA-Z0-9]+", "_", s).strip("_").lower()
    return s or "college"


def _school_short(bs_name: str) -> str:
    """Short slug for a business-school name: e.g. 'Wharton School' -> 'wharton'."""
    if not bs_name:
        return "business"
    # keep only first token that isn't 'School/College/Institute/of/the'
    stop = {"school", "college", "institute", "of", "the", "for", "and", "&"}
    tokens = re.findall(r"[A-Za-z0-9]+", bs_name)
    first = next((t for t in tokens if t.lower() not in stop), tokens[0] if tokens else "business")
    return first.lower()


def build_college_id(college_name: str, bs_name: str) -> str:
    cs = _short(college_name)
    bs = _school_short(bs_name)
    # avoid duplication if school short == college short (e.g. "mit" + "sloan" fine, "nyu"+"stern" fine)
    return f"{cs}_{bs}"


def _rank_numeric(rank_raw) -> int | None:
    if isinstance(rank_raw, int):
        return rank_raw
    if isinstance(rank_raw, str):
        m = re.match(r"\s*(\d+)", rank_raw)
        if m:
            return int(m.group(1))
    return None


def _allowed_host(url: str) -> str:
    return urllib.parse.urlparse(url).netloc


# ── Xlsx loader ──────────────────────────────────────────────────────────────
def resolve_xlsx() -> Path:
    if config.XLSX_PATH.exists():
        return config.XLSX_PATH
    if config.XLSX_FALLBACK.exists():
        return config.XLSX_FALLBACK
    raise SystemExit(
        f"xlsx not found in either location:\n"
        f"  {config.XLSX_PATH}\n  {config.XLSX_FALLBACK}"
    )


def load_from_xlsx() -> list[dict]:
    xlsx = resolve_xlsx()
    print(f"Loading: {xlsx}")
    wb = openpyxl.load_workbook(xlsx, read_only=True, data_only=True)
    ws = wb[config.XLSX_SHEET]
    rows = list(ws.iter_rows(min_row=2, values_only=True))

    now = datetime.now(timezone.utc).isoformat()
    colleges: list[dict] = []
    seen_ids: set[str] = set()

    for r in rows:
        college_name, bs_name, location, state, region, rank_raw, url = r
        if not url:
            print(f"  SKIP (no URL): {college_name} / {bs_name}")
            continue

        college_id = build_college_id(college_name or "", bs_name or "")
        # Disambiguate if slug collides
        base_id = college_id
        n = 2
        while college_id in seen_ids:
            college_id = f"{base_id}_{n}"
            n += 1
        seen_ids.add(college_id)

        colleges.append({
            "college_id":       college_id,
            "college_name":     college_name,
            "business_school":  bs_name,
            "display_name":     f"{bs_name} ({college_name})" if bs_name else college_name,
            "location":         location,
            "state":            state,
            "region":           region,
            "rank_raw":         str(rank_raw) if rank_raw is not None else None,
            "rank_numeric":     _rank_numeric(rank_raw),
            "admission_url":    url,
            "seeds":            [url],                     # bootstrap; discover_seeds.py will expand
            "allowed_hosts":    [_allowed_host(url)],
            "index_status":     "registered",
            "eval_score":       None,
            "last_scraped_at":  None,
            "last_indexed_at":  None,
            "registered_at":    now,
            "source":           "xlsx",
        })

    # Merge in extra_colleges from overrides.yaml (e.g. Kelley as pilot baseline)
    extras = _load_extra_colleges()
    existing_ids = {c["college_id"] for c in colleges}
    for extra in extras:
        if extra["college_id"] in existing_ids:
            print(f"  SKIP extra (duplicate id): {extra['college_id']}")
            continue
        # Fill in defaults so the record matches the xlsx schema
        extra.setdefault("seeds", [extra["admission_url"]])
        extra.setdefault("allowed_hosts", [_allowed_host(extra["admission_url"])])
        extra.setdefault("index_status", "registered")
        extra.setdefault("eval_score", None)
        extra.setdefault("last_scraped_at", None)
        extra.setdefault("last_indexed_at", None)
        extra.setdefault("registered_at", now)
        extra.setdefault("source", "overrides.yaml")
        colleges.append(extra)

    return colleges


def _load_extra_colleges() -> list[dict]:
    if not config.OVERRIDES_YAML.exists():
        return []
    data = yaml.safe_load(config.OVERRIDES_YAML.read_text(encoding="utf-8")) or {}
    return data.get("extra_colleges", []) or []


# ── Firestore upload ─────────────────────────────────────────────────────────
def upload_to_firestore(colleges: list[dict]):
    try:
        import firebase_admin
        from firebase_admin import credentials, firestore
    except ImportError:
        print("\nERROR: firebase-admin not installed. Run: pip install firebase-admin")
        sys.exit(1)

    if not config.FIREBASE_CREDS.exists():
        print(f"\nERROR: Firebase service account key not found: {config.FIREBASE_CREDS}")
        print("Drop serviceAccountKey.json into the CollegeMatch/ folder, then re-run.")
        sys.exit(1)

    print(f"\nUpserting {len(colleges)} colleges to Firestore collection '{config.FIRESTORE_COLLECTION}'...")
    if not firebase_admin._apps:
        cred = credentials.Certificate(str(config.FIREBASE_CREDS))
        firebase_admin.initialize_app(cred)
    db = firestore.client()
    col_ref = db.collection(config.FIRESTORE_COLLECTION)

    batch = db.batch()
    for i, c in enumerate(colleges):
        doc_ref = col_ref.document(c["college_id"])
        batch.set(doc_ref, c, merge=True)  # merge so we don't clobber seeds/scores set later
        if (i + 1) % 400 == 0:
            batch.commit()
            batch = db.batch()
            print(f"  Committed {i + 1} / {len(colleges)}")
    batch.commit()
    print(f"  Committed {len(colleges)} / {len(colleges)}")


def save_backup(colleges: list[dict]):
    LOCAL_BACKUP.write_text(
        json.dumps(colleges, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    print(f"Local backup -> {LOCAL_BACKUP}")


# ── Query helpers (used by other modules) ────────────────────────────────────
def load_local() -> list[dict]:
    """Read the registry JSON backup. Used by discover_seeds.py, scrape_college.py, etc.
    Prefers local JSON to avoid a Firestore round-trip on every build step.
    """
    if not LOCAL_BACKUP.exists():
        raise SystemExit(
            f"No local registry at {LOCAL_BACKUP}. Run: python college_registry.py --dry-run"
        )
    return json.loads(LOCAL_BACKUP.read_text(encoding="utf-8"))


def get_college(college_id: str) -> dict:
    for c in load_local():
        if c["college_id"] == college_id:
            return c
    raise KeyError(f"college_id not found in registry: {college_id}")


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inspect", action="store_true", help="print summary stats + 3 samples, no writes")
    ap.add_argument("--dry-run", action="store_true", help="write local JSON backup only (no Firestore)")
    ap.add_argument("--only", help="filter to a single college_id")
    args = ap.parse_args()

    colleges = load_from_xlsx()
    if args.only:
        colleges = [c for c in colleges if c["college_id"] == args.only]
        if not colleges:
            print(f"No match for --only {args.only}")
            sys.exit(1)

    # Summary
    from collections import Counter
    print(f"\nLoaded {len(colleges)} colleges")
    print(f"  Regions: {dict(Counter(c['region'] for c in colleges))}")
    ranked = [c for c in colleges if c["rank_numeric"] is not None]
    if ranked:
        print(f"  Rank range: {min(c['rank_numeric'] for c in ranked)} to {max(c['rank_numeric'] for c in ranked)}  (ranked={len(ranked)}, NR={len(colleges) - len(ranked)})")
    print(f"  Unique admission hosts: {len(set(c['allowed_hosts'][0] for c in colleges))}")

    # Show 3 samples
    print("\nSample records:")
    for c in colleges[:3]:
        print(f"  - {c['college_id']}: rank={c['rank_raw']}  {c['display_name']}  ->{c['admission_url']}")

    # Show the 5 pilot schools if present
    pilot_ids = {"iu_kelley", "upenn_wharton", "nyu_stern", "umich_ross", "utaustin_mccombs"}
    pilots_in = [c for c in colleges if c["college_id"] in pilot_ids]
    if pilots_in:
        print("\nPilot matches:")
        for c in pilots_in:
            print(f"  - {c['college_id']}: rank={c['rank_raw']}  ->{c['admission_url']}")

    if args.inspect:
        return

    save_backup(colleges)

    if args.dry_run:
        print("\nDry run complete. No Firestore write.")
        return

    upload_to_firestore(colleges)
    print("\nDone.")


if __name__ == "__main__":
    main()
