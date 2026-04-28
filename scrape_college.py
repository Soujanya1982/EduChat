"""Per-college polite crawler.

Refactor of rag_prototype/scrape.py — same crawl + clean logic, but
parameterized by a registry record (seeds, allowed_hosts) instead of
module-level constants. Writes raw HTML + cleaned text into per-college
subfolders and emits a manifest the indexer reads.

Usage:
    python scrape_college.py --pilot                 # all 5 pilots
    python scrape_college.py --only iu_kelley
    python scrape_college.py --college-id upenn_wharton  # alias
"""
import argparse
import hashlib
import json
import re
import time
import urllib.parse
import urllib.robotparser
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

import requests
from bs4 import BeautifulSoup

import config
import college_registry


# ── Helpers (same logic as prototype) ────────────────────────────────────────
def _slug(url: str) -> str:
    h = hashlib.md5(url.encode()).hexdigest()[:10]
    path = urllib.parse.urlparse(url).path.strip("/").replace("/", "_") or "root"
    return f"{path[:60]}_{h}"


_session = requests.Session()
_session.headers.update(config.HTTP_HEADERS)
_robots_cache: dict[str, urllib.robotparser.RobotFileParser | None] = {}


def _robots_ok(url: str) -> bool:
    parsed = urllib.parse.urlparse(url)
    key = f"{parsed.scheme}://{parsed.netloc}"
    if key not in _robots_cache:
        rp = urllib.robotparser.RobotFileParser()
        robots_url = f"{key}/robots.txt"
        try:
            # Fetch robots.txt ourselves so we can inspect the status code.
            # urllib.robotparser.read() treats a 403 response as "deny all",
            # but a 403 on robots.txt itself means the site didn't provide one —
            # the correct interpretation is "allow all" (same as 404 / missing).
            resp = _session.get(robots_url, timeout=10)
            if resp.status_code == 200:
                rp.set_url(robots_url)
                rp.parse(resp.text.splitlines())
            else:
                # 403, 404, 5xx on robots.txt → treat as no restrictions
                rp = None
        except Exception:
            rp = None
        _robots_cache[key] = rp
    rp = _robots_cache[key]
    return rp.can_fetch(config.USER_AGENT, url) if rp else True


# Whole-token prefix check against these — avoids false positives like
# "et-tb-has-footer" matching "footer" as a substring, which was decomposing
# entire WordPress <body> elements (Wharton, McCombs).
_JUNK_CLASS_PREFIXES = (
    "menu", "navigation", "sidebar", "breadcrumb", "cookie",
    "footer", "header", "skip-link", "social-share", "pagination",
)
_WEBVTT_TIMESTAMP_RE = re.compile(r"\d{2}:\d{2}:\d{2}\.\d{3}\s*-->\s*\d{2}:\d{2}:\d{2}\.\d{3}")


def _is_junk_classish(value) -> bool:
    """True if any class/id token equals a junk prefix or starts with '<prefix>-' / '<prefix>_'.

    Matches: "footer", "footer-main", "footer_nav", "site-footer" (no — see below).
    Does NOT match: "has-footer", "et-tb-has-footer", "no-footer-margin".
    Note: we intentionally only match *leading* tokens, so "site-footer" is kept.
    If a site uses that pattern for actual footers we'll handle it via the
    <footer> tag stripping above or the role=contentinfo stripping.
    """
    if not value:
        return False
    tokens = value if isinstance(value, list) else [value]
    for tok in tokens:
        t = str(tok).lower()
        for k in _JUNK_CLASS_PREFIXES:
            if t == k or t.startswith(k + "-") or t.startswith(k + "_"):
                return True
    return False


def _clean_html_to_text(html: str) -> tuple[str, str]:
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "noscript", "nav", "footer", "header", "form", "aside"]):
        tag.decompose()
    for attr in ("navigation", "banner", "contentinfo", "complementary"):
        for el in soup.find_all(attrs={"role": attr}):
            el.decompose()
    # Materialize the list before decomposing so we don't hit NoneType when a
    # parent match detaches a child that's still in the iterator.
    for el in list(soup.find_all(class_=_is_junk_classish)):
        if el.parent is not None:
            el.decompose()
    for el in list(soup.find_all(id=_is_junk_classish)):
        if el.parent is not None:
            el.decompose()

    title = (soup.title.string.strip() if soup.title and soup.title.string else "") or ""
    main = soup.find("article") or soup.find("main") or soup.find(id="main") or soup.body or soup
    text = main.get_text("\n", strip=True)

    lines: list[str] = []
    for ln in text.splitlines():
        ln = ln.strip()
        if not ln:
            continue
        if ln.startswith("WEBVTT") or _WEBVTT_TIMESTAMP_RE.search(ln):
            continue
        lines.append(ln)
    return title, "\n".join(lines)


def _should_crawl(url: str, allowed_hosts: set[str], seeds: list[str]) -> bool:
    parsed = urllib.parse.urlparse(url)
    if parsed.netloc not in allowed_hosts:
        return False
    if url in seeds:
        return True
    path_lower = parsed.path.lower()
    if any(bad in path_lower for bad in config.JUNK_PATH_KEYWORDS):
        return False
    return any(kw in path_lower for kw in config.ALLOWED_PATH_KEYWORDS)


def _fetch(url: str) -> str | None:
    if not _robots_ok(url):
        print(f"    [robots BLOCKED] {url}")
        return None
    try:
        r = _session.get(url, timeout=30, allow_redirects=True)
        if r.status_code == 200:
            ctype = (r.headers.get("content-type") or "").lower()
            if "html" in ctype or "xml" in ctype:
                return r.text
            print(f"    [skip non-html ({ctype})] {url}")
            return None
        print(f"    [HTTP {r.status_code}] {url}")
    except requests.RequestException as e:
        print(f"    [ERR {type(e).__name__}] {url}")
    return None


# ── Per-college crawl ────────────────────────────────────────────────────────
def scrape_college(college: dict, max_pages: int | None = None) -> dict:
    """Crawl one college from its registry record. Returns stats dict."""
    cid = college["college_id"]
    seeds = college["seeds"]
    allowed_hosts = set(college["allowed_hosts"])
    max_pages = max_pages or config.MAX_CRAWL_PAGES

    raw_dir = config.raw_html_dir(cid)
    clean_dir = config.clean_text_dir(cid)
    manifest_path = config.manifest_path(cid)

    visited: set[str] = set()
    queue: deque[str] = deque(seeds)
    manifest: list[dict] = []
    started = time.time()

    print(f"[scrape] {cid} — seeds={len(seeds)}, hosts={sorted(allowed_hosts)}, cap={max_pages}")

    while queue and len(visited) < max_pages:
        url = queue.popleft()
        url = url.split("#")[0].rstrip("/")
        if url in visited:
            continue
        visited.add(url)

        print(f"  [{len(visited)}/{max_pages}] {url}")
        html = _fetch(url)
        time.sleep(config.CRAWL_DELAY_SECONDS)
        if html is None:
            continue

        slug = _slug(url)
        (raw_dir / f"{slug}.html").write_text(html, encoding="utf-8")
        title, text = _clean_html_to_text(html)
        (clean_dir / f"{slug}.txt").write_text(
            f"URL: {url}\nTITLE: {title}\n\n{text}", encoding="utf-8"
        )
        manifest.append({"url": url, "title": title, "slug": slug, "chars": len(text)})

        # Enqueue outgoing links (BFS)
        soup = BeautifulSoup(html, "lxml")
        for a in soup.find_all("a", href=True):
            nxt = urllib.parse.urljoin(url, a["href"]).split("#")[0].rstrip("/")
            if nxt not in visited and _should_crawl(nxt, allowed_hosts, seeds):
                queue.append(nxt)

    elapsed = time.time() - started
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    stats = {
        "college_id": cid,
        "pages": len(manifest),
        "elapsed_sec": round(elapsed, 1),
        "scraped_at": datetime.now(timezone.utc).isoformat(),
    }
    print(f"  -> {len(manifest)} pages in {elapsed:.1f}s -> {manifest_path}")
    return stats


# ── Re-clean (no network) ────────────────────────────────────────────────────
def reclean_college(college: dict) -> dict:
    """Re-run _clean_html_to_text over existing raw HTML without re-fetching.

    Uses the existing manifest for url/title/slug mapping so we keep stable
    filenames. Rewrites clean_text/<cid>/<slug>.txt and the manifest's `chars`.
    """
    cid = college["college_id"]
    raw_dir = config.raw_html_dir(cid)
    clean_dir = config.clean_text_dir(cid)
    manifest_path = config.manifest_path(cid)

    if not manifest_path.exists():
        print(f"[reclean] {cid}: no manifest at {manifest_path} — skip")
        return {"college_id": cid, "pages": 0, "skipped": True}

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not manifest:
        print(f"[reclean] {cid}: empty manifest — skip")
        return {"college_id": cid, "pages": 0, "skipped": True}

    updated = 0
    missing = 0
    total_chars_before = 0
    total_chars_after = 0
    for entry in manifest:
        slug = entry["slug"]
        url = entry["url"]
        html_path = raw_dir / f"{slug}.html"
        if not html_path.exists():
            missing += 1
            continue
        html = html_path.read_text(encoding="utf-8")
        title, text = _clean_html_to_text(html)
        chars_before = entry.get("chars", 0) or 0
        total_chars_before += chars_before
        total_chars_after += len(text)
        # Prefer the freshly-parsed title but fall back to manifest value if blank.
        title_out = title or entry.get("title") or ""
        (clean_dir / f"{slug}.txt").write_text(
            f"URL: {url}\nTITLE: {title_out}\n\n{text}", encoding="utf-8"
        )
        entry["title"] = title_out
        entry["chars"] = len(text)
        updated += 1

    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    delta = total_chars_after - total_chars_before
    print(f"[reclean] {cid}: {updated} pages recleaned "
          f"({missing} raw HTML missing), chars {total_chars_before}->{total_chars_after} "
          f"(delta {delta:+})")
    return {
        "college_id": cid,
        "pages": updated,
        "missing": missing,
        "chars_before": total_chars_before,
        "chars_after": total_chars_after,
    }


# ── Main ─────────────────────────────────────────────────────────────────────
PILOT_IDS = [
    "iu_kelley", "upenn_wharton", "nyu_stern", "uva_mcintire", "utaustin_mccombs",
]


def _load_seeds_registry() -> list[dict]:
    """Prefer registry_seeds.json (after discover_seeds.py); fall back to bare registry."""
    seeds_json = config.DATA_DIR / "registry_seeds.json"
    if seeds_json.exists():
        return json.loads(seeds_json.read_text(encoding="utf-8"))
    print("  WARN: registry_seeds.json not found, using bare registry (only admission_url as seed)")
    return college_registry.load_local()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true", help="all 5 pilot colleges")
    ap.add_argument("--only", help="single college_id")
    ap.add_argument("--college-id", dest="only", help="alias for --only")
    ap.add_argument("--max-pages", type=int, default=None, help="override config.MAX_CRAWL_PAGES")
    ap.add_argument(
        "--reclean",
        action="store_true",
        help="re-run HTML->text cleaner over existing raw HTML (no network, no re-scrape)",
    )
    args = ap.parse_args()

    registry = _load_seeds_registry()
    if args.only:
        registry = [c for c in registry if c["college_id"] == args.only]
    elif args.pilot:
        registry = [c for c in registry if c["college_id"] in PILOT_IDS]
    else:
        ap.error("specify --pilot or --only/--college-id")

    if not registry:
        print("No colleges matched filter.")
        return

    if args.reclean:
        print(f"Re-cleaning {len(registry)} colleges (no network)...\n")
        all_stats = []
        for i, c in enumerate(registry, 1):
            print(f"=== {i}/{len(registry)} ===")
            all_stats.append(reclean_college(c))
        print("\nSummary:")
        for s in all_stats:
            if s.get("skipped"):
                print(f"  {s['college_id']}: SKIPPED")
                continue
            print(f"  {s['college_id']}: {s['pages']} pages, "
                  f"chars {s['chars_before']}->{s['chars_after']}")
        return

    print(f"Scraping {len(registry)} colleges...\n")
    all_stats = []
    for i, c in enumerate(registry, 1):
        print(f"=== {i}/{len(registry)} ===")
        all_stats.append(scrape_college(c, max_pages=args.max_pages))
        print()

    print("Summary:")
    for s in all_stats:
        print(f"  {s['college_id']}: {s['pages']} pages in {s['elapsed_sec']}s")
    total_pages = sum(s["pages"] for s in all_stats)
    total_time = sum(s["elapsed_sec"] for s in all_stats)
    print(f"  TOTAL: {total_pages} pages in {total_time:.1f}s")


if __name__ == "__main__":
    main()
