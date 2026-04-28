"""Hybrid seed discovery: YAML override > sitemap.xml > 1-hop link crawl.

Given a registry record with just the bootstrap admission_url, expand it into
~5-10 deep admission-related URLs that the scraper should start from.

Strategy (first success wins):
  1. overrides.yaml :: seeds[college_id]        → use verbatim, skip discovery
  2. sitemap.xml / sitemap_index.xml            → filter by keyword, dedupe, cap 10
  3. 1-hop link crawl from admission_url       → same filter + cap

Output: rag_backend/data/registry_seeds.json — same schema as registry.json but
with `seeds` and `allowed_hosts` populated. Used by scrape_college.py.

Usage:
    python discover_seeds.py                   # all 115 colleges
    python discover_seeds.py --pilot           # just the 5 pilot schools
    python discover_seeds.py --only iu_kelley  # one
    python discover_seeds.py --dry-run         # don't save, just print
"""
import argparse
import json
import re
import time
import urllib.parse
import urllib.robotparser
import xml.etree.ElementTree as ET
from pathlib import Path

import requests
import yaml
from bs4 import BeautifulSoup

import config
import college_registry


REGISTRY_SEEDS = config.DATA_DIR / "registry_seeds.json"
MAX_SEEDS_PER_COLLEGE = 10
PILOT_IDS = [
    "iu_kelley",           # baseline (override)
    "upenn_wharton",       # private elite
    "nyu_stern",           # private urban
    "uva_mcintire",        # public flagship South
    "utaustin_mccombs",    # public flagship large-state
]
# Note: umich_ross was originally in the pilot but its site returns 403 under
# any UA (Cloudflare Bot-Fight). Requires Selenium/Playwright; defer to Phase 2.


# ── YAML overrides ───────────────────────────────────────────────────────────
def _load_seed_overrides() -> dict:
    if not config.OVERRIDES_YAML.exists():
        return {}
    data = yaml.safe_load(config.OVERRIDES_YAML.read_text(encoding="utf-8")) or {}
    return data.get("seeds", {}) or {}


# ── URL scoring / filters ────────────────────────────────────────────────────
def _score_url(url: str) -> int:
    """Higher = more likely UG-admission-relevant. Negative = junk."""
    u = url.lower()
    path = urllib.parse.urlparse(u).path
    score = 0
    # Big positive: high-value UG-admission tokens
    for kw in config.HIGH_VALUE_PATH_KEYWORDS:
        if kw in path:
            score += 6
    # Smaller positive: any allowed keyword
    for kw in config.ALLOWED_PATH_KEYWORDS:
        if kw in path:
            score += 1
    # Big negative: junk / non-UG / study-abroad
    for junk in config.JUNK_PATH_KEYWORDS:
        if junk in path:
            score -= 10
    # Prefer shorter paths (landing pages) over deep ones
    depth = path.count("/")
    score += max(0, 5 - depth)
    # Penalize file extensions other than html
    if re.search(r"\.(pdf|jpg|png|gif|zip|docx?|xlsx?)$", u):
        score -= 3
    return score


def _is_admission_url(url: str) -> bool:
    u = url.lower()
    path = urllib.parse.urlparse(u).path
    # Reject junk first
    if any(j in path for j in config.JUNK_PATH_KEYWORDS):
        return False
    # Must contain at least one allowed keyword
    if not any(kw in path for kw in config.ALLOWED_PATH_KEYWORDS):
        return False
    # Final guard: discard anything still ranking as net-negative after junk check
    return _score_url(url) > 0


def _same_org(host_a: str, host_b: str) -> bool:
    """Heuristic: same organization if last 2 DNS labels match (e.g. iu.edu)."""
    def tail(h: str) -> str:
        parts = h.lower().split(".")
        return ".".join(parts[-2:]) if len(parts) >= 2 else h.lower()
    return tail(host_a) == tail(host_b)


# ── HTTP helpers ─────────────────────────────────────────────────────────────
_session = requests.Session()
_session.headers.update(config.HTTP_HEADERS)
_robots_cache: dict[str, urllib.robotparser.RobotFileParser] = {}
FETCH_LOG_VERBOSE = False   # toggled on by main() when --verbose


def _robots_ok(url: str) -> bool:
    parsed = urllib.parse.urlparse(url)
    key = f"{parsed.scheme}://{parsed.netloc}"
    if key not in _robots_cache:
        rp = urllib.robotparser.RobotFileParser()
        try:
            rp.set_url(f"{key}/robots.txt")
            rp.read()
        except Exception:
            rp = None
        _robots_cache[key] = rp
    rp = _robots_cache[key]
    return rp.can_fetch(config.USER_AGENT, url) if rp else True


def _fetch(url: str, timeout: int = 15) -> str | None:
    if not _robots_ok(url):
        if FETCH_LOG_VERBOSE:
            print(f"      [fetch BLOCKED by robots.txt] {url}")
        return None
    try:
        r = _session.get(url, timeout=timeout, allow_redirects=True)
        if r.status_code == 200:
            return r.text
        if FETCH_LOG_VERBOSE:
            print(f"      [fetch {r.status_code}] {url}")
    except requests.RequestException as e:
        if FETCH_LOG_VERBOSE:
            print(f"      [fetch ERR {type(e).__name__}] {url}")
        return None
    return None


# ── Strategy 2: sitemap.xml ──────────────────────────────────────────────────
_SITEMAP_NS = "{http://www.sitemaps.org/schemas/sitemap/0.9}"


def _extract_sitemap_urls(xml_text: str, depth: int = 0, max_depth: int = 2) -> list[str]:
    """Parse a sitemap.xml or sitemap_index.xml recursively (1 level of nesting)."""
    if depth > max_depth:
        return []
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return []
    urls: list[str] = []
    # sitemapindex → sub-sitemaps
    for sm in root.findall(f"{_SITEMAP_NS}sitemap"):
        loc = sm.find(f"{_SITEMAP_NS}loc")
        if loc is not None and loc.text:
            sub = _fetch(loc.text)
            if sub:
                urls.extend(_extract_sitemap_urls(sub, depth + 1, max_depth))
                time.sleep(config.CRAWL_DELAY_SECONDS)
    # urlset → actual URLs
    for u in root.findall(f"{_SITEMAP_NS}url"):
        loc = u.find(f"{_SITEMAP_NS}loc")
        if loc is not None and loc.text:
            urls.append(loc.text.strip())
    return urls


def _try_sitemap(admission_url: str) -> list[str]:
    parsed = urllib.parse.urlparse(admission_url)
    base = f"{parsed.scheme}://{parsed.netloc}"
    for candidate in (f"{base}/sitemap.xml", f"{base}/sitemap_index.xml"):
        xml = _fetch(candidate)
        if not xml:
            continue
        urls = _extract_sitemap_urls(xml)
        if urls:
            return urls
    return []


# ── Strategy 3: 1-hop crawl ──────────────────────────────────────────────────
def _one_hop_links(admission_url: str) -> list[str]:
    html = _fetch(admission_url)
    if not html:
        return []
    soup = BeautifulSoup(html, "lxml")
    base_host = urllib.parse.urlparse(admission_url).netloc
    urls: list[str] = []
    for a in soup.find_all("a", href=True):
        href = urllib.parse.urljoin(admission_url, a["href"]).split("#")[0].rstrip("/")
        host = urllib.parse.urlparse(href).netloc
        if not host:
            continue
        # Keep same org (same 2-label domain) — catches subdomains like
        # admissions.virginia.edu ↔ commerce.virginia.edu
        if _same_org(host, base_host):
            urls.append(href)
    return urls


# ── Main discovery ───────────────────────────────────────────────────────────
def discover_for(college: dict, overrides: dict | None = None, verbose: bool = True) -> dict:
    """Returns a new dict with `seeds` and `allowed_hosts` populated.
    Also adds `seeds_source` ∈ {"override","sitemap","onehop","bootstrap"} for provenance.
    """
    overrides = overrides if overrides is not None else _load_seed_overrides()
    out = dict(college)
    cid = college["college_id"]
    admission_url = college["admission_url"]

    # Strategy 1: override
    if cid in overrides:
        ov = overrides[cid]
        out["seeds"] = list(ov.get("seeds", [admission_url]))
        out["allowed_hosts"] = list(ov.get("allowed_hosts", [urllib.parse.urlparse(admission_url).netloc]))
        out["seeds_source"] = "override"
        if verbose:
            print(f"  [override] {cid}: {len(out['seeds'])} seeds across {len(out['allowed_hosts'])} hosts")
        return out

    # Strategy 2: sitemap
    sitemap_urls = _try_sitemap(admission_url)
    time.sleep(config.CRAWL_DELAY_SECONDS)
    candidates = [u for u in sitemap_urls if _is_admission_url(u)]
    source = "sitemap"

    # Strategy 3: fallback to 1-hop crawl
    if len(candidates) < 3:
        if verbose and sitemap_urls:
            print(f"  [sitemap weak] {cid}: only {len(candidates)} admission URLs found, falling back to 1-hop")
        hop = _one_hop_links(admission_url)
        time.sleep(config.CRAWL_DELAY_SECONDS)
        candidates = list(dict.fromkeys(candidates + [u for u in hop if _is_admission_url(u)]))
        source = "onehop" if len(candidates) <= 3 else "sitemap+onehop"

    if not candidates:
        if verbose:
            print(f"  [bootstrap only] {cid}: no extra URLs discovered, using admission_url only")
        out["seeds"] = [admission_url]
        out["allowed_hosts"] = [urllib.parse.urlparse(admission_url).netloc]
        out["seeds_source"] = "bootstrap"
        return out

    # Always include the admission URL as the first seed
    candidates = [admission_url] + [u for u in candidates if u.rstrip("/") != admission_url.rstrip("/")]

    # Dedupe (case-insensitive), score, rank
    seen_lower: set[str] = set()
    uniq = []
    for u in candidates:
        key = u.lower().rstrip("/")
        if key not in seen_lower:
            seen_lower.add(key)
            uniq.append(u)
    uniq.sort(key=lambda u: -_score_url(u))

    top = uniq[:MAX_SEEDS_PER_COLLEGE]
    hosts = sorted({urllib.parse.urlparse(u).netloc for u in top})

    out["seeds"] = top
    out["allowed_hosts"] = hosts
    out["seeds_source"] = source
    if verbose:
        print(f"  [{source}] {cid}: {len(top)} seeds across {len(hosts)} host(s)")
    return out


def save_enriched(colleges: list[dict]):
    REGISTRY_SEEDS.write_text(
        json.dumps(colleges, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    print(f"\nEnriched registry -> {REGISTRY_SEEDS}")


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true", help="only the pilot colleges")
    ap.add_argument("--only", help="single college_id")
    ap.add_argument("--dry-run", action="store_true", help="don't write registry_seeds.json")
    ap.add_argument("--verbose", action="store_true", help="log every HTTP failure")
    args = ap.parse_args()

    global FETCH_LOG_VERBOSE
    FETCH_LOG_VERBOSE = args.verbose

    registry = college_registry.load_local()
    if args.only:
        registry = [c for c in registry if c["college_id"] == args.only]
    elif args.pilot:
        registry = [c for c in registry if c["college_id"] in PILOT_IDS]

    if not registry:
        print("No colleges matched filter.")
        return

    overrides = _load_seed_overrides()
    print(f"Discovering seeds for {len(registry)} colleges...")
    print(f"  overrides.yaml pins: {len(overrides)} college(s)")
    print()

    enriched: list[dict] = []
    for i, c in enumerate(registry, 1):
        print(f"[{i}/{len(registry)}] {c['college_id']}  ({c['display_name']})")
        enriched.append(discover_for(c, overrides))

    # Summary
    from collections import Counter
    sources = Counter(c.get("seeds_source") for c in enriched)
    total_seeds = sum(len(c.get("seeds", [])) for c in enriched)
    print(f"\nSummary:")
    print(f"  Colleges:     {len(enriched)}")
    print(f"  Total seeds:  {total_seeds}  (avg {total_seeds/len(enriched):.1f} per college)")
    print(f"  Sources:      {dict(sources)}")

    bootstrap_only = [c["college_id"] for c in enriched if c.get("seeds_source") == "bootstrap"]
    if bootstrap_only:
        print(f"  WARN: bootstrap-only (only admission URL, no expansion): {bootstrap_only}")

    if args.dry_run:
        print("\nDry run — not saving.")
        return

    # When running for a subset, merge into existing registry_seeds.json instead of overwriting
    if args.only or args.pilot:
        existing = []
        if REGISTRY_SEEDS.exists():
            existing = json.loads(REGISTRY_SEEDS.read_text(encoding="utf-8"))
        by_id = {c["college_id"]: c for c in existing}
        for c in enriched:
            by_id[c["college_id"]] = c
        save_enriched(list(by_id.values()))
    else:
        save_enriched(enriched)


if __name__ == "__main__":
    main()
