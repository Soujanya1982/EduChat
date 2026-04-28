"""Central config for the multi-college RAG backend.

Everything is keyed by `college_id` — a slug built from
College Name + Business School Name (e.g. "upenn_wharton").
"""
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
RAW_HTML_DIR = DATA_DIR / "raw_html"        # data/raw_html/<college_id>/*.html
CLEAN_TEXT_DIR = DATA_DIR / "clean_text"    # data/clean_text/<college_id>/*.txt
CHROMA_DIR = DATA_DIR / "chroma"            # single persistent Chroma store
EVALS_DIR = DATA_DIR / "evals"              # data/evals/<college_id>.md
MANIFESTS_DIR = DATA_DIR / "manifests"      # data/manifests/<college_id>.json

for d in (DATA_DIR, RAW_HTML_DIR, CLEAN_TEXT_DIR, CHROMA_DIR, EVALS_DIR, MANIFESTS_DIR):
    d.mkdir(parents=True, exist_ok=True)


def raw_html_dir(college_id: str) -> Path:
    p = RAW_HTML_DIR / college_id
    p.mkdir(parents=True, exist_ok=True)
    return p


def clean_text_dir(college_id: str) -> Path:
    p = CLEAN_TEXT_DIR / college_id
    p.mkdir(parents=True, exist_ok=True)
    return p


def manifest_path(college_id: str) -> Path:
    return MANIFESTS_DIR / f"{college_id}.json"


def eval_path(college_id: str) -> Path:
    return EVALS_DIR / f"{college_id}.md"


def chroma_collection_name(college_id: str) -> str:
    # Chroma collection names must be [a-zA-Z0-9._-], 3-63 chars
    return f"{college_id}_adm"


# ── Data source ──────────────────────────────────────────────────────────────
XLSX_PATH = (
    Path(__file__).parents[1]
    / "us_undergrad_business_schools_2026_v3.xlsx"
)  # sibling to rag_backend/; will be copied here if not found
XLSX_FALLBACK = (
    Path(__file__).parents[2]
    / "Claude" / "Projects" / "CollegeApplicationResearch"
    / "us_undergrad_business_schools_2026_v3.xlsx"
)
XLSX_SHEET = "Top 100 UG Business Schools"
OVERRIDES_YAML = ROOT / "overrides.yaml"

# ── Firestore ────────────────────────────────────────────────────────────────
FIREBASE_CREDS = Path(__file__).parents[1] / "serviceAccountKey.json"
FIRESTORE_COLLECTION = "business_schools"   # keeps IPEDS's "colleges" collection untouched

# ── Crawl ────────────────────────────────────────────────────────────────────
CRAWL_SAME_PATH_ONLY = False
MAX_CRAWL_PAGES = 40           # per college
CRAWL_DELAY_SECONDS = 1.0      # polite
# Browser-like UA with EduChat identifier suffix. Still identifies us honestly
# to ops teams but avoids naive WAFs that block anything without a Mozilla/ prefix.
# We always honor robots.txt and rate-limit, so this is polite scraping.
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36 "
    "EduChat-RAG/0.2 (+contact: akshaj@educhat.local)"
)
HTTP_HEADERS = {
    "User-Agent": USER_AGENT,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

# Keywords used to decide which links to follow during crawl AND which URLs
# auto-discovery should keep from a sitemap.xml
ALLOWED_PATH_KEYWORDS = [
    "admission", "apply", "international", "cost", "aid", "financial",
    "first-year", "freshman", "transfer", "tuition", "scholarship",
    "deadline", "requirement", "criteria", "test", "sat", "act",
    "toefl", "ielts", "common-app", "coalition", "essay", "fafsa",
    "english-exam", "visa", "prepare-first-year", "undergraduate",
    "undergrad", "bba", "bs-", "ba-", "major", "prospective-students",
    "prospective", "how-to-apply", "applying", "class-profile",
    "faq", "visit", "come-visit", "tour",
]
# These "win" big bonus points in scoring — they're strong UG-admission signals
HIGH_VALUE_PATH_KEYWORDS = [
    "/bba/", "/undergraduate/", "/undergrad/", "/prospective-students",
    "/how-to-apply", "/admissions/", "/apply/", "/first-year",
    "/freshman", "/international/admissions", "/financial-aid",
]
JUNK_PATH_KEYWORDS = [
    # Generic site sections we never want
    "/news/", "/events/", "/blog/", "/calendar/", "/about/", "/research/",
    "/faculty/", "/alumni/", "/giving/", "/donate/", "/podcast/", "/video/",
    "/library/", "/login", "/logout", "/contact/", "/portal-partners/",
    "/social-impact",
    # Non-UG programs (we're indexing undergraduate only)
    "/mba/", "/emba/", "/phd/", "/doctoral/", "/grad/", "/graduate/",
    "/ms-", "/ma-", "/m.s.", "/m.a.", "/master/", "/masters/", "/executive/",
    # Study-abroad programs (about leaving the US, not admission)
    "/international-programs/", "/study-abroad/", "/global-experience",
    "/summer-faculty-led/", "/summer-programs/", "/exchange/",
    "/incoming-exchange/",
]

# ── Chunk + embed ────────────────────────────────────────────────────────────
CHUNK_SIZE_TOKENS = 300
CHUNK_OVERLAP_TOKENS = 50
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# ── LLM / retrieval ──────────────────────────────────────────────────────────
TOP_K = 5
GROQ_MODEL = "llama-3.3-70b-versatile"
GROQ_TEMPERATURE = 0.1
GROQ_MAX_TOKENS = 800

# ── Eval ─────────────────────────────────────────────────────────────────────
EVAL_PASS_THRESHOLD = 8   # PASS if >= 8/10

# ── Pinecone ──────────────────────────────────────────────────────────────────
PINECONE_INDEX_NAME = "collegmatch"   # single serverless index, one namespace per college
PINECONE_CLOUD      = "aws"
PINECONE_REGION     = "us-east-1"
EMBED_DIMENSION     = 384             # all-MiniLM-L6-v2 output dimension
