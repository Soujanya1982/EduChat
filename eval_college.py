"""Per-college RAG quality evaluator.

Asks a fixed set of 10 admission questions for each college via rag.ask(),
auto-scores each answer 0-10 using Groq as LLM judge, and writes a markdown
report to data/evals/<college_id>.md.

PASS = average score >= config.EVAL_PASS_THRESHOLD (default 8).

Token-budget design:
  - Answer generation uses config.GROQ_MODEL (70b) for quality.
  - Judging uses JUDGE_MODEL (8b-instant) — much higher free-tier limit.
  - Each completed question is checkpointed to data/evals/<cid>.jsonl so
    a rate-limit crash mid-eval doesn't wipe progress; re-running resumes
    from where it left off (use --no-cache to start fresh).

Usage:
    python eval_college.py --pilot
    python eval_college.py --only utaustin_mccombs
    python eval_college.py --only iu_kelley --no-cache   # start fresh
"""
import argparse
import io
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Force UTF-8 on Windows so emoji/arrows in print() don't crash cp1252 terminals.
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import config
from rag import RAG, deep_link, snippet

# Use a small fast model for judging — it only needs to output a single digit
# and has a 500k token/day free-tier limit vs. the 70b's 100k limit.
JUDGE_MODEL = "llama-3.1-8b-instant"


# ── Standard 10-question battery ─────────────────────────────────────────────
EVAL_QUESTIONS = [
    "What is the application deadline for first-year (freshman) applicants?",
    "What standardized tests are required or accepted (SAT, ACT)?",
    "What GPA do I need to be a competitive applicant?",
    "How do I submit my application — Common App, Coalition App, or a school-specific portal?",
    "What scholarships or financial aid are available to incoming freshmen?",
    "What is the tuition and total estimated cost of attendance?",
    "What does the admitted student class profile look like (average GPA, test scores, class size)?",
    "What majors or concentrations are offered in the undergraduate business program?",
    "What are the requirements for international students applying as freshmen?",
    "Is there an early decision or early action option, and what are those deadlines?",
]


# ── LLM-as-judge prompts ──────────────────────────────────────────────────────
JUDGE_SYSTEM = """You are evaluating the quality of an AI answer to a college admission question.
Score the answer on a scale of 0 to 10 using this rubric:

10  — Complete, specific answer with at least one inline citation [S#]; directly answers the question.
8-9 — Good answer with specific details; may lack a citation or miss a minor sub-point.
5-7 — Partial answer: addresses the question but is vague, incomplete, or missing citations.
2-4 — Mostly says "I don't have that information" or gives an extremely brief non-answer.
0-1 — Hallucination, factually wrong, or completely off-topic.

Return ONLY a single integer from 0 to 10. No explanation, no punctuation."""

JUDGE_USER = """QUESTION: {question}

ANSWER: {answer}

Score (0-10):"""


# ── Checkpoint helpers ────────────────────────────────────────────────────────
def _ckpt_path(cid: str) -> Path:
    return config.EVALS_DIR / f"{cid}.jsonl"


def _load_checkpoint(cid: str) -> dict[int, dict]:
    """Return {question_index: result_dict} for already-completed questions."""
    p = _ckpt_path(cid)
    done = {}
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                done[entry["idx"]] = entry
            except Exception:
                pass
    return done


def _save_checkpoint(cid: str, idx: int, entry: dict):
    """Append one question result to the checkpoint file."""
    p = _ckpt_path(cid)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps({**entry, "idx": idx}, ensure_ascii=False) + "\n")


def _clear_checkpoint(cid: str):
    p = _ckpt_path(cid)
    if p.exists():
        p.unlink()


# ── Scoring ───────────────────────────────────────────────────────────────────
def _judge_score(rag: RAG, question: str, answer: str) -> int:
    """Ask Groq (small model) to score one Q/A pair. Returns int 0-10."""
    completion = RAG._groq.chat.completions.create(
        model=JUDGE_MODEL,
        temperature=0.0,
        max_tokens=4,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": JUDGE_USER.format(question=question, answer=answer)},
        ],
    )
    raw = completion.choices[0].message.content.strip()
    m = re.search(r"\d+", raw)
    score = int(m.group()) if m else 0
    return max(0, min(10, score))


def _is_daily_limit(exc: Exception) -> bool:
    """True if this 429 is the 24-hour daily cap (not retryable tonight)."""
    return "tokens per day" in str(exc)


def _ask_with_retry(rag: RAG, cid: str, q: str, display: str,
                    max_retries: int = 3) -> tuple:
    """rag.ask() with exponential backoff on hourly 429 errors.
    Raises on daily-limit 429 (those won't recover until midnight UTC).
    """
    for attempt in range(max_retries + 1):
        try:
            r = rag.ask(cid, q, college_display=display)
            return r["answer"], r["sources"]
        except Exception as e:
            if "429" in str(e):
                if _is_daily_limit(e):
                    print(f"       [DAILY LIMIT] {e}")
                    return f"[RATE LIMITED: {e}]", []
                wait = 60 * (2 ** attempt)
                print(f"       [429 hourly, retry in {wait}s] attempt {attempt+1}/{max_retries}")
                time.sleep(wait)
            else:
                print(f"       [ERR] {e}")
                return f"[ERROR: {e}]", []
    return "[MAX RETRIES EXCEEDED]", []


def _judge_with_retry(rag: RAG, q: str, answer: str,
                      max_retries: int = 3) -> int:
    """_judge_score() with exponential backoff on hourly 429 errors."""
    for attempt in range(max_retries + 1):
        try:
            return _judge_score(rag, q, answer)
        except Exception as e:
            if "429" in str(e):
                if _is_daily_limit(e):
                    print(f"       [JUDGE DAILY LIMIT]")
                    return 0
                wait = 30 * (2 ** attempt)
                print(f"       [JUDGE 429 hourly, retry in {wait}s]")
                time.sleep(wait)
            else:
                print(f"       [JUDGE ERR] {e}")
                return 0
    return 0


# ── Per-college eval ──────────────────────────────────────────────────────────
def eval_college(college: dict, rag: RAG, force: bool = False) -> dict:
    """Run all 10 eval questions for one college. Returns summary dict.

    Resumes from checkpoint (.jsonl) if a previous run was interrupted.
    Use force=True to clear the checkpoint and start fresh.
    """
    cid = college["college_id"]
    display = college.get("display_name") or cid
    out_path = config.eval_path(cid)

    # If a complete .md exists and no force flag, skip entirely.
    if out_path.exists() and not force:
        print(f"  [skip] {cid}: eval already exists (use --no-cache to re-run)")
        text = out_path.read_text(encoding="utf-8")
        m = re.search(r"Average score.*?(\d+(?:\.\d+)?)\s*/\s*10", text)
        avg = float(m.group(1)) if m else None
        passed = avg is not None and avg >= config.EVAL_PASS_THRESHOLD
        return {"college_id": cid, "avg": avg, "passed": passed, "skipped": True}

    if force:
        _clear_checkpoint(cid)

    # Load any partial progress from a previous interrupted run.
    done = _load_checkpoint(cid)
    if done:
        print(f"\n  [resume] {cid}: {len(done)}/10 questions already checkpointed")

    print(f"\n{'='*70}")
    print(f"EVAL: {display}")
    print(f"{'='*70}")

    results = []
    for i, q in enumerate(EVAL_QUESTIONS, start=1):
        # Resume: use cached result if available.
        if i in done:
            entry = done[i]
            # Re-inflate sources (stored as list of [doc, meta] pairs).
            sources = [tuple(s) for s in entry.get("sources", [])]
            results.append({
                "question": q,
                "answer": entry["answer"],
                "score": entry["score"],
                "sources": sources,
            })
            verdict = "PASS" if entry["score"] >= config.EVAL_PASS_THRESHOLD else "FAIL"
            print(f"  Q{i:02d}: [cached] score={entry['score']}/10 [{verdict}]")
            continue

        print(f"  Q{i:02d}: {q[:70]}...")
        answer, sources = _ask_with_retry(rag, cid, q, display)
        time.sleep(0.3)
        score = _judge_with_retry(rag, q, answer)

        entry = {
            "question": q,
            "answer": answer,
            "score": score,
            # sources: store as list of [doc, meta] for JSON serialisability
            "sources": [[d, m] for d, m in sources],
        }
        _save_checkpoint(cid, i, entry)
        results.append({**entry, "sources": sources})

        verdict = "PASS" if score >= config.EVAL_PASS_THRESHOLD else "FAIL"
        print(f"       score={score}/10 [{verdict}]")
        time.sleep(0.3)

    scores = [r["score"] for r in results]
    avg = round(sum(scores) / len(scores), 1)
    passed = avg >= config.EVAL_PASS_THRESHOLD
    label = "PASS" if passed else "FAIL"
    print(f"\n  RESULT: avg={avg}/10 -- {label}")

    _write_md(out_path, cid, display, results, avg, passed)
    print(f"  Report -> {out_path}")

    # Clean up checkpoint now that .md is written successfully.
    _clear_checkpoint(cid)

    return {"college_id": cid, "avg": avg, "passed": passed, "skipped": False}


def _write_md(path: Path, cid: str, display: str, results: list, avg: float, passed: bool):
    label = "PASS" if passed else "FAIL"
    lines = [
        f"# Eval: {display}",
        f"",
        f"**College ID:** `{cid}`  ",
        f"**Date:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}  ",
        f"**Average score:** {avg} / 10 - **{label}**  ",
        f"**Threshold:** {config.EVAL_PASS_THRESHOLD} / 10  ",
        f"**Judge model:** {JUDGE_MODEL}  ",
        f"",
        "---",
        "",
    ]
    for i, r in enumerate(results, start=1):
        score = r["score"]
        verdict = "PASS" if score >= config.EVAL_PASS_THRESHOLD else "FAIL"
        lines += [
            f"## Q{i:02d} [{score}/10 {verdict}]",
            f"",
            f"**Question:** {r['question']}",
            f"",
            f"**Answer:**",
            f"",
            r["answer"],
            f"",
            f"**Sources:**",
            f"",
        ]
        for j, (doc, meta) in enumerate(r["sources"], start=1):
            url = meta.get("url", "")
            linked = deep_link(url, doc)
            snip = snippet(doc)
            lines.append(f"- [S{j}] [{url}]({linked})  ")
            lines.append(f'  > "{snip}"')
        lines += ["", "---", ""]

    path.write_text("\n".join(lines), encoding="utf-8")


# ── Main ─────────────────────────────────────────────────────────────────────
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
    ap.add_argument("--pilot", action="store_true", help="all 5 pilot colleges")
    ap.add_argument("--only", help="single college_id")
    ap.add_argument("--no-cache", dest="force", action="store_true",
                    help="clear checkpoint and re-run from scratch")
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
        sys.exit(1)

    rag = RAG()
    all_stats = []
    for c in registry:
        all_stats.append(eval_college(c, rag, force=args.force))

    # ── Summary table ─────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("PILOT EVAL SUMMARY")
    print(f"{'='*70}")
    passed_count = 0
    scored = [s for s in all_stats if s.get("avg") is not None]
    for s in all_stats:
        if s.get("avg") is None:
            print(f"  {s['college_id']}: no score")
            continue
        label = "PASS" if s["passed"] else "FAIL"
        cached = " (cached)" if s.get("skipped") else ""
        print(f"  {s['college_id']}: {s['avg']}/10 - {label}{cached}")
        if s["passed"]:
            passed_count += 1
    if scored:
        overall = round(sum(s["avg"] for s in scored) / len(scored), 1)
        print(f"\n  Overall avg: {overall}/10 ({passed_count}/{len(scored)} colleges PASS)")
        if overall >= config.EVAL_PASS_THRESHOLD:
            print("  -> Pilot PASSED. Safe to run build_all.py.")
        else:
            print("  -> Pilot FAILED. Fix retrieval before scaling to all colleges.")


if __name__ == "__main__":
    main()
