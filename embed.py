"""Thin embedding wrapper — fastembed (ONNX) with a sentence-transformers fallback.

fastembed uses onnxruntime instead of PyTorch, cutting RAM from ~500 MB to
~100 MB — essential for Render's 512 MB free tier.

On Linux (Render) fastembed's optional Rust dependency (py-rust-stemmers)
installs from a pre-built wheel.  On Windows with Python 3.14 the wheel is
not yet available, so we pre-inject a harmless stub that covers the one
import site (SparseTextEmbedding / BM25) we never use.  The stub is benign
on Linux too — it only gets picked up if the real module is absent.
"""
from __future__ import annotations

import sys
import types


def _ensure_fastembed_importable() -> bool:
    """Inject a py_rust_stemmers stub if needed, return True on success."""
    if "py_rust_stemmers" not in sys.modules:
        try:
            import py_rust_stemmers  # noqa: F401 — real module present, nothing to do
        except ImportError:
            # Real module missing (Windows / Python 3.14 w/o pre-built wheel).
            # Stub covers the one usage site we never exercise: SparseTextEmbedding.
            _stub = types.ModuleType("py_rust_stemmers")
            _stub.SnowballStemmer = type(  # type: ignore[attr-defined]
                "SnowballStemmer", (), {"__init__": lambda s, l: None, "stem": lambda s, w: w}
            )
            sys.modules["py_rust_stemmers"] = _stub

    try:
        import fastembed  # noqa: F401
        return True
    except Exception:
        return False


# ── Embedder class ────────────────────────────────────────────────────────────
class Embedder:
    """Unified embedder: fastembed (ONNX) when available, else sentence-transformers."""

    def __init__(self, model_name: str):
        self._backend: str
        if _ensure_fastembed_importable():
            from fastembed import TextEmbedding
            self._model = TextEmbedding(model_name)
            self._backend = "fastembed"
        else:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(model_name)
            self._backend = "sentence-transformers"

    # ── Public API ────────────────────────────────────────────────────────────
    def embed_one(self, text: str) -> list[float]:
        """Return the embedding vector for a single string."""
        if self._backend == "fastembed":
            return list(self._model.embed([text]))[0].tolist()
        return self._model.encode(text).tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Return a list of embedding vectors for a batch of strings."""
        if self._backend == "fastembed":
            return [e.tolist() for e in self._model.embed(texts)]
        return self._model.encode(texts, show_progress_bar=False).tolist()
