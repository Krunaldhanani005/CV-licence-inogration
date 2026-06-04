"""FAISS index wrapper for cosine-similarity face search.

Embeddings are L2-normalised, so an inner-product index (``IndexFlatIP``) yields
cosine similarity directly. A parallel list of person ids maps row -> person.
"""
from __future__ import annotations

import os
import threading
from typing import List, Optional, Tuple

import faiss
import numpy as np

from core.utils import get_logger

logger = get_logger(__name__)


class FaissIndex:
    """Persistent flat inner-product index keyed by person id."""

    def __init__(self, dim: int, index_dir: str) -> None:
        self.dim = dim
        self.index_dir = index_dir
        os.makedirs(index_dir, exist_ok=True)
        self._index_path = os.path.join(index_dir, "faces.index")
        self._ids_path = os.path.join(index_dir, "faces_ids.npy")
        self._lock = threading.RLock()
        self._index = faiss.IndexFlatIP(dim)
        self._ids: List[str] = []
        self.load()

    # --------------------------------------------------------------- helpers
    @staticmethod
    def normalize(vec: np.ndarray) -> np.ndarray:
        vec = np.asarray(vec, dtype=np.float32)
        if vec.ndim == 1:
            vec = vec[None, :]
        norms = np.linalg.norm(vec, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vec / norms

    # ---------------------------------------------------------------- mutate
    def rebuild(self, person_ids: List[str], embeddings: np.ndarray) -> None:
        """Replace the whole index from scratch (called after enrollment edits)."""
        with self._lock:
            self._index = faiss.IndexFlatIP(self.dim)
            self._ids = []
            if len(person_ids) and embeddings is not None and len(embeddings):
                vecs = self.normalize(embeddings)
                self._index.add(vecs)
                self._ids = list(person_ids)
            self.save()
        logger.info("FAISS index rebuilt with %d person(s)", len(self._ids))

    # ---------------------------------------------------------------- search
    def search(self, embedding: np.ndarray, k: int = 1) -> List[Tuple[str, float]]:
        """Return up to ``k`` ``(person_id, similarity)`` matches."""
        with self._lock:
            if self._index.ntotal == 0:
                return []
            vec = self.normalize(embedding)
            k = min(k, self._index.ntotal)
            scores, idxs = self._index.search(vec, k)
            out: List[Tuple[str, float]] = []
            for score, idx in zip(scores[0], idxs[0]):
                if 0 <= idx < len(self._ids):
                    out.append((self._ids[idx], float(score)))
            return out

    @property
    def size(self) -> int:
        with self._lock:
            return self._index.ntotal

    # -------------------------------------------------------------- persist
    def save(self) -> None:
        with self._lock:
            faiss.write_index(self._index, self._index_path)
            np.save(self._ids_path, np.array(self._ids, dtype=object), allow_pickle=True)

    def load(self) -> None:
        with self._lock:
            if os.path.exists(self._index_path) and os.path.exists(self._ids_path):
                try:
                    self._index = faiss.read_index(self._index_path)
                    self._ids = list(np.load(self._ids_path, allow_pickle=True))
                    logger.info("Loaded FAISS index (%d entries)", self._index.ntotal)
                except Exception as exc:  # noqa: BLE001
                    logger.error("Failed loading FAISS index: %s", exc)
                    self._index = faiss.IndexFlatIP(self.dim)
                    self._ids = []
