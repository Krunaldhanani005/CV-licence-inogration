"""InsightFace embedding extraction, enrollment and FAISS-backed matching."""
from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from config import Settings
from core.utils import get_logger
from .face_db import FaceDatabase
from .faiss_index import FaissIndex

logger = get_logger(__name__)

_EMBED_DIM = 512  # InsightFace recognition embedding size


@dataclass
class FaceMatch:
    """Result of a face search."""

    box: Tuple[int, int, int, int]   # face box in frame coordinates
    name: str
    score: float
    is_known: bool
    department: str = ""
    color: str = ""


class FaceRecognizer:
    """Detects faces, extracts embeddings and matches against the FAISS index."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        cfg = settings.section("recognition")
        self.model_name = cfg.get("model_name", "buffalo_sc")
        self.det_size = int(cfg.get("det_size", 640))
        self.threshold = float(cfg.get("threshold", 0.45))
        self.min_face_size = int(cfg.get("min_face_size", 30))
        self.guest_label = cfg.get("guest_label", "Guest")
        self._infer_lock = threading.Lock()

        self.db = FaceDatabase(
            faces_dir=settings.path("faces_dir"),
            embeddings_dir=settings.path("embeddings_dir"),
            configs_dir=settings.path("configs_dir"),
        )
        self.index = FaissIndex(dim=_EMBED_DIM, index_dir=settings.path("faiss_dir"))
        self._app = None
        self._load_model()
        self.refresh_index()

    # --------------------------------------------------------------- model
    def _load_model(self) -> None:
        from insightface.app import FaceAnalysis

        root = self.settings.path("models_dir")
        logger.info("Loading InsightFace pack '%s' (CPU)…", self.model_name)
        self._app = FaceAnalysis(
            name=self.model_name,
            root=root,
            providers=["CPUExecutionProvider"],
        )
        self._app.prepare(ctx_id=-1, det_size=(self.det_size, self.det_size))
        logger.info("InsightFace ready")

    # -------------------------------------------------------------- detect
    def detect_faces(self, image: np.ndarray):
        """Return raw InsightFace face objects for an image (thread-safe).

        The recognition worker thread and the enrollment (Flask) thread can both
        call this; a lock serialises model inference to avoid races.
        """
        if self._app is None:
            return []
        with self._infer_lock:
            return self._app.get(image)

    def _largest_embedding(self, image: np.ndarray) -> Optional[np.ndarray]:
        faces = self.detect_faces(image)
        if not faces:
            return None
        face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        return np.asarray(face.normed_embedding, dtype=np.float32)

    # -------------------------------------------------------------- matching
    def match_embedding(self, embedding: np.ndarray) -> Dict[str, object]:
        """Search the index; return ``{name, department, color, score, is_known}``."""
        from core.utils import departments as dept

        results = self.index.search(embedding, k=1)
        if results:
            person_id, score = results[0]
            if score >= self.threshold:
                rec = self.db.record_of(person_id) or {}
                department = rec.get("department", "")
                return {
                    "name": rec.get("name") or self.guest_label,
                    "department": department,
                    "color": rec.get("color") or dept.color_for(department),
                    "score": float(score),
                    "is_known": True,
                }
            return {"name": self.guest_label, "department": "", "color": dept.GUEST_COLOR,
                    "score": float(score), "is_known": False}
        return {"name": self.guest_label, "department": "", "color": dept.GUEST_COLOR,
                "score": 0.0, "is_known": False}

    def recognize_faces(self, image: np.ndarray) -> List[FaceMatch]:
        """Detect + match every face in an image (used for body association)."""
        matches: List[FaceMatch] = []
        for face in self.detect_faces(image):
            x1, y1, x2, y2 = [int(v) for v in face.bbox]
            if min(x2 - x1, y2 - y1) < self.min_face_size:
                continue
            emb = np.asarray(face.normed_embedding, dtype=np.float32)
            m = self.match_embedding(emb)
            matches.append(FaceMatch(
                box=(x1, y1, x2, y2), name=m["name"], score=m["score"],
                is_known=m["is_known"], department=m["department"], color=m["color"]))
        return matches

    # ------------------------------------------------------------ enrollment
    def enroll_person(self, person_id: str) -> Tuple[int, int]:
        """Recompute a person's averaged embedding from their stored photos.

        Returns ``(used_images, total_images)``.
        """
        record = self.db.get(person_id)
        if not record:
            return 0, 0
        person_dir = self.db.person_dir(person_id)
        embeddings: List[np.ndarray] = []
        images = record.get("images", [])
        for filename in images:
            path = os.path.join(person_dir, filename)
            image = cv2.imread(path)
            if image is None:
                logger.warning("Could not read enrollment image: %s", path)
                continue
            emb = self._largest_embedding(image)
            if emb is not None:
                embeddings.append(emb)
            else:
                logger.warning("No face found in %s", path)

        if not embeddings:
            logger.warning("No usable faces for person %s", person_id)
            self.db.set_embedding_count(person_id, 0)
            return 0, len(images)

        avg = np.mean(np.vstack(embeddings), axis=0)
        avg = avg / (np.linalg.norm(avg) or 1.0)
        self.db.set_embedding(person_id, avg.astype(np.float32))
        self.db.set_embedding_count(person_id, len(embeddings))
        logger.info("Enrolled %s using %d/%d image(s)", person_id, len(embeddings), len(images))
        return len(embeddings), len(images)

    def refresh_index(self) -> None:
        """Rebuild the FAISS index from all stored averaged embeddings."""
        ids, embeddings = self.db.all_embeddings()
        self.index.rebuild(ids, embeddings if embeddings is not None else np.empty((0, _EMBED_DIM)))

    @staticmethod
    def now_iso() -> str:
        return datetime.now().isoformat(timespec="seconds")
