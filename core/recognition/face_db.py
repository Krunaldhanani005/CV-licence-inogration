"""Person metadata + embedding store (JSON + NumPy, no SQL).

Layout::

    data/configs/people.json          # person metadata index
    data/faces/<person_id>/*.jpg      # uploaded face photos
    data/embeddings/<person_id>.npy   # averaged embedding (1, dim)
    data/faiss/faces.index            # FAISS index (managed by FaissIndex)

Each person record::

    {
        "id": "uuid",
        "name": "Krunal",
        "images": ["a.jpg", "b.jpg"],
        "created_at": "...",
        "updated_at": "..."
    }
"""
from __future__ import annotations

import json
import os
import threading
import uuid
from typing import Dict, List, Optional, Tuple

import numpy as np

from core.utils import get_logger

logger = get_logger(__name__)


class FaceDatabase:
    """JSON-backed registry of enrolled people and their averaged embeddings."""

    def __init__(self, faces_dir: str, embeddings_dir: str, configs_dir: str) -> None:
        self.faces_dir = faces_dir
        self.embeddings_dir = embeddings_dir
        self.configs_dir = configs_dir
        for d in (faces_dir, embeddings_dir, configs_dir):
            os.makedirs(d, exist_ok=True)
        self._json_path = os.path.join(configs_dir, "people.json")
        self._lock = threading.RLock()
        self._people: Dict[str, Dict] = {}
        self._load()

    # ----------------------------------------------------------------- io
    def _load(self) -> None:
        with self._lock:
            if os.path.exists(self._json_path):
                try:
                    with open(self._json_path, "r", encoding="utf-8") as fh:
                        self._people = json.load(fh)
                except (OSError, json.JSONDecodeError) as exc:
                    logger.error("Failed loading people.json: %s", exc)
                    self._people = {}
            logger.info("Loaded %d enrolled person(s)", len(self._people))

    def _save(self) -> None:
        with self._lock:
            tmp = self._json_path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(self._people, fh, indent=2)
            os.replace(tmp, self._json_path)

    # ------------------------------------------------------------- accessors
    def list_people(self) -> List[Dict]:
        with self._lock:
            return [self._public(pid, rec) for pid, rec in self._people.items()]

    def get(self, person_id: str) -> Optional[Dict]:
        with self._lock:
            rec = self._people.get(person_id)
            return self._public(person_id, rec) if rec else None

    def name_of(self, person_id: str) -> str:
        with self._lock:
            rec = self._people.get(person_id)
            return rec["name"] if rec else ""

    def _public(self, pid: str, rec: Dict) -> Dict:
        return {
            "id": pid,
            "name": rec.get("name", ""),
            "department": rec.get("department", ""),
            "color": rec.get("color", ""),
            "images": rec.get("images", []),
            "image_count": len(rec.get("images", [])),
            "embedding_count": rec.get("embedding_count", 0),
            "created_at": rec.get("created_at"),
            "updated_at": rec.get("updated_at"),
        }

    def record_of(self, person_id: str) -> Optional[Dict]:
        """Internal record (name/department/color) for a recognised person."""
        with self._lock:
            rec = self._people.get(person_id)
            if not rec:
                return None
            return {
                "name": rec.get("name", ""),
                "department": rec.get("department", ""),
                "color": rec.get("color", ""),
            }

    def person_dir(self, person_id: str) -> str:
        path = os.path.join(self.faces_dir, person_id)
        os.makedirs(path, exist_ok=True)
        return path

    # --------------------------------------------------------------- mutate
    def create(self, name: str, department: str, color: str, timestamp: str) -> str:
        person_id = uuid.uuid4().hex
        with self._lock:
            self._people[person_id] = {
                "name": name.strip(),
                "department": (department or "").strip(),
                "color": color,
                "images": [],
                "embedding_count": 0,
                "created_at": timestamp,
                "updated_at": timestamp,
            }
            self._save()
        os.makedirs(self.person_dir(person_id), exist_ok=True)
        logger.info("Created person '%s' [%s] (%s)", name, department, person_id)
        return person_id

    def update_meta(self, person_id: str, name: Optional[str], department: Optional[str],
                    color: Optional[str], timestamp: str) -> bool:
        with self._lock:
            rec = self._people.get(person_id)
            if not rec:
                return False
            if name is not None and name.strip():
                rec["name"] = name.strip()
            if department is not None:
                rec["department"] = department.strip()
            if color is not None:
                rec["color"] = color
            rec["updated_at"] = timestamp
            self._save()
        return True

    def set_embedding_count(self, person_id: str, count: int) -> None:
        with self._lock:
            if person_id in self._people:
                self._people[person_id]["embedding_count"] = int(count)
                self._save()

    def add_image(self, person_id: str, filename: str, timestamp: str) -> None:
        with self._lock:
            if person_id not in self._people:
                return
            imgs = self._people[person_id].setdefault("images", [])
            if filename not in imgs:
                imgs.append(filename)
            self._people[person_id]["updated_at"] = timestamp
            self._save()

    def set_embedding(self, person_id: str, embedding: np.ndarray) -> None:
        path = os.path.join(self.embeddings_dir, f"{person_id}.npy")
        np.save(path, np.asarray(embedding, dtype=np.float32))

    def get_embedding(self, person_id: str) -> Optional[np.ndarray]:
        path = os.path.join(self.embeddings_dir, f"{person_id}.npy")
        if os.path.exists(path):
            return np.load(path)
        return None

    def delete(self, person_id: str) -> bool:
        import shutil

        with self._lock:
            if person_id not in self._people:
                return False
            del self._people[person_id]
            self._save()
        # Remove face images + embedding file.
        shutil.rmtree(os.path.join(self.faces_dir, person_id), ignore_errors=True)
        emb = os.path.join(self.embeddings_dir, f"{person_id}.npy")
        if os.path.exists(emb):
            os.remove(emb)
        logger.info("Deleted person %s", person_id)
        return True

    # ------------------------------------------------------- index materials
    def all_embeddings(self) -> Tuple[List[str], Optional[np.ndarray]]:
        """Return ``(person_ids, embeddings)`` for people that have embeddings."""
        ids: List[str] = []
        vecs: List[np.ndarray] = []
        with self._lock:
            person_ids = list(self._people.keys())
        for pid in person_ids:
            emb = self.get_embedding(pid)
            if emb is not None:
                ids.append(pid)
                vecs.append(emb.reshape(-1))
        if not vecs:
            return [], None
        return ids, np.vstack(vecs).astype(np.float32)
