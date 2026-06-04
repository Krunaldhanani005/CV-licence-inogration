"""Person enrollment service — orchestrates photo storage + embedding rebuild.

Wraps :class:`FaceRecognizer` (and its :class:`FaceDatabase`) so that routes stay
thin. On any change it regenerates the averaged embedding and rebuilds FAISS.
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from core.recognition import FaceRecognizer
from core.utils import get_logger
from core.utils import departments as dept

logger = get_logger(__name__)

_ALLOWED_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _resolve_department(department: str, custom: str) -> str:
    """If 'Other' is chosen with a custom value, use the custom department."""
    department = (department or "").strip()
    custom = (custom or "").strip()
    if department.lower() == "other" and custom:
        return custom
    return department or "Other"


class PersonService:
    """CRUD + enrollment for people, delegating to the shared recognizer."""

    def __init__(self, recognizer: FaceRecognizer) -> None:
        self.recognizer = recognizer
        self.db = recognizer.db

    # ------------------------------------------------------------------ read
    def list_people(self) -> List[Dict]:
        return self.db.list_people()

    def get_person(self, person_id: str) -> Optional[Dict]:
        return self.db.get(person_id)

    # ---------------------------------------------------------------- create
    def create_person(self, name: str, department: str, custom_department: str,
                      files: List) -> Dict:
        if not name or not name.strip():
            raise ValueError("Name is required.")
        final_dept = _resolve_department(department, custom_department)
        color = dept.color_for(final_dept)
        person_id = self.db.create(name, final_dept, color, self.recognizer.now_iso())
        saved = self._save_images(person_id, files)
        used, total = self.recognizer.enroll_person(person_id)
        self.recognizer.refresh_index()
        return {
            "id": person_id,
            "name": name.strip(),
            "department": final_dept,
            "color": color,
            "saved_images": saved,
            "encoded_faces": used,
            "total_images": total,
        }

    # ----------------------------------------------------------------- update
    def update_person(self, person_id: str, name: Optional[str], department: Optional[str],
                      custom_department: Optional[str], files: List) -> Dict:
        record = self.db.get(person_id)
        if not record:
            raise ValueError("Person not found.")
        final_dept = record.get("department", "")
        if department is not None:
            final_dept = _resolve_department(department, custom_department or "")
        color = dept.color_for(final_dept)
        self.db.update_meta(person_id, name, final_dept, color, self.recognizer.now_iso())
        saved = self._save_images(person_id, files) if files else 0
        used, total = self.recognizer.enroll_person(person_id)
        self.recognizer.refresh_index()
        return {
            "id": person_id,
            "name": (name or record["name"]).strip(),
            "department": final_dept,
            "color": color,
            "saved_images": saved,
            "encoded_faces": used,
            "total_images": total,
        }

    # ----------------------------------------------------------------- clear
    def clear_all(self) -> int:
        """Delete every enrolled person + their embeddings, rebuild empty FAISS."""
        people = self.db.list_people()
        for p in people:
            self.db.delete(p["id"])
        self.recognizer.refresh_index()
        logger.info("Cleared all %d enrolled person(s)", len(people))
        return len(people)

    # ----------------------------------------------------------------- delete
    def delete_person(self, person_id: str) -> bool:
        ok = self.db.delete(person_id)
        if ok:
            self.recognizer.refresh_index()
        return ok

    def delete_image(self, person_id: str, filename: str) -> Dict:
        record = self.db.get(person_id)
        if not record:
            raise ValueError("Person not found.")
        path = os.path.join(self.db.person_dir(person_id), filename)
        if os.path.exists(path):
            os.remove(path)
        # Rewrite metadata image list.
        remaining = [f for f in record.get("images", []) if f != filename]
        self.db._people[person_id]["images"] = remaining  # noqa: SLF001 - intentional
        self.db._save()  # noqa: SLF001
        used, total = self.recognizer.enroll_person(person_id)
        self.recognizer.refresh_index()
        return {"encoded_faces": used, "total_images": total}

    # --------------------------------------------------------------- helpers
    def _save_images(self, person_id: str, files: List) -> int:
        person_dir = self.db.person_dir(person_id)
        saved = 0
        existing = len(self.db.get(person_id).get("images", []))
        for idx, file in enumerate(files or []):
            if not file or not getattr(file, "filename", ""):
                continue
            ext = os.path.splitext(file.filename)[1].lower()
            if ext not in _ALLOWED_EXT:
                logger.warning("Skipping unsupported file: %s", file.filename)
                continue
            filename = f"face_{existing + saved + 1:03d}{ext}"
            dest = os.path.join(person_dir, filename)
            file.save(dest)
            if not self._has_face(dest):
                os.remove(dest)
                logger.warning("No face detected in upload %s — discarded", file.filename)
                continue
            self.db.add_image(person_id, filename, self.recognizer.now_iso())
            saved += 1
        return saved

    def _has_face(self, image_path: str) -> bool:
        img = cv2.imread(image_path)
        if img is None:
            return False
        return len(self.recognizer.detect_faces(img)) > 0
