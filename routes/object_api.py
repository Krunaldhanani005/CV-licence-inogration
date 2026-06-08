"""API endpoints for Object Detection settings and Custom Object management."""
from __future__ import annotations

import json
import os
import shutil
import time
import uuid
from datetime import datetime

from flask import Blueprint, current_app, jsonify, request, send_from_directory

from core.utils import get_logger
from services.object_pipeline import (
    COCO_CLASSES, DEFAULT_ENABLED, load_od_settings, save_od_settings,
)

logger = get_logger(__name__)

object_bp = Blueprint("object", __name__, url_prefix="/api/object")

_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CUSTOM_DIR  = os.path.join(_BASE, "data", "custom_objects")
_REGISTRY    = os.path.join(_CUSTOM_DIR, "registry.json")
_ALLOWED_EXT = {".jpg", ".jpeg", ".png"}


def _mm():
    return current_app.config["MODE_MANAGER"]


def _ok(data=None, **extra):
    p = {"success": True}
    if data is not None:
        p["data"] = data
    p.update(extra)
    return jsonify(p)


def _err(msg: str, status: int = 400):
    return jsonify({"success": False, "message": msg}), status


# ---------------------------------------------------------------- OD settings
@object_bp.route("/settings", methods=["GET"])
def od_settings_get():
    return _ok(load_od_settings())


@object_bp.route("/settings", methods=["POST"])
def od_settings_save():
    body = request.get_json(silent=True) or {}
    current = load_od_settings()
    if isinstance(body.get("detection"), dict):
        current["detection"].update(body["detection"])
    if isinstance(body.get("enabled_classes"), list):
        current["enabled_classes"] = [int(c) for c in body["enabled_classes"]]
    if isinstance(body.get("class_thresholds"), dict):
        current.setdefault("class_thresholds", {})
        current["class_thresholds"].update(
            {int(k): float(v) for k, v in body["class_thresholds"].items()})
    save_od_settings(current)
    _mm().od_pipeline().reload_settings()
    return _ok(message="Object Detection settings saved")


@object_bp.route("/classes", methods=["GET"])
def od_classes():
    enabled = set(load_od_settings().get("enabled_classes", DEFAULT_ENABLED))
    classes = [
        {"id": cid, "name": name, "enabled": cid in enabled}
        for cid, name in sorted(COCO_CLASSES.items(), key=lambda x: x[1])
    ]
    return _ok({"classes": classes})


# -------------------------------------------------------- custom objects CRUD
def _load_reg() -> dict:
    if not os.path.exists(_REGISTRY):
        return {"objects": []}
    try:
        with open(_REGISTRY, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return {"objects": []}


def _save_reg(reg: dict) -> None:
    os.makedirs(_CUSTOM_DIR, exist_ok=True)
    with open(_REGISTRY, "w", encoding="utf-8") as fh:
        json.dump(reg, fh, indent=2)


def _image_count(obj_id: str) -> int:
    d = os.path.join(_CUSTOM_DIR, obj_id)
    if not os.path.isdir(d):
        return 0
    return sum(1 for f in os.listdir(d)
               if os.path.splitext(f)[1].lower() in _ALLOWED_EXT)


@object_bp.route("/custom", methods=["GET"])
def custom_list():
    objects = _load_reg().get("objects", [])
    for obj in objects:
        obj["image_count"] = _image_count(obj["id"])
    return _ok(objects)


@object_bp.route("/custom", methods=["POST"])
def custom_create():
    name = (request.form.get("name") or "").strip()
    desc = (request.form.get("description") or "").strip()
    if not name:
        return _err("Object name is required")

    reg = _load_reg()
    if any(o["name"].lower() == name.lower() for o in reg.get("objects", [])):
        return _err(f"A custom object named '{name}' already exists")

    obj_id  = str(uuid.uuid4())[:8]
    obj_dir = os.path.join(_CUSTOM_DIR, obj_id)
    os.makedirs(obj_dir, exist_ok=True)

    saved = []
    for f in request.files.getlist("images"):
        if not f or not f.filename:
            continue
        ext = os.path.splitext(f.filename)[1].lower()
        if ext not in _ALLOWED_EXT:
            continue
        fname = f"{len(saved):04d}_{int(time.time())}{ext}"
        f.save(os.path.join(obj_dir, fname))
        saved.append(fname)

    entry = {
        "id": obj_id, "name": name, "description": desc,
        "images": saved, "image_count": len(saved),
        "created_at": datetime.utcnow().isoformat(),
        "status": "stored",   # future: "training" / "ready"
    }
    reg.setdefault("objects", []).append(entry)
    _save_reg(reg)
    return _ok(entry, message=f"Custom object '{name}' saved ({len(saved)} images)")


@object_bp.route("/custom/<obj_id>", methods=["GET"])
def custom_get(obj_id: str):
    obj = next((o for o in _load_reg().get("objects", [])
                if o["id"] == obj_id), None)
    if not obj:
        return _err("Not found", 404)
    obj_dir = os.path.join(_CUSTOM_DIR, obj_id)
    if os.path.isdir(obj_dir):
        imgs = sorted(f for f in os.listdir(obj_dir)
                      if os.path.splitext(f)[1].lower() in _ALLOWED_EXT)
        obj["images"] = imgs
        obj["image_count"] = len(imgs)
    return _ok(obj)


@object_bp.route("/custom/<obj_id>", methods=["DELETE"])
def custom_delete(obj_id: str):
    reg = _load_reg()
    objs = reg.get("objects", [])
    idx  = next((i for i, o in enumerate(objs) if o["id"] == obj_id), None)
    if idx is None:
        return _err("Not found", 404)
    name = objs[idx]["name"]
    objs.pop(idx)
    reg["objects"] = objs
    _save_reg(reg)
    obj_dir = os.path.join(_CUSTOM_DIR, obj_id)
    if os.path.isdir(obj_dir):
        shutil.rmtree(obj_dir, ignore_errors=True)
    return _ok(message=f"Custom object '{name}' deleted")


@object_bp.route("/custom/<obj_id>/images/<filename>")
def custom_image(obj_id: str, filename: str):
    return send_from_directory(os.path.join(_CUSTOM_DIR, obj_id), filename)


@object_bp.route("/custom/<obj_id>/images/<filename>", methods=["DELETE"])
def custom_delete_image(obj_id: str, filename: str):
    path = os.path.join(_CUSTOM_DIR, obj_id, filename)
    if not os.path.isfile(path):
        return _err("Image not found", 404)
    os.remove(path)
    reg = _load_reg()
    for obj in reg.get("objects", []):
        if obj["id"] == obj_id:
            obj["images"] = [i for i in obj.get("images", []) if i != filename]
            break
    _save_reg(reg)
    return _ok(message="Image deleted")
