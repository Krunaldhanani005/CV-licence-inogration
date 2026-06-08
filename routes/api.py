"""JSON + streaming API used by the front-end."""
from __future__ import annotations

import os
import time
from typing import Tuple

from flask import (Blueprint, Response, current_app, jsonify, request,
                   send_from_directory)

from core.utils import get_logger

logger = get_logger(__name__)

api_bp = Blueprint("api", __name__, url_prefix="/api")


# --------------------------------------------------------------------- helpers
def _pipeline():
    """Return whichever pipeline is currently active (FR or OD)."""
    return current_app.config["MODE_MANAGER"].pipeline()


def _mode_manager():
    return current_app.config["MODE_MANAGER"]


def _people():
    return current_app.config["PERSON_SERVICE"]


def _settings():
    return current_app.config["SETTINGS"]


def _ok(data=None, **extra):
    payload = {"success": True}
    if data is not None:
        payload["data"] = data
    payload.update(extra)
    return jsonify(payload)


def _err(message: str, status: int = 400):
    return jsonify({"success": False, "message": message}), status


# ------------------------------------------------------------------ mode switch
@api_bp.route("/mode", methods=["GET"])
def mode_get():
    return _ok({"mode": _mode_manager().mode})


@api_bp.route("/mode", methods=["POST"])
def mode_switch():
    body = request.get_json(silent=True) or {}
    mode = str(body.get("mode", "")).strip()
    if mode not in ("fr", "od"):
        return _err("mode must be 'fr' or 'od'")
    result = _mode_manager().switch(mode)
    return jsonify(result), (200 if result["success"] else 400)


# ------------------------------------------------------------------ live feed
@api_bp.route("/video_feed")
def video_feed():
    pipeline = _pipeline()

    def generate():
        boundary = b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
        while True:
            jpeg = pipeline.get_jpeg()
            if jpeg is None:
                time.sleep(0.05)
                continue
            yield boundary + jpeg + b"\r\n"
            time.sleep(0.03)  # ~30 fps cap for the browser

    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")


@api_bp.route("/stats")
def stats():
    return _ok(_pipeline().get_stats())


# --------------------------------------------------------------- pipeline ctrl
@api_bp.route("/pipeline/restart", methods=["POST"])
def pipeline_restart():
    _pipeline().restart()
    return _ok(message="Pipeline restarted")


@api_bp.route("/pipeline/status")
def pipeline_status():
    p = _pipeline()
    return _ok({"running": p.running, **p.get_stats()})


# -------------------------------------------------- camera management system
_CAMERA_FIELDS = {
    "camera_type", "camera_index", "width", "height", "fps", "name",
    "camera_name", "rtsp_url", "ip_address", "username", "password",
    "channel", "port", "brand", "stream_path", "file_path", "reconnect_delay_sec",
}
_CAMERA_INT_FIELDS = {"camera_index", "width", "height", "fps", "channel", "port"}


def _camera_cfg(body: dict) -> dict:
    cfg = {k: _coerce_camera(k, v) for k, v in body.items() if k in _CAMERA_FIELDS}
    cfg["camera_type"] = str(body.get("camera_type") or "usb").lower()
    return cfg


@api_bp.route("/cameras", methods=["GET"])
def cameras_list():
    """Full inventory for the Camera Settings page: scanned USB + saved RTSP."""
    data = _pipeline().cam_manager.get_available_cameras(max_index=5)
    return _ok(data)


@api_bp.route("/cameras/scan", methods=["POST"])
def cameras_scan():
    """Re-scan available cameras (Refresh button)."""
    data = _pipeline().cam_manager.get_available_cameras(max_index=5)
    return _ok(data)


@api_bp.route("/cameras/select", methods=["POST"])
def cameras_select():
    """Set the active camera (persists to camera_config.json).

    This does NOT start detection — the Camera Settings page never streams. If
    detection happens to be running, the live source is switched seamlessly.
    """
    p = _pipeline()
    cfg = _camera_cfg(request.get_json(silent=True) or {})

    if p.detection_running:
        result = p.select_source(cfg, persist=True)          # switch live source
        return jsonify({"success": result.get("success", False), **result}), \
            (200 if result.get("success") else 400)

    # Detection stopped: just persist + validate with a quick snapshot (<=400x250).
    p.cam_manager.save_active_camera(cfg)
    test = p.cam_manager.test_camera(cfg)
    return jsonify({
        "success": True,
        "message": "Active camera saved" if test.get("success") else "Saved (could not preview)",
        "label": cfg.get("name") or cfg.get("camera_name") or test.get("label"),
        "resolution": test.get("resolution"),
        "snapshot": test.get("snapshot"),
    })


@api_bp.route("/cameras/preview", methods=["POST"])
def cameras_preview():
    """One-off snapshot preview of a camera without making it active."""
    cfg = _camera_cfg(request.get_json(silent=True) or {})
    result = _pipeline().cam_manager.get_camera_preview(cfg)
    return jsonify(result), (200 if result.get("success") else 400)


@api_bp.route("/cameras/test", methods=["POST"])
def cameras_test():
    """Test an RTSP/IP camera connection and return a preview snapshot."""
    cfg = _camera_cfg(request.get_json(silent=True) or {})
    result = _pipeline().cam_manager.test_camera(cfg)
    return jsonify(result), (200 if result.get("success") else 400)


@api_bp.route("/cameras/rtsp", methods=["POST"])
def cameras_add_rtsp():
    """Save an IP/RTSP camera to the available-cameras list (after a successful test)."""
    mgr = _pipeline().cam_manager
    cfg = _camera_cfg(request.get_json(silent=True) or {})
    # Accept either a full RTSP URL or IP details (the URL is then built).
    url = (cfg.get("rtsp_url") or "").strip() or mgr.build_rtsp_url(cfg)
    if not url:
        return _err("Provide an IP address or a full RTSP URL.")
    cfg["rtsp_url"] = url
    entry = mgr.add_rtsp_camera(cfg)
    return _ok(entry, message="IP camera saved")


@api_bp.route("/cameras/rtsp", methods=["DELETE"])
def cameras_remove_rtsp():
    url = (request.args.get("rtsp_url") or "").strip()
    ok = _pipeline().cam_manager.remove_rtsp_camera(url)
    return _ok(message="Removed") if ok else _err("Not found", 404)


@api_bp.route("/camera/active", methods=["GET"])
def camera_active():
    """Active camera summary (label / resolution / status) for the dashboard."""
    p = _pipeline()
    # While streaming, report the live spec (actual capture resolution); otherwise
    # the persisted active-camera summary.
    info = (p.cam_manager.describe() if p.detection_running
            else p.cam_manager.active_summary()) or {
        "label": "None", "resolution": "—", "camera_type": None}
    stats = p.get_stats()
    info["status"] = stats.get("camera_status", "no_source")
    info["has_source"] = p.has_source
    info["detection_running"] = p.detection_running
    info["live_fps"] = stats.get("fps", 0.0)
    return _ok(info)


# ----------------------------------------------------- detection start / stop
@api_bp.route("/detection/start", methods=["POST"])
def detection_start():
    """Start camera + detection/recognition/tracking (dashboard button)."""
    result = _pipeline().start_detection()
    return jsonify({"success": result.get("success", False), **result}), \
        (200 if result.get("success") else 400)


@api_bp.route("/detection/stop", methods=["POST"])
def detection_stop():
    """Stop all processing and release the camera (CPU -> near zero)."""
    return _ok(_pipeline().stop_detection())


@api_bp.route("/detection/status", methods=["GET"])
def detection_status():
    p = _pipeline()
    return _ok({"running": p.detection_running, **p.get_stats()})


def _coerce_camera(key: str, value):
    if key in _CAMERA_INT_FIELDS:
        try:
            return int(value)
        except (TypeError, ValueError):
            return value
    return value


# -------------------------------------------------------------------- people
@api_bp.route("/people", methods=["GET"])
def people_list():
    return _ok(_people().list_people())


@api_bp.route("/departments", methods=["GET"])
def departments_list():
    from core.utils import departments as dept
    return _ok({"departments": dept.catalog()})


@api_bp.route("/people/clear", methods=["POST"])
def people_clear():
    count = _people().clear_all()
    _pipeline().reload_recognition()
    return _ok({"removed": count}, message=f"Removed {count} person(s)")


@api_bp.route("/people", methods=["POST"])
def people_create():
    name = request.form.get("name", "").strip()
    department = request.form.get("department", "").strip()
    custom = request.form.get("custom_department", "").strip()
    files = request.files.getlist("photos")
    try:
        result = _people().create_person(name, department, custom, files)
    except ValueError as exc:
        return _err(str(exc))
    _pipeline().reload_recognition()
    return _ok(result, message="Person added")


@api_bp.route("/people/<person_id>", methods=["GET"])
def people_get(person_id: str):
    person = _people().get_person(person_id)
    if not person:
        return _err("Person not found", 404)
    return _ok(person)


@api_bp.route("/people/<person_id>", methods=["POST"])
def people_update(person_id: str):
    name = request.form.get("name", "").strip()
    department = request.form.get("department", None)
    custom = request.form.get("custom_department", "")
    files = request.files.getlist("photos")
    try:
        result = _people().update_person(person_id, name, department, custom, files)
    except ValueError as exc:
        return _err(str(exc), 404)
    _pipeline().reload_recognition()
    return _ok(result, message="Person updated")


@api_bp.route("/people/<person_id>", methods=["DELETE"])
def people_delete(person_id: str):
    if not _people().delete_person(person_id):
        return _err("Person not found", 404)
    _pipeline().reload_recognition()
    return _ok(message="Person deleted")


@api_bp.route("/people/<person_id>/images/<filename>", methods=["DELETE"])
def people_delete_image(person_id: str, filename: str):
    try:
        result = _people().delete_image(person_id, filename)
    except ValueError as exc:
        return _err(str(exc), 404)
    _pipeline().reload_recognition()
    return _ok(result, message="Image removed")


@api_bp.route("/people/<person_id>/image/<filename>", methods=["GET"])
def people_image(person_id: str, filename: str):
    faces_dir = _settings().path("faces_dir")
    directory = os.path.join(faces_dir, person_id)
    return send_from_directory(directory, filename)


@api_bp.route("/people/<person_id>/reenroll", methods=["POST"])
def people_reenroll(person_id: str):
    svc = _people()
    record = svc.get_person(person_id)
    if not record:
        return _err("Person not found", 404)
    # Force re-computation: remove cached per-image embeddings so enroll_person reprocesses all
    import shutil
    emb_dir = os.path.join(svc.recognizer.db.embeddings_dir, person_id)
    shutil.rmtree(emb_dir, ignore_errors=True)
    used, total = svc.recognizer.enroll_person(person_id)
    svc.recognizer.refresh_index()
    _pipeline().reload_recognition()
    return _ok({"encoded_faces": used, "total_images": total},
               message=f"{used}/{total} face(s) encoded")


@api_bp.route("/people/reenroll-all", methods=["POST"])
def people_reenroll_all():
    svc = _people()
    import shutil
    results = []
    for person in svc.list_people():
        pid = person["id"]
        # Remove per-image cache to force full reprocessing
        emb_dir = os.path.join(svc.recognizer.db.embeddings_dir, pid)
        shutil.rmtree(emb_dir, ignore_errors=True)
        used, total = svc.recognizer.enroll_person(pid)
        results.append({"id": pid, "name": person["name"], "encoded": used, "total": total})
    svc.recognizer.refresh_index()
    _pipeline().reload_recognition()
    total_ok = sum(1 for r in results if r["encoded"] == r["total"])
    return _ok(results, message=f"Re-encoded {len(results)} people — {total_ok} fully encoded")


# ------------------------------------------------------------ system settings
_SETTINGS_SECTIONS = ("detection", "recognition", "pose", "pipeline", "tracking")


@api_bp.route("/settings", methods=["GET"])
def settings_get():
    s = _settings()
    return _ok({sec: s.section(sec) for sec in _SETTINGS_SECTIONS})


@api_bp.route("/settings", methods=["POST"])
def settings_save():
    body = request.get_json(silent=True) or {}
    s = _settings()
    for section, values in body.items():
        if section in _SETTINGS_SECTIONS and isinstance(values, dict):
            s.update_section(section, _coerce(values))
    # Recognition threshold / detection params take effect after a restart.
    _pipeline().restart()
    return _ok(message="Settings saved")


def _coerce(values: dict) -> dict:
    """Best-effort numeric/boolean coercion of JSON form values."""
    out = {}
    for k, v in values.items():
        if isinstance(v, str):
            low = v.strip().lower()
            if low in ("true", "false"):
                out[k] = (low == "true")
                continue
            try:
                out[k] = int(v) if v.isdigit() else float(v)
                continue
            except ValueError:
                pass
        out[k] = v
    return out
