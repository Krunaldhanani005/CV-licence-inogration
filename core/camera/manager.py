"""Camera source management.

``CameraManager`` is the *single* place that owns a video source. Every other
module (detection, recognition, tracking, pose, display) receives frames through
this manager — no module opens ``cv2.VideoCapture`` directly.

It supports multiple source types behind one clean OOP interface so new sources
can be added without touching the pipeline:

    usb    -> cv2.VideoCapture(<index>)        (default)
    webcam -> alias of usb
    rtsp   -> cv2.VideoCapture(<rtsp_url>)
    file   -> cv2.VideoCapture(<video path>)
"""
from __future__ import annotations

import base64
import os
import threading
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
from urllib.parse import quote

import cv2
import numpy as np

from config import Settings
from core.utils import get_logger
from .stream import CameraStream

logger = get_logger(__name__)


@dataclass
class SourceSpec:
    """Resolved, type-agnostic description of a capture source."""

    camera_type: str
    handle: object                 # int (usb) or str (rtsp/file)
    width: int
    height: int
    fps: int
    label: str                     # human readable, e.g. "USB Camera #0"


class CameraManager:
    """Owns the active :class:`CameraStream` and resolves source configuration."""

    SUPPORTED_TYPES = ("usb", "webcam", "rtsp", "file")

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._lock = threading.RLock()
        self._stream: Optional[CameraStream] = None
        self._spec: Optional[SourceSpec] = None
        # Low-latency RTSP: TCP transport, 5 s fail-fast timeout, plus
        # fflags;nobuffer and flags;low_delay to remove FFmpeg's internal
        # receive buffer — cuts end-to-end RTSP latency from ~1 s to near-zero.
        os.environ.setdefault(
            "OPENCV_FFMPEG_CAPTURE_OPTIONS",
            "rtsp_transport;tcp|stimeout;5000000|fflags;nobuffer|flags;low_delay",
        )
        # Persistent camera config (active camera + saved RTSP cameras).
        self._config_path = os.path.join(settings.path("configs_dir"), "camera_config.json")
        self._store = self._load_store()

    # ------------------------------------------------------------- config io
    def config(self) -> Dict:
        return self.settings.section("camera")

    # ------------------------------------------------- persistent camera store
    def _load_store(self) -> Dict:
        import json
        if os.path.exists(self._config_path):
            try:
                with open(self._config_path, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                    data.setdefault("active_camera", None)
                    data.setdefault("rtsp_cameras", [])
                    return data
            except (OSError, ValueError) as exc:
                logger.warning("Could not read camera_config.json: %s", exc)
        return {"active_camera": None, "rtsp_cameras": []}

    def _save_store(self) -> None:
        import json
        with self._lock:
            os.makedirs(os.path.dirname(self._config_path), exist_ok=True)
            tmp = self._config_path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(self._store, fh, indent=2)
            os.replace(tmp, self._config_path)

    # ---- active camera ------------------------------------------------------
    def get_active_camera(self) -> Optional[Dict]:
        """Return the persisted active-camera config (or ``None``)."""
        with self._lock:
            active = self._store.get("active_camera")
            return dict(active) if active else None

    def save_active_camera(self, cfg: Dict) -> None:
        """Persist the active-camera selection to ``camera_config.json``."""
        with self._lock:
            self._store["active_camera"] = dict(cfg)
        self._save_store()
        logger.info("Active camera saved: %s", cfg.get("name") or cfg.get("camera_name") or cfg)

    def clear_active_camera(self) -> None:
        with self._lock:
            self._store["active_camera"] = None
        self._save_store()

    # ---- saved RTSP cameras -------------------------------------------------
    def list_rtsp_cameras(self) -> list:
        with self._lock:
            return [dict(c) for c in self._store.get("rtsp_cameras", [])]

    def add_rtsp_camera(self, cfg: Dict) -> Dict:
        """Persist an RTSP camera so it appears in the available-cameras list."""
        # Resolve to a concrete URL so saved cameras always reopen the same way.
        url = (cfg.get("rtsp_url") or "").strip() or self.build_rtsp_url(cfg)
        entry = {
            "camera_type": "rtsp",
            "camera_name": (cfg.get("camera_name") or "IP Camera").strip(),
            "rtsp_url": url,
            "ip_address": (cfg.get("ip_address") or "").strip(),
            "port": int(cfg.get("port", 554) or 554),
            "username": (cfg.get("username") or "").strip(),
            "password": cfg.get("password") or "",
            "brand": str(cfg.get("brand") or "hikvision").lower(),
            "stream_path": (cfg.get("stream_path") or "").strip(),
        }
        with self._lock:
            cams = self._store.setdefault("rtsp_cameras", [])
            # De-dupe by resolved URL.
            cams = [c for c in cams if (c.get("rtsp_url") or "") != url]
            cams.append(entry)
            self._store["rtsp_cameras"] = cams
        self._save_store()
        logger.info("Saved RTSP camera: %s", entry["camera_name"])
        return entry

    def remove_rtsp_camera(self, url: str) -> bool:
        with self._lock:
            cams = self._store.get("rtsp_cameras", [])
            new = [c for c in cams if (c.get("rtsp_url") or "") != url]
            removed = len(new) != len(cams)
            self._store["rtsp_cameras"] = new
        if removed:
            self._save_store()
        return removed

    @staticmethod
    def parse_resolution(value: str) -> Tuple[int, int]:
        try:
            w, h = str(value).lower().split("x")
            return int(w), int(h)
        except (ValueError, AttributeError):
            return 1280, 720

    @staticmethod
    def _dimensions(cfg: Dict) -> Tuple[int, int]:
        """Resolution from width/height, falling back to a legacy 'resolution'."""
        if cfg.get("width") and cfg.get("height"):
            return int(cfg["width"]), int(cfg["height"])
        if cfg.get("resolution"):
            return CameraManager.parse_resolution(cfg["resolution"])
        return 1280, 720

    # ----------------------------------------------------------- url helpers
    # Common IP-camera RTSP path templates ({channel} is substituted).
    BRAND_PATHS: Dict[str, str] = {
        "hikvision": "/Streaming/Channels/{channel}01",
        "dahua": "/cam/realmonitor?channel={channel}&subtype=0",
        "amcrest": "/cam/realmonitor?channel={channel}&subtype=0",
        "onvif": "/onvif1",
        "generic": "/11",
        "axis": "/axis-media/media.amp",
        "reolink": "/h264Preview_0{channel}_main",
    }

    def build_rtsp_url(self, cfg: Dict) -> str:
        """Return an explicit RTSP url (overrides ip), or build one from ip/creds.

        Supports IP cameras of different brands via ``brand`` / ``stream_path``
        and a configurable ``port`` (default 554).
        """
        explicit = (cfg.get("rtsp_url") or "").strip()
        if explicit:
            return explicit
        ip = (cfg.get("ip_address") or "").strip()
        if not ip:
            return ""
        port = int(cfg.get("port", 554) or 554)
        user = quote(str(cfg.get("username") or ""), safe="")
        pwd = quote(str(cfg.get("password") or ""), safe="")
        auth = f"{user}:{pwd}@" if user else ""

        path = (cfg.get("stream_path") or "").strip()
        if not path:
            brand = str(cfg.get("brand") or "hikvision").lower()
            channel = cfg.get("channel", 1)
            template = self.BRAND_PATHS.get(brand, self.BRAND_PATHS["hikvision"])
            path = template.format(channel=channel)
        if not path.startswith("/"):
            path = "/" + path
        return f"rtsp://{auth}{ip}:{port}{path}"

    # ------------------------------------------------------------- factory
    def resolve_spec(self, cfg: Optional[Dict] = None) -> SourceSpec:
        """Resolve a configuration dict into a concrete :class:`SourceSpec`."""
        cfg = cfg or self.config()
        ctype = str(cfg.get("camera_type") or "usb").lower()
        if ctype not in self.SUPPORTED_TYPES:
            logger.warning("Unknown camera_type '%s' -> defaulting to usb", ctype)
            ctype = "usb"
        width, height = self._dimensions(cfg)
        fps = int(cfg.get("fps", 30) or 0)

        if ctype in ("usb", "webcam"):
            index = int(cfg.get("camera_index", 0) or 0)
            return SourceSpec(ctype, index, width, height, fps, f"USB Camera #{index}")

        if ctype == "file":
            path = (cfg.get("file_path") or "").strip()
            label = f"Video File ({os.path.basename(path)})" if path else "Video File"
            return SourceSpec(ctype, path, width, height, fps, label)

        # rtsp / ip camera
        url = self.build_rtsp_url(cfg)
        name = (cfg.get("camera_name") or cfg.get("name") or "RTSP Camera").strip()
        return SourceSpec("rtsp", url, width, height, fps, name or "RTSP Camera")

    # ------------------------------------------------------------- discovery
    def scan_cameras(self, max_index: int = 5, thumb_w: int = 320) -> list:
        """Scan local camera indexes, returning only devices that yield frames.

        Duplicate V4L2 nodes of the same physical device (same name) are collapsed
        to the lowest working index for a clean list. Each entry::

            {type, index, name, width, height, resolution, status, thumbnail}
        """
        cameras = []
        seen_names = set()
        for idx in range(max_index + 1):
            info = self._probe_index(idx, thumb_w)
            if not info:
                continue
            if info["name"] in seen_names:           # collapse duplicate nodes
                continue
            seen_names.add(info["name"])
            cameras.append(info)
        logger.info("Camera scan found %d device(s)", len(cameras))
        return cameras

    # Backwards-compatible alias.
    def discover_cameras(self, max_index: int = 5, thumb_w: int = 320) -> list:
        return self.scan_cameras(max_index, thumb_w)

    def get_available_cameras(self, max_index: int = 5) -> Dict:
        """Full camera inventory for the management page.

        Returns ``{"available": [...], "active": {...}}``. USB cameras are scanned
        live; the currently active USB device is *not* re-opened (it is busy) —
        its card is rebuilt from the live stream instead. Saved RTSP cameras are
        appended from the persistent store.
        """
        active = self.get_active_camera() or {}
        active_type = active.get("camera_type")
        active_idx = active.get("camera_index") if active_type in ("usb", "webcam") else None
        active_url = active.get("rtsp_url") if active_type == "rtsp" else None

        cameras = []
        seen_names = set()
        # Pre-seed the active device's real V4L2 name so its duplicate nodes are
        # collapsed even when the active card uses a custom display name.
        if active_idx is not None:
            real = self._device_name(active_idx)
            if real:
                seen_names.add(real)
        for idx in range(max_index + 1):
            if active_idx is not None and idx == active_idx and self._stream and self.connected:
                card = self._active_usb_card(active)
                cameras.append(card)
                seen_names.add(card["name"])
                continue
            info = self._probe_index(idx, 320)
            if not info or info["name"] in seen_names:
                continue
            seen_names.add(info["name"])
            info["active"] = False
            cameras.append(info)

        # Saved RTSP cameras (no live probe — shown with a glyph thumbnail).
        for rc in self.list_rtsp_cameras():
            cameras.append({
                "type": "rtsp",
                "index": None,
                "name": rc.get("camera_name", "RTSP Camera"),
                "rtsp_url": rc.get("rtsp_url", ""),
                "resolution": "—",
                "status": "Saved",
                "thumbnail": None,
                "active": bool(active_url and rc.get("rtsp_url") == active_url),
                "config": {k: rc.get(k) for k in
                           ("camera_type", "camera_name", "rtsp_url", "ip_address", "username")},
            })

        return {"available": cameras, "active": self.active_summary()}

    def _active_usb_card(self, active: Dict) -> Dict:
        frame = self.read()
        thumb = self._encode_snapshot(frame, 320) if frame is not None else None
        w, h = self.resolution()
        return {
            "type": "usb",
            "index": active.get("camera_index", 0),
            "name": active.get("name") or f"USB Camera {active.get('camera_index', 0)}",
            "width": w, "height": h,
            "resolution": f"{w}x{h}",
            "status": "Connected",
            "thumbnail": thumb,
            "active": True,
        }

    def _probe_index(self, idx: int, thumb_w: int) -> Optional[Dict]:
        cap = cv2.VideoCapture(idx)
        try:
            if not cap.isOpened():
                return None
            frame = None
            for _ in range(3):           # first USB frame is often empty
                ok, f = cap.read()
                if ok and f is not None:
                    frame = f
                    break
            if frame is None:
                return None
            h, w = frame.shape[:2]
            return {
                "type": "usb",
                "index": idx,
                "name": self._device_name(idx) or f"USB Camera {idx}",
                "width": w,
                "height": h,
                "resolution": f"{w}x{h}",
                "status": "Connected",
                "thumbnail": self._encode_snapshot(frame, thumb_w),
                "active": False,
            }
        except Exception as exc:  # noqa: BLE001
            logger.debug("Probe index %d failed: %s", idx, exc)
            return None
        finally:
            cap.release()

    @staticmethod
    def _device_name(idx: int) -> Optional[str]:
        """Best-effort friendly device name (Linux v4l2)."""
        path = f"/sys/class/video4linux/video{idx}/name"
        try:
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as fh:
                    name = fh.read().strip()
                    return name or None
        except OSError:
            pass
        return None

    # ------------------------------------------------------------- lifecycle
    def open(self, cfg: Optional[Dict] = None) -> CameraStream:
        """Start (or restart) the capture stream.

        ``cfg`` is an explicit, *session* camera configuration (from the
        selection dialog). When ``None`` the persisted settings are used.
        """
        with self._lock:
            self.stop()
            spec = self.resolve_spec(cfg)
            self._spec = spec
            reconnect = float((cfg or self.config()).get("reconnect_delay_sec", 3.0))
            self._stream = CameraStream(
                source=spec.handle,
                resolution=(spec.width, spec.height),
                fps=spec.fps,
                reconnect_delay=reconnect,
                name=spec.label,
            ).start()
            logger.info("CameraManager active source: %s (handle=%r)", spec.label, spec.handle)
            return self._stream

    def wait_until_connected(self, timeout: float = 4.0) -> bool:
        """Block briefly until the active stream produces frames."""
        import time
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.connected:
                return True
            time.sleep(0.1)
        return self.connected

    def set_active_camera(self, cfg: Dict) -> Dict:
        """Open ``cfg`` as the active camera and, on success, persist it.

        Returns ``{success, message, ...source info}``. Used by the pipeline when
        the user selects a camera on the management page.
        """
        try:
            self.open(cfg)
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to open selected camera: %s", exc)
            self.stop()
            return {"success": False, "message": "Camera Connection Failed"}

        if not self.wait_until_connected(timeout=5.0):
            self.stop()
            return {"success": False, "message": "Camera Connection Failed"}

        self.save_active_camera(cfg)
        info = self.describe()
        info.update({"success": True, "message": "Camera Connected"})
        return info

    def get_camera_preview(self, cfg: Dict) -> Dict:
        """Return a one-off snapshot preview for ``cfg`` without making it active."""
        return self.test_camera(cfg)

    def test_camera(self, cfg: Dict) -> Dict:
        """Open ``cfg`` briefly, grab a frame, return status + snapshot preview."""
        return self.test_connection(cfg)

    def read(self) -> Optional[np.ndarray]:
        """Return the most recent frame from the active source (or ``None``)."""
        with self._lock:
            return self._stream.read() if self._stream else None

    def stop(self) -> None:
        with self._lock:
            if self._stream is not None:
                self._stream.stop()
                self._stream = None

    @property
    def connected(self) -> bool:
        with self._lock:
            return bool(self._stream and self._stream.connected)

    @property
    def spec(self) -> Optional[SourceSpec]:
        return self._spec

    def resolution(self) -> Tuple[int, int]:
        if self._spec:
            return self._spec.width, self._spec.height
        return self._dimensions(self.config())

    # ------------------------------------------------------------- dashboard
    def describe(self) -> Dict[str, object]:
        """Active-source summary for the dashboard / live header."""
        spec = self._spec or self.resolve_spec()
        return {
            "camera_type": spec.camera_type,
            "label": spec.label,
            "resolution": f"{spec.width}x{spec.height}",
            "width": spec.width,
            "height": spec.height,
            "fps": spec.fps,
            "status": "connected" if self.connected else "disconnected",
        }

    def active_summary(self) -> Optional[Dict[str, object]]:
        """Canonical active-camera info from the *persisted config*.

        Works whether or not detection is currently streaming, so the dashboard
        and camera page show the chosen camera even while detection is stopped.
        """
        cfg = self.get_active_camera()
        if not cfg:
            return None
        spec = self.resolve_spec(cfg)
        return {
            "camera_type": spec.camera_type,
            "label": spec.label,
            "resolution": f"{spec.width}x{spec.height}" if spec.camera_type != "rtsp" else "—",
            "width": spec.width,
            "height": spec.height,
            "fps": spec.fps,
        }

    # ------------------------------------------------------------ connectivity
    def test_connection(self, cfg: Dict) -> Dict[str, object]:
        """Open the selected source, grab a frame, and return a preview snapshot.

        If the requested source is the one already running, the live frame is
        reused instead of opening a second handle (avoids "device busy" on USB).
        """
        spec = self.resolve_spec(cfg)
        if spec.camera_type == "rtsp" and not str(spec.handle):
            return {"success": False, "message": "No RTSP URL or IP address configured."}
        if spec.camera_type == "file" and not str(spec.handle):
            return {"success": False, "message": "No video file path configured."}

        frame = self._grab_active_if_same(spec)
        opened_temp = False
        if frame is None:
            frame, opened_temp = self._grab_once(spec)

        if frame is None:
            return {
                "success": False,
                "message": "Camera Connection Failed",
                "label": spec.label,
            }

        h, w = frame.shape[:2]
        return {
            "success": True,
            "message": "Camera Connected",
            "label": spec.label,
            "width": w,
            "height": h,
            "resolution": f"{w}x{h}",
            # Camera Settings preview is capped at 400x250 — snapshot only, no
            # detection processing happens here.
            "snapshot": self._encode_snapshot(frame, max_w=400, max_h=250),
            "reused_live": not opened_temp,
        }

    def _grab_active_if_same(self, spec: SourceSpec) -> Optional[np.ndarray]:
        with self._lock:
            if self._stream and self._spec and self._spec.handle == spec.handle \
                    and self._stream.connected:
                return self._stream.read()
        return None

    def _grab_once(self, spec: SourceSpec) -> Tuple[Optional[np.ndarray], bool]:
        """Open a temporary capture, grab a single frame, release it."""
        handle = spec.handle
        cap = cv2.VideoCapture(handle, cv2.CAP_FFMPEG) if isinstance(handle, str) \
            else cv2.VideoCapture(handle)
        try:
            if isinstance(handle, str):
                # Fail fast on unreachable IP cameras instead of hanging ~30s.
                for prop in ("CAP_PROP_OPEN_TIMEOUT_MSEC", "CAP_PROP_READ_TIMEOUT_MSEC"):
                    if hasattr(cv2, prop):
                        cap.set(getattr(cv2, prop), 5000)
            if isinstance(handle, int):
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, spec.width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, spec.height)
            if not cap.isOpened():
                return None, True
            # A couple of warm-up reads — first frame is often empty on USB.
            frame = None
            for _ in range(3):
                ok, f = cap.read()
                if ok and f is not None:
                    frame = f
                    break
            return frame, True
        except Exception as exc:  # noqa: BLE001
            logger.error("Camera test error: %s", exc)
            return None, True
        finally:
            cap.release()

    @staticmethod
    def _encode_snapshot(frame: np.ndarray, max_w: int = 640,
                         max_h: Optional[int] = None) -> str:
        """Return a base64 data-URI JPEG (downscaled) for the UI preview."""
        h, w = frame.shape[:2]
        scale = 1.0
        if w > max_w:
            scale = max_w / float(w)
        if max_h is not None and h * scale > max_h:
            scale = max_h / float(h)
        if scale < 1.0:
            frame = cv2.resize(frame, (max(1, int(w * scale)), max(1, int(h * scale))))
        ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
        if not ok:
            return ""
        return "data:image/jpeg;base64," + base64.b64encode(buf).decode("ascii")
