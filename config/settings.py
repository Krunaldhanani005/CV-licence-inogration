"""Central configuration management.

Loads ``config/default.yaml`` and merges any runtime overrides persisted by the
Admin Panel into ``data/configs/runtime.json``. Exposes a singleton accessor so
every module shares the same configuration instance.
"""
from __future__ import annotations

import json
import logging
import os
import threading
from copy import deepcopy
from typing import Any, Dict, Optional

import yaml

logger = logging.getLogger(__name__)

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEFAULT_YAML = os.path.join(_BASE_DIR, "config", "default.yaml")
_RUNTIME_JSON = os.path.join(_BASE_DIR, "data", "configs", "runtime.json")


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge ``override`` into ``base`` (returns a new dict)."""
    result = deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


class Settings:
    """Thread-safe, dot/section accessible configuration container."""

    def __init__(self, base_dir: str = _BASE_DIR) -> None:
        self.base_dir = base_dir
        self._lock = threading.RLock()
        self._data: Dict[str, Any] = {}
        self.reload()

    # ----------------------------------------------------------------- loading
    def reload(self) -> None:
        with self._lock:
            with open(_DEFAULT_YAML, "r", encoding="utf-8") as fh:
                data = yaml.safe_load(fh) or {}
            overrides = self._load_runtime()
            self._data = _deep_merge(data, overrides)
        logger.info("Configuration loaded (overrides=%s)", bool(overrides))
        self._ensure_dirs()

    def _load_runtime(self) -> Dict[str, Any]:
        if not os.path.exists(_RUNTIME_JSON):
            return {}
        try:
            with open(_RUNTIME_JSON, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Failed reading runtime overrides: %s", exc)
            return {}

    def _ensure_dirs(self) -> None:
        for key in ("data_dir", "faces_dir", "embeddings_dir", "faiss_dir",
                    "configs_dir", "models_dir", "logs_dir"):
            path = self.path(key)
            if path:
                os.makedirs(path, exist_ok=True)

    # ----------------------------------------------------------------- access
    def section(self, name: str) -> Dict[str, Any]:
        with self._lock:
            return deepcopy(self._data.get(name, {}))

    def get(self, section: str, key: str, default: Any = None) -> Any:
        with self._lock:
            return self._data.get(section, {}).get(key, default)

    def path(self, key: str) -> str:
        """Return an absolute path for a configured directory/file key."""
        rel = self._data.get("paths", {}).get(key, "")
        if not rel:
            return ""
        return rel if os.path.isabs(rel) else os.path.join(self.base_dir, rel)

    @property
    def all(self) -> Dict[str, Any]:
        with self._lock:
            return deepcopy(self._data)

    # ----------------------------------------------------------------- updates
    def update_section(self, name: str, values: Dict[str, Any]) -> None:
        """Persist runtime overrides for a section and hot-reload."""
        with self._lock:
            overrides = self._load_runtime()
            overrides.setdefault(name, {})
            overrides[name].update(values)
            os.makedirs(os.path.dirname(_RUNTIME_JSON), exist_ok=True)
            with open(_RUNTIME_JSON, "w", encoding="utf-8") as fh:
                json.dump(overrides, fh, indent=2)
        self.reload()
        logger.info("Updated config section '%s'", name)


_settings_singleton: Optional[Settings] = None
_singleton_lock = threading.Lock()


def get_settings() -> Settings:
    """Return the process-wide :class:`Settings` singleton."""
    global _settings_singleton
    if _settings_singleton is None:
        with _singleton_lock:
            if _settings_singleton is None:
                _settings_singleton = Settings()
    return _settings_singleton
