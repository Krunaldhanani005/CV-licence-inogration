"""Department definitions + consistent colour management.

A single source of truth for department colours so that bounding boxes, name
labels, dashboard statistics and department cards all match exactly. Fixed
departments have curated colours; custom departments get a deterministic colour
derived from their name (stable across restarts).
"""
from __future__ import annotations

import hashlib
from typing import Dict, List, Tuple

# Canonical department list (order used by dropdowns).
DEPARTMENTS: List[str] = [
    "Sales",
    "Technical",
    "Robotics",
    "Marketing",
    "Finance",
    "Human Resources",
    "Management",
    "Other",
]

# Curated colours (hex). Keys are lower-cased for lookup.
_FIXED_COLORS: Dict[str, str] = {
    "sales": "#22C55E",            # green
    "technical": "#3B82F6",        # blue
    "robotics": "#8B5CF6",         # purple
    "marketing": "#F97316",        # orange
    "finance": "#EAB308",          # yellow
    "human resources": "#EC4899",  # pink
    "management": "#EF4444",       # red
}

# Modern department icons (frontend uses these names to pick an SVG).
_ICONS: Dict[str, str] = {
    "sales": "trending",
    "technical": "code",
    "robotics": "robot",
    "marketing": "megaphone",
    "finance": "currency",
    "human resources": "people",
    "management": "briefcase",
    "other": "tag",
}

GUEST_COLOR = "#64748B"   # neutral slate for guests / unknown


def normalize(department: str) -> str:
    return (department or "").strip()


def color_for(department: str) -> str:
    """Return a stable hex colour for any department (fixed or custom)."""
    dept = normalize(department)
    if not dept:
        return GUEST_COLOR
    key = dept.lower()
    if key in _FIXED_COLORS:
        return _FIXED_COLORS[key]
    # Deterministic, pleasant colour from the name hash (HSL with fixed S/L).
    digest = hashlib.md5(key.encode("utf-8")).hexdigest()
    hue = int(digest[:8], 16) % 360
    return _hsl_to_hex(hue, 0.58, 0.55)


def icon_for(department: str) -> str:
    return _ICONS.get(normalize(department).lower(), "tag")


def hex_to_bgr(hex_color: str) -> Tuple[int, int, int]:
    """Convert ``#RRGGBB`` to an OpenCV BGR tuple."""
    h = (hex_color or "").lstrip("#")
    if len(h) != 6:
        return (139, 92, 86)  # fallback
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return (b, g, r)


def catalog() -> List[Dict[str, str]]:
    """Return the department catalogue (name, color, icon) for the frontend."""
    out = []
    for name in DEPARTMENTS:
        out.append({"name": name, "color": color_for(name), "icon": icon_for(name)})
    return out


def _hsl_to_hex(h: float, s: float, l: float) -> str:
    """HSL (h in degrees, s/l in 0..1) -> #RRGGBB."""
    c = (1 - abs(2 * l - 1)) * s
    x = c * (1 - abs((h / 60.0) % 2 - 1))
    m = l - c / 2
    if h < 60:    r, g, b = c, x, 0
    elif h < 120: r, g, b = x, c, 0
    elif h < 180: r, g, b = 0, c, x
    elif h < 240: r, g, b = 0, x, c
    elif h < 300: r, g, b = x, 0, c
    else:         r, g, b = c, 0, x
    return "#{:02X}{:02X}{:02X}".format(
        int((r + m) * 255), int((g + m) * 255), int((b + m) * 255))
