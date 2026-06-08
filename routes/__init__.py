"""Flask blueprints."""
from .pages import pages_bp
from .api import api_bp
from .object_api import object_bp

__all__ = ["pages_bp", "api_bp", "object_bp"]
