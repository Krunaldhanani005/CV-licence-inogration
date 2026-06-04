"""HTML page routes (server-rendered shells; data is fetched via the API)."""
from __future__ import annotations

from flask import Blueprint, render_template

pages_bp = Blueprint("pages", __name__)


@pages_bp.route("/")
def dashboard():
    return render_template("dashboard.html", active="dashboard")


@pages_bp.route("/camera")
def camera():
    return render_template("camera.html", active="camera")


@pages_bp.route("/people")
def people():
    return render_template("people.html", active="people")


@pages_bp.route("/settings")
def settings():
    return render_template("settings.html", active="settings")
