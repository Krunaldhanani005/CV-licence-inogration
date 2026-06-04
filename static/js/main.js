/* Shared helpers: API fetch, toast, clock, global camera/FPS poll. */
const API = {
  async get(url) { return this._json(await fetch(url)); },
  async post(url, body, isForm = false) {
    const opts = { method: "POST" };
    if (isForm) { opts.body = body; }
    else { opts.headers = { "Content-Type": "application/json" }; opts.body = JSON.stringify(body || {}); }
    return this._json(await fetch(url, opts));
  },
  async del(url) { return this._json(await fetch(url, { method: "DELETE" })); },
  async _json(res) {
    let data = {};
    try { data = await res.json(); } catch (e) { /* ignore */ }
    return { ok: res.ok, status: res.status, ...data };
  },
};

function toast(message, kind = "ok") {
  const el = document.getElementById("toast");
  if (!el) return;
  el.textContent = message;
  el.className = `toast show ${kind}`;
  clearTimeout(el._t);
  el._t = setTimeout(() => { el.className = "toast"; }, 3200);
}

function tickClock() {
  const el = document.getElementById("clock");
  if (!el) return;
  const d = new Date();
  el.textContent = d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
}

async function pollGlobalStatus() {
  try {
    const r = await API.get("/api/stats");
    const s = r.data || {};
    const fps = document.getElementById("fpsPill");
    if (fps) fps.textContent = (s.fps ?? 0).toFixed ? s.fps.toFixed(1) : s.fps;
    const dot = document.getElementById("globalCamDot");
    const lbl = document.getElementById("globalCamStatus");
    const connected = s.camera_status === "connected";
    if (dot) dot.className = "status-dot " + (connected ? "on" : "off");
    if (lbl) lbl.textContent = connected ? "Camera online" : "Camera offline";
    window.__lastStats = s;
  } catch (e) { /* ignore */ }
}

setInterval(tickClock, 1000); tickClock();
setInterval(pollGlobalStatus, 2000); pollGlobalStatus();
