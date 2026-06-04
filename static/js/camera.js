/* Camera Management page: auto-scan, snapshot preview, select active, add RTSP.
   No live detection stream here — preview is a single still frame (max 400x250). */
const grid = document.getElementById("cameraGrid");
const snapPreview = document.getElementById("snapPreview");
const prevEmpty = document.getElementById("prevEmpty");
let cameras = [];

// ------------------------------------------------------------- scan
async function loadCameras(rescan) {
  grid.innerHTML = `<div class="cam-loading"><span class="spinner"></span> Scanning cameras…</div>`;
  let data = {};
  try {
    const r = rescan ? await API.post("/api/cameras/scan") : await API.get("/api/cameras");
    data = r.data || {};
  } catch (e) { /* ignore */ }
  cameras = data.available || [];
  renderCameras();
  renderActive(data.active);
}

function renderCameras() {
  grid.innerHTML = "";
  if (!cameras.length) {
    grid.innerHTML = `<div class="cam-empty">No cameras detected. Connect a USB camera and Rescan, or add an RTSP camera.</div>`;
    return;
  }
  cameras.forEach(c => grid.appendChild(cameraCard(c)));
}

function cameraCard(c) {
  const el = document.createElement("div");
  el.className = "card camera-card" + (c.active ? " active" : "");
  const isRtsp = c.type === "rtsp";
  const icon = isRtsp ? "🌐" : "📷";
  const thumb = c.thumbnail
    ? `<img class="cc-thumb" src="${c.thumbnail}"/>`
    : `<div class="cc-thumb glyph">${icon}</div>`;
  const sub = isRtsp ? "RTSP / IP Camera" : `USB · index ${c.index}`;
  el.innerHTML = `
    <div class="cc-top">
      ${thumb}
      <span class="badge ${c.active ? 'ok' : (c.status==='Connected'?'ok':'')}" >
        ${c.active ? 'Currently Active' : (c.status || 'Ready')}</span>
    </div>
    <div class="cc-name">${icon} ${escapeHtml(c.name)}</div>
    <div class="cc-sub">${sub} · ${c.resolution || '—'}</div>
    <div class="cc-actions">
      <button class="btn ghost cc-prev">Preview</button>
      <button class="btn ${c.active ? 'secondary' : 'green'} cc-sel" ${c.active ? 'disabled' : ''}>
        ${c.active ? 'Active' : 'Select'}</button>
    </div>
    ${isRtsp ? '<button class="cc-del" title="Remove">×</button>' : ''}`;

  el.querySelector(".cc-prev").onclick = () => previewCamera(c);
  el.querySelector(".cc-sel").onclick = () => selectCamera(c);
  const del = el.querySelector(".cc-del");
  if (del) del.onclick = (e) => { e.stopPropagation(); removeRtsp(c); };
  return el;
}

function cfgOf(c) {
  if (c.type === "rtsp") {
    const cc = c.config || {};
    return { camera_type: "rtsp", camera_name: c.name, rtsp_url: c.rtsp_url || cc.rtsp_url,
             ip_address: cc.ip_address || "", username: cc.username || "" };
  }
  return { camera_type: "usb", camera_index: c.index, name: c.name,
           width: c.width, height: c.height, fps: 30 };
}

// ------------------------------------------------------------- preview (still)
async function previewCamera(c) {
  setPreviewMeta(c.name, c.resolution, "Loading…", false);
  const r = await API.post("/api/cameras/preview", cfgOf(c));
  if (r.success && r.snapshot) {
    showSnapshot(r.snapshot);
    setPreviewMeta(c.name, r.resolution || c.resolution, "Connected", true);
  } else if (c.thumbnail) {
    showSnapshot(c.thumbnail);
    setPreviewMeta(c.name, c.resolution, "Preview", true);
  } else {
    toast(r.message || "Preview failed", "err");
    setPreviewMeta(c.name, c.resolution, "Failed", false);
  }
}

// ------------------------------------------------------------- select active
async function selectCamera(c) {
  const r = await API.post("/api/cameras/select", cfgOf(c));
  if (r.success) {
    toast(`${c.name} saved as the active camera`, "ok");
    if (r.snapshot) { showSnapshot(r.snapshot); setPreviewMeta(c.name, r.resolution || c.resolution, "Active", true); }
    loadCameras(false);
  } else {
    toast(r.message || "Camera Connection Failed", "err");
  }
}

async function removeRtsp(c) {
  if (!confirm(`Remove ${c.name}?`)) return;
  await API.del("/api/cameras/rtsp?rtsp_url=" + encodeURIComponent(c.rtsp_url || ""));
  loadCameras(false);
}

// ------------------------------------------------------------- preview helpers
function showSnapshot(src) {
  prevEmpty.style.display = "none";
  snapPreview.src = src; snapPreview.style.display = "block";
}
function setPreviewMeta(name, res, status, ok) {
  prevName.textContent = name || "—";
  prevRes.textContent = res || "—";
  prevStatus.textContent = status || "—";
  document.getElementById("prevDot").className = "status-dot " + (ok ? "on" : "off");
  document.getElementById("prevConn").textContent = ok ? "Connected" : "Preview";
}

function renderActive(active) {
  const box = document.getElementById("activeBox");
  if (active && active.label) {
    box.className = "active-box on";
    document.getElementById("activeDot").className = "status-dot on";
    document.getElementById("activeName").textContent = active.label;
    document.getElementById("activeSub").textContent =
      `${active.resolution} · ${active.status === "connected" ? "Live" : active.status}`;
  } else {
    box.className = "active-box none";
    document.getElementById("activeDot").className = "status-dot off";
    document.getElementById("activeName").textContent = "No active camera";
    document.getElementById("activeSub").textContent = "Select a camera to begin detection";
  }
}

// ------------------------------------------------------------- IP camera modal
const rtspModal = document.getElementById("rtspModal");
let rtspTested = null;

// Brand RTSP path templates (must mirror the backend BRAND_PATHS).
const BRAND_PATHS = {
  hikvision: "/Streaming/Channels/{ch}01",
  dahua: "/cam/realmonitor?channel={ch}&subtype=0",
  reolink: "/h264Preview_0{ch}_main",
  axis: "/axis-media/media.amp",
  onvif: "/onvif1",
  generic: "/11",
};

function ipMode() {
  return document.querySelector('input[name="ipmode"]:checked').value;
}
function applyMode() {
  const url = ipMode() === "url";
  document.getElementById("ipUrl").style.display = url ? "block" : "none";
  document.getElementById("ipBuild").style.display = url ? "none" : "block";
  document.querySelectorAll("#modeSeg .seg-opt").forEach(o =>
    o.classList.toggle("active", o.querySelector("input").checked));
  updateResolved();
}
document.querySelectorAll('input[name="ipmode"]').forEach(r => r.addEventListener("change", applyMode));

document.getElementById("rBrand").onchange = () => {
  document.getElementById("pathWrap").style.display =
    (rBrand.value === "custom") ? "block" : "none";
  updateResolved();
};
["rIp","rPort","rUser","rPass","rChannel","rPath","rUrl"].forEach(id =>
  document.getElementById(id).addEventListener("input", updateResolved));

function buildUrl() {
  if (ipMode() === "url") return rUrl.value.trim();
  const ip = rIp.value.trim();
  if (!ip) return "";
  const port = rPort.value.trim() || "554";
  const user = rUser.value.trim(), pass = rPass.value;
  const auth = user ? `${encodeURIComponent(user)}:${encodeURIComponent(pass)}@` : "";
  const ch = rChannel.value.trim() || "1";
  let path = (rBrand.value === "custom")
    ? (rPath.value.trim() || "/")
    : (BRAND_PATHS[rBrand.value] || BRAND_PATHS.hikvision).replace("{ch}", ch);
  if (!path.startsWith("/")) path = "/" + path;
  return `rtsp://${auth}${ip}:${port}${path}`;
}
function updateResolved() { document.getElementById("rResolved").value = buildUrl(); }

function rtspCfg() {
  const url = buildUrl();
  return {
    camera_type: "rtsp",
    camera_name: rName.value.trim() || "IP Camera",
    ip_address: ipMode() === "url" ? "" : rIp.value.trim(),
    port: parseInt(rPort.value || "554", 10),
    channel: parseInt(rChannel.value || "1", 10),
    username: ipMode() === "url" ? "" : rUser.value.trim(),
    password: ipMode() === "url" ? "" : rPass.value,
    brand: rBrand.value,
    stream_path: rBrand.value === "custom" ? rPath.value.trim() : "",
    rtsp_url: url,
  };
}

document.getElementById("addRtspBtn").onclick = () => {
  ["rName","rIp","rUser","rPass","rUrl","rPath"].forEach(id => document.getElementById(id).value = "");
  rPort.value = 554; rChannel.value = 1; rBrand.value = "hikvision";
  document.querySelector('input[name="ipmode"][value="build"]').checked = true;
  document.getElementById("pathWrap").style.display = "none";
  rtspTested = null; rtspSave.disabled = true;
  rtspMsg.style.display = "none"; document.getElementById("rtspPrevWrap").style.display = "none";
  applyMode();
  rtspModal.classList.add("show");
};
document.getElementById("rtspCancel").onclick = () => rtspModal.classList.remove("show");
rtspModal.addEventListener("click", e => { if (e.target === rtspModal) rtspModal.classList.remove("show"); });

document.getElementById("rtspTest").onclick = async () => {
  const cfg = rtspCfg();
  if (!cfg.rtsp_url) { banner("Enter an IP address (or full RTSP URL).", false); return; }
  const btn = document.getElementById("rtspTest");
  btn.disabled = true; btn.textContent = "Connecting…";
  const r = await API.post("/api/cameras/test", cfg);
  btn.disabled = false; btn.textContent = "Test Connection";
  banner((r.success ? "✓ " : "✕ ") + (r.message || "") + (r.resolution ? " · " + r.resolution : ""), r.success);
  const wrap = document.getElementById("rtspPrevWrap");
  if (r.success && r.snapshot) { document.getElementById("rtspPrevImg").src = r.snapshot; wrap.style.display = "block"; }
  rtspTested = r.success ? cfg : null;
  rtspSave.disabled = !r.success;
};
document.getElementById("rtspSave").onclick = async () => {
  if (!rtspTested) return;
  const r = await API.post("/api/cameras/rtsp", rtspTested);
  toast(r.message || "Saved", r.success ? "ok" : "err");
  if (r.success) { rtspModal.classList.remove("show"); loadCameras(false); }
};
function banner(text, ok) {
  rtspMsg.style.display = "block";
  rtspMsg.className = "test-banner " + (ok ? "ok" : "err");
  rtspMsg.textContent = text;
}

document.getElementById("rescanBtn").onclick = () => loadCameras(true);

function escapeHtml(s) {
  return (s || "").replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

loadCameras(false);
