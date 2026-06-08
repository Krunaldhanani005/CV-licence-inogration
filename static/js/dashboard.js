/* Dashboard: mode selector (FR / OD), Start/Stop, analytics, fullscreen, reception. */
const toggleBtn      = document.getElementById("toggleBtn");
const toggleLabel    = document.getElementById("toggleLabel");
const ovStart        = document.getElementById("ovStart");
const fsBtn          = document.getElementById("fsBtn");
const receptionBtn   = document.getElementById("receptionBtn");
const exitReceptionBtn = document.getElementById("exitReceptionBtn");
const videoWrap      = document.getElementById("videoWrap");
const overlay        = document.getElementById("streamOverlay");

let running     = false;
let busy        = false;
let currentMode = "fr";   // "fr" | "od"

// ------------------------------------------------------- mode switching
async function switchMode(mode) {
  if (mode === currentMode) return;
  const prevMode = currentMode;
  currentMode = mode;
  applyModeUI();
  const r = await API.post("/api/mode", { mode });
  if (!r.success) {
    currentMode = prevMode;
    applyModeUI();
    toast(r.message || "Could not switch mode", "err");
    return;
  }
  toast(mode === "od" ? "Object Detection mode" : "Face Recognition mode", "ok");
  refresh();
}

function applyModeUI() {
  const isFR = currentMode === "fr";
  // Mode buttons
  document.getElementById("modeFR").classList.toggle("active", isFR);
  document.getElementById("modeOD").classList.toggle("active", !isFR);
  // KPI grids
  document.getElementById("frKpis").style.display     = isFR ? "" : "none";
  document.getElementById("odKpis").style.display     = isFR ? "none" : "";
  // Status rows
  document.getElementById("frStatusRows").style.display = isFR ? "" : "none";
  document.getElementById("odStatusRows").style.display = isFR ? "none" : "";
  // Bottom cards
  document.getElementById("deptCard").style.display   = isFR ? "" : "none";
  document.getElementById("classCard").style.display  = isFR ? "none" : "";
  // Analytics strip
  document.getElementById("frStrip").style.display    = isFR ? "" : "none";
  document.getElementById("odStrip").style.display    = isFR ? "none" : "";
  // Dash title
  document.getElementById("dashTitle").textContent = isFR
    ? "AI Reception Monitoring"
    : "Object Detection";
  document.getElementById("dashSub").textContent = isFR
    ? "Real-time person detection, face recognition & activity"
    : "Real-time multi-class object detection & tracking";
}

async function loadInitialMode() {
  try {
    const r = await API.get("/api/mode");
    if (r.success && r.data && r.data.mode) {
      currentMode = r.data.mode;
      applyModeUI();
    }
  } catch (e) {}
}

// ------------------------------------------------------- start/stop toggle
async function startDetection() {
  if (busy) return; busy = true;
  setToggle("starting");
  const r = await API.post("/api/detection/start");
  busy = false;
  if (!r.success) { toast(r.message || "Could not start detection", "err"); }
  else { toast("Detection started", "ok"); }
  refresh();
}
async function stopDetection() {
  if (busy) return; busy = true;
  setToggle("stopping");
  await API.post("/api/detection/stop");
  busy = false;
  toast("Detection stopped — CPU idle", "ok");
  refresh();
}
function onToggle() { running ? stopDetection() : startDetection(); }
toggleBtn.onclick = onToggle;
ovStart.onclick   = startDetection;

function setToggle(state) {
  if (state === "starting") {
    toggleBtn.className = "btn secondary toggle-btn";
    toggleLabel.textContent = "Starting…";
    toggleBtn.querySelector(".t-ico").textContent = "…"; return;
  }
  if (state === "stopping") {
    toggleBtn.className = "btn secondary toggle-btn";
    toggleLabel.textContent = "Stopping…";
    toggleBtn.querySelector(".t-ico").textContent = "…"; return;
  }
  if (running) {
    toggleBtn.className = "btn danger toggle-btn";
    toggleBtn.querySelector(".t-ico").textContent = "■";
    toggleLabel.textContent = "Stop Detection";
  } else {
    toggleBtn.className = "btn green toggle-btn";
    toggleBtn.querySelector(".t-ico").textContent = "▶";
    toggleLabel.textContent = "Start Detection";
  }
}

// ------------------------------------------------------- fullscreen
fsBtn.onclick = () => {
  const fsEl = document.fullscreenElement || document.webkitFullscreenElement;
  if (fsEl) (document.exitFullscreen || document.webkitExitFullscreen).call(document);
  else (videoWrap.requestFullscreen || videoWrap.webkitRequestFullscreen).call(videoWrap);
};
document.addEventListener("fullscreenchange", () =>
  videoWrap.classList.toggle("is-fullscreen", !!document.fullscreenElement));

// ------------------------------------------------------- reception mode
function enterReception() { document.body.classList.add("reception"); exitReceptionBtn.style.display = "block"; }
function exitReception()  { document.body.classList.remove("reception"); exitReceptionBtn.style.display = "none"; }
receptionBtn.onclick  = enterReception;
exitReceptionBtn.onclick = exitReception;
document.addEventListener("keydown", e => { if (e.key === "Escape") exitReception(); });

// ------------------------------------------------------- overlay states
function setOverlay(status) {
  const map = {
    no_source:  { show: true, icon: "⏻", title: "Detection Stopped", sub: "Press start to begin live monitoring", start: true },
    lost:       { show: true, icon: "⚠", title: "Camera Lost", sub: "Reconnecting to the camera…", start: false },
    connecting: { show: true, icon: "◌", title: "Connecting…", sub: "Opening the camera stream", start: false },
    connected:  { show: false },
  };
  const s = map[status] || map.no_source;
  overlay.style.display = s.show ? "flex" : "none";
  if (s.show) {
    document.getElementById("ovIcon").textContent  = s.icon;
    document.getElementById("ovTitle").textContent = s.title;
    document.getElementById("ovSub").textContent   = s.sub;
    ovStart.style.display = s.start ? "inline-flex" : "none";
  }
}

// ------------------------------------------------------- FR stats update
function updateFRStats(s) {
  const people = s.people_count ?? 0, rec = s.recognized_count ?? 0, guest = s.guest_count ?? 0;
  setText("cPeople",    people); setText("sPeople", people);
  setText("cRecognized", rec);   setText("sRec",    rec);
  setText("cGuests",    guest);  setText("sGuest",  guest);
}

// ------------------------------------------------------- OD stats update
function updateODStats(s) {
  const total = s.total_objects ?? 0, types = s.unique_classes ?? 0;
  setText("cObjects", total);  setText("sObjects", total);
  setText("cTypes",   types);  setText("sTypes",   types);
  setText("cOdFps",  (s.fps ?? 0).toFixed ? (s.fps ?? 0).toFixed(1) : s.fps);

  // Class counts breakdown
  const counts = s.class_counts || {};
  const el = document.getElementById("classCounts");
  const entries = Object.entries(counts).sort((a, b) => b[1] - a[1]);
  if (!entries.length) {
    el.innerHTML = '<div class="help">No objects detected yet.</div>';
  } else {
    el.innerHTML = entries.map(([name, cnt]) => {
      const colors = ["#3b82f6","#22c55e","#f59e0b","#ef4444","#8b5cf6",
                      "#06b6d4","#ec4899","#84cc16","#f97316"];
      const color  = colors[Math.abs(hashStr(name)) % colors.length];
      return `<div class="dept-stat" style="border-left-color:${color}">
        <span style="font-size:18px;width:30px;text-align:center;color:${color}">◈</span>
        <span class="ds-name">${escapeHtml(name)}</span>
        <span class="ds-count" style="color:${color}">${cnt}</span>
      </div>`;
    }).join("");
  }
}

function hashStr(s) {
  let h = 0;
  for (let i = 0; i < s.length; i++) h = (Math.imul(31, h) + s.charCodeAt(i)) | 0;
  return h;
}

// ------------------------------------------------------- OD enabled classes count
async function loadEnabledClassCount() {
  try {
    const r = await API.get("/api/object/classes");
    if (!r.success) return;
    const count = (r.data.classes || []).filter(c => c.enabled).length;
    setText("cEnabledClasses", count);
  } catch (e) {}
}

// ------------------------------------------------------- poll
async function refresh() {
  let s = {}, a = {};
  try { s = (await API.get("/api/stats")).data || {}; } catch (e) {}
  try { a = (await API.get("/api/camera/active")).data || {}; } catch (e) {}

  const fps    = s.fps ?? 0;
  const fpsTxt = fps.toFixed ? fps.toFixed(1) : fps;
  setText("cFps", fpsTxt); setText("sFps", fpsTxt);

  if (currentMode === "od") {
    updateODStats(s);
  } else {
    updateFRStats(s);
  }

  const status = s.camera_status || "no_source";
  const label  = { connected: "Running", lost: "Camera Lost", no_source: "Stopped" }[status] || status;
  setText("sCam", label);
  const cam = document.getElementById("cCam");
  cam.textContent  = label;
  cam.className    = "badge " + (status === "connected" ? "ok" : "bad");

  const on = status === "connected";
  document.getElementById("liveDot").className      = "status-dot " + (on ? "on" : "off");
  document.getElementById("liveStatus").textContent =
    on ? "LIVE" : (status === "lost" ? "Camera Lost" : "Stopped");
  document.getElementById("liveBadge").classList.toggle("live", on);

  if (a.has_source || (a.label && a.label !== "None")) {
    setText("streamCam", a.label || "Camera");
    setText("streamSub", (a.resolution || "") + (a.camera_type ? " · " + a.camera_type.toUpperCase() : ""));
  } else {
    setText("streamCam", "No camera selected");
    setText("streamSub", "Configure in Camera Settings");
  }

  running = !!a.detection_running;
  if (!busy) setToggle();

  let ovState = "no_source";
  if (running) ovState = on ? "connected" : (status === "lost" ? "lost" : "connecting");
  setOverlay(ovState);
}

// ------------------------------------------------------- FR dept stats
const DEPT_SVG = {
  trending: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="3 17 9 11 13 15 21 7"/><polyline points="15 7 21 7 21 13"/></svg>',
  code: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="16 18 22 12 16 6"/><polyline points="8 6 2 12 8 18"/></svg>',
  robot: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="4" y="8" width="16" height="11" rx="2"/><path d="M12 4v4"/><circle cx="9" cy="13" r="1"/><circle cx="15" cy="13" r="1"/><path d="M2 13v2M22 13v2"/></svg>',
  megaphone: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 11l15-6v14l-15-6z"/><path d="M3 11v3a2 2 0 0 0 2 2h2"/></svg>',
  currency: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="9"/><path d="M14.5 9a2.5 2.5 0 0 0-2.5-1.5C10.5 7.5 9.5 8.4 9.5 9.7c0 2.8 5 1.6 5 4.6 0 1.3-1 2.2-2.5 2.2A2.6 2.6 0 0 1 9.4 15M12 6v12"/></svg>',
  people: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/><path d="M23 21v-2a4 4 0 0 0-3-3.87M16 3.13A4 4 0 0 1 16 11"/></svg>',
  briefcase: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="2" y="7" width="20" height="14" rx="2"/><path d="M16 21V5a2 2 0 0 0-2-2h-4a2 2 0 0 0-2 2v16"/></svg>',
  tag: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M20.6 13.4 11 3.8a2 2 0 0 0-1.4-.6H4a1 1 0 0 0-1 1v5.6a2 2 0 0 0 .6 1.4l9.6 9.6a2 2 0 0 0 2.8 0l4.6-4.6a2 2 0 0 0 0-2.8z"/><circle cx="7.5" cy="7.5" r="1"/></svg>',
};
let deptCatalog = [];
async function loadDeptCatalog() {
  try { deptCatalog = ((await API.get("/api/departments")).data || {}).departments || []; } catch (e) {}
}
function deptMeta(name) {
  const d = deptCatalog.find(x => x.name.toLowerCase() === (name || "").toLowerCase());
  return d || { color: "#64748B", icon: "tag" };
}
async function loadEnrolled() {
  let people = [];
  try { people = (await API.get("/api/people")).data || []; } catch (e) {}
  setText("cEnrolled", people.length);
  const counts = {};
  people.forEach(p => { const d = p.department || "Other"; counts[d] = (counts[d] || 0) + 1; });
  const el = document.getElementById("deptStats");
  const entries = Object.entries(counts).sort((a, b) => b[1] - a[1]);
  if (!entries.length) { el.innerHTML = '<div class="help">No employees enrolled yet.</div>'; return; }
  el.innerHTML = entries.map(([name, count]) => {
    const m = deptMeta(name);
    const color = (people.find(p => (p.department || "Other") === name) || {}).color || m.color;
    return `<div class="dept-stat" style="border-left-color:${color}">
      <span class="ds-ico" style="color:${color}">${DEPT_SVG[m.icon] || DEPT_SVG.tag}</span>
      <span class="ds-name">${escapeHtml(name)}</span>
      <span class="ds-count" style="color:${color}">${count}</span>
    </div>`;
  }).join("");
}

function escapeHtml(s) { return (s || "").replace(/[&<>"']/g, c => ({ "&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;" }[c])); }
function setText(id, v) { const el = document.getElementById(id); if (el) el.textContent = v; }

(async () => {
  await loadInitialMode();
  await loadDeptCatalog();
  refresh();
  loadEnrolled();
  if (currentMode === "od") loadEnabledClassCount();
})();
setInterval(refresh, 1200);
setInterval(() => {
  if (currentMode === "fr") loadEnrolled();
  else loadEnabledClassCount();
}, 5000);
