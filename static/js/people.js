/* Person management: table view, department dropdown + custom, clear-all. */
const modal = document.getElementById("modal");
const form = document.getElementById("personForm");
const body = document.getElementById("peopleBody");
const drop = document.getElementById("drop");
const photos = document.getElementById("photos");
const deptSelect = document.getElementById("deptSelect");
const customWrap = document.getElementById("customWrap");
let editingId = null;
let departments = [];

// ---------------------------------------------------------- departments
async function loadDepartments() {
  try {
    const r = await API.get("/api/departments");
    departments = (r.data && r.data.departments) || [];
  } catch (e) { departments = []; }
  deptSelect.innerHTML = departments.map(d => `<option value="${d.name}">${d.name}</option>`).join("");
  toggleCustom();
}
deptSelect.onchange = toggleCustom;
function toggleCustom() {
  customWrap.style.display = (deptSelect.value === "Other") ? "block" : "none";
}
function deptColor(name) {
  const d = departments.find(x => x.name.toLowerCase() === (name || "").toLowerCase());
  return d ? d.color : "#64748B";
}

// ---------------------------------------------------------- modal
function openModal(title) { document.getElementById("modalTitle").textContent = title; modal.classList.add("show"); }
function closeModal() {
  modal.classList.remove("show"); form.reset(); editingId = null;
  document.getElementById("existingThumbs").innerHTML = "";
  document.getElementById("newThumbs").innerHTML = "";
  photos.value = ""; toggleCustom();
}
document.getElementById("addBtn").onclick = () => { closeModal(); openModal("Add Person"); };
document.getElementById("cancelBtn").onclick = closeModal;
modal.addEventListener("click", e => { if (e.target === modal) closeModal(); });

drop.onclick = () => photos.click();
photos.onchange = previewNew;
["dragover","dragenter"].forEach(ev => drop.addEventListener(ev, e => { e.preventDefault(); drop.classList.add("drag"); }));
["dragleave","drop"].forEach(ev => drop.addEventListener(ev, e => { e.preventDefault(); drop.classList.remove("drag"); }));
drop.addEventListener("drop", e => { photos.files = e.dataTransfer.files; previewNew(); });
function previewNew() {
  const boxEl = document.getElementById("newThumbs"); boxEl.innerHTML = "";
  [...photos.files].forEach(f => {
    const img = document.createElement("img"); img.src = URL.createObjectURL(f);
    const wrap = document.createElement("div"); wrap.className = "t"; wrap.appendChild(img); boxEl.appendChild(wrap);
  });
}

// ---------------------------------------------------------- list (table)
async function loadPeople() {
  const r = await API.get("/api/people");
  const people = r.data || [];
  body.innerHTML = "";
  document.getElementById("emptyState").style.display = people.length ? "none" : "block";
  people.forEach(p => body.appendChild(personRow(p)));
}

function personRow(p) {
  const tr = document.createElement("tr");
  const color = p.color || deptColor(p.department);
  const avatar = p.image_count
    ? `<img class="row-thumb" src="/api/people/${p.id}/image/${p.images[0]}"/>`
    : `<div class="row-thumb glyph">☺</div>`;
  const deptChip = p.department
    ? `<span class="dept-chip"><span class="dot" style="background:${color}"></span>${escapeHtml(p.department)}</span>`
    : `<span class="help">—</span>`;
  const allEncoded = p.embedding_count >= p.image_count && p.embedding_count > 0;
  const emb = p.embedding_count > 0
    ? `<span class="badge ${allEncoded ? 'ok' : 'warn'}">${p.embedding_count} encoded</span>`
    : `<span class="badge bad">none</span>`;
  const reencodeBtn = !allEncoded && p.image_count > 0
    ? `<button class="btn secondary sm" data-reencode title="Re-detect faces in all ${p.image_count} photos">Re-encode</button>`
    : "";
  tr.innerHTML = `
    <td><div class="row-name">${avatar}<span>${escapeHtml(p.name)}</span></div></td>
    <td>${deptChip}</td>
    <td>${p.image_count}</td>
    <td>${emb}</td>
    <td style="text-align:right;">
      ${reencodeBtn}
      <button class="btn secondary sm" data-edit>Edit</button>
      <button class="btn danger sm" data-del>Delete</button>
    </td>`;
  tr.querySelector("[data-edit]").onclick = () => editPerson(p.id);
  tr.querySelector("[data-del]").onclick = () => delPerson(p.id, p.name);
  const reBtn = tr.querySelector("[data-reencode]");
  if (reBtn) reBtn.onclick = () => reencodePerson(p.id, p.name, reBtn);
  return tr;
}

// ---------------------------------------------------------- edit
async function editPerson(id) {
  const r = await API.get(`/api/people/${id}`);
  if (!r.success) return toast("Could not load person", "err");
  const p = r.data;
  closeModal(); editingId = id;
  form.name.value = p.name;
  // department: if it matches a known dept use it, else select Other + custom.
  const known = departments.some(d => d.name === p.department);
  deptSelect.value = known ? p.department : "Other";
  if (!known && p.department) { document.getElementById("customDept").value = p.department; }
  toggleCustom();
  openModal("Edit Person");
  const boxEl = document.getElementById("existingThumbs"); boxEl.innerHTML = "";
  (p.images || []).forEach(fn => {
    const wrap = document.createElement("div"); wrap.className = "t";
    wrap.innerHTML = `<img src="/api/people/${id}/image/${fn}"/><button type="button" class="rm">×</button>`;
    wrap.querySelector(".rm").onclick = async () => {
      const d = await API.del(`/api/people/${id}/images/${fn}`);
      toast(d.message || "Removed", d.success ? "ok" : "err");
      if (d.success) wrap.remove();
    };
    boxEl.appendChild(wrap);
  });
}

async function delPerson(id, name) {
  if (!confirm(`Delete ${name}? This removes their photos and embedding.`)) return;
  const r = await API.del(`/api/people/${id}`);
  toast(r.message || "Deleted", r.success ? "ok" : "err");
  if (r.success) loadPeople();
}

async function reencodePerson(id, name, btn) {
  btn.disabled = true; btn.textContent = "Re-encoding…";
  const r = await API.post(`/api/people/${id}/reenroll`);
  btn.disabled = false; btn.textContent = "Re-encode";
  toast(`${name}: ${r.message || (r.success ? "Done" : "Error")}`, r.success ? "ok" : "err");
  if (r.success) loadPeople();
}

document.getElementById("reencodeAllBtn").onclick = async () => {
  const btn = document.getElementById("reencodeAllBtn");
  btn.disabled = true; btn.textContent = "Re-encoding…";
  const r = await API.post("/api/people/reenroll-all");
  btn.disabled = false; btn.textContent = "Re-encode All";
  toast(r.message || (r.success ? "Done" : "Error"), r.success ? "ok" : "err");
  if (r.success) loadPeople();
};

document.getElementById("clearAllBtn").onclick = async () => {
  if (!confirm("Remove ALL enrolled people, embeddings and rebuild the index? This cannot be undone.")) return;
  const r = await API.post("/api/people/clear");
  toast(r.message || "Cleared", r.success ? "ok" : "err");
  loadPeople();
};

// ---------------------------------------------------------- save
form.onsubmit = async (e) => {
  e.preventDefault();
  const name = form.name.value.trim();
  if (!name) return toast("Name is required", "err");
  const fd = new FormData();
  fd.append("name", name);
  fd.append("department", deptSelect.value);
  fd.append("custom_department", document.getElementById("customDept").value.trim());
  [...photos.files].forEach(f => fd.append("photos", f));
  const saveBtn = document.getElementById("saveBtn");
  saveBtn.disabled = true; saveBtn.textContent = "Saving…";
  const url = editingId ? `/api/people/${editingId}` : "/api/people";
  const r = await API.post(url, fd, true);
  saveBtn.disabled = false; saveBtn.textContent = "Save";
  if (r.success) {
    const d = r.data || {};
    toast(`${r.message}. ${d.encoded_faces}/${d.total_images} face(s) encoded.`, "ok");
    closeModal(); loadPeople();
  } else { toast(r.message || "Error saving", "err"); }
};

function escapeHtml(s) {
  return (s || "").replace(/[&<>"']/g, c => ({ "&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;" }[c]));
}

(async () => { await loadDepartments(); loadPeople(); })();
