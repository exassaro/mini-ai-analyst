/* ═══════════════════════════════════════════════════════════
   Mini AI Analyst — Frontend Logic
   ═══════════════════════════════════════════════════════════ */

const API = window.location.origin;

/* ── State ─────────────────────────────────────────────────── */
let fileId   = null;
let modelId  = null;
let columns  = [];

/* ── DOM refs ──────────────────────────────────────────────── */
const $       = (sel) => document.querySelector(sel);
const dropzone      = $("#dropzone");
const fileInput     = $("#fileInput");
const browseBtn     = $("#browseBtn");
const uploadStatus  = $("#uploadStatus");
const uploadSpinner = $("#uploadSpinner");
const uploadMsg     = $("#uploadMsg");

const profileBtn    = $("#profileBtn");
const profileResult = $("#profileResult");

const targetSelect  = $("#targetSelect");
const trainBtn      = $("#trainBtn");
const trainResult   = $("#trainResult");

const predictInput  = $("#predictInput");
const predictBtn    = $("#predictBtn");
const predictResult = $("#predictResult");

const summaryBtn    = $("#summaryBtn");
const summaryResult = $("#summaryResult");

const toast         = $("#toast");

/* ═══════════════════ HELPERS ════════════════════════════════ */

function showToast(msg, isError = false) {
  toast.textContent = msg;
  toast.classList.toggle("error", isError);
  toast.classList.remove("hidden");
  clearTimeout(toast._timer);
  toast._timer = setTimeout(() => toast.classList.add("hidden"), 4000);
}

function enableSection(id) {
  $(id).classList.remove("disabled");
}

function setLoading(btn, loading) {
  btn.disabled = loading;
  btn.dataset.origText = btn.dataset.origText || btn.textContent;
  btn.textContent = loading ? "Working…" : btn.dataset.origText;
}

async function api(method, path, body) {
  const opts = { method, headers: {} };
  if (body instanceof FormData) {
    opts.body = body;
  } else if (body) {
    opts.headers["Content-Type"] = "application/json";
    opts.body = JSON.stringify(body);
  }
  const res = await fetch(`${API}${path}`, opts);
  const data = await res.json();
  if (!res.ok) throw new Error(data.detail || "Request failed");
  return data;
}

/* ═══════════════════ 1. UPLOAD ══════════════════════════════ */

function handleFile(file) {
  if (!file) return;
  if (!file.name.endsWith(".csv")) {
    showToast("Please select a .csv file", true);
    return;
  }
  uploadFile(file);
}

async function uploadFile(file) {
  uploadStatus.classList.remove("hidden");
  uploadSpinner.style.display = "";
  uploadMsg.textContent = `Uploading ${file.name}…`;

  try {
    const fd = new FormData();
    fd.append("file", file);
    const data = await api("POST", "/upload", fd);
    fileId = data.file_id;

    uploadSpinner.style.display = "none";
    uploadMsg.innerHTML = `<strong>✓</strong> Uploaded — <code>${data.filename}</code> &nbsp;·&nbsp; file_id: <code>${fileId}</code>`;
    showToast("File uploaded successfully");

    // unlock step 2
    enableSection("#section-profile");
    profileBtn.disabled = false;
  } catch (err) {
    uploadSpinner.style.display = "none";
    uploadMsg.textContent = `Upload failed: ${err.message}`;
    showToast(err.message, true);
  }
}

// Drag & drop
dropzone.addEventListener("click", () => fileInput.click());
browseBtn.addEventListener("click", (e) => { e.stopPropagation(); fileInput.click(); });
fileInput.addEventListener("change", () => handleFile(fileInput.files[0]));

dropzone.addEventListener("dragover", (e) => { e.preventDefault(); dropzone.classList.add("drag-over"); });
dropzone.addEventListener("dragleave", () => dropzone.classList.remove("drag-over"));
dropzone.addEventListener("drop", (e) => {
  e.preventDefault();
  dropzone.classList.remove("drag-over");
  handleFile(e.dataTransfer.files[0]);
});

/* ═══════════════════ 2. PROFILE ═════════════════════════════ */

profileBtn.addEventListener("click", async () => {
  setLoading(profileBtn, true);
  profileResult.classList.add("hidden");

  try {
    const data = await api("GET", `/profile?file_id=${fileId}`);
    columns = data.columns || [];
    renderProfile(data);
    profileResult.classList.remove("hidden");

    // populate target select & unlock step 3
    targetSelect.innerHTML = columns.map(c => `<option value="${c}">${c}</option>`).join("");
    targetSelect.disabled = false;
    enableSection("#section-train");
    trainBtn.disabled = false;
    showToast("Profile complete");
  } catch (err) {
    showToast(err.message, true);
  } finally {
    setLoading(profileBtn, false);
  }
});

function renderProfile(d) {
  let html = `<h3>📊 Dataset Profile</h3>`;
  html += `<p><strong>Shape:</strong> ${d.shape[0].toLocaleString()} rows × ${d.shape[1]} columns</p>`;

  // Column types table
  html += `<table><thead><tr><th>Column</th><th>Type</th><th>Nulls %</th><th>Unique</th><th>Outliers</th></tr></thead><tbody>`;
  for (const col of d.columns) {
    const outlier = d.outlier_counts[col] ?? "—";
    html += `<tr>
      <td>${col}</td>
      <td>${d.column_types[col]}</td>
      <td>${d.null_percentage[col]}%</td>
      <td>${d.unique_counts[col]}</td>
      <td>${outlier}</td>
    </tr>`;
  }
  html += `</tbody></table>`;

  // Flags
  if (d.high_cardinality_columns.length)
    html += `<p style="margin-top:.7rem">⚠️ <strong>High cardinality:</strong> ${d.high_cardinality_columns.join(", ")}</p>`;
  if (d.constant_columns.length)
    html += `<p>⚠️ <strong>Constant columns:</strong> ${d.constant_columns.join(", ")}</p>`;

  profileResult.innerHTML = html;
}

/* ═══════════════════ 3. TRAIN ═══════════════════════════════ */

trainBtn.addEventListener("click", async () => {
  const target = targetSelect.value;
  if (!target) { showToast("Select a target column", true); return; }

  setLoading(trainBtn, true);
  trainResult.classList.add("hidden");

  try {
    const data = await api("POST", "/train", { file_id: fileId, target_column: target });
    modelId = data.model_id;
    renderTrain(data);
    trainResult.classList.remove("hidden");

    // unlock step 4 & 5
    enableSection("#section-predict");
    predictInput.disabled = false;
    predictBtn.disabled = false;

    enableSection("#section-summary");
    summaryBtn.disabled = false;
    showToast("Model trained!");
  } catch (err) {
    showToast(err.message, true);
  } finally {
    setLoading(trainBtn, false);
  }
});

function renderTrain(d) {
  let html = `<h3>🤖 Training Result</h3>`;
  html += `<p><strong>Problem type:</strong> ${d.problem_type}</p>`;
  html += `<p><strong>Model ID:</strong> <code>${d.model_id}</code></p>`;
  html += `<div style="margin-top:.6rem">`;
  for (const [k, v] of Object.entries(d.metrics)) {
    const cls = metricClass(k, v);
    html += `<span class="metric-pill ${cls}">${k}: ${v}</span>`;
  }
  html += `</div>`;
  trainResult.innerHTML = html;
}

function metricClass(key, val) {
  if (key === "accuracy" || key === "r2" || key === "f1_weighted") {
    if (val >= 0.8) return "metric-pill--good";
    if (val >= 0.5) return "metric-pill--warn";
    return "metric-pill--bad";
  }
  return "metric-pill--warn";
}

/* ═══════════════════ 4. PREDICT ═════════════════════════════ */

predictBtn.addEventListener("click", async () => {
  let rows;
  try {
    rows = JSON.parse(predictInput.value);
    if (!Array.isArray(rows)) throw 0;
  } catch {
    showToast("Invalid JSON — must be an array of objects", true);
    return;
  }

  setLoading(predictBtn, true);
  predictResult.classList.add("hidden");

  try {
    const data = await api("POST", "/predict", { model_id: modelId, data: rows });
    renderPredict(data);
    predictResult.classList.remove("hidden");
    showToast("Prediction complete");
  } catch (err) {
    showToast(err.message, true);
  } finally {
    setLoading(predictBtn, false);
  }
});

function renderPredict(d) {
  let html = `<h3>🎯 Predictions</h3>`;
  html += `<table><thead><tr><th>#</th><th>Prediction</th>`;
  if (d.probabilities) html += `<th>Probabilities</th>`;
  html += `</tr></thead><tbody>`;
  d.predictions.forEach((p, i) => {
    html += `<tr><td>${i + 1}</td><td>${p}</td>`;
    if (d.probabilities) html += `<td>${d.probabilities[i].map(v => v.toFixed(3)).join(", ")}</td>`;
    html += `</tr>`;
  });
  html += `</tbody></table>`;
  predictResult.innerHTML = html;
}

/* ═══════════════════ 5. SUMMARY ═════════════════════════════ */

summaryBtn.addEventListener("click", async () => {
  setLoading(summaryBtn, true);
  summaryResult.classList.add("hidden");

  try {
    const data = await api("GET", `/summary?file_id=${fileId}&model_id=${modelId}`);
    renderSummary(data);
    summaryResult.classList.remove("hidden");
    showToast("Summary ready");
  } catch (err) {
    showToast(err.message, true);
  } finally {
    setLoading(summaryBtn, false);
  }
});

function renderSummary(d) {
  let html = `<h3>📝 Analysis Summary</h3>`;
  html += `<div class="summary-text">${d.summary_text}</div>`;

  if (d.top_correlated_features.length) {
    html += `<table style="margin-top:1rem"><thead><tr><th>Feature</th><th>Correlation</th></tr></thead><tbody>`;
    d.top_correlated_features.forEach(f => {
      html += `<tr><td>${f.feature}</td><td>${f.correlation}</td></tr>`;
    });
    html += `</tbody></table>`;
  }

  html += `<div style="margin-top:.8rem">`;
  for (const [k, v] of Object.entries(d.model_performance)) {
    const cls = metricClass(k, v);
    html += `<span class="metric-pill ${cls}">${k}: ${v}</span>`;
  }
  html += `</div>`;

  summaryResult.innerHTML = html;
}
