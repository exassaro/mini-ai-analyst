/* ═══════════════════════════════════════════════════════════
   Mini AI Analyst — Frontend Logic (Dashboard Version)
   ═══════════════════════════════════════════════════════════ */

const API = window.location.origin;

/* ── State ─────────────────────────────────────────────────── */
let fileId   = null;
let modelId  = null;
let columns  = [];

/* ── DOM refs ──────────────────────────────────────────────── */
const $       = (sel) => document.querySelector(sel);
const $$      = (sel) => document.querySelectorAll(sel);

// Sidebar & Layout
const sidebar = $("#sidebar");
const mobileToggle = $("#mobileToggle");
const navItems = $$(".nav-item");
const sections = $$(".content-section");
const topbarStatus = $("#topbarStatus");

// Upload
const dropzone      = $("#dropzone");
const fileInput     = $("#fileInput");
const browseBtn     = $("#browseBtn");
const uploadStatus  = $("#uploadStatus");
const uploadSpinner = $("#uploadSpinner");
const uploadMsg     = $("#uploadMsg");

// Profile
const profileBtn    = $("#profileBtn");
const profileResult = $("#profileResult");
const profileResultCard = $("#profileResultCard");

// Train
const targetSelect  = $("#targetSelect");
const trainBtn      = $("#trainBtn");
const trainResult   = $("#trainResult");

// Predict
const predictInput  = $("#predictInput");
const predictBtn    = $("#predictBtn");
const predictResult = $("#predictResult");

// Summary
const summaryBtn    = $("#summaryBtn");
const summaryResult = $("#summaryResult");
const summaryResultCard = $("#summaryResultCard");

const toast         = $("#toast");

/* ═══════════════════ HELPERS ════════════════════════════════ */

function showToast(msg, isError = false) {
  toast.textContent = msg;
  toast.classList.toggle("error", isError);
  toast.classList.remove("hidden");
  clearTimeout(toast._timer);
  toast._timer = setTimeout(() => toast.classList.add("hidden"), 4000);
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

/* ═══════════════════ NAVIGATION ═════════════════════════════ */

mobileToggle.addEventListener("click", () => {
    sidebar.classList.toggle("open");
});

function navigateTo(targetId) {
    // Hide all sections
    sections.forEach(sec => sec.classList.remove("active"));
    // Deactivate all nav items
    navItems.forEach(nav => nav.classList.remove("active"));
    
    // Show target section
    $(`#section-${targetId}`).classList.add("active");
    // Activate target nav
    $(`.nav-item[data-target="${targetId}"]`).classList.add("active");

    // Close sidebar on mobile after click
    if (window.innerWidth <= 768) {
        sidebar.classList.remove("open");
    }
}

navItems.forEach(nav => {
    nav.addEventListener("click", () => {
        if (nav.disabled) return;
        navigateTo(nav.dataset.target);
    });
});

function unlockStep(targetId) {
    const nav = $(`#nav-${targetId}`);
    if (nav) nav.disabled = false;
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
    uploadMsg.innerHTML = `<strong>File uploaded successfully.</strong> (${data.filename})`;
    topbarStatus.innerHTML = `Active Dataset: <strong>${data.filename}</strong>`;
    showToast("File uploaded successfully");

    unlockStep("profile");
    setTimeout(() => { navigateTo("profile"); }, 800);
  } catch (err) {
    uploadSpinner.style.display = "none";
    uploadMsg.textContent = `Upload failed: ${err.message}`;
    showToast(err.message, true);
  }
}

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
  profileResultCard.classList.add("hidden");

  try {
    const data = await api("GET", `/profile?file_id=${fileId}`);
    columns = data.columns || [];
    renderProfile(data);
    profileResultCard.classList.remove("hidden");

    targetSelect.innerHTML = columns.map(c => `<option value="${c}">${c}</option>`).join("");
    targetSelect.disabled = false;
    
    unlockStep("train");
    showToast("Profile complete");
  } catch (err) {
    showToast(err.message, true);
  } finally {
    setLoading(profileBtn, false);
  }
});

function renderProfile(d) {
  let html = `<h3>Dataset Characteristics</h3>`;
  html += `<p style="margin-bottom: 1rem;"><strong>Shape:</strong> ${d.shape[0].toLocaleString()} rows × ${d.shape[1]} columns</p>`;

  // Column types table
  html += `<table><thead><tr><th>Column</th><th>Type</th><th>Nulls %</th><th>Unique</th><th>Skewness</th><th>Outliers</th></tr></thead><tbody>`;
  for (const col of d.columns) {
    const outlier = d.outlier_counts[col] ?? "—";
    const skew = d.skewness[col] !== undefined && d.skewness[col] !== null ? d.skewness[col] : "—";
    html += `<tr>
      <td><strong>${col}</strong></td>
      <td>${d.column_types[col]}</td>
      <td>${d.null_percentage[col]}%</td>
      <td>${d.unique_counts[col]}</td>
      <td>${skew}</td>
      <td>${outlier}</td>
    </tr>`;
  }
  html += `</tbody></table>`;

  // Flags
  if (d.imbalanced_columns && d.imbalanced_columns.length)
    html += `<div style="margin-top:1rem; padding: 0.75rem; background: #FEF3C7; color: #92400E; border-radius: 6px;"><strong>Imbalanced categorical:</strong> ${d.imbalanced_columns.join(", ")}</div>`;
  if (d.high_cardinality_columns.length)
    html += `<div style="margin-top:0.5rem; padding: 0.75rem; background: #FEF3C7; color: #92400E; border-radius: 6px;"><strong>High cardinality:</strong> ${d.high_cardinality_columns.join(", ")}</div>`;
  if (d.constant_columns.length)
    html += `<div style="margin-top:0.5rem; padding: 0.75rem; background: #FEE2E2; color: #991B1B; border-radius: 6px;"><strong>Constant columns:</strong> ${d.constant_columns.join(", ")}</div>`;

  // Correlation Matrix
  if (d.numeric_correlations && Object.keys(d.numeric_correlations).length > 0) {
    const numCols = Object.keys(d.numeric_correlations);
    html += `<h3 style="margin-top: 2rem;">Pairwise Correlation Matrix</h3><table style="font-size: 0.85em"><thead><tr><th></th>`;
    for (const col of numCols) {
        html += `<th>${col}</th>`;
    }
    html += `</tr></thead><tbody>`;
    for (const rowCol of numCols) {
        html += `<tr><td><strong>${rowCol}</strong></td>`;
        for (const col of numCols) {
            const val = d.numeric_correlations[rowCol][col];
            html += `<td>${val !== null && val !== undefined ? val : '—'}</td>`;
        }
        html += `</tr>`;
    }
    html += `</tbody></table>`;
  }

  // Distribution Chart
  if (d.histograms && Object.keys(d.histograms).length > 0) {
    const feature = Object.keys(d.histograms)[0];
    html += `
      <div style="margin-top: 2rem; background: var(--surface); border: 1px solid var(--border); padding: 1.5rem; border-radius: 8px;">
        <h3 style="margin-bottom: 1rem;">Basic Feature Distribution: ${feature}</h3>
        <canvas id="distChart" width="400" height="200"></canvas>
      </div>
    `;
    setTimeout(() => {
        const ctx = document.getElementById("distChart");
        if (ctx) {
            const hist = d.histograms[feature];
            const labels = [];
            for (let i = 0; i < hist.bins.length - 1; i++) {
                labels.push(`${hist.bins[i].toFixed(1)} - ${hist.bins[i+1].toFixed(1)}`);
            }
            new Chart(ctx, {
                type: 'bar',
                data: {
                    labels: labels,
                    datasets: [{
                        label: 'Frequency',
                        data: hist.counts,
                        backgroundColor: 'rgba(37, 99, 235, 0.5)',
                        borderColor: 'rgba(37, 99, 235, 1)',
                        borderWidth: 1
                    }]
                },
                options: { scales: { y: { beginAtZero: true } } }
            });
        }
    }, 100);
  }

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

    // fetch model info to populate form
    const modelInfo = await api("GET", `/model-info?model_id=${modelId}`);
    
    // populate prediction form dynamically
    const fieldsContainer = $("#dynamicFormFields");
    fieldsContainer.innerHTML = "";
    modelInfo.features.forEach(feat => {
       fieldsContainer.innerHTML += `
         <div style="display:flex; flex-direction:column;">
           <label>${feat}</label>
           <input type="text" name="${feat}" placeholder="Enter value" />
         </div>
       `;
    });

    unlockStep("predict");
    unlockStep("summary");
    showToast("Model trained successfully!");
  } catch (err) {
    showToast(err.message, true);
  } finally {
    setLoading(trainBtn, false);
  }
});

function renderTrain(d) {
  let html = `<h3>Training Results</h3>`;
  html += `<p style="margin-bottom: 0.25rem;"><strong>Problem type:</strong> <span style="text-transform: capitalize;">${d.problem_type}</span></p>`;
  html += `<p style="margin-bottom: 1rem;"><strong>Model ID:</strong> <code>${d.model_id}</code></p>`;
  html += `<div>`;
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

document.querySelectorAll('input[name="predictMethod"]').forEach(el => {
  el.addEventListener("change", (e) => {
    if (e.target.value === "json") {
      $("#predictJsonContainer").classList.remove("hidden");
      $("#predictFormContainer").classList.add("hidden");
    } else {
      $("#predictJsonContainer").classList.add("hidden");
      $("#predictFormContainer").classList.remove("hidden");
    }
  });
});

predictBtn.addEventListener("click", async () => {
  let rows;
  const isForm = $("input[name='predictMethod']:checked").value === "form";
  if (isForm) {
    const formData = {};
    document.querySelectorAll('#dynamicFormFields input').forEach(inp => {
       formData[inp.name] = isNaN(Number(inp.value)) || inp.value.trim() === "" ? inp.value : Number(inp.value);
    });
    rows = [formData];
  } else {
    try {
      rows = JSON.parse(predictInput.value);
      if (!Array.isArray(rows)) throw 0;
    } catch {
      showToast("Invalid JSON payload — must be an array of objects", true);
      return;
    }
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
  let html = `<h3>Prediction Output</h3>`;
  
  if (!d.predictions || d.predictions.length === 0) {
      html += `<p>No predictions returned.</p>`;
      predictResult.innerHTML = html;
      return;
  }

  const rowOne = d.predictions[0];
  const keys = Object.keys(rowOne);
  const targetCol = keys.find(k => k !== "confidence");
  const hasConfidence = keys.includes("confidence");

  html += `<table><thead><tr><th style="width: 50px;">#</th><th>Predicted ${targetCol}</th>`;
  if (hasConfidence) html += `<th>Confidence</th>`;
  html += `</tr></thead><tbody>`;

  d.predictions.forEach((p, i) => {
    html += `<tr><td>${i + 1}</td><td><strong>${p[targetCol]}</strong></td>`;
    if (p.confidence !== undefined) html += `<td>${(p.confidence * 100).toFixed(1)}%</td>`;
    html += `</tr>`;
  });
  html += `</tbody></table>`;
  predictResult.innerHTML = html;
}

/* ═══════════════════ 5. SUMMARY ═════════════════════════════ */

summaryBtn.addEventListener("click", async () => {
  setLoading(summaryBtn, true);
  summaryResultCard.classList.add("hidden");

  try {
    const data = await api("GET", `/summary?file_id=${fileId}&model_id=${modelId}`);
    renderSummary(data);
    summaryResultCard.classList.remove("hidden");
    showToast("Summary report generated");
  } catch (err) {
    showToast(err.message, true);
  } finally {
    setLoading(summaryBtn, false);
  }
});

function renderSummary(d) {
  let html = `
    <div style="padding: 1.5rem; background: var(--bg); border: 1px solid var(--border); border-left: 4px solid var(--primary); border-radius: 6px;">
        <h3 style="margin-bottom: 0.5rem; color: var(--text);">Executive Overview</h3>
        <p style="font-size: 0.95rem; line-height: 1.7; color: var(--text);">${d.summary_text}</p>
    </div>
  `;

  if (d.top_correlated_features.length) {
    html += `<h3 style="margin-top: 2rem;">Top Correlated Features</h3>`;
    html += `<table><thead><tr><th>Feature</th><th>Correlation Score</th></tr></thead><tbody>`;
    d.top_correlated_features.forEach(f => {
      html += `<tr><td><strong>${f.feature}</strong></td><td>${f.correlation}</td></tr>`;
    });
    html += `</tbody></table>`;
  }

  html += `<h3 style="margin-top: 2rem;">Model Performance Metrics</h3>`;
  html += `<div style="margin-top: 0.5rem;">`;
  for (const [k, v] of Object.entries(d.model_performance)) {
    const cls = metricClass(k, v);
    html += `<span class="metric-pill ${cls}">${k}: ${v}</span>`;
  }
  html += `</div>`;

  summaryResult.innerHTML = html;
}
