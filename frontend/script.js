/* ═══════════════════════════════════════════════════════════
   Mini AI Analyst — Frontend Logic (Dashboard Version)
   Aligned with hardened backend API responses.
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
  const opts = { method, headers: { "session-token": "valid-token" } };
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

/** Render a list of warning strings as alert banners */
function renderWarnings(warnings) {
  if (!warnings || warnings.length === 0) return "";
  return warnings.map(w =>
    `<div class="alert alert--info">
       <i data-lucide="info" style="width:16px;height:16px;flex-shrink:0;"></i>
       <span>${escapeHtml(w)}</span>
     </div>`
  ).join("");
}

/** Render data leakage warnings as danger alerts */
function renderLeakageWarnings(warnings) {
  if (!warnings || warnings.length === 0) return "";
  return warnings.map(w =>
    `<div class="alert alert--danger">
       <i data-lucide="alert-triangle" style="width:16px;height:16px;flex-shrink:0;"></i>
       <span>${escapeHtml(w)}</span>
     </div>`
  ).join("");
}

/** Simple HTML escaping */
function escapeHtml(str) {
  const el = document.createElement("span");
  el.textContent = str;
  return el.innerHTML;
}

/** Re-initialize Lucide icons for dynamically added elements */
function refreshIcons() {
  if (typeof lucide !== "undefined") {
    lucide.createIcons();
  }
}

/** Semantic type badge color */
function typeBadgeClass(type) {
  switch (type) {
    case "numerical":   return "badge--blue";
    case "categorical": return "badge--purple";
    case "datetime":    return "badge--teal";
    case "boolean":     return "badge--amber";
    case "text":        return "badge--gray";
    default:            return "badge--gray";
  }
}

/* ═══════════════════ NAVIGATION ═════════════════════════════ */

mobileToggle.addEventListener("click", () => {
    sidebar.classList.toggle("open");
});

function navigateTo(targetId) {
    sections.forEach(sec => sec.classList.remove("active"));
    navItems.forEach(nav => nav.classList.remove("active"));
    $(`#section-${targetId}`).classList.add("active");
    $(`.nav-item[data-target="${targetId}"]`).classList.add("active");
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

  // Hide previous schema preview
  const schemaCard = $("#schemaPreviewCard");
  if (schemaCard) schemaCard.classList.add("hidden");

  try {
    const fd = new FormData();
    fd.append("file", file);
    const data = await api("POST", "/upload", fd);
    fileId = data.file_id;
    columns = data.columns || [];

    uploadSpinner.style.display = "none";
    uploadMsg.innerHTML = `<strong>File uploaded successfully.</strong> (${escapeHtml(data.filename)})`;
    
    // Update topbar with shape info
    const shapeText = data.shape && data.shape.length === 2
      ? ` — ${data.shape[0].toLocaleString()} rows × ${data.shape[1]} cols`
      : "";
    topbarStatus.innerHTML = `Active Dataset: <strong>${escapeHtml(data.filename)}</strong>${shapeText}`;

    // Render schema preview if available
    if (data.column_info && Object.keys(data.column_info).length > 0) {
      renderSchemaPreview(data);
      schemaCard.classList.remove("hidden");
    }

    showToast("File uploaded successfully");
    unlockStep("profile");
    setTimeout(() => { navigateTo("profile"); }, 800);
  } catch (err) {
    uploadSpinner.style.display = "none";
    uploadMsg.textContent = `Upload failed: ${err.message}`;
    showToast(err.message, true);
  }
}

/** Render the schema preview table shown right after upload */
function renderSchemaPreview(data) {
  const shapeEl = $("#schemaShape");
  const tableEl = $("#schemaTable");
  const flagsEl = $("#schemaFlags");

  if (data.shape) {
    shapeEl.innerHTML = `<strong>${data.shape[0].toLocaleString()}</strong> rows × <strong>${data.shape[1]}</strong> columns`;
  }

  // Build column info table
  let html = `<table><thead><tr>
    <th>Column</th><th>Type</th><th>Raw Dtype</th><th>Nulls %</th><th>Unique</th><th>Flags</th>
  </tr></thead><tbody>`;

  for (const col of (data.columns || [])) {
    const info = data.column_info[col];
    if (!info) continue;

    const flags = [];
    if (info.is_high_cardinality) flags.push('<span class="flag flag--warn">High Card.</span>');
    if (info.is_constant) flags.push('<span class="flag flag--danger">Constant</span>');

    html += `<tr>
      <td><strong>${escapeHtml(col)}</strong></td>
      <td><span class="badge ${typeBadgeClass(info.semantic_type)}">${info.semantic_type}</span></td>
      <td style="color: var(--text-muted); font-size: 0.8rem;">${escapeHtml(info.dtype)}</td>
      <td>${info.null_percentage}%</td>
      <td>${info.unique_count}</td>
      <td>${flags.length ? flags.join(" ") : "—"}</td>
    </tr>`;
  }
  html += `</tbody></table>`;
  tableEl.innerHTML = html;

  // Flags summary
  let flagHtml = "";
  if (data.high_cardinality_columns && data.high_cardinality_columns.length) {
    flagHtml += `<div class="alert alert--warn"><i data-lucide="alert-circle" style="width:16px;height:16px;flex-shrink:0;"></i><span><strong>High cardinality:</strong> ${data.high_cardinality_columns.map(escapeHtml).join(", ")}</span></div>`;
  }
  if (data.constant_columns && data.constant_columns.length) {
    flagHtml += `<div class="alert alert--danger"><i data-lucide="alert-triangle" style="width:16px;height:16px;flex-shrink:0;"></i><span><strong>Constant columns:</strong> ${data.constant_columns.map(escapeHtml).join(", ")}</span></div>`;
  }
  flagsEl.innerHTML = flagHtml;
  refreshIcons();
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
    columns = data.schema_info.columns || [];
    renderProfile(data);
    profileResultCard.classList.remove("hidden");

    targetSelect.innerHTML = columns.map(c => `<option value="${escapeHtml(c)}">${escapeHtml(c)}</option>`).join("");
    targetSelect.disabled = false;
    
    // Generate feature checkboxes
    const featureBox = $("#featureCheckboxes");
    featureBox.innerHTML = columns.map(c => `
      <label style="display:flex; align-items:center; gap:0.4rem; font-weight:normal;">
        <input type="checkbox" name="featureCol" value="${escapeHtml(c)}" checked />
        ${escapeHtml(c)}
      </label>
    `).join("");

    updateDisabledTargetCheckbox();
    
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
  const schema = d.schema_info;
  html += `<p style="margin-bottom: 1rem;"><strong>Shape:</strong> ${schema.shape[0].toLocaleString()} rows × ${schema.shape[1]} columns</p>`;
  if (d.task_type) {
      html += `<p style="margin-bottom: 1rem;"><strong>Inferred Task:</strong> <span class="badge badge--purple">${escapeHtml(d.task_type)}</span></p>`;
  }

  // Insights
  if (d.insights && d.insights.length > 0) {
      html += `<h3 style="margin-top: 2rem; margin-bottom: 1rem;">Key Insights</h3><div style="display:flex; flex-direction:column; gap:0.75rem;">`;
      d.insights.forEach(ins => {
          const alertCls = ins.severity === 'danger' ? 'alert--danger' : (ins.severity === 'warning' ? 'alert--warn' : 'alert--info');
          const icon = ins.severity === 'danger' ? 'alert-triangle' : (ins.severity === 'warning' ? 'alert-circle' : 'info');
          html += `<div class="alert ${alertCls}">
             <i data-lucide="${icon}" style="width:20px;height:20px;flex-shrink:0;margin-top:2px;"></i>
             <div style="flex:1;">
               <strong style="display:block;margin-bottom:0.2rem;">${escapeHtml(ins.title)}</strong>
               <p style="color:inherit;margin-bottom:0.3rem;">${escapeHtml(ins.message)}</p>
               ${ins.affected_columns && ins.affected_columns.length > 0 ? `<p style="color:inherit;font-size:0.8rem;opacity:0.8;">Affected: ${escapeHtml(ins.affected_columns.join(", "))}</p>` : ''}
               ${ins.recommendation ? `<p style="color:inherit;font-size:0.85rem;margin-top:0.3rem;font-weight:500;">Tip: ${escapeHtml(ins.recommendation)}</p>` : ''}
             </div>
          </div>`;
      });
      html += `</div>`;
  }

  // Column types table
  html += `<h3 style="margin-top: 2rem;">Schema Details</h3><table><thead><tr>
    <th>Column</th><th>Type</th><th>Nulls %</th><th>Unique</th>
  </tr></thead><tbody>`;
  for (const col of schema.columns) {
    const colType = schema.column_types[col] || "—";
    html += `<tr>
      <td><strong>${escapeHtml(col)}</strong></td>
      <td><span class="badge ${typeBadgeClass(colType)}">${colType}</span></td>
      <td>${schema.null_percentage[col]}%</td>
      <td>${schema.unique_counts[col]}</td>
    </tr>`;
  }
  html += `</tbody></table>`;

  // Numeric correlations snippet (if available)
  const corr = d.numeric_stats.pairwise_correlations;
  if (corr && Object.keys(corr).length > 0) {
      const numCols = Object.keys(corr);
      html += `<h3 style="margin-top: 2rem;">Pairwise Correlation Table</h3><div style="overflow-x: auto; width: 100%; border: 1px solid var(--border); border-radius: 6px;"><table style="font-size: 0.85em; margin-top: 0; min-width: max-content;"><thead><tr><th></th>`;
      for (const col of numCols) {
          html += `<th>${escapeHtml(col)}</th>`;
      }
      html += `</tr></thead><tbody>`;
      for (const rowCol of numCols) {
          html += `<tr><td><strong>${escapeHtml(rowCol)}</strong></td>`;
          for (const col of numCols) {
              const val = corr[rowCol][col];
              const displayVal = val !== null && val !== undefined ? val : '—';
              let style = "";
              if (val !== null && val !== undefined && rowCol !== col) {
                const absVal = Math.abs(val);
                if (absVal > 0.9) style = 'background: #FEE2E2; color: #991B1B; font-weight: 600;';
                else if (absVal > 0.7) style = 'background: #FEF3C7; color: #92400E;';
              }
              html += `<td style="${style}">${displayVal}</td>`;
          }
          html += `</tr>`;
      }
      html += `</tbody></table></div>`;
  }

  // Extract data using normalization layer
  const chartData = extractChartData(d);

  html += `<h3 style="margin-top: 2rem;">Data Visualizations</h3>
  <div style="display:grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); gap:1.5rem; margin-top:1rem;">`;


  // 2. Correlation Heatmap
  html += `<div style="background:var(--surface); border:1px solid var(--border); padding:1rem; border-radius:8px;">
      <h4 style="margin-bottom:1rem;font-size:1rem;">Correlation Heatmap</h4>
      <div style="height:250px; position:relative;">
          ${chartData.correlations.matrix.length > 0 ? `<canvas id="corrChart"></canvas>` : `<div style="height:100%;display:flex;align-items:center;justify-content:center;color:var(--text-muted);font-size:0.9rem;text-align:center;">Insufficient numeric data to generate a correlation heatmap.</div>`}
      </div>
  </div>`;

  html += `</div>`;
  profileResult.innerHTML = html;
  refreshIcons();

  // Render Charts
  setTimeout(() => {
      // Render Correlation Heatmap using Matrix plugin
      if (chartData.correlations.matrix.length > 0) {
          const ctxCorr = document.getElementById("corrChart");
          if (ctxCorr) {
              new Chart(ctxCorr, {
                  type: 'matrix',
                  data: {
                      datasets: [{
                          label: 'Correlation',
                          data: chartData.correlations.matrix,
                          backgroundColor(context) {
                              const value = context.dataset.data[context.dataIndex].v;
                              const alpha = Math.abs(value);
                              return value < 0 ? `rgba(239, 68, 68, ${alpha})` : `rgba(59, 130, 246, ${alpha})`;
                          },
                          borderColor: 'rgba(255, 255, 255, 0.5)',
                          borderWidth: 1,
                          width: ({chart}) => (chart.chartArea || {}).width / chartData.correlations.features.length - 1,
                          height: ({chart}) => (chart.chartArea || {}).height / chartData.correlations.features.length - 1
                      }]
                  },
                  options: {
                      responsive: true,
                      maintainAspectRatio: false,
                      scales: {
                          x: { type: 'category', labels: chartData.correlations.features, offset: true, ticks: { autoSkip: false }, grid: { display: false } },
                          y: { type: 'category', labels: chartData.correlations.features, offset: true, ticks: { autoSkip: false }, grid: { display: false } }
                      },
                      plugins: {
                          legend: { display: false },
                          tooltip: {
                              callbacks: {
                                  title: () => '',
                                  label: (ctx) => `${ctx.raw.x} & ${ctx.raw.y}: ${ctx.raw.v}`
                              }
                          }
                      }
                  }
              });
          }
      }
  }, 100);
}

/** Normalize backend response for dynamic UI charts */
function extractChartData(d) {
    const data = {
        nulls: { labels: [], values: [] },
        correlations: { features: [], matrix: [] }
    };
    
    // Parse Nulls
    if (d.schema_info && d.schema_info.null_percentage) {
        for (const [col, pct] of Object.entries(d.schema_info.null_percentage)) {
            data.nulls.labels.push(col);
            data.nulls.values.push(pct);
        }
    }
    
    // Parse Correlations
    if (d.numeric_stats && d.numeric_stats.pairwise_correlations) {
        const corr = d.numeric_stats.pairwise_correlations;
        const cols = Object.keys(corr);
        if (cols.length >= 2) {
            data.correlations.features = cols;
            cols.forEach((colY) => {
                cols.forEach((colX) => {
                    const val = corr[colY][colX];
                    if (val !== null && val !== undefined) {
                        data.correlations.matrix.push({ x: colX, y: colY, v: val });
                    }
                });
            });
        }
    }
    return data;
}

/* ═══════════════════ 3. TRAIN ═══════════════════════════════ */

targetSelect.addEventListener("change", updateDisabledTargetCheckbox);

function updateDisabledTargetCheckbox() {
  const target = targetSelect.value;
  document.querySelectorAll('input[name="featureCol"]').forEach(chk => {
    if (chk.value === target) {
      chk.checked = false;
      chk.disabled = true;
    } else {
      chk.disabled = false;
    }
  });
}

trainBtn.addEventListener("click", async () => {
  const target = targetSelect.value;
  if (!target) { showToast("Select a target column", true); return; }

  const selectedFeatures = Array.from(document.querySelectorAll('input[name="featureCol"]:checked')).map(chk => chk.value);
  if (selectedFeatures.length === 0) { showToast("Select at least one feature column", true); return; }

  setLoading(trainBtn, true);
  trainResult.classList.add("hidden");

  try {
    const data = await api("POST", "/train", { 
      file_id: fileId, 
      target_column: target, 
      features: selectedFeatures 
    });
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
           <label>${escapeHtml(feat)}</label>
           <input type="text" name="${escapeHtml(feat)}" placeholder="Enter value" />
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
  html += `<p style="margin-bottom: 0.25rem;"><strong>Problem type:</strong> <span style="text-transform: capitalize;">${escapeHtml(d.problem_type)}</span></p>`;
  html += `<p style="margin-bottom: 0.25rem;"><strong>Target:</strong> ${escapeHtml(d.target)}</p>`;
  html += `<p style="margin-bottom: 1rem;"><strong>Model ID:</strong> <code>${escapeHtml(d.model_id)}</code></p>`;

  // Metrics pills
  html += `<h4 style="margin-bottom: 0.5rem; font-size: 0.9rem; color: var(--text-muted);">Evaluation Metrics</h4>`;
  html += `<div>`;
  for (const [k, v] of Object.entries(d.metrics)) {
    const cls = metricClass(k, v);
    html += `<span class="metric-pill ${cls}">${k}: ${v}</span>`;
  }
  html += `</div>`;

  // Feature importances
  if (d.feature_importances && Object.keys(d.feature_importances).length > 0) {
    html += `<h4 style="margin-top: 1.5rem; margin-bottom: 0.75rem; font-size: 0.9rem; color: var(--text-muted);">Feature Importances</h4>`;
    html += renderFeatureImportances(d.feature_importances);
  }

  // Features used
  if (d.features && d.features.length > 0) {
    html += `<h4 style="margin-top: 1.5rem; margin-bottom: 0.5rem; font-size: 0.9rem; color: var(--text-muted);">Features Used (${d.features.length})</h4>`;
    html += `<div style="display: flex; flex-wrap: wrap; gap: 0.35rem;">`;
    d.features.forEach(f => {
      html += `<span class="badge badge--gray">${escapeHtml(f)}</span>`;
    });
    html += `</div>`;
  }

  trainResult.innerHTML = html;
  refreshIcons();
}

/** Render feature importances as horizontal bar chart using divs */
function renderFeatureImportances(importances) {
  const entries = Object.entries(importances).slice(0, 10); // top 10
  if (entries.length === 0) return "";

  const maxVal = Math.max(...entries.map(([, v]) => v));

  let html = `<div class="importance-chart">`;
  for (const [name, value] of entries) {
    const pct = maxVal > 0 ? (value / maxVal * 100) : 0;
    const displayPct = (value * 100).toFixed(1);
    html += `
      <div class="importance-row">
        <span class="importance-label">${escapeHtml(name)}</span>
        <div class="importance-bar-bg">
          <div class="importance-bar" style="width: ${pct}%;"></div>
        </div>
        <span class="importance-value">${displayPct}%</span>
      </div>`;
  }
  html += `</div>`;
  return html;
}

function metricClass(key, val) {
  if (["accuracy", "balanced_accuracy", "precision", "recall", "f1_weighted", "f1_macro", "r2"].includes(key)) {
    if (val >= 0.8) return "metric-pill--good";
    if (val >= 0.5) return "metric-pill--warn";
    return "metric-pill--bad";
  }
  if (["mae", "rmse", "mse", "mape", "max_error"].includes(key)) {
    return "metric-pill--warn"; // can't judge good/bad without context
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

  html += `<table><thead><tr><th style="width: 50px;">#</th><th>Predicted ${escapeHtml(targetCol)}</th>`;
  if (hasConfidence) html += `<th>Confidence</th>`;
  html += `</tr></thead><tbody>`;

  d.predictions.forEach((p, i) => {
    html += `<tr><td>${i + 1}</td><td><strong>${p[targetCol]}</strong></td>`;
    if (p.confidence !== undefined && p.confidence !== null) {
      const confPct = (p.confidence * 100).toFixed(1);
      const confClass = p.confidence >= 0.8 ? "metric-pill--good" : p.confidence >= 0.5 ? "metric-pill--warn" : "metric-pill--bad";
      html += `<td><span class="metric-pill ${confClass}">${confPct}%</span></td>`;
    }
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
        <p style="font-size: 0.95rem; line-height: 1.7; color: var(--text);">${escapeHtml(d.summary_text)}</p>
    </div>
  `;

  // Feature importances from model
  if (d.feature_importances && Object.keys(d.feature_importances).length > 0) {
    html += `<h3 style="margin-top: 2rem;">Top Predictors (Model Importance)</h3>`;
    html += renderFeatureImportances(d.feature_importances);
  }

  // Top correlated features
  if (d.top_correlated_features && d.top_correlated_features.length) {
    html += `<h3 style="margin-top: 2rem;">Top Correlated Features</h3>`;
    html += `<table><thead><tr><th>Feature</th><th>Correlation Score</th></tr></thead><tbody>`;
    d.top_correlated_features.forEach(f => {
      html += `<tr><td><strong>${escapeHtml(f.feature)}</strong></td><td>${f.correlation}</td></tr>`;
    });
    html += `</tbody></table>`;
  }

  // Model performance metrics
  html += `<h3 style="margin-top: 2rem;">Model Performance Metrics</h3>`;
  html += `<div style="margin-top: 0.5rem;">`;
  for (const [k, v] of Object.entries(d.model_performance)) {
    const cls = metricClass(k, v);
    html += `<span class="metric-pill ${cls}">${k}: ${v}</span>`;
  }
  html += `</div>`;

  // Warnings
  if (d.warnings && d.warnings.length) {
    html += `<h3 style="margin-top: 2rem;">ℹ Analysis Notes</h3>`;
    html += renderWarnings(d.warnings);
  }

  summaryResult.innerHTML = html;
  refreshIcons();
}
