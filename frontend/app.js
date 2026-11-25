/* SmearScan web UI - talks only to the real backend API on the same origin. */
"use strict";
const $ = (s) => document.querySelector(s);
const $$ = (s) => [...document.querySelectorAll(s)];
const fmt = (v, d = 3) => (v === null || v === undefined || Number.isNaN(v) ? "—" : Number(v).toFixed(d));
const pct = (v, d = 1) => (v === null || v === undefined ? "—" : (100 * v).toFixed(d) + "%");
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const charts = {};
const PURPLE = "#5b2a6e", PINK = "#d9818c", GREEN = "#4f9a72", GREY = "#b8b2bf";

async function api(path, opts = {}) {
  const res = await fetch(path, opts);
  let body = null;
  try { body = await res.json(); } catch { /* non-JSON */ }
  if (!res.ok) {
    const d = body && body.detail;
    throw new Error(typeof d === "string" ? d : d ? JSON.stringify(d) : `${res.status} ${res.statusText}`);
  }
  return body;
}
function toast(msg) { const t = $("#toast"); t.textContent = msg; t.classList.add("show"); clearTimeout(t._h); t._h = setTimeout(() => t.classList.remove("show"), 3500); }
function setState(el, cls, text) { el.className = "state " + (cls || ""); el.textContent = text || ""; }
function uptime(s) { const d = Math.floor(s / 86400), h = Math.floor(s % 86400 / 3600), m = Math.floor(s % 3600 / 60); return d ? `${d}d ${h}h` : h ? `${h}h ${m}m` : `${m}m ${Math.floor(s % 60)}s`; }
function chart(id, cfg) { if (charts[id]) charts[id].destroy(); charts[id] = new Chart(document.getElementById(id), cfg); return charts[id]; }
const baseOpts = { responsive: true, maintainAspectRatio: false, animation: false, plugins: { legend: { labels: { boxWidth: 12, font: { size: 11 } } } } };

/* ---------------------------------------------------------------- tabs */
const loaders = { predict: () => {}, data: loadData, monitor: loadMonitoring, models: loadModels, insights: loadInsights };
$$(".tabs button").forEach((b) => b.addEventListener("click", () => {
  $$(".tabs button").forEach((x) => x.classList.toggle("active", x === b));
  $$(".tab").forEach((t) => t.classList.toggle("active", t.id === "tab-" + b.dataset.tab));
  history.replaceState(null, "", "#" + b.dataset.tab);
  loaders[b.dataset.tab]();
}));

/* ---------------------------------------------------------------- status bar */
async function refreshStatus() {
  try {
    const h = await api("/health");
    $("#st-health").className = "pill ok"; $("#st-health").textContent = `API healthy · ${h.replica}`;
    $("#st-model").className = h.model_loaded ? "pill info" : "pill bad";
    $("#st-model").textContent = h.model_loaded ? `model ${h.model_version}` : "no model loaded";
    $("#st-uptime").textContent = "uptime " + uptime(h.uptime_s);
  } catch (e) {
    $("#st-health").className = "pill bad"; $("#st-health").textContent = "API unreachable";
  }
}

/* ---------------------------------------------------------------- predict */
let predFile = null, lastPredId = null;
const dropEl = $("#pred-drop");
function choosePredFile(file) {
  predFile = file; $("#pred-btn").disabled = !file;
  $("#pred-drop-text").textContent = file ? `${file.name} (${(file.size / 1024).toFixed(1)} KB)` : "Drop an image here or browse";
}
$("#pred-file").addEventListener("change", (e) => choosePredFile(e.target.files[0]));
["dragover", "dragenter"].forEach((ev) => dropEl.addEventListener(ev, (e) => { e.preventDefault(); dropEl.classList.add("over"); }));
["dragleave", "drop"].forEach((ev) => dropEl.addEventListener(ev, () => dropEl.classList.remove("over")));
dropEl.addEventListener("drop", (e) => { e.preventDefault(); if (e.dataTransfer.files[0]) choosePredFile(e.dataTransfer.files[0]); });
$$("[data-example]").forEach((b) => b.addEventListener("click", async () => {
  const res = await fetch(`/api/examples/${b.dataset.example}`);
  if (!res.ok) return toast("Could not load example");
  const name = res.headers.get("X-Example-Name") || `${b.dataset.example}.png`;
  choosePredFile(new File([await res.blob()], name, { type: "image/png" }));
  toast(`Loaded ${b.dataset.example} test example (true label shown in file name)`);
}));
$("#pred-btn").addEventListener("click", async () => {
  if (!predFile) return;
  const st = $("#pred-state"); setState(st, "busy", "Scoring…"); $("#pred-btn").disabled = true;
  const fd = new FormData(); fd.append("file", predFile);
  try {
    const r = await api("/api/predict", { method: "POST", body: fd });
    lastPredId = r.prediction_id; renderPrediction(r);
    setState(st, "ok", `Done in ${r.latency_ms} ms`);
  } catch (e) { setState(st, "bad", e.message); }
  finally { $("#pred-btn").disabled = false; }
});
function renderPrediction(r) {
  $("#pred-empty").hidden = true; $("#pred-result").hidden = false;
  $("#pred-preview").src = URL.createObjectURL(predFile);
  const pos = r.predicted_class === "Parasitized";
  $("#pred-label").textContent = r.predicted_class; $("#pred-label").className = "verdict " + (pos ? "pos" : "neg");
  $("#pred-conf").textContent = `confidence ${pct(r.confidence, 2)}${r.low_confidence ? " · LOW CONFIDENCE" : ""}`;
  $("#pred-rec").textContent = r.recommendation;
  $("#pred-bar").style.width = (100 * r.probabilities.Parasitized).toFixed(1) + "%";
  $(".bar .thr").style.left = (100 * r.threshold) + "%";
  const s = r.image_stats;
  $("#pred-kv").innerHTML = [
    ["P(Parasitized)", fmt(r.probabilities.Parasitized, 4)], ["P(Uninfected)", fmt(r.probabilities.Uninfected, 4)],
    ["Model version", r.model_version], ["Prediction id", r.prediction_id],
    ["Image size", `${s.width} × ${s.height} px`], ["Purple-stain fraction", pct(s.stain_fraction, 2)],
    ["Cell brightness", fmt(s.brightness, 1)], ["Server latency", `${r.latency_ms} ms`],
  ].map(([k, v]) => `<tr><td>${k}</td><td>${esc(v)}</td></tr>`).join("");
  setState($("#fb-state"), "", "");
}
$$("[data-fb]").forEach((b) => b.addEventListener("click", async () => {
  if (!lastPredId) return;
  try {
    await api(`/api/predictions/${lastPredId}/feedback`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ true_label: b.dataset.fb }) });
    setState($("#fb-state"), "ok", `recorded: ${b.dataset.fb}`);
  } catch (e) { setState($("#fb-state"), "bad", e.message); }
}));

/* ---------------------------------------------------------------- data & retraining */
$("#up-btn").addEventListener("click", async () => {
  const files = $("#up-files").files, zip = $("#up-zip").files[0], label = $("#up-label").value;
  const st = $("#up-state");
  if (!files.length && !zip) return setState(st, "bad", "Choose images or a ZIP first");
  if (files.length && !label) return setState(st, "bad", "Choose a label for the selected images");
  const fd = new FormData();
  [...files].forEach((f) => fd.append("files", f));
  if (files.length) fd.append("label", label);
  if (zip) fd.append("archive", zip);
  fd.append("note", $("#up-note").value);
  setState(st, "busy", `Uploading ${files.length + (zip ? 1 : 0)} item(s)…`); $("#up-btn").disabled = true;
  try {
    const r = await api("/api/uploads", { method: "POST", body: fd });
    setState(st, "ok", `Batch ${r.batch_id} stored`);
    $("#up-report").innerHTML = `<p><span class="report-ok">${r.accepted} accepted and preprocessed</span> · ${r.duplicates} duplicate(s) skipped · <span class="${r.rejected ? "report-bad" : ""}">${r.rejected} rejected</span> of ${r.received} received.</p>` +
      (r.rejected_items.length ? `<table class="tbl"><tr><th>Rejected file</th><th>Reason</th></tr>${r.rejected_items.slice(0, 20).map((x) => `<tr><td>${esc(x.file)}</td><td>${esc(x.reason)}</td></tr>`).join("")}</table>` : "");
    $("#up-files").value = ""; $("#up-zip").value = "";
    loadData();
  } catch (e) { setState(st, "bad", e.message); }
  finally { $("#up-btn").disabled = false; }
});

async function loadData() {
  try {
    const [d, s, j] = await Promise.all([api("/api/datasets"), api("/api/samples?limit=48"), api("/api/jobs?limit=1")]);
    const by = (lbl) => d.by_label_status.filter((r) => r.label === lbl).reduce((a, r) => a + r.n, 0);
    $("#ds-stats").innerHTML = [["Stored samples", d.total_samples], ["Waiting for training", d.pending_for_training], ["Parasitized / Uninfected", `${by("Parasitized")} / ${by("Uninfected")}`]]
      .map(([k, v]) => `<div class="stat"><b>${v}</b><span>${k}</span></div>`).join("");
    $("#batches").innerHTML = d.batches.length ? `<table class="tbl"><tr><th>Batch</th><th>Uploaded (UTC)</th><th>Received</th><th>Accepted</th><th>Duplicates</th><th>Rejected</th><th>Note</th></tr>` +
      d.batches.map((b) => `<tr><td>${b.id}</td><td>${b.created_at.replace("T", " ").slice(0, 19)}</td><td>${b.n_received}</td><td>${b.n_accepted}</td><td>${b.n_duplicate}</td><td>${b.n_rejected}</td><td>${esc(b.note)}</td></tr>`).join("") + "</table>"
      : `<p class="muted">No uploads yet.</p>`;
    $("#thumbs").innerHTML = s.items.map((x) => `<figure title="${esc(x.original_name)} · ${x.status}${x.used_in_version ? " in " + x.used_in_version : ""}"><img loading="lazy" src="/api/samples/${x.id}/image" alt=""><figcaption>${x.label.slice(0, 5)} · ${x.used_in_version || x.status.slice(0, 4)}</figcaption></figure>`).join("");
    if (j.items[0]) showJob(j.items[0]);
  } catch (e) { toast("Could not load data: " + e.message); }
}

$("#rt-btn").addEventListener("click", async () => {
  const st = $("#rt-state"); setState(st, "busy", "Starting…"); $("#rt-btn").disabled = true;
  try {
    const r = await api("/api/retrain", { method: "POST", headers: { "Content-Type": "application/json" }, body: "{}" });
    setState(st, "ok", `Job ${r.job_id} started on ${r.pending_samples} new samples`);
    pollJob(r.job_id);
  } catch (e) { setState(st, "bad", e.message); $("#rt-btn").disabled = false; }
});
const STAGES = ["queued", "loading_data", "preprocessing", "loading_model", "training", "evaluating", "registering", "done"];
let pollHandle = null;
function pollJob(id) {
  clearTimeout(pollHandle);
  const tick = async () => {
    try {
      const j = await api(`/api/jobs/${id}`); showJob(j);
      if (["queued", "running"].includes(j.status)) pollHandle = setTimeout(tick, 1500);
      else { $("#rt-btn").disabled = false; refreshStatus(); loadData(); toast(`Retraining ${j.status}`); }
    } catch (e) { pollHandle = setTimeout(tick, 3000); }
  };
  tick();
}
function showJob(j) {
  $("#job-panel").hidden = false; $("#job-id").textContent = `${j.id} · parent ${j.parent_version || "—"}`;
  const cls = { succeeded: "ok", failed: "bad", rejected: "warn", running: "info", queued: "info" }[j.status] || "";
  $("#job-status").className = "pill " + cls; $("#job-status").textContent = j.status;
  const cur = STAGES.indexOf(j.stage);
  $("#job-stages").innerHTML = STAGES.slice(1).map((s, i) => {
    const idx = i + 1; let c = idx < cur || j.stage === "done" ? "done" : idx === cur ? "now" : "";
    if (j.status === "failed" && idx === cur) c = "fail";
    return `<li class="${c}">${s.replace("_", " ")}</li>`;
  }).join("") + (j.status === "failed" ? `<li class="fail">failed</li>` : "");
  $("#job-progress").style.width = (100 * (j.progress || 0)) + "%";
  $("#job-log").innerHTML = (j.log || []).map((l) => `<li>${l.at.slice(11, 19)} ${esc(l.message)}</li>`).reverse().join("");
  const ep = j.epochs || [];
  chart("job-chart", { type: "line", data: { labels: ep.map((e) => "ep " + e.epoch), datasets: [
    { label: "train loss", data: ep.map((e) => e.loss), borderColor: PINK, tension: .2 },
    { label: "val loss", data: ep.map((e) => e.val_loss), borderColor: PURPLE, tension: .2 },
    { label: "val accuracy", data: ep.map((e) => e.val_accuracy), borderColor: GREEN, yAxisID: "y1", tension: .2 }] },
    options: { ...baseOpts, scales: { y: { title: { display: true, text: "loss" } }, y1: { position: "right", grid: { drawOnChartArea: false }, title: { display: true, text: "accuracy" } } } } });
  const r = j.result;
  if (r && r.gate) {
    const m = r.metrics;
    $("#job-result").innerHTML = `<p><strong>${r.promoted ? `✅ ${r.version} promoted to active` : `⛔ ${r.version} rejected — ${r.parent} keeps serving`}</strong> (${r.seconds}s)</p>
      <table class="tbl gate"><tr><th>Acceptance check</th><th>Active ${r.parent}</th><th>Candidate ${r.version}</th><th>Rule</th><th></th></tr>
      ${r.gate.checks.map((c) => `<tr><td>${c.name}</td><td>${fmt(c.active, 4)}</td><td>${fmt(c.candidate, 4)}</td><td>${c.rule}</td><td class="${c.passed ? "pass" : "fail"}">${c.passed ? "pass" : "fail"}</td></tr>`).join("")}</table>
      <table class="tbl"><tr><th>Gold test set (n=${m.candidate.n})</th><th>accuracy</th><th>precision</th><th>recall</th><th>F1</th><th>ROC-AUC</th></tr>
      ${[["active " + r.parent, m.active], ["candidate " + r.version, m.candidate]].map(([n, x]) => `<tr><td>${n}</td><td>${fmt(x.accuracy, 4)}</td><td>${fmt(x.precision, 4)}</td><td>${fmt(x.recall, 4)}</td><td>${fmt(x.f1, 4)}</td><td>${fmt(x.roc_auc, 4)}</td></tr>`).join("")}</table>`;
  } else if (j.status === "failed") {
    $("#job-result").innerHTML = `<p class="report-bad">Failed: ${esc(j.error)}</p>`;
  } else $("#job-result").innerHTML = "";
  if (["queued", "running"].includes(j.status)) { $("#rt-btn").disabled = true; if (!pollHandle) pollJob(j.id); }
}

/* ---------------------------------------------------------------- monitoring */
async function loadMonitoring() {
  try {
    const m = await api("/api/monitoring");
    const r = m.requests, p = m.predictions;
    const pred = r.endpoints.find((e) => e.path === "/api/predict") || {};
    $("#kpis").innerHTML = [
      ["Uptime (this replica)", uptime(m.system.uptime_s)], ["Requests (1 h)", r.total], ["5xx error rate", pct(r.error_rate_5xx, 2)],
      ["Predict p95", pred.p95_ms ? pred.p95_ms + " ms" : "—"], ["Predictions (all time)", p.total_predictions],
      ["Low-confidence rate", pct(p.low_confidence_rate)],
    ].map(([k, v]) => `<div class="kpi"><b>${v}</b><span>${k}</span></div>`).join("");
    $("#mon-time").textContent = `updated ${new Date().toLocaleTimeString()} · replicas seen: ${Object.keys(r.by_replica).join(", ") || "—"}`;
    chart("ch-traffic", { type: "bar", data: { labels: r.timeline.map((t) => new Date(t.t * 1000).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })),
      datasets: [{ label: "requests/min", data: r.timeline.map((t) => t.requests), backgroundColor: "#cdb6d8", yAxisID: "y", order: 2 },
        { type: "line", label: "predict p95 (ms)", data: r.timeline.map((t) => t.p95_ms), borderColor: PURPLE, backgroundColor: PURPLE, yAxisID: "y1", pointRadius: 2, order: 1 }] },
      options: { ...baseOpts, scales: { y: { position: "left", beginAtZero: true, title: { display: true, text: "requests / min" } },
        y1: { position: "right", beginAtZero: true, grid: { drawOnChartArea: false }, title: { display: true, text: "p95 ms" } } } } });
    chart("ch-conf", { type: "bar", data: { labels: p.confidence_hist.edges.slice(0, -1).map((e, i) => `${e}-${p.confidence_hist.edges[i + 1]}`),
      datasets: [{ label: "predictions", data: p.confidence_hist.counts, backgroundColor: "#d9a3ab" }] },
      options: { ...baseOpts, scales: { y: { beginAtZero: true, ticks: { precision: 0 }, title: { display: true, text: "predictions" } }, x: { title: { display: true, text: "confidence" } } } } });
    $("#tbl-endpoints").innerHTML = `<tr><th>Endpoint</th><th>Count</th><th>p50</th><th>p95</th><th>p99</th><th>4xx</th><th>5xx</th></tr>` +
      r.endpoints.map((e) => `<tr><td>${e.path}</td><td>${e.count}</td><td>${e.p50_ms ?? "—"}</td><td>${e.p95_ms ?? "—"}</td><td>${e.p99_ms ?? "—"}</td><td>${e.errors_4xx}</td><td>${e.errors_5xx}</td></tr>`).join("");
    const drift = p.drift || {};
    $("#tbl-drift").innerHTML = `<tr><th>Feature</th><th>n</th><th>PSI</th><th>KS p</th><th>Recent mean</th><th>Train mean</th><th>Status</th></tr>` +
      Object.entries(drift).map(([f, d]) => `<tr><td>${f}${d.alerting === false ? " <span class=\"muted small\">(info)</span>" : ""}</td><td>${d.n}</td><td>${fmt(d.psi)}</td><td>${fmt(d.ks_pvalue, 4)}</td><td>${fmt(d.recent_mean)}</td><td>${fmt(d.reference_mean)}</td><td><span class="pill ${d.status === "drift" ? "bad" : d.status === "moderate" ? "warn" : d.status === "stable" ? "ok" : "info"}">${d.status.replace("_", " ")}</span></td></tr>`).join("");
    const q = p.labelled_quality;
    $("#quality").innerHTML = q ? `<div class="stats"><div class="stat"><b>${pct(q.accuracy)}</b><span>accuracy on ${q.n_labelled} labelled predictions</span></div><div class="stat"><b>${pct(q.recall)}</b><span>recall (Parasitized)</span></div><div class="stat"><b>${pct(q.expected_accuracy_from_test)}</b><span>accuracy expected from test set</span></div></div>
      <p class="muted small">Computed only over predictions that received a confirmed label via feedback. Small samples are noisy.</p>`
      : `<p class="muted">No ground-truth feedback yet. Production accuracy cannot be measured without confirmed labels — use the feedback buttons after a prediction, or <code>POST /api/predictions/{id}/feedback</code> from a lab system.</p>`;
    const sg = m.retraining_signal;
    $("#signal").innerHTML = `<p><span class="pill ${sg.recommended ? "warn" : "ok"}">${sg.recommended ? "retraining recommended" : "no action needed"}</span></p>` +
      (sg.reasons.length ? `<ul>${sg.reasons.map((x) => `<li>${esc(x)}</li>`).join("")}</ul>` : `<p class="muted">No drift alert, no accuracy drop on labelled data, and fewer new samples than the retraining minimum.</p>`) +
      `<p class="muted small">Model versions serving the recent window: ${Object.entries(p.model_versions).map(([v, n]) => `${v} (${n})`).join(", ") || "—"}</p>`;
  } catch (e) { toast("Monitoring failed: " + e.message); }
}
$("#mon-refresh").addEventListener("click", loadMonitoring);

/* ---------------------------------------------------------------- models */
async function loadModels() {
  try {
    const [info, models, jobs] = await Promise.all([api("/api/model"), api("/api/models"), api("/api/jobs?limit=20")]);
    const t = info.metrics.test;
    $("#active-model").innerHTML = `<table class="kv">
      ${[["Version", info.version], ["Architecture", info.architecture], ["Framework", info.framework], ["Created", info.created_at],
        ["Trained by", info.trigger], ["Parent", info.parent_version || "— (ImageNet MobileNetV2)"], ["Decision threshold", info.decision_threshold],
        ["Training data", Object.entries(info.training_data || {}).filter(([k]) => !Array.isArray(info.training_data[k])).map(([k, v]) => `${k}: ${typeof v === "object" ? JSON.stringify(v) : v}`).join(" · ")]]
        .map(([k, v]) => `<tr><td>${k}</td><td>${esc(v)}</td></tr>`).join("")}</table>`;
    const cm = t.confusion_matrix.matrix;
    const shade = (v) => `background: rgba(91,42,110,${(0.08 + 0.8 * v / Math.max(...cm.flat())).toFixed(2)}); color:${v / Math.max(...cm.flat()) > .5 ? "#fff" : "inherit"}`;
    $("#cm").innerHTML = `<div class="cm"><div class="h"></div><div class="h">pred Uninfected</div><div class="h">pred Parasitized</div>
      <div class="h">true Uninfected</div><div style="${shade(cm[0][0])}">${cm[0][0]}</div><div style="${shade(cm[0][1])}">${cm[0][1]}</div>
      <div class="h">true Parasitized</div><div style="${shade(cm[1][0])}">${cm[1][0]}</div><div style="${shade(cm[1][1])}">${cm[1][1]}</div></div>
      <p class="muted small">n = ${t.n} cells from held-out patients</p>`;
    const keys = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"];
    const ds = [{ label: info.version, data: keys.map((k) => t[k]), backgroundColor: PURPLE }];
    if (info.baseline_metrics) ds.push({ label: "custom CNN baseline", data: keys.map((k) => info.baseline_metrics.test[k]), backgroundColor: GREY });
    chart("ch-metrics", { type: "bar", data: { labels: keys, datasets: ds }, options: { ...baseOpts, scales: { y: { min: 0.85, max: 1 } } } });
    const vs = Object.entries(models.versions).sort((a, b) => Number(b[0].slice(1)) - Number(a[0].slice(1)));
    $("#tbl-versions").innerHTML = `<tr><th>Version</th><th>Status</th><th>Created (UTC)</th><th>Parent</th><th>Test acc</th><th>Test F1</th><th>Recall</th><th>ROC-AUC</th><th></th></tr>` +
      vs.map(([v, x]) => `<tr class="${v === models.active ? "active-row" : ""}"><td><strong>${v}</strong></td><td><span class="pill ${x.status === "active" ? "ok" : x.status === "rejected" ? "bad" : ""}">${x.status}</span></td><td>${x.created_at.replace("T", " ").slice(0, 19)}</td><td>${x.parent || "—"}</td><td>${fmt(x.summary.test_accuracy, 4)}</td><td>${fmt(x.summary.test_f1, 4)}</td><td>${fmt(x.summary.test_recall, 4)}</td><td>${fmt(x.summary.test_roc_auc, 4)}</td>
      <td>${v !== models.active && x.status !== "rejected" ? `<button class="ghost small" data-activate="${v}">Activate</button>` : ""}</td></tr>`).join("");
    $$("[data-activate]").forEach((b) => b.addEventListener("click", async () => {
      try { await api(`/api/models/${b.dataset.activate}/activate`, { method: "POST" }); toast(`${b.dataset.activate} is now active`); refreshStatus(); loadModels(); } catch (e) { toast(e.message); }
    }));
    $("#tbl-jobs").innerHTML = `<tr><th>Job</th><th>Status</th><th>Started (UTC)</th><th>Parent → candidate</th><th>Epochs</th><th>Outcome</th></tr>` +
      jobs.items.map((j) => `<tr><td>${j.id}</td><td>${j.status}</td><td>${(j.started_at || j.created_at).replace("T", " ").slice(0, 19)}</td><td>${j.parent_version || "—"} → ${j.candidate_version || "—"}</td><td>${(j.epochs || []).length}</td><td>${esc(j.error || (j.log || []).slice(-1).map((l) => l.message)[0] || "")}</td></tr>`).join("");
  } catch (e) { toast("Could not load models: " + e.message); }
}
$("#rb-btn").addEventListener("click", async () => {
  const st = $("#rb-state"); setState(st, "busy", "Rolling back…");
  try { const r = await api("/api/models/rollback", { method: "POST" }); setState(st, "ok", `Active model is now ${r.active}`); refreshStatus(); loadModels(); }
  catch (e) { setState(st, "bad", e.message); }
});

/* ---------------------------------------------------------------- insights */
let insightsLoaded = false;
async function loadInsights() {
  if (insightsLoaded) return;
  try {
    const d = await api("/api/insights");
    const c = d.card;
    $("#card-info").innerHTML = `<table class="kv">${[["Dataset", c.name], ["Provenance", c.provenance], ["Licence", c.licence], ["Citation", c.citation],
      ["Images / patients", `${c.n_images.toLocaleString()} / ${c.n_patients}`], ["Split", c.split_strategy],
      ["Train / val / test cells", ["train", "val", "test"].map((s) => Object.values(c.counts[s]).reduce((a, b) => a + b, 0)).join(" / ")],
      ["Patients per split", Object.values(c.patients_per_split).join(" / ")]].map(([k, v]) => `<tr><td>${k}</td><td>${esc(v)}</td></tr>`).join("")}</table>`;
    $("#insights").innerHTML = d.features.map((f, i) => `<div class="card"><h2>${i + 1}. ${esc(f.title)}</h2><div class="insight">
      <div class="chart-box"><canvas id="ins-${f.key}"></canvas></div>
      <div><p><strong>What the data shows.</strong> ${esc(f.finding)}</p><p class="why"><strong>Why it matters.</strong> ${esc(f.why_it_matters)}</p></div></div></div>`).join("");
    d.features.forEach((f) => {
      if (f.key === "patients") {
        chart("ins-" + f.key, { type: "bar", data: { labels: f.chart.values.map((_, i) => i + 1), datasets: [{ label: "cells per patient", data: f.chart.values, backgroundColor: "#b9a2c6" }] },
          options: { ...baseOpts, scales: { x: { title: { display: true, text: f.x_label }, ticks: { display: false } } } } });
      } else {
        const b = f.chart.bins, labels = b.slice(0, -1).map((v, i) => ((v + b[i + 1]) / 2).toFixed(f.key === "stain_fraction" ? 3 : 0));
        const norm = (a) => { const s = a.reduce((x, y) => x + y, 0); return a.map((v) => v / s); };
        chart("ins-" + f.key, { type: "line", data: { labels, datasets: [
          { label: "Parasitized", data: norm(f.chart.Parasitized), borderColor: PURPLE, backgroundColor: "rgba(91,42,110,.15)", fill: true, pointRadius: 0, tension: .3 },
          { label: "Uninfected", data: norm(f.chart.Uninfected), borderColor: PINK, backgroundColor: "rgba(217,129,140,.15)", fill: true, pointRadius: 0, tension: .3 }] },
          options: { ...baseOpts, scales: { x: { title: { display: true, text: f.x_label } }, y: { title: { display: true, text: "share of cells" } } } } });
      }
    });
    insightsLoaded = true;
  } catch (e) { $("#insights").innerHTML = `<p class="report-bad">Could not load insights: ${esc(e.message)}</p>`; }
}

/* ---------------------------------------------------------------- boot */
refreshStatus(); setInterval(refreshStatus, 15000);
const start = location.hash.slice(1);
if (start && loaders[start]) $(`.tabs button[data-tab="${start}"]`).click();
