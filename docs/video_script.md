# Demo video script (camera on, ~8–10 minutes)

Record with your webcam visible (OBS "Display + Video Capture" scene, or Zoom/Meet screen-share recording).
Everything below uses the deployed app; keep a terminal open for the two commands in step 7–8.

**Before recording**
- Open the deployed UI and confirm the status bar shows *API healthy* and *model v1* (or your current version).
- Use `demo_upload/demo_batch.zip` from the repository: 40 known-label cells (class folders) from validation-split
  patients — never used to train v1 and not part of the gold test set the acceptance gate scores on.
- Have `locust/results/scaling.png` and `locust/results/summary.csv` open in another tab.

---

**1. Introduction (30 s)** — on camera
> "Hi, I'm Ogayo Andrew. This is SmearScan, a malaria parasite screening service. Reading a blood smear under the
> microscope is slow, skilled work, and Africa carries 94% of the world's malaria cases. SmearScan classifies single red-blood-cell images as parasitized or
> uninfected so a slide can be triaged automatically. I'll show the deployed app end to end: prediction, uploading new
> labelled data, real retraining from my saved model, monitoring, and load testing."

**2. Architecture in one breath (30 s)** — README architecture diagram
> "FastAPI serves the model and this web UI. Uploads and jobs are stored in SQLite and files on a persistent volume.
> Retraining runs in a separate worker process, fine-tunes the active MobileNetV2 model, and only promotes the new
> version if it passes an acceptance gate on held-out patients. Everything is one Docker image."

**3. Single prediction (1 min)** — *Predict* tab
- Click **Load a parasitized example** (the file name shows its true label) → **Predict**.
- Point at: predicted class, confidence, probability bar against the threshold, model version, latency, stain fraction.
> "The true label is Parasitized and the model agrees with high confidence. Results near the threshold are flagged
> for microscopist review."
- Click **Parasitized** under *Confirmed by microscopy?* → "this ground truth feeds production-accuracy monitoring."
- Repeat once with **Load an uninfected example**.

**4. Dataset insights (1 min)** — *Dataset insights* tab
- Walk through the four cards: stain fraction (the key signal, AUC 0.97 alone), brightness and the slide batch effect
  (why I split by patient and monitor drift), crop size (no shortcut), cells per patient.

**5. Bulk upload + persistence (1 min)** — *Data & retraining* tab
- Choose `demo_batch.zip` under *ZIP archive*, note "demo — clinic batch", **Upload**.
- Show the report: accepted / duplicates / rejected. Then upload one corrupted or non-image file to show rejection.
- Scroll to *Stored uploads*: batch table + thumbnails. Refresh the page → "the data is persisted, not in memory."

**6. Retraining (2–3 min)** — same tab
- Point at *Waiting for training*. Click **Start retraining**.
- Narrate the stages as they light up: loading data → preprocessing (shows new + replayed sample counts) →
  loading model ("it loads the currently active model's saved weights") → training (epoch chart updates live) →
  evaluating → registering.
- Read the result: acceptance-gate table (F1, recall, new-data holdout), candidate vs active metrics, promoted or rejected.
> "If the candidate had degraded recall on held-out patients, it would be rejected and the old model would keep serving."
- Status bar now shows the new model version. *Models* tab: version history with parent, metrics, status;
  show **Roll back** exists (optionally roll back and re-activate).

**7. Monitoring (1 min)** — *Monitoring* tab
- KPIs (uptime, request count, error rate, p95), traffic chart, endpoint table, confidence histogram.
- Drift table. Optional: in the terminal run
  `python scripts/simulate_drift.py --host <URL> --n 100 --feedback`, refresh, and show brightness/stain PSI go to
  *drift* and the retraining recommendation change.
> "Production accuracy is only computed over predictions that received confirmed labels; otherwise the app shows none."

**8. Load testing (1 min)** — results tab/terminal
- Show `scaling.png` and the summary table: throughput, p50/p95/p99, failures for 1, 2 and 3 containers.
- Explain the trend honestly (see README "Load testing" interpretation) and where the bottleneck is.

**9. Close (20 s)** — on camera
> "The repository has the notebook with the full evaluation, the tests including the upload-to-retraining workflow,
> Docker and deployment instructions, and the load-test scripts. Thanks for watching."
