"""Section 9 of the notebook (significance, robustness, cost) and section 10 (discussion).

Kept separate so the cells can be appended to an already-executed notebook: they only
depend on the split arrays and on the *serialised* models written in section 8.
"""
SECTION_MD = """
## 9. Is transfer learning worth it here?

On the test patients the two models are close, and the baseline is slightly ahead. Three questions decide which one
should be deployed: is the gap real, how do the models behave when slides look different from the training slides
(the batch effect found in the EDA), and what does each cost at inference time? These cells reload both models from
their saved artifacts, which also confirms the serialised files are the ones evaluated above.
"""

SECTION_CODE = r'''
tl_saved = M.load_keras(S.models_dir / "v1" / "model.keras")
base_saved = M.load_keras(S.models_dir / "baseline_cnn.keras")
p_tl_s, p_base_s = M.predict_proba(tl_saved, x_te), M.predict_proba(base_saved, x_te)

# (a) Patient-level bootstrap of the accuracy difference: resample whole test patients, because cells from one
#     slide are correlated and a cell-level bootstrap would understate the uncertainty.
pid = np.array([r["patient_id"] for r in rows_te]); patients = np.unique(pid)
cells_of = {p: np.where(pid == p)[0] for p in patients}
rng = np.random.default_rng(0); diffs = []
for _ in range(2000):
    s = np.concatenate([cells_of[p] for p in rng.choice(patients, len(patients))])
    diffs.append(((p_tl_s[s] >= .5) == y_te[s]).mean() - ((p_base_s[s] >= .5) == y_te[s]).mean())
diffs = np.array(diffs)
point = ((p_tl_s >= .5) == y_te).mean() - ((p_base_s >= .5) == y_te).mean()
print(f"accuracy(transfer) - accuracy(baseline) = {point:+.4f}; 95% patient-bootstrap CI "
      f"[{np.percentile(diffs, 2.5):+.4f}, {np.percentile(diffs, 97.5):+.4f}]; P(transfer better) = {(diffs > 0).mean():.2f}")

# (b) Robustness to a stain / illumination shift (darker, bluer slides: thicker smear, different Giemsa batch or lamp)
from PIL import ImageEnhance
def shift_split(brightness, red, blue):
    out = np.empty_like(x_te)
    for i, r in enumerate(rows_te):
        img = ImageEnhance.Brightness(Image.open(S.data_dir / r["path"]).convert("RGB")).enhance(brightness)
        R, G, B = img.split()
        img = Image.merge("RGB", (R.point(lambda v: v * red), G, B.point(lambda v: min(255, v * blue))))
        out[i] = to_uint8_array(img, S.image_size)
    return out
robust = []
for name, args in [("none", None), ("mild", (0.85, 0.95, 1.06)), ("strong", (0.72, 0.90, 1.12))]:
    xs = x_te if args is None else shift_split(*args)
    for mname, mdl in [("MobileNetV2 transfer", tl_saved), ("Custom CNN baseline", base_saved)]:
        mm = M.evaluate_probs(y_te, M.predict_proba(mdl, xs))
        robust.append({"shift": name, "model": mname, "accuracy": mm["accuracy"], "recall": mm["recall"],
                       "specificity": mm["specificity"], "f1": mm["f1"], "roc_auc": mm["roc_auc"]})
robust = pd.DataFrame(robust)
fig, ax = plt.subplots(figsize=(7, 3.6))
sns.barplot(data=robust, x="shift", y="accuracy", hue="model", ax=ax); ax.set_ylim(0.5, 1); ax.set_title("Test accuracy under stain/illumination shift")
plt.tight_layout(); plt.savefig(FIG / "robustness.png", dpi=110); plt.show()

# (c) CPU inference cost for one image (what the API does per request)
lat = {}
for mname, mdl in [("MobileNetV2 transfer", tl_saved), ("Custom CNN baseline", base_saved)]:
    f = tf.function(lambda a, m=mdl: m(a, training=False)); one = x_te[:1].astype("float32") / 127.5 - 1; f(one)
    t0 = time.perf_counter(); [f(one) for _ in range(200)]; lat[mname] = (time.perf_counter() - t0) / 200 * 1000
print("single-image CPU latency (ms):", {k: round(v, 2) for k, v in lat.items()})
robust.round(4)
'''

DISCUSSION_MD = """
## 10. Discussion

**Results.** On 4,179 cells from 30 patients the model never saw, the deployed MobileNetV2 transfer model reaches
95.79% accuracy, precision 0.968, recall 0.948, F1 0.958, ROC-AUC 0.991 and PR-AUC 0.992 (loss 0.118). The custom CNN
baseline scores 96.12% / 0.975 / 0.948 / 0.961 / 0.992. The 0.34-point gap is not meaningful: the patient-level 95%
bootstrap interval for the difference spans roughly -1.2 to +0.7 points. Both models clear the frozen-feature heads from the
search (~93.8% validation accuracy), so fine-tuning the top MobileNetV2 blocks was worth about three points.

**Why the transfer model is the one deployed.** Under a simulated stain/illumination shift the baseline breaks
down - its accuracy drops to about 94% for a mild shift and 75% for a strong one, because it starts calling almost
every darker cell parasitized (recall up, specificity collapsing) - while the transfer model stays at about 95-96%.
The EDA showed that slide-to-slide staining variation is the largest non-biological effect in this data, and new
clinics will add more of it, so robustness to exactly this shift matters more than a third of a point on in-distribution
test cells. The price is latency (8.8 ms vs 2.2 ms per image, measured on this CPU), which the load tests show is
affordable. A small, from-scratch CNN can memorise this dataset's colour statistics; ImageNet features generalise better
to colour changes.

**Errors and generalisation.** Errors are concentrated by patient: 6 of 30 test patients have no errors, and the worst
three account for 32 of the 176 errors; one slide (c92p53) has an 18% error rate. Validation and test accuracy agree
within about one point, and training accuracy is not far above them, so the model is not overfitting cells - the
remaining variance is between patients. The confident mistakes match the stain analysis: missed parasitized cells
have almost no purple stain (median stain fraction 0.0004 vs 0.012 for correctly detected ones; early ring forms,
defocused parasites, or label noise), while false positives carry stain specks (median 0.0036 vs 0.000; platelets or
stain debris over the cell). Some of these look mislabelled on inspection, consistent with known noise in this dataset.

**Class balance and threshold.** The classes are balanced, so no reweighting was needed. The threshold is a
clinical choice: at 0.5 the model misses 110 of 2,114 infected cells (recall 0.948, specificity 0.968); at 0.3 it misses
75 (recall 0.965) for a specificity of 0.947. Because the service flags results within 0.15 of the threshold for review,
the deployment keeps 0.5 and routes uncertain cells to a microscopist; a screening programme that prefers sensitivity can
lower `SMEAR_DECISION_THRESHOLD` without retraining.

**Limitations.** One hospital, one microscope setup and one species (*P. falciparum*); pre-segmented cells; patient-level
results come from 30 test patients, so per-patient estimates are noisy; the shift experiment is synthetic. A cell-level
classifier is one component of slide-level diagnosis, which also needs cell segmentation and parasitaemia estimation.
Before any clinical use the model would need prospective evaluation on slides from the clinics that would use it - the
retraining and feedback loop in this system is built for exactly that.
"""
