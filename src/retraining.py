"""Retraining worker: fine-tune the active model on newly uploaded labelled data.

Run as a separate OS process (``python -m src.retraining --job <id>``) launched by the
API, so training never blocks request handling and its memory is returned to the OS
when it finishes.

Strategy
1. Data: every uploaded sample not yet used by a model (status ``preprocessed``),
   split 80/20 stratified into new-train / new-holdout. To avoid catastrophic
   forgetting, new-train is mixed with a *replay* sample from a fixed pool of
   original training cells (``retrain_replay_ratio`` x the new data, at least
   ``retrain_replay_min``). Early stopping watches new-holdout + a held-back slice of
   the replay pool, so it reflects both new and old distributions.
2. Model: load the **currently active registry version** (initially the notebook's
   MobileNetV2 transfer model) with its weights, keep the backbone's lower blocks
   frozen, and fine-tune blocks 13-16 + head with Adam at a low learning rate
   (1e-5 default) and early stopping with best-weight restoration.
3. Gate: candidate and active model are both scored on the frozen *gold* test set
   (held-out patients, never trained on). The candidate is promoted only if
     F1(cand)     >= F1(active)     - gate_max_f1_drop
     recall(cand) >= recall(active) - gate_max_recall_drop   (missed infections matter most)
     acc_new(cand) >= acc_new(active) - 0.01                (on new-holdout, if >= 10 samples)
   Otherwise it is registered as ``rejected`` and the active model keeps serving.
4. Every outcome is versioned with full lineage (parent, sample ids, gate results).
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np

from .config import CLASS_NAMES, Settings, get_settings
from .data_store import Store, utcnow
from .drift import DRIFT_FEATURES, build_reference
from .preprocessing import load_image, to_uint8_array
from .registry import ModelRegistry

log = logging.getLogger("retraining")
LABEL_TO_INDEX = {c: i for i, c in enumerate(CLASS_NAMES)}


# ----------------------------------------------------------- reference datasets
def _load_pngs(paths: list[Path], size: int) -> np.ndarray:
    x = np.empty((len(paths), size, size, 3), np.uint8)
    for i, p in enumerate(paths):
        x[i] = to_uint8_array(load_image(p.read_bytes()), size)
    return x


def gold_test_set(s: Settings, limit: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Held-out-patient test images from data/test, cached as uint8 on the volume."""
    cache = s.storage_dir / "cache" / f"gold_test_{s.image_size}{'_' + str(limit) if limit else ''}.npz"
    if cache.exists():
        z = np.load(cache)
        return z["x"], z["y"]
    files = sorted((s.data_dir / "test").glob("*/*.png"))
    if limit:
        rng = np.random.default_rng(0)
        files = [files[i] for i in sorted(rng.choice(len(files), min(limit, len(files)), replace=False))]
    if not files:
        raise RuntimeError(f"gold test set missing under {s.data_dir / 'test'}")
    y = np.array([LABEL_TO_INDEX[f.parent.name] for f in files], np.int32)
    x = _load_pngs(files, s.image_size)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, x=x, y=y)
    return x, y


def replay_pool(s: Settings) -> tuple[np.ndarray, np.ndarray]:
    """Fixed, stratified pool of original training cells listed in data/replay_pool.csv."""
    cache = s.storage_dir / "cache" / f"replay_pool_{s.image_size}.npz"
    if cache.exists():
        z = np.load(cache)
        return z["x"], z["y"]
    with open(s.data_dir / "replay_pool.csv", newline="") as fh:
        rows = list(csv.DictReader(fh))
    x = _load_pngs([s.data_dir / r["path"] for r in rows], s.image_size)
    y = np.array([LABEL_TO_INDEX[r["label"]] for r in rows], np.int32)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, x=x, y=y)
    return x, y


def stratified_split(y: np.ndarray, frac: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    a, b = [], []
    for c in np.unique(y):
        idx = rng.permutation(np.where(y == c)[0])
        k = int(round(len(idx) * frac)) if len(idx) > 1 else 0
        b += idx[:k].tolist()
        a += idx[k:].tolist()
    return np.array(sorted(a), int), np.array(sorted(b), int)


def gate(active: dict, cand: dict, active_new: dict | None, cand_new: dict | None, s: Settings) -> dict:
    checks = [
        {"name": "gold F1 not degraded", "active": active["f1"], "candidate": cand["f1"],
         "rule": f">= active - {s.gate_max_f1_drop}", "passed": cand["f1"] >= active["f1"] - s.gate_max_f1_drop},
        {"name": "gold recall (Parasitized) not degraded", "active": active["recall"], "candidate": cand["recall"],
         "rule": f">= active - {s.gate_max_recall_drop}",
         "passed": cand["recall"] >= active["recall"] - s.gate_max_recall_drop},
    ]
    if active_new and cand_new and active_new["n"] >= 10:
        checks.append({"name": "accuracy on new-data holdout", "active": active_new["accuracy"],
                       "candidate": cand_new["accuracy"], "rule": ">= active - 0.01",
                       "passed": cand_new["accuracy"] >= active_new["accuracy"] - 0.01})
    return {"passed": all(c["passed"] for c in checks), "checks": checks}


# ------------------------------------------------------------------------- run
def run(job_id: str, s: Settings | None = None) -> dict:
    s = s or get_settings()
    store = Store(s.db_path)
    reg = ModelRegistry(s.models_dir)
    t_start = time.time()

    def stage(name: str, progress: float, msg: str) -> None:
        store.update_job(job_id, stage=name, progress=round(progress, 3))
        store.append_job_log(job_id, msg)
        log.info("[%s] %s", name, msg)

    store.update_job(job_id, status="running", started_at=utcnow(), pid=os.getpid())
    try:
        import tensorflow as tf

        from . import model as M

        params = (store.get_job(job_id) or {}).get("params") or {}
        seed = int(params.get("seed", 42))
        tf.keras.utils.set_random_seed(seed)
        rng = np.random.default_rng(seed)

        # 1. data -------------------------------------------------------------
        stage("loading_data", 0.05, "collecting uploaded samples that no model has used yet")
        samples = store.trainable_samples()
        if len(samples) < s.retrain_min_new_samples:
            raise ValueError(f"need at least {s.retrain_min_new_samples} new labelled samples, have {len(samples)}")
        x_new = np.stack([np.load(s.storage_dir / r["array_path"]) for r in samples])
        y_new = np.array([LABEL_TO_INDEX[r["label"]] for r in samples], np.int32)
        tr_i, ho_i = stratified_split(y_new, 0.2, rng)
        counts = {c: int((y_new == i).sum()) for i, c in enumerate(CLASS_NAMES)}
        stage("loading_data", 0.10, f"{len(samples)} new samples {counts}: {len(tr_i)} train / {len(ho_i)} holdout")

        x_pool, y_pool = replay_pool(s)
        pool_tr, pool_val = stratified_split(y_pool, 0.2, np.random.default_rng(0))  # fixed held-back slice
        n_replay = int(min(len(pool_tr), max(s.retrain_replay_min, s.retrain_replay_ratio * len(tr_i))))
        rep_i = rng.choice(pool_tr, size=n_replay, replace=False)
        x_train = np.concatenate([x_new[tr_i], x_pool[rep_i]])
        y_train = np.concatenate([y_new[tr_i], y_pool[rep_i]])
        val_take = rng.choice(pool_val, size=min(len(pool_val), 300), replace=False)
        x_val = np.concatenate([x_new[ho_i], x_pool[val_take]])
        y_val = np.concatenate([y_new[ho_i], y_pool[val_take]])
        stage("preprocessing", 0.15, f"training set: {len(tr_i)} new + {n_replay} replayed original cells; "
                                     f"validation: {len(ho_i)} new + {len(val_take)} original")

        # 2. model ------------------------------------------------------------
        parent = reg.active_version()
        if parent is None:
            raise RuntimeError("no active model to fine-tune")
        store.update_job(job_id, parent_version=parent)
        stage("loading_model", 0.20, f"loading active model {parent} as the starting point")
        model = M.load_keras(reg.model_path(parent))
        n_trainable = M.unfreeze_top(model, from_block=13)
        lr = float(params.get("learning_rate", s.retrain_learning_rate))
        epochs = int(params.get("epochs", s.retrain_max_epochs))
        M.compile_model(model, lr, "adam")

        # 3. train ------------------------------------------------------------
        stage("training", 0.25, f"fine-tuning {n_trainable} backbone layers + head, lr={lr}, up to {epochs} epochs")

        def on_epoch(epoch: int, logs: dict) -> None:
            entry = {"epoch": epoch + 1, **{k: round(v, 5) for k, v in logs.items()}}
            store.append_job_log(job_id, "", key="epochs_json", entry=entry)
            store.update_job(job_id, progress=round(0.25 + 0.45 * (epoch + 1) / epochs, 3))

        train_ds = M.make_dataset(x_train, y_train, s.retrain_batch_size, augment=True, shuffle=True, seed=seed)
        val_ds = M.make_dataset(x_val, y_val, 128)
        fit = M.fit(model, train_ds, val_ds, epochs=epochs, patience=s.retrain_patience,
                    extra_callbacks=[M.ProgressCallback(on_epoch)], verbose=0)
        stage("training", 0.70, f"stopped after {fit.epochs_run} epochs in {fit.seconds:.0f}s (best weights restored)")

        # 4. evaluate + gate ----------------------------------------------------
        stage("evaluating", 0.72, "scoring candidate and active model on the gold test set (held-out patients)")
        x_gold, y_gold = gold_test_set(s, s.gate_eval_limit)
        active_model = M.load_keras(reg.model_path(parent))
        cache = s.storage_dir / "cache" / f"gold_probs_{parent}_{len(y_gold)}.npy"
        if cache.exists():
            p_act = np.load(cache)
        else:
            p_act = M.predict_proba(active_model, x_gold)
            np.save(cache, p_act)
        p_cand = M.predict_proba(model, x_gold)
        m_act, m_cand = M.evaluate_probs(y_gold, p_act), M.evaluate_probs(y_gold, p_cand)
        m_act_new = m_cand_new = None
        if len(ho_i):
            m_act_new = M.evaluate_probs(y_new[ho_i], M.predict_proba(active_model, x_new[ho_i]))
            m_cand_new = M.evaluate_probs(y_new[ho_i], M.predict_proba(model, x_new[ho_i]))
        decision = gate(m_act, m_cand, m_act_new, m_cand_new, s)
        stage("evaluating", 0.85, f"gold F1 {m_act['f1']:.4f} -> {m_cand['f1']:.4f}, recall {m_act['recall']:.4f} -> "
                                  f"{m_cand['recall']:.4f}; gate {'PASSED' if decision['passed'] else 'FAILED'}")

        # 5. register / promote -------------------------------------------------
        stage("registering", 0.90, "saving versioned artifact and metadata")
        parent_meta = reg.metadata(parent)
        new_stats = {f: [json.loads(r["stats_json"])[f] for r in samples] for f in DRIFT_FEATURES}
        ref = parent_meta.get("drift_reference") or {}
        if len(samples) >= 200:  # widen the reference once the new data is a meaningful sample
            pool = {f: ref[f]["sample"] + new_stats[f] for f in DRIFT_FEATURES if f in ref}
            ref = build_reference(pool) if pool else ref
        meta = {
            **{k: parent_meta[k] for k in ("architecture", "image_size", "class_names", "positive_class", "preprocessing")
               if k in parent_meta},
            "framework": f"tensorflow {tf.__version__}",
            "parent_version": parent,
            "trigger": f"retraining job {job_id}",
            "hyperparameters": {"learning_rate": lr, "max_epochs": epochs, "epochs_run": fit.epochs_run,
                                "batch_size": s.retrain_batch_size, "finetune_from_block": 13,
                                "replay_samples": n_replay, "patience": s.retrain_patience, "seed": seed},
            "training_data": {"new_samples": len(samples), "new_train": int(len(tr_i)), "new_holdout": int(len(ho_i)),
                              "new_by_class": counts, "replay_samples": n_replay,
                              "sample_ids": [r["id"] for r in samples],
                              "upload_batches": sorted({r["batch_id"] for r in samples})},
            "metrics": {"test": m_cand, "new_holdout": m_cand_new},
            "parent_metrics_same_eval": {"test": m_act, "new_holdout": m_act_new},
            "gate": decision,
            "history": {"retrain": fit.history},
            "drift_reference": ref,
            "summary": {"test_accuracy": m_cand["accuracy"], "test_f1": m_cand["f1"], "test_recall": m_cand["recall"],
                        "test_roc_auc": m_cand["roc_auc"], "gate_passed": decision["passed"]},
        }
        version = reg.register(model, meta, status="candidate")
        store.update_job(job_id, candidate_version=version)
        if decision["passed"]:
            reg.promote(version, reason=f"passed acceptance gate (job {job_id})")
            store.mark_samples_used([r["id"] for r in samples], version)
            status, msg = "succeeded", f"{version} promoted to active; {parent} archived (rollback available)"
        else:
            reg.set_status(version, "rejected", reason="failed acceptance gate")
            status, msg = "rejected", f"{version} rejected by the acceptance gate; {parent} keeps serving"
        stage("done", 1.0, msg)
        result = {"version": version, "parent": parent, "promoted": decision["passed"], "gate": decision,
                  "metrics": {"candidate": m_cand, "active": m_act, "candidate_new": m_cand_new, "active_new": m_act_new},
                  "seconds": round(time.time() - t_start, 1)}
        store.update_job(job_id, status=status, finished_at=utcnow(), result_json=result)
        return result
    except Exception as exc:  # recorded for the UI, then re-raised for the exit code
        store.append_job_log(job_id, f"failed: {exc}")
        store.update_job(job_id, status="failed", stage="failed", error=f"{exc.__class__.__name__}: {exc}",
                         finished_at=utcnow(), result_json={"traceback": traceback.format_exc()[-4000:]})
        raise


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", required=True)
    args = ap.parse_args()
    logging.basicConfig(level=os.environ.get("SMEAR_LOG_LEVEL", "INFO"),
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")
    try:
        run(args.job)
    except Exception:
        sys.exit(1)


if __name__ == "__main__":
    main()
