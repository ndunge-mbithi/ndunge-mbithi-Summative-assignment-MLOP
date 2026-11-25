"""Non-interactive training pipeline: data -> train -> evaluate -> register.

Mirrors the notebook's final configuration so a model can be rebuilt in CI or on a
training machine without Jupyter:

    python -m src.train                      # full run, registers + promotes a new version
    python -m src.train --quick              # smoke run on a subset (minutes on CPU)
    python -m src.train --no-promote         # register as candidate only

Assumes ``scripts/acquire_data.py`` has produced data/manifest.csv and the splits.
"""
from __future__ import annotations

import argparse
import json
import logging

import numpy as np
import pandas as pd
import tensorflow as tf

from . import model as M
from .config import CLASS_NAMES, get_settings
from .drift import DRIFT_FEATURES, build_reference
from .registry import ModelRegistry

log = logging.getLogger("train")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="small subset, 1+1 epochs (smoke test)")
    ap.add_argument("--no-promote", action="store_true")
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--units", type=int, default=64)
    ap.add_argument("--head-lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    logging.basicConfig(level="INFO", format="%(asctime)s %(levelname)s %(message)s")
    s = get_settings()
    tf.keras.utils.set_random_seed(args.seed)

    limit = 600 if args.quick else 0
    x_tr, y_tr, _ = M.load_split_arrays(s.data_dir, "train", s.image_size, limit=limit)
    x_va, y_va, _ = M.load_split_arrays(s.data_dir, "val", s.image_size, limit=limit // 2)
    x_te, y_te, _ = M.load_split_arrays(s.data_dir, "test", s.image_size, limit=limit // 2)
    log.info("data: train %d val %d test %d", len(y_tr), len(y_va), len(y_te))

    train_ds = M.make_dataset(x_tr, y_tr, 64, augment=True, shuffle=True, seed=args.seed)
    val_ds = M.make_dataset(x_va, y_va, 128)
    model = M.build_transfer_model(s.image_size, s.pretrained_weights, dropout=args.dropout, dense_units=args.units)
    M.compile_model(model, args.head_lr)
    e1, e2 = (1, 1) if args.quick else (6, 6)
    f1 = M.fit(model, train_ds, val_ds, epochs=e1, patience=2)
    M.unfreeze_top(model, 13)
    steps = int(np.ceil(len(x_tr) / 64)) * e2
    M.compile_model(model, tf.keras.optimizers.schedules.CosineDecay(1e-4, steps, alpha=0.05))
    f2 = M.fit(model, train_ds, val_ds, epochs=e2, patience=3)

    m_test = M.evaluate_probs(y_te, M.predict_proba(model, x_te))
    m_val = M.evaluate_probs(y_va, M.predict_proba(model, x_va))
    log.info("test: %s", json.dumps({k: round(m_test[k], 4) for k in ("accuracy", "precision", "recall", "f1", "roc_auc")}))

    manifest = pd.read_csv(s.data_dir / "manifest.csv")
    train_rows = manifest[manifest.split == "train"]
    reg = ModelRegistry(s.models_dir)
    version = reg.register(model, {
        "architecture": "MobileNetV2 (ImageNet, alpha=1.0, 128px) + GAP/Dropout/Dense head",
        "framework": f"tensorflow {tf.__version__}", "image_size": s.image_size, "class_names": list(CLASS_NAMES),
        "positive_class": "Parasitized", "preprocessing": "RGB, aspect-preserving resize to 128 + black pad, scale to [-1,1]",
        "hyperparameters": {"head": {"dropout": args.dropout, "units": args.units, "lr": args.head_lr, "optimizer": "adam"},
                            "stage1_epochs": f1.epochs_run, "stage2_epochs": f2.epochs_run, "finetune_from_block": 13,
                            "quick": args.quick},
        "training_data": {"source": "NIH/NLM malaria cell images", "n_train": int(len(y_tr)), "n_val": int(len(y_va)),
                          "n_test": int(len(y_te)), "uploaded_samples": 0},
        "metrics": {"test": m_test, "val": m_val},
        "history": {"stage1": f1.history, "stage2": f2.history},
        "drift_reference": build_reference({f: train_rows[f].tolist() for f in DRIFT_FEATURES}),
        "parent_version": None, "trigger": "src.train CLI" + (" --quick" if args.quick else ""),
        "summary": {"test_accuracy": m_test["accuracy"], "test_f1": m_test["f1"], "test_recall": m_test["recall"],
                    "test_roc_auc": m_test["roc_auc"]},
    })
    if not args.no_promote:
        reg.promote(version, reason="src.train CLI")
    log.info("registered %s (%s)", version, "promoted" if not args.no_promote else "candidate")


if __name__ == "__main__":
    main()
