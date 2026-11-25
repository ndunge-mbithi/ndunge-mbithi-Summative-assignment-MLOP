import json

import numpy as np
import pytest

from src.config import get_settings
from src.model import build_baseline_cnn, evaluate_probs
from src.prediction import Predictor
from src.registry import ModelRegistry, RegistryError


def test_active_model_loads_and_has_metadata():
    s = get_settings()
    p = Predictor(s)
    assert p.refresh(force=True)
    assert p.ready and p.version == ModelRegistry(s.models_dir).active_version()
    for key in ("metrics", "drift_reference", "hyperparameters", "training_data", "architecture"):
        assert key in p.metadata
    assert p.metadata["metrics"]["test"]["accuracy"] > 0.9


def test_predictions_match_known_labels(test_images):
    """The served model must classify known held-out-patient images correctly."""
    p = Predictor(get_settings())
    results = [(p.predict_bytes(path.read_bytes()), label) for path, label in test_images]
    correct = sum(r["predicted_class"] == label for r, label in results)
    assert correct / len(results) >= 0.9, f"only {correct}/{len(results)} correct"
    r = results[0][0]
    assert set(r["probabilities"]) == {"Parasitized", "Uninfected"}
    assert abs(sum(r["probabilities"].values()) - 1) < 1e-5
    assert r["confidence"] == max(r["probabilities"].values())


def test_registry_versioning_promotion_and_rollback(tmp_path):
    reg = ModelRegistry(tmp_path)
    m = build_baseline_cnn(32)
    v1 = reg.register(m, {"summary": {"test_f1": 0.5}})
    reg.promote(v1)
    v2 = reg.register(m, {"summary": {"test_f1": 0.6}, "parent_version": v1})
    assert (v1, v2) == ("v1", "v2")
    assert json.loads((tmp_path / v2 / "metadata.json").read_text())["parent_version"] == v1
    reg.promote(v2)
    st = reg.state()
    assert st["active"] == "v2" and st["versions"]["v1"]["status"] == "archived"
    assert reg.rollback() == "v1"
    assert reg.state()["active"] == "v1" and reg.state()["versions"]["v2"]["status"] == "archived"
    with pytest.raises(RegistryError):
        reg.register(m, {}, version="v1")  # versions are immutable
    with pytest.raises(RegistryError):
        reg.delete_version("v1")  # active version is protected


def test_evaluate_probs_metrics():
    y = np.array([0, 0, 1, 1, 1])
    p = np.array([0.1, 0.6, 0.8, 0.4, 0.9])
    m = evaluate_probs(y, p)
    assert m["confusion_matrix"]["matrix"] == [[1, 1], [1, 2]]
    assert m["precision"] == pytest.approx(2 / 3) and m["recall"] == pytest.approx(2 / 3)
    assert m["specificity"] == pytest.approx(0.5) and 0 <= m["roc_auc"] <= 1
