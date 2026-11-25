from src.config import get_settings
from src.retraining import gate


def m(f1, recall, acc=0.95, n=50):
    return {"f1": f1, "recall": recall, "accuracy": acc, "n": n}


def test_gate_passes_when_not_degraded():
    s = get_settings()
    d = gate(m(0.96, 0.97), m(0.958, 0.965), m(0.9, 0.9, 0.90), m(0.9, 0.9, 0.92), s)
    assert d["passed"] and len(d["checks"]) == 3


def test_gate_blocks_recall_drop():
    d = gate(m(0.96, 0.97), m(0.97, 0.94), None, None, get_settings())
    assert not d["passed"]
    assert [c["name"] for c in d["checks"] if not c["passed"]] == ["gold recall (Parasitized) not degraded"]


def test_gate_blocks_f1_drop_and_skips_tiny_holdout():
    d = gate(m(0.96, 0.97), m(0.94, 0.97), m(0.9, 0.9, 0.9, n=5), m(0.1, 0.1, 0.1, n=5), get_settings())
    assert not d["passed"] and len(d["checks"]) == 2
