"""Locust load test for the prediction endpoint.

Every simulated user cycles deterministically through the same 50 held-out test
images (25 per class), so all experiments send identical payloads. 95% of requests
are predictions; 5% are /health checks, as a monitoring system or UI would add.

  locust -f locust/locustfile.py --host http://localhost:8080 --headless -u 50 -r 50 -t 60s --csv out
"""
import itertools
import os
from pathlib import Path

from locust import HttpUser, between, events, task

ROOT = Path(__file__).resolve().parents[1]
N_PER_CLASS = int(os.environ.get("LOCUST_IMAGES_PER_CLASS", 25))
IMAGES: list[tuple[str, bytes, str]] = []
for cls in ("Parasitized", "Uninfected"):
    files = sorted((ROOT / "data" / "test" / cls).glob("*.png"))[::50][:N_PER_CLASS]
    IMAGES += [(f.name, f.read_bytes(), cls) for f in files]
assert IMAGES, "no test images found under data/test"


@events.init.add_listener
def _(environment, **_kw):
    environment.stats.correct = 0
    environment.stats.scored = 0


class ClinicUser(HttpUser):
    wait_time = between(0.5, 1.5)  # a technician submitting cells, not a tight loop

    def on_start(self):
        offset = id(self) % len(IMAGES)
        self.cycle = itertools.cycle(IMAGES[offset:] + IMAGES[:offset])

    @task(19)
    def predict(self):
        name, data, label = next(self.cycle)
        with self.client.post("/api/predict", files={"file": (name, data, "image/png")}, name="/api/predict",
                              catch_response=True) as r:
            if r.status_code != 200:
                r.failure(f"HTTP {r.status_code}")
                return
            body = r.json()
            stats = self.environment.stats
            stats.scored += 1
            stats.correct += body.get("predicted_class") == label
            r.success()

    @task(1)
    def health(self):
        self.client.get("/health", name="/health")


@events.quitting.add_listener
def _report(environment, **_kw):
    s = environment.stats
    if s.scored:
        print(f"prediction correctness under load: {s.correct}/{s.scored} = {s.correct / s.scored:.4f}")
