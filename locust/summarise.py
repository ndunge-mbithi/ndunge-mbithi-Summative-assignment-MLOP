"""Aggregate Locust CSVs + docker stats samples into a table and charts."""
import glob
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

OUT = Path(__file__).resolve().parent / "results"
rows = []
for f in sorted(glob.glob(str(OUT / "r*_u*_stats.csv"))):
    m = re.search(r"r(\d+)_u(\d+)_stats\.csv$", f)
    r, u = int(m.group(1)), int(m.group(2))
    df = pd.read_csv(f)
    p = df[df.Name == "/api/predict"].iloc[0]
    agg = df[df.Name == "Aggregated"].iloc[0]
    cpu, mem = [], []
    stats_file = OUT / f"r{r}_u{u}_docker_stats.txt"
    if stats_file.exists():
        snap = pd.read_csv(stats_file, names=["name", "cpu", "mem"])
        api = snap[snap.name.str.contains("-api-")]
        cpu = api.cpu.str.rstrip("%").astype(float)
        mem = api["mem"].str.extract(r"([\d.]+)MiB")[0].astype(float)
    rows.append({
        "replicas": r, "users": u, "requests": int(agg["Request Count"]), "failures": int(agg["Failure Count"]),
        "failure_rate": round(agg["Failure Count"] / max(agg["Request Count"], 1), 4),
        "throughput_rps": round(agg["Requests/s"], 2), "predict_avg_ms": round(p["Average Response Time"], 1),
        "predict_p50_ms": p["50%"], "predict_p95_ms": p["95%"], "predict_p99_ms": p["99%"], "predict_max_ms": round(p["Max Response Time"]),
        "api_cpu_pct_mean_per_container": round(float(pd.Series(cpu).mean()), 1) if len(cpu) else None,
        "api_mem_mib_mean_per_container": round(float(pd.Series(mem).mean()), 0) if len(mem) else None,
    })
res = pd.DataFrame(rows).sort_values(["replicas", "users"])
res.to_csv(OUT / "summary.csv", index=False)
print(res.to_string(index=False))

fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))
for r, g in res.groupby("replicas"):
    axes[0].plot(g.users, g.throughput_rps, "o-", label=f"{r} container(s)")
    axes[1].plot(g.users, g.predict_p50_ms, "o--", alpha=.6, color=f"C{r - 1}")
    axes[1].plot(g.users, g.predict_p95_ms, "o-", label=f"{r} container(s) p95", color=f"C{r - 1}")
    axes[2].plot(g.users, 100 * g.failure_rate, "o-", label=f"{r} container(s)")
axes[0].set(title="Throughput", xlabel="concurrent users", ylabel="requests / s")
axes[1].set(title="Predict latency (dashed p50, solid p95)", xlabel="concurrent users", ylabel="ms")
axes[2].set(title="Failure rate", xlabel="concurrent users", ylabel="% of requests", ylim=(0, max(1.0, 100 * res.failure_rate.max() * 1.2)))
for a in axes:
    a.grid(alpha=.3)
    a.legend(fontsize=8)
plt.tight_layout()
plt.savefig(OUT / "scaling.png", dpi=120)
print("wrote", OUT / "summary.csv", "and", OUT / "scaling.png")
