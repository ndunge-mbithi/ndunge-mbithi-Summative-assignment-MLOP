#!/usr/bin/env bash
# Repeatable scaling experiment: for each replica count, (re)start the stack, then run
# Locust at each user level for a fixed duration, sampling container CPU/memory.
#
#   locust/run_experiments.sh                     # defaults: replicas "1 2 3", users "10 25 50", 60s
#   REPLICAS="1 2" USERS="20" DURATION=90s locust/run_experiments.sh
#
# Results: locust/results/r<replicas>_u<users>_{stats,stats_history,failures}.csv,
#          r<replicas>_u<users>_docker_stats.txt, then summarise with
#          python locust/summarise.py
set -euo pipefail
cd "$(dirname "$0")/.."
REPLICAS=${REPLICAS:-"1 2 3"}
USERS=${USERS:-"10 25 50"}
DURATION=${DURATION:-60s}
HOST=${HOST:-http://localhost:${LB_PORT:-8080}}
OUT=locust/results
mkdir -p "$OUT"

wait_healthy() {
  for _ in $(seq 1 90); do
    n=$(docker compose ps api --format '{{.Health}}' | grep -c '^healthy$' || true)
    [ "$n" -ge "$1" ] && curl -fs "$HOST/health" >/dev/null && return 0
    sleep 2
  done
  echo "stack did not become healthy" >&2; return 1
}

for r in $REPLICAS; do
  echo "=== $r replica(s)"
  docker compose up -d --scale api="$r" --no-recreate api
  wait_healthy "$r"
  docker compose up -d --no-deps --force-recreate lb   # nginx re-resolves the replica set (no-deps keeps the scale)
  wait_healthy "$r"
  for _ in $(seq 1 20); do curl -fs -o /dev/null -F "file=@$(ls data/test/Parasitized/*.png | head -1)" "$HOST/api/predict"; done  # warm-up
  for u in $USERS; do
    tag="r${r}_u${u}"
    echo "--- $tag ($DURATION)"
    ( while true; do docker stats --no-stream --format '{{.Name}},{{.CPUPerc}},{{.MemUsage}}' >> "$OUT/${tag}_docker_stats.txt"; sleep 3; done ) &
    sampler=$!
    locust -f locust/locustfile.py --host "$HOST" --headless -u "$u" -r "$u" -t "$DURATION" \
      --csv "$OUT/$tag" --only-summary 2>&1 | tail -n 3 | tee "$OUT/${tag}_summary.txt"
    kill $sampler 2>/dev/null || true
    sleep 5
  done
done
python locust/summarise.py
