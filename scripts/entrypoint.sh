#!/bin/sh
# Container entrypoint: seed the persistent volume with the baked model registry on
# first start (never overwriting a registry that retraining has already changed),
# then start the API. One uvicorn worker per container: scale out with containers.
set -eu
mkdir -p "$SMEAR_MODELS_DIR" "$SMEAR_STORAGE_DIR"
if [ ! -f "$SMEAR_MODELS_DIR/registry.json" ]; then
  echo "seeding model registry into $SMEAR_MODELS_DIR"
  cp -r /app/models_seed/. "$SMEAR_MODELS_DIR/"
fi
exec uvicorn api.main:app --host 0.0.0.0 --port "${PORT:-8000}" --workers "${WEB_CONCURRENCY:-1}" \
  --proxy-headers --forwarded-allow-ips='*' --timeout-keep-alive 30
