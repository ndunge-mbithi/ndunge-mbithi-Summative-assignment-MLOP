# SmearScan API + web UI + retraining worker (one image; the worker runs as a subprocess).
#
#   docker build -t smearscan:latest .
#   docker run -p 8000:8000 -v smearscan-data:/data smearscan:latest
#
# BASE_IMAGE is overridable for registries other than Docker Hub; it must provide
# Python 3.12 with venv.
ARG BASE_IMAGE=python:3.12-slim

# ---- stage 1: dependencies --------------------------------------------------
FROM ${BASE_IMAGE} AS deps
RUN python3 -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
COPY requirements.txt /tmp/requirements.txt
# Optional build secret "ca": a CA bundle for builds behind a TLS-intercepting proxy.
RUN --mount=type=secret,id=ca,required=false \
    if [ -f /run/secrets/ca ]; then export PIP_CERT=/run/secrets/ca; fi; \
    pip install --upgrade pip && pip install -r /tmp/requirements.txt

# ---- stage 2: data that ships with the service ------------------------------
# Only the gold test set and the replay pool are needed at runtime; the build context
# may contain the full training set, which is pruned here so it never reaches the image.
FROM ${BASE_IMAGE} AS data
WORKDIR /src
COPY data/ ./data/
COPY scripts/prune_build_data.py ./
RUN python3 prune_build_data.py data

# ---- stage 3: runtime --------------------------------------------------------
FROM ${BASE_IMAGE}
ENV PATH=/opt/venv/bin:$PATH PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 TF_CPP_MIN_LOG_LEVEL=2 \
    SMEAR_DATA_DIR=/app/data SMEAR_MODELS_DIR=/data/models SMEAR_STORAGE_DIR=/data/storage \
    PORT=8000 WEB_CONCURRENCY=1
RUN (getent passwd 1000 >/dev/null || useradd --create-home --uid 1000 app) && mkdir -p /data && chown 1000 /data
WORKDIR /app
COPY --from=deps /opt/venv /opt/venv
COPY --from=data /src/data ./data
COPY src ./src
COPY api ./api
COPY frontend ./frontend
COPY scripts/entrypoint.sh ./scripts/entrypoint.sh
# The registry baked into the image seeds the persistent volume on first start.
COPY --chown=1000:1000 models/registry.json ./models_seed/registry.json
COPY --chown=1000:1000 models/v1 ./models_seed/v1
USER 1000
EXPOSE 8000
VOLUME ["/data"]
HEALTHCHECK --interval=20s --timeout=5s --start-period=60s --retries=3 \
  CMD python -c "import os,urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ.get(\"PORT\",\"8000\")}/health', timeout=4)" || exit 1
ENTRYPOINT ["sh", "scripts/entrypoint.sh"]
