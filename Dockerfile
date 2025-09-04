# Multi-stage to keep the final image smaller(ish)
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# System deps: ffmpeg for audio, build tools for some wheels, libsndfile for TTS
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg git build-essential libsndfile1 wget ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirement spec first to leverage docker layer caching
COPY requirements.txt ./requirements.txt
RUN pip install -r requirements.txt

# Install Argos models offline
COPY models/ /models/
RUN python - <<'PY'
import glob
import argostranslate.package as pkg
paths = glob.glob("/models/*.argosmodel")
assert paths, "No .argosmodel files found in /models"
for p in paths:
    print("Installing Argos package:", p)
    pkg.install_from_path(p)
print("Argos packages installed.")
PY


# --- Prefetch models at build time so first run is fast/offline-ish ---
# 1) Argos Translate: install French->Spanish package
# 2) Coqui TTS: pre-download a Spanish voice model
COPY scripts/prefetch_models.py ./scripts/prefetch_models.py
RUN python ./scripts/prefetch_models.py

# App code
COPY pipeline.py ./pipeline.py
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# By default, show help
ENTRYPOINT ["/entrypoint.sh"]