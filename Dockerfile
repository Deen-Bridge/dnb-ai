# ----------  Deen Bridge AI – Dockerfile  ----------
# Pinned slim base, non-root, layer-cached deps, health-check.
#
# Build:
#   docker build -t deenbridge-ai .
#
# Run (pass your Gemini key):
#   docker run --env-file .env -p 8000:8000 deenbridge-ai
#   docker run -e GEMINI_API_KEY=... -p 8000:8000 deenbridge-ai
#
# Render: to switch from Python buildpack to Docker, change render.yaml:
#   env: python  →  runtime: docker
# and update buildCommand/startCommand to use this image.
# -----------------------------------------------

FROM python:3.12-slim AS base

# Prevent .pyc files and force unbuffered stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install OS-level build deps (gcc needed by some wheels), then remove
# after pip install to keep the final layer small.
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc && \
    rm -rf /var/lib/apt/lists/*

# --- Dependency layer (cached unless requirements.txt changes) ---
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Remove build tools — nothing else compiles at runtime
RUN apt-get purge -y --auto-remove gcc

# --- Application layer ---
COPY . .

# Create a non-root user
RUN addgroup --system app && adduser --system --ingroup app app && \
    chown -R app:app /app

# Runtime data directories that the app writes to
RUN mkdir -p /app/data/review && chown -R app:app /app/data

USER app

EXPOSE 8000

# Render injects PORT; respect it with a default of 8000
ENV PORT=8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:${PORT}/ping')" || exit 1

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT}"]
