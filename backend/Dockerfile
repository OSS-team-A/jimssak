# API server image. The backend does not use ai/; the AI analyzer is a separate container
# that talks to this API over HTTP. Build from the repository root, where pyproject.toml is:
#   docker build -f backend/Dockerfile -t jimssak-api .
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

COPY pyproject.toml ./
COPY backend ./backend
RUN pip install -e . \
    && useradd --system --uid 10001 --home-dir /app app \
    && mkdir -p /data/media \
    && chown -R app:app /data

# SQLite database and uploaded videos live on the /data volume so they survive redeploys.
ENV AUTH_DB_PATH=/data/auth.sqlite3 \
    MEDIA_DIR=/data/media
VOLUME ["/data"]

USER app
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"

# Run a single process: the SQLite database is not meant for several writers.
CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
