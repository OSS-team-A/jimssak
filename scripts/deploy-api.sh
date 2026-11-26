#!/usr/bin/env bash
# Runs on the EC2 server. Loads a gzipped API image and replaces the running container.
# The deploy workflow copies this script and the image to the server, then runs:
#   bash ~/deploy-api.sh ~/jimssak-api.tar.gz
# Settings and API keys stay on the server in ENV_FILE; they are never sent by CI.
set -euo pipefail

IMAGE_TAR="${1:-$HOME/jimssak-api.tar.gz}"
NAME=jimssak-api
IMAGE=jimssak-api:latest
ENV_FILE="$HOME/jimssak.env"
DATA_DIR="$HOME/jimssak-data"
HEALTH_URL=http://127.0.0.1:8000/health

if [ ! -f "$ENV_FILE" ]; then
  echo "Missing $ENV_FILE (copy backend/.env to the server first)" >&2
  exit 1
fi

gunzip -c "$IMAGE_TAR" | docker load
rm -f "$IMAGE_TAR"

# The container runs as uid 10001; the SQLite database and videos live here.
mkdir -p "$DATA_DIR"
sudo chown 10001:10001 "$DATA_DIR"

docker rm -f "$NAME" >/dev/null 2>&1 || true
docker run -d --name "$NAME" \
  --restart unless-stopped \
  --env-file "$ENV_FILE" \
  -v "$DATA_DIR:/data" \
  -p 127.0.0.1:8000:8000 \
  --log-opt max-size=10m --log-opt max-file=3 \
  "$IMAGE"

for _ in $(seq 1 30); do
  if curl -fsS "$HEALTH_URL" >/dev/null 2>&1; then
    echo "API is healthy"
    docker image prune -f >/dev/null
    exit 0
  fi
  sleep 2
done

echo "API did not become healthy" >&2
docker logs --tail 50 "$NAME" >&2
exit 1
