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
NETWORK=jimssak
DB_NAME=jimssak-db
DB_ENV_FILE="$HOME/jimssak-db.env"
HEALTH_URL=http://127.0.0.1:8000/health

if [ ! -f "$ENV_FILE" ]; then
  echo "Missing $ENV_FILE (copy backend/.env to the server first)" >&2
  exit 1
fi

gunzip -c "$IMAGE_TAR" | docker load
rm -f "$IMAGE_TAR"

# The container runs as uid 10001; uploaded videos live here.
mkdir -p "$DATA_DIR"
sudo chown 10001:10001 "$DATA_DIR"

# PostgreSQL runs in its own container, started on the first deploy and left running after
# that; its data is in the jimssak-pgdata volume. The password is made once and stays in
# DB_ENV_FILE on the server. Putting DATABASE_URL in ENV_FILE uses another database instead.
docker network inspect "$NETWORK" >/dev/null 2>&1 || docker network create "$NETWORK" >/dev/null
if [ ! -f "$DB_ENV_FILE" ]; then
  (umask 077 && printf 'POSTGRES_USER=jimssak\nPOSTGRES_DB=jimssak\nPOSTGRES_PASSWORD=%s\n' \
    "$(openssl rand -hex 24)" > "$DB_ENV_FILE")
fi
if ! docker container inspect "$DB_NAME" >/dev/null 2>&1; then
  docker run -d --name "$DB_NAME" \
    --restart unless-stopped \
    --network "$NETWORK" \
    --env-file "$DB_ENV_FILE" \
    -v jimssak-pgdata:/var/lib/postgresql/data \
    --log-opt max-size=10m --log-opt max-file=3 \
    postgres:16-alpine >/dev/null
fi
docker start "$DB_NAME" >/dev/null
for _ in $(seq 1 30); do
  docker exec "$DB_NAME" pg_isready -U jimssak -d jimssak >/dev/null 2>&1 && break
  sleep 2
done

DB_ARGS=()
if ! grep -q '^DATABASE_URL=' "$ENV_FILE"; then
  DB_PASSWORD=$(grep '^POSTGRES_PASSWORD=' "$DB_ENV_FILE" | cut -d= -f2-)
  DB_ARGS=(-e "DATABASE_URL=postgresql://jimssak:$DB_PASSWORD@$DB_NAME:5432/jimssak")
fi

docker rm -f "$NAME" >/dev/null 2>&1 || true
docker run -d --name "$NAME" \
  --restart unless-stopped \
  --network "$NETWORK" \
  --env-file "$ENV_FILE" \
  "${DB_ARGS[@]}" \
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
