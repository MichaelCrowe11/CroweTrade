#!/usr/bin/env bash
set -euo pipefail

log() { printf "[%s] %s\n" "$(date +%H:%M:%S)" "$*"; }

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

deploy_fly() {
  log "Checking flyctl..."
  if ! command -v flyctl >/dev/null 2>&1; then
    log "Installing flyctl..."
    curl -fsSL https://fly.io/install.sh | sh -s -- -b /usr/local/bin
  fi
  log "flyctl: $(flyctl version || echo not-installed)"

  if ! command -v jq >/dev/null 2>&1; then
    log "Installing jq..."
    if command -v apt-get >/dev/null 2>&1; then
      sudo apt-get update -y >/dev/null 2>&1 || true
      sudo apt-get install -y jq >/dev/null 2>&1 || true
    else
      log "jq not found and automatic install unsupported on this host"; return 2
    fi
  fi

  if [ -n "${FLYIO_TOKEN:-}" ]; then
    export FLYCTL_ACCESS_TOKEN="$FLYIO_TOKEN"
    export FLY_API_TOKEN="$FLYIO_TOKEN"
    log "Using FLYIO_TOKEN for Fly auth"
  elif [ -n "${FLY_API_TOKEN:-}" ]; then
    export FLYCTL_ACCESS_TOKEN="$FLY_API_TOKEN"
    log "Using FLY_API_TOKEN for Fly auth"
  else
    log "No Fly token in env; export FLYIO_TOKEN to enable Fly deploy"
    return 2
  fi

  local APP_EXEC="crowetrade-execution"
  local APP_PORT="crowetrade-portfolio"
  local ORG_OPT=""
  if [ -n "${FLY_ORG:-}" ]; then ORG_OPT="--org $FLY_ORG"; fi

  # Ensure apps exist
  for APP in "$APP_EXEC" "$APP_PORT"; do
    if flyctl apps list --json | jq -e --arg a "$APP" '.[] | select(.Name==$a)' >/dev/null 2>&1; then
      log "App $APP exists"
    else
      log "Creating app $APP ..."
      flyctl apps create "$APP" $ORG_OPT || true
    fi
  done

  # Deploy both apps
  log "Deploying execution app ..."
  flyctl deploy -c fly.execution.toml --remote-only --now
  log "Deploying portfolio app ..."
  flyctl deploy -c fly.portfolio.toml --remote-only --now

  # Health checks
  for APP in "$APP_EXEC" "$APP_PORT"; do
    local HOST
    HOST=$(flyctl status -a "$APP" --json | jq -r '.Hostname // empty')
    if [ -z "$HOST" ] || [ "$HOST" = "null" ]; then
      log "Could not resolve hostname for $APP"; return 4
    fi
    log "Health: https://$HOST/health"
    curl -fsS --retry 10 --retry-delay 2 "https://$HOST/health" >/dev/null
    log "$APP healthy"
  done

  log "Fly.io deployment successful."
}

fallback_docker_compose() {
  if ! command -v docker >/dev/null 2>&1; then
    log "Docker not installed; cannot run local fallback"
    return 3
  fi
  if ! command -v docker compose >/dev/null 2>&1 && ! docker compose version >/dev/null 2>&1; then
    log "Docker Compose v2 not available; trying 'docker-compose'"
    if ! command -v docker-compose >/dev/null 2>&1; then
      log "docker-compose not installed"; return 3
    fi
    DOCKER_COMPOSE_CMD="docker-compose"
  else
    DOCKER_COMPOSE_CMD="docker compose"
  fi

  log "Building and starting local stack via Compose ..."
  $DOCKER_COMPOSE_CMD -f docker-compose.yml up -d --build
  log "Waiting for local health endpoints ..."
  curl -fsS --retry 20 --retry-delay 1 http://localhost:18080/health >/dev/null
  curl -fsS --retry 20 --retry-delay 1 http://localhost:18081/health >/dev/null
  log "Local Docker stack healthy at http://localhost:18080 and :18081"
}

case "${1:-}" in
  fly)
    deploy_fly
    ;;
  local|docker)
    fallback_docker_compose
    ;;
  *)
    log "Attempting Fly deploy first, then local fallback..."
    if deploy_fly; then exit 0; fi
    log "Falling back to local Docker..."
    fallback_docker_compose
    ;;
esac
