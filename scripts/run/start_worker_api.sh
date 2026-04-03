#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 <experiment_config> <worker_id>"
  exit 1
fi

CONFIG_PATH="$1"
WORKER_ID="$2"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

export PYTHONPATH="$REPO_ROOT"

echo "[start_worker_api] repo_root=$REPO_ROOT"
echo "[start_worker_api] config=$CONFIG_PATH"
echo "[start_worker_api] worker_id=$WORKER_ID"

python -m src.distributed.api.app \
  --config "$CONFIG_PATH" \
  --worker-id "$WORKER_ID"