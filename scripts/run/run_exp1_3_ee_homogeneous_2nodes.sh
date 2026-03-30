#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

export PYTHONPATH="$PROJECT_ROOT"

CONFIG_PATH="${PROJECT_ROOT}/configs/experiments/exp1_3_ee_homogeneous_2nodes.yaml"

# -------------------------------------------------------------------
# Remote execution settings
# -------------------------------------------------------------------
# Adjust these if needed, or override them as environment variables
# before running the script.
#
# Example:
#   SSH_USER=pi \
#   WORKER1_HOST=192.168.1.101 \
#   WORKER2_HOST=192.168.1.102 \
#   REMOTE_PROJECT_ROOT=/home/pi/thesis-project \
#   ./scripts/run/run_exp1_3_ee_homogeneous_2nodes.sh
# -------------------------------------------------------------------

SSH_USER="${SSH_USER:-$USER}"
WORKER1_HOST="${WORKER1_HOST:-192.168.1.101}"
WORKER2_HOST="${WORKER2_HOST:-192.168.1.102}"

REMOTE_PROJECT_ROOT="${REMOTE_PROJECT_ROOT:-$PROJECT_ROOT}"
REMOTE_PYTHON="${REMOTE_PYTHON:-python3}"

WORKER1_ID="worker1"
WORKER2_ID="worker2"

WORKER1_LOG="${REMOTE_PROJECT_ROOT}/logs/${WORKER1_ID}.log"
WORKER2_LOG="${REMOTE_PROJECT_ROOT}/logs/${WORKER2_ID}.log"

STARTUP_WAIT_SEC="${STARTUP_WAIT_SEC:-3}"

# Pattern used to stop exactly the worker service for this experiment config
WORKER1_PATTERN="src.distributed.worker_server --config ${CONFIG_PATH} --worker-id ${WORKER1_ID}"
WORKER2_PATTERN="src.distributed.worker_server --config ${CONFIG_PATH} --worker-id ${WORKER2_ID}"

remote_exec() {
  local host="$1"
  shift
  ssh -o BatchMode=yes "${SSH_USER}@${host}" "$@"
}

start_remote_worker() {
  local host="$1"
  local worker_id="$2"
  local log_path="$3"

  echo "[run_exp1_3] Starting ${worker_id} on ${SSH_USER}@${host}"

  remote_exec "$host" "
    set -euo pipefail
    mkdir -p '${REMOTE_PROJECT_ROOT}/logs'
    cd '${REMOTE_PROJECT_ROOT}'
    export PYTHONPATH='${REMOTE_PROJECT_ROOT}'

    # Stop any previous instance of the same worker command
    pkill -f \"src.distributed.worker_server --config ${CONFIG_PATH} --worker-id ${worker_id}\" || true

    nohup ${REMOTE_PYTHON} -m src.distributed.worker_server \
      --config '${CONFIG_PATH}' \
      --worker-id '${worker_id}' \
      > '${log_path}' 2>&1 < /dev/null &
  "
}

stop_remote_worker() {
  local host="$1"
  local worker_id="$2"

  echo "[run_exp1_3] Stopping ${worker_id} on ${SSH_USER}@${host}"
  remote_exec "$host" "
    set -euo pipefail
    pkill -f \"src.distributed.worker_server --config ${CONFIG_PATH} --worker-id ${worker_id}\" || true
  " || true
}

cleanup() {
  echo "[run_exp1_3] Cleaning up remote worker services..."
  stop_remote_worker "$WORKER1_HOST" "$WORKER1_ID"
  stop_remote_worker "$WORKER2_HOST" "$WORKER2_ID"
}

trap cleanup EXIT

echo "[run_exp1_3] Project root: $PROJECT_ROOT"
echo "[run_exp1_3] Config: $CONFIG_PATH"
echo "[run_exp1_3] Worker1: ${SSH_USER}@${WORKER1_HOST}"
echo "[run_exp1_3] Worker2: ${SSH_USER}@${WORKER2_HOST}"

start_remote_worker "$WORKER1_HOST" "$WORKER1_ID" "$WORKER1_LOG"
start_remote_worker "$WORKER2_HOST" "$WORKER2_ID" "$WORKER2_LOG"

echo "[run_exp1_3] Waiting ${STARTUP_WAIT_SEC}s for worker servers to start..."
sleep "$STARTUP_WAIT_SEC"

echo "[run_exp1_3] Running distributed master client from master node..."
python -m src.distributed.master_client \
  --config "$CONFIG_PATH"

echo "[run_exp1_3] Experiment 1.3 completed successfully."