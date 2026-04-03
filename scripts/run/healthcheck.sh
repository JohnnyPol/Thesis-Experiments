#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:-configs/experiments/exp1_3_ee_homogeneous_2nodes.yaml}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

WORKER1_HOST="192.168.0.102"
WORKER1_PORT="9101"
WORKER2_HOST="192.168.0.104"
WORKER2_PORT="9102"

echo "[run_exp1_3] checking worker1 health..."
curl --fail --silent "http://${WORKER1_HOST}:${WORKER1_PORT}/health" >/dev/null
echo "[run_exp1_3] worker1 is reachable"

echo "[run_exp1_3] checking worker2 health..."
curl --fail --silent "http://${WORKER2_HOST}:${WORKER2_PORT}/health" >/dev/null
echo "[run_exp1_3] worker2 is reachable"