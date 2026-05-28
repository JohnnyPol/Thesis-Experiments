#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:-configs/experiments/exp1_3_resnet18_ee_3nodes_cifar10.yaml}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

export PYTHONPATH="$REPO_ROOT"

source "${REPO_ROOT}/scripts/run/distributed_common.sh"

echo "[healthcheck] config=$CONFIG_PATH"
check_workers_for_config "[healthcheck]" "$CONFIG_PATH"
