#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

export PYTHONPATH="$PROJECT_ROOT"

CONFIG_PATH="${PROJECT_ROOT}/configs/experiments/exp1_1_baseline_single_node.yaml"

python -m src.inference.single_node \
  --config "$CONFIG_PATH"